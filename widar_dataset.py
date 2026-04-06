import numpy as np
import torch
from torch.utils.data import Dataset
import os
import fnmatch
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from tqdm import tqdm
import multiprocessing as mp



def find_npz_files(dir_path : str, recursive=True):
    file_paths = []
    if recursive:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.npz'):
                    file_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(dir_path):
            if file.endswith('.npz'):
                file_paths.append(os.path.join(dir_path, file))
    return file_paths


# Porting from Widar3.0 "get_doppler_spectrum.m" MATLAB function
# Input : csi_data: (T, 30*rx_acnt) complex
# Output : conj_mult_f: (T, 30*(rx_acnt-1)) complex
def process_csi_section(csi_data, rx_acnt):
    m = np.mean(np.abs(csi_data), axis=0)
    v = np.std(np.abs(csi_data), axis=0, ddof=0)
    ratio = m / (v + 1e-12)
    ratio_group = ratio.reshape(30, rx_acnt).mean(axis=0)
    idx = int(np.argmax(ratio_group)) + 1
    
    T, C = csi_data.shape
    assert C == 30 * rx_acnt

    # 3.4 Amp Adjust (IndoTrack)
    csi_data_adj = np.zeros_like(csi_data, dtype=np.complex128)
    # csi_data_ref = np.tile(csi_data[:, (idx-1)*30:idx*30], (1, rx_acnt))
    # csi_data_ref_adj = np.zeros_like(csi_data_ref, dtype=np.complex128)

    alpha_sum = 0.0
    for jj in range(30 * rx_acnt):
        amp = np.abs(csi_data[:, jj])
        nonzero_amp = amp[amp != 0]
        if nonzero_amp.size == 0:
            alpha = 0.0
        else:
            alpha = np.min(nonzero_amp)
        alpha_sum += alpha
        x = csi_data[:, jj]
        csi_data_adj[:, jj] = np.abs(np.abs(x) - alpha) * np.exp(1j * np.angle(x))
    
    return csi_data_adj

    beta = 1000.0 * alpha_sum / (30.0 * rx_acnt)
    for jj in range(30 * rx_acnt):
        x = csi_data_ref[:, jj]
        csi_data_ref_adj[:, jj] = (np.abs(x) + beta) * np.exp(1j * np.angle(x))

    # 3.5 Conj Mult
    conj_mult = csi_data_adj * np.conj(csi_data_ref_adj)

    # 기준 idx 그룹 열 제외 (MATLAB은 1:30*(idx-1), 30*idx+1:90)
    s1 = conj_mult[:, :30*(idx-1)]
    s2 = conj_mult[:, 30*idx:]
    conj_mult = np.concatenate((s1, s2), axis=1)

    # 3.6 Filter out static + high freq
    fs = 1000.0
    low_high_cut = 2.0
    low_low_cut = 60.0

    # high-pass(>2Hz), low-pass(<60Hz) 이중 필터
    b_lo, a_lo = butter(6, low_low_cut / (fs / 2.0), btype='low')
    b_hi, a_hi = butter(3, low_high_cut / (fs / 2.0), btype='high')

    # 각 채널별 filtfilt
    conj_mult_f = np.zeros_like(conj_mult, dtype=np.complex128)
    for jj in range(conj_mult.shape[1]):
        y = conj_mult[:, jj]
        # 실수/허수 분리해서 처리 (MATLAB filter은 복소 처리 가능)
        yr = filtfilt(b_lo, a_lo, y.real)
        yr = filtfilt(b_hi, a_hi, yr)
        yi = filtfilt(b_lo, a_lo, y.imag)
        yi = filtfilt(b_hi, a_hi, yi)
        conj_mult_f[:, jj] = yr + 1j * yi

    return conj_mult_f

def fit_data_size(data, target_size):
    T, C = data.shape
    if T > target_size:
        return data[:target_size, :]
    elif T < target_size:
        padding = np.zeros((target_size - T, C), dtype=data.dtype)
        return np.concatenate((data, padding), axis=0)
    else:
        return data
    
# def interpolate_data(data, target_size):
#     T, C = data.shape
#     if T == target_size:
#         return data
#     x_old = np.arange(T)
#     x_new = np.linspace(0, T-1, target_size)
#     interpolated_data = np.zeros((target_size, C), dtype=data.dtype)
#     for c in range(C):
#         interpolated_data[:, c] = np.interp(x_new, x_old, data[:, c])
#     return interpolated_data

def interpolate_data(data, target_size):
    T, C = data.shape
    if T == target_size:
        return data
        
    # 0부터 T-1까지 target_size개 만큼 균일하게 나눈 후, 
    # 가장 가까운 정수 인덱스로 반올림(round)합니다.
    x_new = np.linspace(0, T - 1, target_size)
    nearest_indices = np.round(x_new).astype(int)
    
    # 계산된 인덱스를 사용해 원본 데이터에서 값을 그대로 가져옵니다.
    interpolated_data = data[nearest_indices, :]
    
    return interpolated_data


def preprocess_csi_gen(csi, noise_sigma, target_size=512):
    process_csi = csi/noise_sigma

    # Subsampling Process
    # process_csi = fit_data_size(process_csi, target_size=2048)
    process_csi = interpolate_data(process_csi, target_size=target_size)

    process_csi = process_csi_section(process_csi, rx_acnt=3)

    process_csi = np.abs(process_csi)  # (time steps, features)
    process_csi = process_csi.astype(np.float32)

    process_csi = process_csi - np.min(process_csi, axis=0)

    return process_csi

def _preprocess_worker(args):
    file_path, min_data_len, target_size = args
    data = np.load(file_path)
    csi = np.array(data['csi_data'])
    if csi.shape[0] < min_data_len:
        return file_path, None
    noise_sigma = np.array(data['noise_array'])
    processed_csi = preprocess_csi_gen(csi, noise_sigma, target_size=target_size)
    return file_path, processed_csi

class WiDARDataset(Dataset):
    def __init__(self, dir_path, target_size=512, min_data_len=1024, transform=None, split_ratio=0.8, split_seed=42, must_have=None, must_not_have=None):
        self.dir_path = dir_path
        self.target_size = target_size
        self.min_data_len = min_data_len
        self.transform = transform
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.must_have = must_have
        self.must_not_have = must_not_have
        self.file_paths = []
        self.max_T = 0

        if isinstance(dir_path, str):
            # find .npz files in recursive way
            self.file_paths = find_npz_files(dir_path)
        elif isinstance(dir_path, list):
            for path in dir_path:
                self.file_paths.extend(find_npz_files(path))
        else:
            raise ValueError("dir_path should be a string or a list of strings.")
        
        self.file_paths = self.filter_files(self.file_paths)

        temp_results = {}
        worker_args = [(fp, self.min_data_len, self.target_size) for fp in self.file_paths]
        print("Filtering and preprocessing data using multiprocessing...")
        valid_file_paths = []
        self.max_T = 0
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for fp, p_csi in tqdm(pool.imap_unordered(_preprocess_worker, worker_args), total=len(worker_args), desc="Processing"):
                if p_csi is not None:
                    temp_results[fp] = p_csi
                    valid_file_paths.append(fp)
                    if p_csi.shape[0] > self.max_T:
                        self.max_T = p_csi.shape[0]
        print(f"Max time steps after preprocessing: {self.max_T}")
        self.max_T = 512 # 고정된 시퀀스 길이로 설정 (모델 입력 크기에 맞게)

        self.file_paths = valid_file_paths
        print(f"Total files after filtering: {len(self.file_paths)}")

        print("Stacking preprocessed data into a single shared tensor...")
        self.path_to_idx = {fp: i for i, fp in enumerate(self.file_paths)}
        
        # 첫 번째 데이터로 feature 크기 확인
        first_csi = next(iter(temp_results.values()))
        num_features = first_csi.shape[1]
        
        # 전체 데이터를 담을 단일 텐서 메모리 할당 (DataLoader 워커 간 메모리 공유 목적)
        self.preprocessed_data = torch.zeros((len(self.file_paths), self.max_T, num_features), dtype=torch.float32)
        for fp, p_csi in temp_results.items():
            idx = self.path_to_idx[fp]
            T = p_csi.shape[0]
            T = min(T, self.max_T)  # 시퀀스 길이 제한
            self.preprocessed_data[idx, :T, :] = torch.tensor(p_csi, dtype=torch.float32)[:T, :]
            
        del temp_results # 복제 방지를 위해 임시 딕셔너리 메모리 해제

        self.file_paths, self.non_selected_paths = self.split_files(self.file_paths)

    def flip_splits(self):
        tmp = self.file_paths
        self.file_paths = self.non_selected_paths
        self.non_selected_paths = tmp
        
    def __len__(self):
        return len(self.file_paths)
    
    def filter_files(self, file_paths):
        filtered_paths = []
        
        must_have = [self.must_have] if isinstance(self.must_have, str) else self.must_have
        must_not_have = [self.must_not_have] if isinstance(self.must_not_have, str) else self.must_not_have
        for path in file_paths:
            os.path.normpath(path)
            if must_have and not any(fnmatch.fnmatch(path, f'*{feature}*') for feature in must_have):
                continue
            if must_not_have and any(fnmatch.fnmatch(path, f'*{feature}*') for feature in must_not_have):
                continue
            filtered_paths.append(path)
        return filtered_paths
    
    def split_files(self, file_paths):
        file_paths = sorted(file_paths)
        np.random.seed(self.split_seed)
        np.random.shuffle(file_paths)
        split_index = int(self.split_ratio * len(file_paths))
        return file_paths[:split_index], file_paths[split_index:]
    
    def get_gesture_num_from_path(self, file_path):
        # Extract gesture label from file path (assuming format includes '-gestureX-')
        base_name = os.path.basename(file_path)
        parts = base_name.split('-')
        for part in parts:
            if part.startswith('gesture'):
                return int(part.replace('gesture', ''))
        return None  # Return None if no gesture label is found
    
    def get_label_from_gesture_num(self, gesture_num):
        if gesture_num is None:
            return None
        
        label_list = [0, 1, 2, 3, 17, 18]  # Example mapping of gesture numbers to labels
        label = np.zeros(len(label_list), dtype=np.float32)
        if gesture_num in label_list:
            label[label_list.index(gesture_num)] = 1.0
        return label
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        real_idx = self.path_to_idx[file_path]
        process_csi = self.preprocessed_data[real_idx]

        # Get label from file path
        gesture_num = self.get_gesture_num_from_path(file_path)
        label = self.get_label_from_gesture_num(gesture_num)

        return process_csi, label
        # return csi.astype(np.float32)

if __name__ == "__main__":
    must_have=[f'-gesture{i}-' for i in range(4)]+['-gesture17-','-gesture18-']
    dataset = WiDARDataset('widar_data', must_have=must_have)
    len_list = []
    more_than_1536 = 0
    less_than_1024 = 0
    print(f"Dataset size: {len(dataset)}")
    for data in tqdm(dataset, desc="Processing files"): # type: ignore
        len_list.append(data.shape[0])
        if data.shape[0] > 1536:
            more_than_1536 += 1
        if data.shape[0] < 1024:
            less_than_1024 += 1
    print(f"Files with more than 1536 time steps: {more_than_1536}")
    print(f"Files with less than 1024 time steps: {less_than_1024}")
    
    print(f"Min Lengths: {min(len_list)}")
    print(f"Max Lengths: {max(len_list)}")
    print(f"Mean Lengths: {np.mean(len_list)}")
    print(f"Median Lengths: {np.median(len_list)}")
    print(f"Std Lengths: {np.std(len_list)}")