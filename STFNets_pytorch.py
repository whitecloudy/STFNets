import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
from sklearn.metrics import f1_score

# ==========================================
# Configuration & Constants
# ==========================================

BATCH_SIZE = 64
GEN_FFT_N = [16, 32, 64, 128]
GEN_FFT_STEP = GEN_FFT_N 
FILTER_LEN = [3, 3, 3, 3]
DILATION_LEN = [1, 2, 4, 8]

GEN_FFT_N2 = [12, 24, 48, 96]
SERIES_SIZE2 = 384
GEN_FFT_STEP2 = GEN_FFT_N2

GEN_C_OUT = 64
KEEP_PROB = 0.8

# Selection Logic
SELECT = 'wifi' # Default
if len(sys.argv) > 1:
    SELECT = sys.argv[1]

if SELECT == 'wifi':
    SERIES_SIZE = 512
    SENSOR_AXIS = 30
    SENSOR_NUM = 2
    OUT_DIM = 6
    ACT_DOMAIN = 'freq'
    FILTER_FLAG = False
    FREQ_CONV_FLAG = True
    ADAM_B1 = 0.9
    ADAM_B2 = 0.99
elif SELECT == 'hhar':
    SERIES_SIZE = 512
    SENSOR_AXIS = 3
    SENSOR_NUM = 2
    OUT_DIM = 6
    ACT_DOMAIN = 'time'
    FILTER_FLAG = True
    FREQ_CONV_FLAG = False
    ADAM_B1 = 0.5
    ADAM_B2 = 0.9
else:
    print("Select wifi or hhar")
    sys.exit(1)

DROP_FLAG = True
INPUT_COMPLEX_NORM_FLAG = True
CLIP_FLAG = False
ADAM_LR = 1e-4
GLOBAL_KERNEL_SIZE = 32
FILTER_EXP_SEL = 'linear_interp'
FILTER_INIT = 'real'
CONV_KERNEL_INIT = 'freq'
MERGE_INIT = 'zero'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# Layers & Model
# ==========================================

class ComplexLayerNorm(nn.Module):
    def __init__(self, c_size, epsilon=1e-4):
        super(ComplexLayerNorm, self).__init__()
        self.epsilon = epsilon
        self.c_size = c_size
        
        # Parameters initialized to match TF code
        self.beta_r = nn.Parameter(torch.zeros(1, 1, 1, c_size))
        self.beta_i = nn.Parameter(torch.zeros(1, 1, 1, c_size))
        self.gamma_rr = nn.Parameter(torch.full((1, 1, 1, c_size), 0.70710678118))
        self.gamma_ii = nn.Parameter(torch.full((1, 1, 1, c_size), 0.70710678118))
        self.gamma_ri = nn.Parameter(torch.zeros(1, 1, 1, c_size))

    def forward(self, in_r, in_i):
        # Input: [Batch, Time, Freq, Channels]
        # TF reduces over [1, 2, 3] -> global mean per sample
        r_mean = torch.mean(in_r, dim=[1, 2, 3], keepdim=True)
        i_mean = torch.mean(in_i, dim=[1, 2, 3], keepdim=True)
        
        r_center = in_r - r_mean
        i_center = in_i - i_mean
        
        conv_rr = torch.mean(r_center**2, dim=[1, 2, 3], keepdim=True) + self.epsilon
        conv_ii = torch.mean(i_center**2, dim=[1, 2, 3], keepdim=True) + self.epsilon
        conv_ri = torch.mean(r_center * i_center, dim=[1, 2, 3], keepdim=True) + self.epsilon
        
        tau = conv_rr + conv_ii
        delta = conv_rr * conv_ii - conv_ri**2
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)
        inverse_st = 1.0 / (s * t)
        
        Wrr = (conv_ii + s) * inverse_st
        Wii = (conv_rr + s) * inverse_st
        Wri = -conv_ri * inverse_st
        
        r_norm = Wrr * r_center + Wri * i_center
        i_norm = Wri * r_center + Wii * i_center
        
        out_r = self.gamma_rr * r_norm + self.gamma_ri * i_norm + self.beta_r
        out_i = self.gamma_ri * r_norm + self.gamma_ii * i_norm + self.beta_i
        
        return out_r, out_i

class STFLayer(nn.Module):
    def __init__(self, fft_list, f_step_list, kernel_len_list, dilation_len_list, c_in, c_out, 
                 out_fft_list=None, ser_size=SERIES_SIZE, pooling=False):
        super(STFLayer, self).__init__()
        self.fft_list = fft_list
        self.f_step_list = f_step_list
        self.kernel_len_list = kernel_len_list
        self.dilation_len_list = dilation_len_list
        self.c_in = c_in
        self.c_out = c_out
        self.pooling = pooling
        self.ser_size = ser_size
        
        if pooling:
            self.fft_n_list = out_fft_list
        else:
            self.fft_n_list = fft_list
            
        self.c_out_per_fft = c_out // len(self.fft_n_list)
        
        # 1. Patch Filter (FILTER_FLAG)
        if FILTER_FLAG:
            self.global_kernel_size = GLOBAL_KERNEL_SIZE
            # Base kernel for resizing
            self.patch_kernel_r = nn.Parameter(torch.randn(1, c_in * self.c_out_per_fft, 1, self.global_kernel_size // 2 + 1))
            self.patch_kernel_i = nn.Parameter(torch.randn(1, c_in * self.c_out_per_fft, 1, self.global_kernel_size // 2 + 1))
            nn.init.xavier_uniform_(self.patch_kernel_r)
            nn.init.xavier_uniform_(self.patch_kernel_i)
            
            self.patch_bias_r = nn.Parameter(torch.zeros(self.c_out_per_fft * len(self.fft_list)))

        # 2. Spectral Filter (FREQ_CONV_FLAG)
        if FREQ_CONV_FLAG:
            basic_len = kernel_len_list[0]
            # Weight: [c_out_per_fft, c_in, 1, basic_len]
            self.conv_kernel_r = nn.Parameter(torch.randn(self.c_out_per_fft, c_in, 1, basic_len))
            self.conv_kernel_i = nn.Parameter(torch.randn(self.c_out_per_fft, c_in, 1, basic_len))
            nn.init.xavier_uniform_(self.conv_kernel_r)
            nn.init.xavier_uniform_(self.conv_kernel_i)
            
        # 3. Merge Kernels
        self.merge_kernels = nn.ParameterDict()
        self.merge_biases = nn.ParameterDict()
        for fft_n in self.fft_n_list:
            for tar_fft_n in self.fft_n_list:
                if tar_fft_n > fft_n:
                    ratio = tar_fft_n // fft_n
                    key = f"{fft_n}_{tar_fft_n}"
                    self.merge_kernels[key + '_r'] = nn.Parameter(torch.zeros(ratio, ratio))
                    self.merge_kernels[key + '_i'] = nn.Parameter(torch.zeros(ratio, ratio))
                    self.merge_biases[key + '_r'] = nn.Parameter(torch.zeros(ratio))
                    self.merge_biases[key + '_i'] = nn.Parameter(torch.zeros(ratio))
        
        if INPUT_COMPLEX_NORM_FLAG:
            self.norms = nn.ModuleList([ComplexLayerNorm(c_in) for _ in range(len(self.fft_n_list))])

    def forward(self, x):
        # x: [Batch, c_in, time]
        
        patch_fft_list = [0.] * len(self.fft_n_list)
        
        # STFT and Merge
        for i, fft_n in enumerate(self.fft_n_list):
            if self.pooling:
                frame_len = self.fft_list[i]
                frame_step = self.fft_list[i]
            else:
                frame_len = fft_n
                frame_step = fft_n
            
            batch, c_in, time_len = x.shape
            x_flat = x.reshape(batch * c_in, time_len)
            
            # Rectangular window
            window = torch.ones(frame_len, device=x.device)
            stft_out = torch.stft(x_flat, n_fft=frame_len, hop_length=frame_step, win_length=frame_len, 
                                  window=window, center=False, return_complex=False)
            # stft_out: [Batch*c_in, freq, frames, 2]
            
            stft_out = stft_out.view(batch, c_in, stft_out.size(1), stft_out.size(2), 2)
            # Permute to [Batch, frames, freq, c_in]
            patch_fft_r = stft_out[..., 0].permute(0, 3, 2, 1) 
            patch_fft_i = stft_out[..., 1].permute(0, 3, 2, 1)
            
            patch_fft = torch.complex(patch_fft_r, patch_fft_i) # [B, T, F, C]
            
            for j, tar_fft_n in enumerate(self.fft_n_list):
                if tar_fft_n < fft_n:
                    continue
                elif tar_fft_n == fft_n:
                    if isinstance(patch_fft_list[j], float):
                        patch_fft_list[j] = patch_fft
                    else:
                        patch_fft_list[j] = patch_fft_list[j] + patch_fft
                else:
                    # Merge logic
                    ratio = tar_fft_n // fft_n
                    B, T, F, C = patch_fft.shape
                    T_new = T // ratio
                    patch_fft_mod = patch_fft.view(B, T_new, ratio, F, C)
                    patch_fft_mod = patch_fft_mod.permute(0, 1, 3, 4, 2) # [B, T_new, F, C, ratio]
                    
                    key = f"{fft_n}_{tar_fft_n}"
                    k = torch.complex(self.merge_kernels[key + '_r'], self.merge_kernels[key + '_i'])
                    b = torch.complex(self.merge_biases[key + '_r'], self.merge_biases[key + '_i'])
                    
                    patch_atten = torch.matmul(patch_fft_mod, k)
                    patch_atten = patch_atten + b
                    patch_atten = torch.abs(patch_atten)
                    patch_atten = F.softmax(patch_atten, dim=-1)
                    patch_atten = torch.complex(patch_atten, torch.zeros_like(patch_atten))
                    
                    patch_merged = torch.sum(patch_fft_mod * patch_atten, dim=-1) # [B, T_new, F, C]
                    patch_merged = patch_merged * float(ratio)
                    
                    # Zero Interp (Pad frequency)
                    pm_shape = patch_merged.shape
                    zeros = torch.zeros(pm_shape[0], pm_shape[1], pm_shape[2], ratio-1, pm_shape[3], 
                                        device=x.device, dtype=patch_merged.dtype)
                    pm_exp = patch_merged.unsqueeze(3)
                    pm_concat = torch.cat([pm_exp, zeros], dim=3)
                    pm_reshaped = pm_concat.view(pm_shape[0], pm_shape[1], pm_shape[2]*ratio, pm_shape[3])
                    
                    target_F = tar_fft_n // 2 + 1
                    patch_final = pm_reshaped[:, :, :target_F, :]
                    
                    if isinstance(patch_fft_list[j], float):
                        patch_fft_list[j] = patch_final
                    else:
                        patch_fft_list[j] = patch_fft_list[j] + patch_final

        # Convolution and Output Generation
        patch_time_list = []
        for i, fft_n in enumerate(self.fft_n_list):
            patch_fft = patch_fft_list[i] # [B, T, F, C]
            
            if INPUT_COMPLEX_NORM_FLAG:
                p_r = patch_fft.real
                p_i = patch_fft.imag
                p_r, p_i = self.norms[i](p_r, p_i)
                patch_fft = torch.complex(p_r, p_i)
            
            if FREQ_CONV_FLAG:
                k_len = self.kernel_len_list[i]
                d_len = self.dilation_len_list[i]
                pad_total = k_len * d_len - d_len
                pad_l = pad_total // 2
                pad_r = pad_total - pad_l
                
                p_r = patch_fft.real
                p_i = patch_fft.imag
                
                # Manual Padding (Reflection/Reverse)
                left_pad_r = torch.flip(p_r[:, :, 1:1+pad_l, :], dims=[2])
                right_pad_r = torch.flip(p_r[:, :, -1-pad_r:-1, :], dims=[2])
                p_r_padded = torch.cat([left_pad_r, p_r, right_pad_r], dim=2)
                
                left_pad_i = torch.flip(p_i[:, :, 1:1+pad_l, :], dims=[2])
                right_pad_i = torch.flip(p_i[:, :, -1-pad_r:-1, :], dims=[2])
                p_i_padded = torch.cat([-left_pad_i, p_i, -right_pad_i], dim=2)
                
                p_r_in = p_r_padded.permute(0, 3, 1, 2) # [B, C, T, F]
                p_i_in = p_i_padded.permute(0, 3, 1, 2)
                
                # Kernel Resize
                curr_k_r = F.interpolate(self.conv_kernel_r, size=(1, k_len), mode='bilinear', align_corners=True)
                curr_k_i = F.interpolate(self.conv_kernel_i, size=(1, k_len), mode='bilinear', align_corners=True)
                
                dilation = d_len
                
                out_rr = F.conv2d(p_r_in, curr_k_r, stride=1, dilation=(1, dilation))
                out_ri = F.conv2d(p_r_in, curr_k_i, stride=1, dilation=(1, dilation))
                out_ir = F.conv2d(p_i_in, curr_k_r, stride=1, dilation=(1, dilation))
                out_ii = F.conv2d(p_i_in, curr_k_i, stride=1, dilation=(1, dilation))
                
                p_out_r = out_rr - out_ii
                p_out_i = out_ri + out_ir
                
                p_out_r = p_out_r.permute(0, 2, 3, 1) # [B, T, F, C]
                p_out_i = p_out_i.permute(0, 2, 3, 1)
                
                patch_out_r = p_out_r
                patch_out_i = p_out_i
            else:
                patch_out_r = patch_fft.real
                patch_out_i = patch_fft.imag

            if FILTER_FLAG:
                target_F = fft_n // 2 + 1
                k_r = F.interpolate(self.patch_kernel_r, size=(1, target_F), mode='bilinear', align_corners=True)
                k_i = F.interpolate(self.patch_kernel_i, size=(1, target_F), mode='bilinear', align_corners=True)
                
                k_r = k_r.view(1, 1, self.c_in, self.c_out_per_fft, target_F).permute(0, 1, 4, 2, 3)
                k_i = k_i.view(1, 1, self.c_in, self.c_out_per_fft, target_F).permute(0, 1, 4, 2, 3)
                
                p_r = patch_out_r.unsqueeze(4)
                p_i = patch_out_i.unsqueeze(4)
                
                real = p_r * k_r - p_i * k_i
                imag = p_r * k_i + p_i * k_r
                
                patch_out_r = torch.sum(real, dim=3)
                patch_out_i = torch.sum(imag, dim=3)
                
                patch_out_r = patch_out_r + self.patch_bias_r[i*self.c_out_per_fft : (i+1)*self.c_out_per_fft]

            if ACT_DOMAIN == 'freq':
                patch_out_r = F.leaky_relu(patch_out_r)
                patch_out_i = F.leaky_relu(patch_out_i)
                
            # ISTFT
            p_complex = torch.complex(patch_out_r, patch_out_i)
            p_complex = p_complex.permute(0, 3, 2, 1) # [B, C, F, T]
            
            B, C, F, T = p_complex.shape
            p_flat = p_complex.reshape(B*C, F, T)
            
            window = torch.ones(fft_n, device=x.device)
            time_out = torch.istft(p_flat, n_fft=fft_n, hop_length=fft_n, win_length=fft_n, window=window, center=False)
            
            time_out = time_out.view(B, C, -1) # [B, C, time]
            time_out = time_out.permute(0, 2, 1) # [B, time, C]
            
            patch_time_list.append(time_out)
            
        patch_time_final = torch.cat(patch_time_list, dim=2) # [B, time, total_C]
        patch_time_final = patch_time_final.permute(0, 2, 1) # [B, total_C, time]
        
        if ACT_DOMAIN == 'time':
            patch_time_final = F.leaky_relu(patch_time_final)
            
        return patch_time_final

class STFNet(nn.Module):
    def __init__(self):
        super(STFNet, self).__init__()
        
        self.acc_layer1 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, SENSOR_AXIS, GEN_C_OUT)
        self.acc_dropout1 = nn.Dropout(1 - KEEP_PROB)
        
        self.acc_layer2 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT)
        self.acc_dropout2 = nn.Dropout(1 - KEEP_PROB)
        
        self.acc_layer3 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT//2)
        self.acc_dropout3 = nn.Dropout(1 - KEEP_PROB)
        
        self.gyro_layer1 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, SENSOR_AXIS, GEN_C_OUT)
        self.gyro_dropout1 = nn.Dropout(1 - KEEP_PROB)
        
        self.gyro_layer2 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT)
        self.gyro_dropout2 = nn.Dropout(1 - KEEP_PROB)
        
        self.gyro_layer3 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT//2)
        self.gyro_dropout3 = nn.Dropout(1 - KEEP_PROB)
        
        self.sensor_layer1 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT,
                                      out_fft_list=GEN_FFT_N2, ser_size=SERIES_SIZE2, pooling=True)
        self.sensor_dropout1 = nn.Dropout(1 - KEEP_PROB)
        
        self.sensor_layer2 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT,
                                      ser_size=SERIES_SIZE2)
        self.sensor_dropout2 = nn.Dropout(1 - KEEP_PROB)
        
        self.sensor_layer3 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT,
                                      ser_size=SERIES_SIZE2)
        self.sensor_dropout3 = nn.Dropout(1 - KEEP_PROB)
        
        self.fc = nn.Linear(GEN_C_OUT, OUT_DIM)

    def forward(self, x):
        # x: [B, SERIES_SIZE, SENSOR_AXIS*SENSOR_NUM]
        acc_in = x[:, :, :SENSOR_AXIS]
        gyro_in = x[:, :, SENSOR_AXIS:]
        
        acc_in = acc_in.permute(0, 2, 1)
        gyro_in = gyro_in.permute(0, 2, 1)
        
        a1 = self.acc_dropout1(self.acc_layer1(acc_in))
        a2 = self.acc_dropout2(self.acc_layer2(a1))
        a3 = self.acc_dropout3(self.acc_layer3(a2))
        
        g1 = self.gyro_dropout1(self.gyro_layer1(gyro_in))
        g2 = self.gyro_dropout2(self.gyro_layer2(g1))
        g3 = self.gyro_dropout3(self.gyro_layer3(g2))
        
        s_in = torch.cat([a3, g3], dim=1)
        
        s1 = self.sensor_dropout1(self.sensor_layer1(s_in))
        s2 = self.sensor_dropout2(self.sensor_layer2(s1))
        s3 = self.sensor_dropout3(self.sensor_layer3(s2))
        
        out = torch.mean(s3, dim=2)
        logits = self.fc(out)
        return logits

# ==========================================
# Data Loading & Training Loop
# ==========================================

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size, length, channels, classes):
        self.size = size
        self.length = length
        self.channels = channels
        self.classes = classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random data for demonstration
        x = torch.randn(self.length, self.channels)
        y = torch.zeros(self.classes)
        y[torch.randint(0, self.classes, (1,))] = 1
        return x, y

if __name__ == "__main__":
    # Initialize Model
    model = STFNet().to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR, betas=(ADAM_B1, ADAM_B2))
    criterion = nn.CrossEntropyLoss()
    
    # Data Loaders (Using dummy data as TFRecords are not available)
    # In a real scenario, implement a Dataset that reads the CSVs or TFRecords
    train_dataset = DummyDataset(1000, SERIES_SIZE, SENSOR_AXIS*SENSOR_NUM, OUT_DIM)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    eval_dataset = DummyDataset(200, SERIES_SIZE, SENSOR_AXIS*SENSOR_NUM, OUT_DIM)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    TOTAL_ITER_NUM = 1000
    
    print("Starting training...")
    model.train()
    
    iter_count = 0
    while iter_count < TOTAL_ITER_NUM:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Target is one-hot in TF code, CrossEntropyLoss expects class indices
            target_indices = torch.argmax(target, dim=1)
            loss = criterion(output, target_indices)
            
            # L2 Regularization
            l2_reg = 0.0
            for name, param in model.named_parameters():
                if 'angle' not in name:
                    l2_reg += torch.norm(param)
            loss += 5e-4 * l2_reg
            
            loss.backward()
            
            if CLIP_FLAG:
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.3)
                
            optimizer.step()
            
            if iter_count % 10 == 0:
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target_indices.view_as(pred)).sum().item()
                acc = correct / len(data)
                print(f"Iter {iter_count}: Loss {loss.item():.4f}, Acc {acc:.4f}")
            
            iter_count += 1
            if iter_count >= TOTAL_ITER_NUM:
                break
