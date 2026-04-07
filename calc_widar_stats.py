import os
import torch
import click
import tqdm
from widar_dataset import WiDARDataset
from torch.utils.data import DataLoader

@click.command()
@click.option('--data_dir', type=str, default='widar_data', help='Path to WiDAR dataset')
@click.option('--series_size', type=int, default=256, help='Target time steps (SERIES_SIZE)')
@click.option('--batch_size', type=int, default=128, help='Batch size for processing')
def main(data_dir, series_size, batch_size):
    # STFNets_pytorch.py 와 동일한 필터링 조건 적용
    must_have = [f'-gesture{i}-' for i in range(4)] + ['-gesture17-', '-gesture18-']
    must_not_have = ["-user5-"]

    print(f"Loading dataset from: {data_dir}...")
    dataset = WiDARDataset(
        dir_path=data_dir, 
        target_size=series_size, 
        min_data_len=0, 
        split_ratio=1.0, 
        must_have=must_have, 
        must_not_have=must_not_have
    )

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    
    # 채널(Feature)별 평균을 구하기 위한 변수
    channel_sum = None
    channel_sq_sum = None
    num_channels = 0

    print("Calculating statistics (Mean and Std)...")
    with torch.no_grad():
        for data, _ in tqdm.tqdm(loader, desc="Processing"):
            # data shape: [Batch, Time, Channels]
            if channel_sum is None:
                num_channels = data.shape[-1]
                channel_sum = torch.zeros(num_channels, dtype=torch.float64)
                channel_sq_sum = torch.zeros(num_channels, dtype=torch.float64)
            
            # 전체(Global) 계산용
            total_sum += data.sum().item()
            total_sq_sum += (data ** 2).sum().item()
            total_count += data.numel()
            
            # 채널(Per-channel) 계산용
            channel_sum += data.sum(dim=[0, 1]).double()
            channel_sq_sum += (data ** 2).sum(dim=[0, 1]).double()

    if total_count == 0:
        print("데이터를 찾을 수 없습니다.")
        return

    overall_mean = total_sum / total_count
    overall_std = (total_sq_sum / total_count - overall_mean ** 2) ** 0.5
    
    channel_count = total_count / num_channels
    channel_mean = channel_sum / channel_count
    channel_std = torch.sqrt(channel_sq_sum / channel_count - channel_mean ** 2)

    print("=" * 50)
    print(f"Total elements : {total_count}")
    print(f"Overall Mean   : {overall_mean:.6f}")
    print(f"Overall Std    : {overall_std:.6f}")
    print("-" * 50)
    print(f"Channel-wise Mean (First 5 of {num_channels} channels):")
    print(channel_mean[:5].numpy())
    print(f"Channel-wise Std  (First 5 of {num_channels} channels):")
    print(channel_std[:5].numpy())
    print("=" * 50)
    
    # 계산된 채널별 통계값을 파일로 저장해 향후 데이터 정규화에 사용할 수 있습니다.
    torch.save({'mean': channel_mean, 'std': channel_std}, 'widar_stats.pt')
    print("Saved statistics to 'widar_stats.pt'")

if __name__ == '__main__':
    main()