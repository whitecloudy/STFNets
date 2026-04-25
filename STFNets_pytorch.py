import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
import math
import os
import sys
from sklearn.metrics import f1_score
from widar_dataset import WiDARDataset, SyntheticWiDARDataset
import tqdm
import pandas as pd
import json

# ==========================================
# Configuration & Constants
# ==========================================

BATCH_SIZE = 128

GEN_C_OUT = 64
KEEP_PROB = 0.8

# Selection Logic
SELECT = 'wifi' # Default

if SELECT == 'wifi':
    SERIES_SIZE = 256
    SENSOR_AXIS = 30
    SENSOR_NUM = 3
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

GENERAL_FFT_STEP = [32, 16, 8, 4]
# GENERAL_FFT_STEP = [SERIES_SIZE//16, SERIES_SIZE//32, SERIES_SIZE//64, SERIES_SIZE//128]

GEN_FFT_N = [SERIES_SIZE//n for n in GENERAL_FFT_STEP]
GEN_FFT_STEP = GEN_FFT_N 
FILTER_LEN = [3, 3, 3, 3]
DILATION_LEN = [1, 2, 4, 8]

SERIES_SIZE2 = SERIES_SIZE // 4 * 3
GEN_FFT_N2 = [SERIES_SIZE2//n for n in GENERAL_FFT_STEP]
# GEN_FFT_N2 = [12, 24, 48, 96]
# SERIES_SIZE2 = 384
GEN_FFT_STEP2 = GEN_FFT_N2

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
        self.gamma_rr = nn.Parameter(torch.full((1, 1, 1, c_size), 1/math.sqrt(2)))
        self.gamma_ii = nn.Parameter(torch.full((1, 1, 1, c_size), 1/math.sqrt(2)))
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
            
            if FILTER_INIT == 'real':
                # Initialize in time domain (matches complex_glorot_uniform with FILTER_INIT='real')
                self.patch_kernel_time = nn.Parameter(torch.randn(1, c_in * self.c_out_per_fft, 1, self.global_kernel_size))
                nn.init.xavier_uniform_(self.patch_kernel_time)
            else:
                # Base kernel for resizing (Frequency domain init)
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
        patch_mask_list = [[] for _ in range(len(self.fft_n_list))]
        
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
            
            if self.pooling:
                patch_fft = patch_fft[:, :, :fft_n // 2 + 1, :]

            for j, tar_fft_n in enumerate(self.fft_n_list):
                if tar_fft_n < fft_n:
                    continue
                elif tar_fft_n == fft_n:
                    patch_mask = torch.ones_like(patch_fft)
                    for exist_mask in patch_mask_list[j]:
                        # print(exist_mask.shape, patch_mask.shape)
                        patch_mask = patch_mask - exist_mask
                    
                    if isinstance(patch_fft_list[j], float):
                        patch_fft_list[j] = patch_fft * patch_mask
                    else:
                        patch_fft_list[j] = patch_fft_list[j] + patch_fft * patch_mask
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
                    patch_atten = Func.softmax(patch_atten, dim=-1)
                    patch_atten = torch.complex(patch_atten, torch.zeros_like(patch_atten))
                    
                    patch_merged = torch.sum(patch_fft_mod * patch_atten, dim=-1) # [B, T_new, F, C]
                    patch_merged = patch_merged * float(ratio)
                    
                    target_F = tar_fft_n // 2 + 1

                    def zero_interp(in_tensor, ratio, target_F):
                        pm_shape = in_tensor.shape
                        zeros = torch.zeros(pm_shape[0], pm_shape[1], pm_shape[2], ratio-1, pm_shape[3], 
                                            device=in_tensor.device, dtype=in_tensor.dtype)
                        pm_exp = in_tensor.unsqueeze(3)
                        pm_concat = torch.cat([pm_exp, zeros], dim=3)
                        pm_reshaped = pm_concat.view(pm_shape[0], pm_shape[1], pm_shape[2]*ratio, pm_shape[3])
                        return pm_reshaped[:, :, :target_F, :]
                    
                    patch_mask = zero_interp(torch.ones_like(patch_merged), ratio, target_F)
                    for exist_mask in patch_mask_list[j]:
                        patch_mask = patch_mask - exist_mask
                    patch_mask_list[j].append(patch_mask)
                    
                    patch_final = zero_interp(patch_merged, ratio, target_F)
                    
                    if isinstance(patch_fft_list[j], float):
                        patch_fft_list[j] = patch_final * patch_mask
                    else:
                        patch_fft_list[j] = patch_fft_list[j] + patch_final * patch_mask

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
                curr_k_r = Func.interpolate(self.conv_kernel_r, size=(1, k_len), mode='bilinear', align_corners=True)
                curr_k_i = Func.interpolate(self.conv_kernel_i, size=(1, k_len), mode='bilinear', align_corners=True)
                
                dilation = d_len
                
                out_rr = Func.conv2d(p_r_in, curr_k_r, stride=1, dilation=(1, dilation))
                out_ri = Func.conv2d(p_r_in, curr_k_i, stride=1, dilation=(1, dilation))
                out_ir = Func.conv2d(p_i_in, curr_k_r, stride=1, dilation=(1, dilation))
                out_ii = Func.conv2d(p_i_in, curr_k_i, stride=1, dilation=(1, dilation))
                
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
                
                if FILTER_INIT == 'real':
                    # Compute FFT of time-domain kernel
                    k_time = self.patch_kernel_time.squeeze(2) # [1, Channels, Time]
                    k_complex = torch.fft.rfft(k_time, n=self.global_kernel_size, dim=-1) # [1, Channels, Freq]
                    base_k_r = k_complex.real.unsqueeze(2) # [1, Channels, 1, Freq]
                    base_k_i = k_complex.imag.unsqueeze(2)
                else:
                    base_k_r = self.patch_kernel_r
                    base_k_i = self.patch_kernel_i

                k_r = Func.interpolate(base_k_r, size=(1, target_F), mode='bilinear', align_corners=True)
                k_i = Func.interpolate(base_k_i, size=(1, target_F), mode='bilinear', align_corners=True)
                
                k_r = k_r.view(1, 1, self.c_in, self.c_out_per_fft, target_F).permute(0, 1, 4, 2, 3)
                k_i = k_i.view(1, 1, self.c_in, self.c_out_per_fft, target_F).permute(0, 1, 4, 2, 3)
                
                p_r = patch_out_r.unsqueeze(4)
                p_i = patch_out_i.unsqueeze(4)
                
                real = p_r * k_r - p_i * k_i
                imag = p_r * k_i + p_i * k_r
                
                patch_out_r = torch.sum(real, dim=3)
                patch_out_i = torch.sum(imag, dim=3)

            if ACT_DOMAIN == 'freq':
                patch_out_r = Func.leaky_relu(patch_out_r)
                patch_out_i = Func.leaky_relu(patch_out_i)
                
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
        
        if FILTER_FLAG:
            patch_time_final = patch_time_final + self.patch_bias_r.view(1, -1, 1)
        
        if ACT_DOMAIN == 'time':
            patch_time_final = Func.leaky_relu(patch_time_final)
            
        return patch_time_final

class SpatialDropout(nn.Module):
    def __init__(self, p):
        super(SpatialDropout, self).__init__()
        self.dropout = nn.Dropout2d(p)
    
    def forward(self, x):
        x = x.unsqueeze(-1)    # [B, C, T] -> [B, C, T, 1]
        x = self.dropout(x)
        x = x.squeeze(-1)      # [B, C, T, 1] -> [B, C, T]
        return x
from collections import OrderedDict

class STFNet(nn.Module):
    def __init__(self, input_size=SERIES_SIZE, sensor_axis=SENSOR_AXIS, sensor_num=SENSOR_NUM, out_dim=OUT_DIM):
        super(STFNet, self).__init__()
        self.input_size = input_size
        self.sensor_axis = sensor_axis
        self.sensor_num = sensor_num
        self.out_dim = out_dim

        if DROP_FLAG:
            self.dropout = SpatialDropout(1 - KEEP_PROB)
        else:
            self.dropout = nn.Identity()

        self.acc_layers = nn.ModuleList()
        
        for s in range(self.sensor_num):
            self.acc_layers.append(
                nn.Sequential(OrderedDict([
                    (f'acc_layer1_s{s}', STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, self.sensor_axis, GEN_C_OUT, ser_size=self.input_size)),
                    (f'dropout1_s{s}', self.dropout),
                    (f'acc_layer2_s{s}', STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT, ser_size=self.input_size)),
                    (f'dropout2_s{s}', self.dropout),
                    (f'acc_layer3_s{s}', STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT//2, ser_size=self.input_size)),
                    (f'dropout3_s{s}', self.dropout)
                ]))
            )

        GEN_C_OUT_MERGE = (GEN_C_OUT//2) * self.sensor_num

        # self.acc_layer1 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, SENSOR_AXIS, GEN_C_OUT)
        # self.acc_layer2 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT)
        # self.acc_layer3 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT//2)
        
        # self.gyro_layer1 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, SENSOR_AXIS, GEN_C_OUT)
        # self.gyro_layer2 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT)
        # self.gyro_layer3 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT, GEN_C_OUT//2)
        
        self.sensor_layer1 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT_MERGE, GEN_C_OUT_MERGE,
                                      out_fft_list=GEN_FFT_N2, ser_size=SERIES_SIZE2, pooling=True)
        self.sensor_layer2 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT_MERGE, GEN_C_OUT_MERGE,
                                      ser_size=SERIES_SIZE2)
        self.sensor_layer3 = STFLayer(GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, GEN_C_OUT_MERGE, GEN_C_OUT_MERGE,
                                      ser_size=SERIES_SIZE2)
    
        self.fc = nn.Linear(GEN_C_OUT_MERGE, self.out_dim)

    def forward(self, x):
        # x: [B, SERIES_SIZE, SENSOR_AXIS*SENSOR_NUM]
        # acc_in = x[:, :, :SENSOR_AXIS]
        # gyro_in = x[:, :, SENSOR_AXIS:]

        x = x.permute(0, 2, 1) # [B, C, T]
        x_list = torch.split(x, self.sensor_axis, dim=1) # List of [B, SENSOR_AXIS, T]
        assert len(x_list) == self.sensor_num, "Sensor split does not match SENSOR_NUM"
        processed_list = []
        for i in range(self.sensor_num):
            out = self.acc_layers[i](x_list[i])
            processed_list.append(out)
        
        # acc_in = acc_in.permute(0, 2, 1)
        # gyro_in = gyro_in.permute(0, 2, 1)
        
        # a1 = self.dropout(self.acc_layer1(acc_in))
        # a2 = self.dropout(self.acc_layer2(a1))
        # a3 = self.dropout(self.acc_layer3(a2))
        
        # g1 = self.dropout(self.gyro_layer1(gyro_in))
        # g2 = self.dropout(self.gyro_layer2(g1))
        # g3 = self.dropout(self.gyro_layer3(g2))
        
        # s_in = torch.cat([a3, g3], dim=1)
        s_in = torch.cat(processed_list, dim=1) # [B, total_C, T]
        
        s1 = self.dropout(self.sensor_layer1(s_in))
        s2 = self.dropout(self.sensor_layer2(s1))
        s3 = self.dropout(self.sensor_layer3(s2))
        
        out = torch.mean(s3, dim=2)
        logits = self.fc(out)
        return logits

# ==========================================
# Data Loading & Training Loop
# ==========================================

class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, series_size, sensor_axis, sensor_num):
        # Load npz file
        with np.load(npz_path) as data:
            # Convert data to PyTorch tensor
            self.x = torch.tensor(data['example'], dtype=torch.float32)
            self.y = torch.tensor(data['label'], dtype=torch.float32)
        
        # Adjust dimensions to model input shape [B, SERIES_SIZE, SENSOR_AXIS*SENSOR_NUM]
        expected_dim = series_size * sensor_axis * sensor_num
        if self.x.dim() == 2 and self.x.shape[1] == expected_dim:
            self.x = self.x.view(-1, series_size, sensor_axis * sensor_num)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

import click

@click.command()
@click.option('--select', type=click.Choice(['wifi', 'hhar']), default='wifi', help='Dataset selection: wifi or hhar')
@click.option('--train_dir', type=str, default='widar_data/training_set', help='Path to training npz file')
@click.option('--eval_dir', type=str, default='widar_data/validation_set', help='Path to evaluation npz file')
@click.option('--synth_dir', type=str, default=None, help='Path to synthetic npz file')
@click.option('--synth_ratio', type=float, default=1.0, help='Ratio of synthetic data to use in training (if synth_dir is provided)')
@click.option('--seed', type=int, default=42, help='Random seed for reproducibility')
@click.option('--total_epoch_num', type=int, default=1000, help='Total number of training epochs')
@click.option('--output_dir', type=str, default='model_output', help='Directory to save model checkpoints and logs')
@click.option('--zero_to_one_norm', is_flag=True, help='Whether to apply zero-to-one normalization to input data')
@click.option('--data_norm', type=float, default=1.0, help='Normalization factor for input data (if zero_to_one_norm is not used)')
@click.option('--additive_noise', type=float, default=1.0, help='Standard deviation of additive Gaussian noise to apply to synthetic input data')
@click.option('--additive_noise_ratio', type=float, default=0.0, help='Ratio of samples in synthetic dataset to which additive noise will be applied')
@click.option('--synth_additive_noise', type=float, default=1.0, help='Standard deviation of additive Gaussian noise to apply to synthetic input data')
@click.option('--synth_additive_noise_ratio', type=float, default=0.0, help='Ratio of samples in synthetic dataset to which additive noise will be applied')

def main(**argv):
    # select, train_dir, eval_dir, seed, total_epoch_num, output_dir, zero_to_one_norm, data_norm = argv.values()
    select = argv.get('select')
    train_dir = argv.get('train_dir')
    eval_dir = argv.get('eval_dir')
    synth_dir = argv.get('synth_dir')
    synth_ratio = argv.get('synth_ratio')
    seed = argv.get('seed')
    total_epoch_num = argv.get('total_epoch_num')
    output_dir = argv.get('output_dir')
    zero_to_one_norm = argv.get('zero_to_one_norm')
    data_norm = argv.get('data_norm')
    additive_noise = float(argv.get('additive_noise'))
    additive_noise_ratio = float(argv.get('additive_noise_ratio'))    
    synth_additive_noise = float(argv.get('synth_additive_noise'))
    synth_additive_noise_ratio = float(argv.get('synth_additive_noise_ratio'))
    

    global SELECT
    SELECT = select
    np.random.seed(seed*12+4454)
    torch.manual_seed(seed*11+116481)

    os.makedirs(output_dir, exist_ok=True)

    # train_dataset = NPZDataset(train_npz_path, SERIES_SIZE, SENSOR_AXIS, SENSOR_NUM)
    must_have=[f'-gesture{i}-' for i in range(4)]+['-gesture17-','-gesture18-']
    # must_have=[f'-user10-gesture{i}-' for i in range(4)]+['-user10-gesture17-','-user10-gesture18-']
    must_not_have=["-user5-",]

    # must_have=["-gesture0-",]
    # must_not_have=["-user10-", "-user5-", "-user11-", "-user12-"]

    series_size = SERIES_SIZE

    train_dataset = WiDARDataset(train_dir, target_size=series_size, min_data_len=0, split_ratio=1.0, must_have=must_have, must_not_have=must_not_have, additive_noise_ratio=additive_noise_ratio, additive_noise_std=additive_noise)
    if synth_dir is not None:
        synth_dataset = SyntheticWiDARDataset(synth_dir, usage_ratio=synth_ratio, target_size=series_size, additive_noise_std=synth_additive_noise, additive_noise_ratio=synth_additive_noise_ratio)
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([train_dataset, synth_dataset])

    # import copy
    # eval_dataset = copy.deepcopy(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, persistent_workers=True, pin_memory=True, num_workers=4, prefetch_factor=4)
    
    # eval_dataset.flip_splits()
    # eval_dataset = NPZDataset(eval_npz_path, SERIES_SIZE, SENSOR_AXIS, SENSOR_NUM)
    eval_dataset = WiDARDataset(eval_dir, target_size=series_size, min_data_len=0, split_ratio=1.0, must_have=must_have, must_not_have=must_not_have)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE*4, shuffle=False, persistent_workers=True, pin_memory=True, num_workers=4, prefetch_factor=4)

    # Initialize Model
    model = STFNet(input_size=series_size).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR, betas=(ADAM_B1, ADAM_B2))
    criterion = nn.CrossEntropyLoss()
    
    # # Data Loaders (Set npz file path)
    # train_npz_path = os.path.join(SELECT, 'train.npz')
    # eval_npz_path = os.path.join(SELECT, 'eval.npz')
    
    # if not os.path.exists(train_npz_path) or not os.path.exists(eval_npz_path):
    #     print(f"No Data files {train_npz_path}, {eval_npz_path} found.")
    #     sys.exit(1)
        
    
    # TOTAL_ITER_NUM = 200000
    TOTAL_EPOCH_NUM = total_epoch_num
    
    print("Start training...")
    
    epoch_count = 0
    max_accuracy = 0.0
    max_f1_score = 0.0
    
    # PyTorch usually runs by epoch, but create an infinite iterator to maintain the iteration method of the original TF code
    # train_iter = iter(train_loader)

    result_csv_log = []
    json.dump(argv, open(os.path.join(output_dir, 'config.json'), 'w'), indent=4)
    
    while epoch_count < TOTAL_EPOCH_NUM:
        model.train()

        train_loss = []
        
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch_count+1}/{TOTAL_EPOCH_NUM}") as t:
            for data, target in t:
                data, target = data.to(device), target.to(device)
                if zero_to_one_norm:
                    data = (data - data.min()) / (data.max() - data.min() + 1e-12)
                elif data_norm != 1.0:
                    data = data / data_norm

                optimizer.zero_grad()
                output = model(data)
            
                # Target comes as one-hot vector, so convert to index
                target_indices = torch.argmax(target, dim=1)
                loss = criterion(output, target_indices)
                
                # L2 Regularization
                l2_reg = 0.0
                for name, param in model.named_parameters():
                    if 'angle' not in name:
                        l2_reg += 0.5 * torch.sum(param ** 2)
                loss += 5e-4 * l2_reg
                
                loss.backward()
                
                if CLIP_FLAG:
                    torch.nn.utils.clip_grad_value_(model.parameters(), 0.3)
                    
                optimizer.step()

                t.set_description(f'Loss: {loss.item():.4f}')
                train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        # Perform validation every 50 iterations like the original code
        model.eval()
        eval_loss = 0
        correct = 0
        total = 0
        total_labels = []
        total_preds = []
        
        with torch.inference_mode():
            with tqdm.tqdm(eval_loader, desc="Evaluating") as eval_t:
                for eval_data, eval_target in eval_t:
                    eval_data, eval_target = eval_data.to(device), eval_target.to(device)
                    eval_output = model(eval_data)
                    
                    eval_target_indices = torch.argmax(eval_target, dim=1)
                    batch_loss = criterion(eval_output, eval_target_indices)
                    eval_loss += batch_loss.item()
                    
                    pred = eval_output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(eval_target_indices.view_as(pred)).sum().item()
                    total += eval_data.size(0)
                    
                    total_labels.extend(eval_target_indices.cpu().numpy())
                    total_preds.extend(pred.cpu().numpy().flatten())
        
        dev_accuracy = correct / total
        dev_cross_entropy = eval_loss / len(eval_loader)
        dev_macro_f1 = f1_score(total_labels, total_preds, average='macro')
        
        print(f"Epoch {epoch_count+1}: Train Loss {train_loss:.4f} | Dev Acc {dev_accuracy:.4f} | Dev Loss {dev_cross_entropy:.4f} | F1 {dev_macro_f1:.4f}")
        result_csv_log.append({
            'epoch': epoch_count+1,
            'train_loss': train_loss,
            'dev_accuracy': dev_accuracy,
            'dev_loss': dev_cross_entropy,
            'dev_macro_f1': dev_macro_f1
        })
        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, 'latest_model.pth'))
        
        if dev_macro_f1 > max_f1_score:
            max_f1_score = dev_macro_f1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"--> Best performance updated! Model saved (F1: {max_f1_score:.4f})")
        
        epoch_count += 1

        # Save training results to CSV
        result_df = pd.DataFrame(result_csv_log)
        result_df.to_csv(os.path.join(output_dir, 'training_results.csv'), index=False)



if __name__ == "__main__":
    main()