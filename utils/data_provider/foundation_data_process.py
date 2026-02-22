import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from utils.data_provider.expert_imputation_process import sample_mask
import torch.distributed as dist
from utils.common_tools.initialize import is_main_process


def generate_foundation_pretrain_datasets(pretrain_data_path, spatial_window, temporal_window, stride=1, scaling_ratio=1.0):
    """
    所有训练数据集生成综合的预训练数据集
    """
    
    samples = []
    
    if is_main_process():
        iterator = tqdm(os.listdir(pretrain_data_path))
    else:
        iterator = os.listdir(pretrain_data_path)
    
    for name in iterator:
        
        data_file = f"{pretrain_data_path}/{name}/{name}_temporal.pkl"
        index_file = f"{pretrain_data_path}/{name}/index_{spatial_window}.pkl"
        
        with open(data_file, 'rb') as f:
            df = pickle.load(f)
            raw_data = torch.tensor(df.values).unsqueeze(-1) # [T, N, 1]
            f.close()
        
        with open(index_file, 'rb') as f:
            spatial_indices = pickle.load(f) # [N // spatial_window + 1, spatial_window]
            f.close()
        
        T, N, _ = raw_data.shape
        
        # Add time-based features
        input_data = torch.cat([
            raw_data,
            torch.tensor(np.tile((df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D'), [1, N, 1]).transpose((2, 1, 0)), dtype=torch.float32),
            torch.tensor(np.tile(df.index.dayofweek, [1, N, 1]).transpose((2, 1, 0)) / 7, dtype=torch.float32)
        ], dim=-1)
        
        # Slide temporal window first
        for t_start in range(0, T - temporal_window + 1, stride):
            t_end = t_start + temporal_window
            # For each temporal window, iterate over spatial regions
            for s_indices in spatial_indices:
                sample = input_data[t_start:t_end, s_indices, :]  # Shape: [temporal_window, spatial_window, f]
                samples.append(sample.permute(2, 1, 0)) # Shape: [F, N, T]
        
    pretrain_datasets = {'train_x': samples}
    
    return pretrain_datasets


def generate_forecasting_samples(data, idx, temporal_window, spatial_indices, mask_indices, stride, mask_flag=False):
    """"
    据给定的索引 idx, 从输入数据 data 中生成输入 x 和输出 y 的数据集
    """
    
    res = data[idx]  # 根据索引获取数据
    length = len(idx)-1  # 获取索引长度并减1
    samples = []
    masks = []
    
    for t_end in range(length, 0, -stride):
        if t_end - temporal_window >= 0:
            t_start = t_end - temporal_window
            for s_indices, m_indices in zip(spatial_indices, mask_indices):
                sample = res[t_start:t_end, s_indices, :]  # Shape: [temporal_window, spatial_window, f]
                samples.append(sample.permute(2, 1, 0)) # Shape: [F, N, T]
                if mask_flag:
                    T = sample.shape[0]
                    masks.append(torch.tensor(m_indices).unsqueeze(0).unsqueeze(2).repeat(1, 1, T))
    
    return samples, masks


def generate_foundation_downstream_forecasting_datasets(downstream_data_path, spatial_window, temporal_window, stride=1, train_val_test_rate=[0.6, 0.2, 0.2], few_shot_ratio=0.1):
    """
    所有评估数据集生成综合的下游 few-shot / zero-shot 数据集
    """
    downstream_forecasting_datasets = {}
    
    # for name in os.listdir(downstream_data_path):
    # for name in tqdm(["pems03_flow", "pems04_flow", "pems07_flow", "pems08_flow", "occpairs_occupancy", "occhamburg_occupancy", "pemsbay_speed", "metrla_speed", "trafficsh_speed", "bikenyc_inflow", "taxinyc_inflow", "tdrive_inflow"]):
    if is_main_process():
        # iterator = tqdm(["pems03_flow", "pems04_flow", "pems07_flow", "pems08_flow"])
        iterator = tqdm(["pems08_flow"])
    else:
        # iterator = ["pems03_flow", "pems04_flow", "pems07_flow", "pems08_flow"]
        iterator = ["pems08_flow"]
    
    for name in iterator:
        for t_window in temporal_window:
            
            if t_window == 24:
                eval_name = f"{name}_short_term_forecasting"
            elif t_window == 48:
                eval_name = f"{name}_long_term_forecasting"
            
            train_samples, valid_samples, test_samples = [], [], []
            
            # save_path = f"{downstream_data_path}/{name}/{name}_processed_{spatial_window}_{t_window}_{few_shot_ratio*100}.pkl"
            
            # if os.path.exists(save_path):
            #     with open(save_path, 'rb') as f:
            #         downstream_forecasting_datasets[eval_name] = pickle.load(f)
            #         f.close()
            #     continue
            
            data_file = f"{downstream_data_path}/{name}/{name}_temporal.pkl"
            index_file = f"{downstream_data_path}/{name}/index_{spatial_window}.pkl"
            mask_file = f"{downstream_data_path}/{name}/mask_{spatial_window}.pkl"
            
            with open(data_file, 'rb') as f:
                df = pickle.load(f)
                raw_data = torch.tensor(df.values).unsqueeze(-1) # [T, N, 1]
                f.close()
            
            with open(index_file, 'rb') as f:
                spatial_indices = pickle.load(f) # [N // spatial_window + 1, spatial_window]
                f.close()
            
            with open(mask_file, 'rb') as f:
                mask_indices = pickle.load(f) # [N // spatial_window + 1, spatial_window]
                f.close()
            
            T, N, _ = raw_data.shape
            
            # Add time-based features
            input_data = torch.cat([
                raw_data,
                torch.tensor(np.tile((df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D'), [1, N, 1]).transpose((2, 1, 0)), dtype=torch.float32),
                torch.tensor(np.tile(df.index.dayofweek, [1, N, 1]).transpose((2, 1, 0)) / 7, dtype=torch.float32)
            ], dim=-1)
            
            train_rate = few_shot_ratio if few_shot_ratio <= train_val_test_rate[0] else train_val_test_rate[0]
            valid_rate = train_val_test_rate[1]
            test_rate = train_val_test_rate[2]
            
            # 根据比例划分训练、验证和测试集索引
            train_idx = [i for i in range(int(T * train_rate))] # for few-shot setting
            valid_idx = [i for i in range(int(T * (1 - valid_rate - test_rate)), int(T * (1 - test_rate)))]
            test_idx = [i for i in range(int(T * (1 - test_rate)), T)]
            
            # 根据索引获取训练、验证和测试数据和标签
            train_samples, _ = generate_forecasting_samples(input_data, train_idx, t_window, spatial_indices, mask_indices, stride)
            valid_samples, _ = generate_forecasting_samples(input_data, valid_idx, t_window, spatial_indices, mask_indices, stride)
            test_samples, test_masks = generate_forecasting_samples(input_data, test_idx, t_window, spatial_indices, mask_indices, stride, mask_flag=True)
            
            downstream_forecasting_datasets[eval_name] = {'train_x': train_samples, 'valid_x': valid_samples, 'test_x': test_samples, 'test_mask': test_masks}
            
            # with open(save_path, 'wb') as f:
            #     pickle.dump(downstream_forecasting_datasets[eval_name], f)
            #     f.close()
    
    return downstream_forecasting_datasets



def generate_imputation_samples(input_data, eval_position_data, idx, temporal_window, spatial_indices, mask_indices, stride, mask_flag=False):
    """"
    据给定的索引 idx, 从输入数据 data 中生成输入 x 和输出 y 的数据集
    """
    
    res = input_data[idx]  # 根据索引获取数据
    eval_position = eval_position_data[idx]  # 根据索引获取评估掩码
    length = len(idx)-1  # 获取索引长度并减1
    samples = []
    test_eval_position = []
    masks = []
    
    for t_end in range(length, 0, -stride):
        if t_end - temporal_window >= 0:
            t_start = t_end - temporal_window
            for s_indices, m_indices in zip(spatial_indices, mask_indices):
                sample = res[t_start:t_end, s_indices, :]  # Shape: [temporal_window, spatial_window, f]
                position = eval_position[t_start:t_end, s_indices, :]
                eval_sample = sample * (1 - position)  # Apply mask
                samples.append(eval_sample.permute(2, 1, 0)) # Shape: [F, N, T]
                test_eval_position.append(position.permute(2, 1, 0)[0].unsqueeze(0)) # Shape: [F, N, T] -> [1, N, T]
                if mask_flag:
                    T = sample.shape[0]
                    masks.append(torch.tensor(m_indices).unsqueeze(0).unsqueeze(2).repeat(1, 1, T))
    
    return samples, test_eval_position, masks



def generate_foundation_downstream_imputation_datasets(downstream_data_path, spatial_window, temporal_window, imputation_type, train_val_test_rate=[0.6, 0.2, 0.2]):
    
    downstream_imputation_datasets = {}
    
    if is_main_process():
        iterator = tqdm(["pems03_flow", "pems04_flow", "pems07_flow", "pems08_flow", "pemsbay_speed", "metrla_speed"])
    else:
        iterator = ["pems03_flow", "pems04_flow", "pems07_flow", "pems08_flow", "pemsbay_speed", "metrla_speed"]
    
    for name in iterator:
        for imputation in imputation_type:
            
            if imputation == 'point_imputation':
                p_fault, p_noise = 0.0015, 0.05
            elif imputation == 'block_imputation':
                p_fault, p_noise = 0, 0.25
            
            eval_name = f"{name}_{imputation}"
            
            # save_path = f"{downstream_data_path}/{name}/{name}_processed_{spatial_window}_{imputation}.pkl"
            
            # if os.path.exists(save_path):
            #     with open(save_path, 'rb') as f:
            #         downstream_imputation_datasets[eval_name] = pickle.load(f)
            #         f.close()
            #     continue
            
            data_file = f"{downstream_data_path}/{name}/{name}_temporal.pkl"
            index_file = f"{downstream_data_path}/{name}/index_{spatial_window}.pkl"
            mask_file = f"{downstream_data_path}/{name}/mask_{spatial_window}.pkl"
            
            with open(data_file, 'rb') as f:
                df_temporal = pickle.load(f)
                raw_data = torch.tensor(df_temporal.values).unsqueeze(-1) # [T, N, 1]
                f.close()
            
            with open(index_file, 'rb') as f:
                spatial_indices = pickle.load(f) # [N // spatial_window + 1, spatial_window]
                f.close()
            
            with open(mask_file, 'rb') as f:
                mask_indices = pickle.load(f) # [N // spatial_window + 1, spatial_window]
                f.close()
            
            T, N, _ = raw_data.shape
            
            # Add time-based features
            input_data = torch.cat([
                raw_data,
                torch.tensor(np.tile((df_temporal.index.values - df_temporal.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D'), [1, N, 1]).transpose((2, 1, 0)), dtype=torch.float32),
                torch.tensor(np.tile(df_temporal.index.dayofweek, [1, N, 1]).transpose((2, 1, 0)) / 7, dtype=torch.float32)
            ], dim=-1)
            
            # 评估掩码（1评估，0不评估）
            eval_mask = sample_mask(df_temporal.values.shape, p=p_fault, p_noise=p_noise, min_seq=12, max_seq=12 * 4, rng=np.random.default_rng(56789)).astype('uint8')
            # 沿着最后一个维度复制3次
            eval_position_data = torch.tensor(np.repeat(np.expand_dims(eval_mask, axis=-1), 3, axis=-1))
            
            test_rate = train_val_test_rate[2]
            test_idx = [i for i in range(int(T * (1 - test_rate)), T)]
            
            test_samples, test_eval_position, test_masks = generate_imputation_samples(input_data, eval_position_data, test_idx, temporal_window, spatial_indices, mask_indices, temporal_window, mask_flag=True)
            
            
            downstream_imputation_datasets[eval_name] = {'test_x': test_samples, 'test_eval': test_eval_position, 'test_mask': test_masks}
            
            # with open(save_path, 'wb') as f:
            #     pickle.dump(downstream_imputation_datasets[eval_name], f)
            #     f.close()
    
    return downstream_imputation_datasets


