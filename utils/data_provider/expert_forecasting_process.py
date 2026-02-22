import os
import pickle
import torch
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from utils.data_provider.spatial_adj_calculate import generate_graph_adjacency, generate_raster_adjacency

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-6)
    
    def inverse_transform(self, data):
        return (data * (self.std + 1e-6)) + self.mean


def generate_expert_samples(data, idx, past_steps, future_steps, stride):
    """"
    据给定的索引 idx, 从输入数据 data 中生成输入 x 和输出 y 的数据集
    """
    _, N, F = data.shape  # 获取数据的形状
    
    res = data[idx]  # 根据索引获取数据    
    length = len(idx)-1  # 获取索引长度并减1
    x_index, y_index = [], []  # 初始化x和y的索引列表
    
    # 遍历索引生成x和y的索引
    for i in tqdm(range(length, 0, -stride)):
        if i - past_steps - future_steps >= 0:
            x_index.extend(list(range(i - past_steps - future_steps, i - future_steps)))
            y_index.extend(list(range(i - future_steps, i)))
    
    # 将索引转换为数组
    x_index = np.asarray(x_index)
    y_index = np.asarray(y_index)
    
    # 重塑数据
    x = res[x_index].reshape(-1, past_steps, N, F).transpose(0, 3, 2, 1)
    y = res[y_index].reshape(-1, future_steps, N, F).transpose(0, 3, 2, 1)[:, :1, ...]
    
    return x, y


def generate_expert_forecasting_datasets(dataset_path, dataset_name, dataset_type, train_val_test_rate=[0.6, 0.2, 0.2], past_steps=12, future_steps=12, stride=1, few_shot_ratio=1.0):
    """
    单个数据集生成训练 验证 和 测试数据集, 并保存为 .npz 文件
    """
    # spatial_data_path = f'{dataset_path}/{dataset_name}_spatial.parquet'  # TODO
    # temporal_data_path = f'{dataset_path}/{dataset_name}_temporal.parquet'
    
    temporal_data_path = f'{dataset_path}/{dataset_name}/{dataset_name}_temporal.pkl'
    
    spatial_data_path = f'{dataset_path}/{dataset_name}/{dataset_name}_spatial.pkl'
        
    # save_path = f'{dataset_path}/{dataset_name}/{dataset_name}_expert_{few_shot_ratio*100}%_{past_steps}_{future_steps}.npz'
    
    # if os.path.exists(save_path):
    #     return np.load(save_path, allow_pickle=True)
    
    # with open(spatial_data_path, 'rb') as f:  # TODO
    #     spatial_data = pq.ParquetFile('file.parquet').read().to_pandas()
    
    # with open(temporal_data_path, 'rb') as f:
    #     temporal_data = pq.ParquetFile('file.parquet').read().to_pandas()
    #     T = temporal_data.shape[0]
    
    with open(spatial_data_path, 'rb') as f:
        df_spatial = pickle.load(f)
        if dataset_type == 'st_graph':
            adj = generate_graph_adjacency(df_spatial)
        elif dataset_type == 'st_raster':
            adj = generate_raster_adjacency(df_spatial)
        f.close()
    
    with open(temporal_data_path, 'rb') as f:
        df_temporal = pickle.load(f)
        raw_temporal_data = torch.tensor(df_temporal.values).unsqueeze(-1)
        f.close()
    
    T, N, _ = raw_temporal_data.shape
    
    # Add time-based features if specified
    feature_list = [raw_temporal_data]
    
    # add_time_of_day
    time_ind = (df_temporal.index.values - df_temporal.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
    time_of_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
    feature_list.append(torch.tensor(time_of_day, dtype=torch.float32))
    
    # add_day_of_week:
    dow = df_temporal.index.dayofweek
    dow_tiled = np.tile(dow, [1, N, 1]).transpose((2, 1, 0))
    day_of_week = dow_tiled / 7
    feature_list.append(torch.tensor(day_of_week, dtype=torch.float32))
    
    temporal_data = torch.cat(feature_list, dim=-1).numpy()  # Concatenate features along the channel dimension
    
    # train_rate = few_shot_ratio if few_shot_ratio <= train_val_test_rate[0] else train_val_test_rate[0]
    train_rate = train_val_test_rate[0] * few_shot_ratio
    valid_rate = train_val_test_rate[1]
    test_rate = train_val_test_rate[2]
    
    # 根据比例划分训练、验证和测试集索引
    train_idx = [i for i in range(int(T * train_rate))] # for few-shot setting
    valid_idx = [i for i in range(int(T * (1 - valid_rate - test_rate)), int(T * (1 - test_rate)))]
    test_idx = [i for i in range(int(T * (1 - test_rate)), T)]
    
    # 根据索引获取训练、验证和测试数据和标签
    train_x, train_y = generate_expert_samples(temporal_data, train_idx, past_steps, future_steps, stride)
    val_x, val_y = generate_expert_samples(temporal_data, valid_idx, past_steps, future_steps, stride)
    test_x, test_y = generate_expert_samples(temporal_data, test_idx, past_steps, future_steps, stride)
    
    # 对数据归一化
    mean, std = np.mean(train_x[:, 0, :, :]), np.std(train_x[:, 0, :, :])
    
    scaler = StandardScaler(mean, std)
    train_x[:, 0, :, :] = scaler.transform(train_x[:, 0, :, :]) 
    val_x[:, 0, :, :] = scaler.transform(val_x[:, 0, :, :])
    test_x[:, 0, :, :] = scaler.transform(test_x[:, 0, :, :])
    
    # # 保存数据到文件
    # np.savez(save_path, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, mean=mean, std=std)
    
    expert_datasets = {'train_x': train_x, 'train_y': train_y, 'val_x': val_x, 'val_y': val_y, 'test_x': test_x, 'test_y': test_y, 'adj':adj, 'mean': mean, 'std': std}
    
    return expert_datasets


if __name__ == '__main__':
    # demo test
    temporal_data = np.random.randn(100, 10)
    train_idx = [i for i in range(30)]
    train_x, train_y = generate_expert_samples(temporal_data, train_idx, 12, 12, 2)
    