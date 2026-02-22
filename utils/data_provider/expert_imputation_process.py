import pickle
import numpy as np
from utils.data_provider.spatial_adj_calculate import generate_graph_adjacency, generate_raster_adjacency


def sample_mask(shape, p=0.002, p_noise=0., max_seq=1, min_seq=1, rng=None):
    """
    定义sample_mask函数, 用于生成随机掩码 (mask), 模拟数据中的缺失值或噪声
    """
    # 如果未提供随机数生成器（rng），则使用numpy的默认随机数生成器
    if rng is None:
        rand = np.random.random  # 随机生成浮点数
        randint = np.random.randint  # 随机生成整数
    else:
        rand = rng.random  # 使用提供的随机数生成器生成浮点数
        randint = rng.integers  # 使用提供的随机数生成器生成整数

    # 初始化掩码矩阵，形状与输入数据相同，初始值为False=0，表示不掩码，而1则表示掩码
    mask = rand(shape) < p  # 以概率p随机生成True或False

    # 遍历掩码矩阵的每一列
    for col in range(mask.shape[1]):
        # 获取当前列中为True的索引
        idxs = np.flatnonzero(mask[:, col])
        # 如果当前列没有True值，则跳过
        if not len(idxs):
            continue

        # 随机生成一个故障序列长度在最小长度min_seq到最大长度max_seq之间
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))

        # 为每个True值扩展故障序列长度
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        # 去除重复的索引值
        idxs = np.unique(idxs_ext)
        # 限制索引范围，防止超出矩阵边界
        idxs = np.clip(idxs, 0, shape[0] - 1)
        # 将扩展后的索引位置设置为True
        mask[idxs, col] = True

    # 在整个掩码矩阵中以概率p_noise随机添加噪声（True值）
    mask = mask | (rand(mask.shape) < p_noise)

    # 返回最终的掩码矩阵，类型为uint8（0或1）
    return mask.astype('uint8')


def generate_classical_expert_imputation_datasets(dataset_path, dataset_name, dataset_type, train_val_test_rate=[0.6, 0.2, 0.2], p_fault=0.0015, p_noise=0.05):
    """
    单个数据集生成训练 和 测试的补全数据集
    """
    
    temporal_data_path = f'{dataset_path}/{dataset_name}/{dataset_name}_temporal.pkl'
    
    spatial_data_path = f'{dataset_path}/{dataset_name}/{dataset_name}_spatial.pkl'
    
    with open(temporal_data_path, 'rb') as f:
        df_temporal = pickle.load(f)
        # datetime_idx = sorted(df_temporal.index)
        # import pandas as pd
        # date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5min')
        # df_temporal = df_temporal.reindex(index=date_range)
        f.close()
    
    with open(spatial_data_path, 'rb') as f:
        df_spatial = pickle.load(f)
        if dataset_type == 'st_graph':
            adj = generate_graph_adjacency(df_spatial)
        elif dataset_type == 'st_raster':
            adj = generate_raster_adjacency(df_spatial)
        f.close()
    
    test_rate = train_val_test_rate[2]
    
    train_slice = np.zeros(len(df_temporal)).astype(bool)
    train_slice[:-int(test_rate * len(df_temporal))] = True
    test_slice = ~train_slice
    
    # 评估掩码（1评估，0不评估）
    eval_mask = sample_mask(df_temporal.values.shape, p=p_fault, p_noise=p_noise, min_seq=12, max_seq=12 * 4, rng=np.random.default_rng(56789)).astype('uint8')  # seed = 1e9, 56789, 9101112
    train_mask = 1 - eval_mask
    test_eval_mask = eval_mask[test_slice]
    
    x_train, mask_train = df_temporal.values[train_slice], train_mask[train_slice]
    x_test, mask_test = df_temporal.values[test_slice], train_mask[test_slice]
    
    expert_datasets = {'x_train': x_train, 'mask_train': mask_train, 'x_test': x_test, 'mask_test': mask_test, 'test_eval_mask': test_eval_mask, 'adj': adj}
    
    return expert_datasets
