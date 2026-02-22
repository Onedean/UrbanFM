import timesfm
import pickle
import torch
import numpy as np
import argparse
import logging, sys
import os.path as osp


def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse

def generate_expert_samples(data, idx, past_steps, future_steps, stride):
    """"
    据给定的索引 idx, 从输入数据 data 中生成输入 x 和输出 y 的数据集
    """
    _, N, F = data.shape  # 获取数据的形状
    
    res = data[idx]  # 根据索引获取数据    
    length = len(idx)-1  # 获取索引长度并减1
    x_index, y_index = [], []  # 初始化x和y的索引列表
    
    # 遍历索引生成x和y的索引
    for i in range(length, 0, -stride):
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


def main(dataset, num_steps, logger):
    
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=4,
            horizon_len=num_steps,
            num_layers=50,
            use_positional_embedding=False,
            context_len=2048,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )
    
    folder_path = "/data/weichen/ST-Library/datasets/eval_datasets"

    logger.info('*'*30)
    logger.info(f"Dataset: {dataset}")
    logger.info('*'*30)
    
    
    with open(f"{folder_path}/{dataset}/{dataset}_temporal.pkl", 'rb') as f:
        df = pickle.load(f)
        raw_temporal_data = torch.tensor(df.values).unsqueeze(-1)
        f.close()

    T, N, _ = raw_temporal_data.shape

    # train_rate = few_shot_ratio if few_shot_ratio <= train_val_test_rate[0] else train_val_test_rate[0]
    test_rate = 0.2
    test_idx = [i for i in range(int(T * (1 - test_rate)), T)]
    
    test_x, test_y = generate_expert_samples(raw_temporal_data.numpy(), test_idx, past_steps=num_steps, future_steps=num_steps, stride=1)
    
    n = test_x.shape[0]
    random_indices = np.random.choice(n, size=500, replace=False)
    
    test_x = test_x[random_indices]
    test_y = test_y[random_indices]
    
    # test_x = test_x[:1]
    # test_y = test_y[:1]
    
    B, F, N, T = test_x.shape
    
    # 使用TimesFM进行预测
    forecast_input = test_x.reshape(B * N, T)
    frequency_input = [0] * B * N
    
    point_forecast, _ = tfm.forecast(
        forecast_input,
        freq=frequency_input,
    )
    preds = point_forecast.reshape(B, N, T)
    labels = test_y.reshape(B, N, T)
    
    labels = torch.Tensor(labels).permute(0, 2, 1)
    preds = torch.Tensor(preds).permute(0, 2, 1)

    # handle the precision issue when performing inverse transform to label
    mask_value = torch.tensor(0)

    test_mae = []
    test_mape = []
    test_rmse = []

    # Calculate metrics
    for i in range(num_steps):
        res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
        test_mae.append(res[0])
        test_mape.append(res[1] * 100)
        test_rmse.append(res[2])

    mae_mean = np.mean(test_mae)
    mae_std = 0
    rmse_mean = np.mean(test_rmse)
    rmse_std = 0
    mape_mean = np.mean(test_mape)
    mape_std = 0
    
    if num_steps == 12:
        logger.info('TimesFM - Short Forecasting' + "\t\t MAE:" + "& $" + f"{mae_mean:.2f}" + "\\textcolor{gray}{\\text{\scriptsize±" + f'{mae_std:.2f}'+"}}$" + "\t\t RMSE:" + "& $" + f"{rmse_mean:.2f}" + "\\textcolor{gray}{\\text{\scriptsize±" + f'{rmse_std:.2f}'+"}}$" + "\t\t MAPE:" + "& $" + f"{mape_mean:.2f}" + "\\textcolor{gray}{\\text{\scriptsize±" + f'{mape_std:.2f}'+"}}$")
    else:
        logger.info('TimesFM - Long Forecasting' + "\t\t MAE:" + "& $" + f"{mae_mean:.2f}" + "\\textcolor{gray}{\\text{\scriptsize±" + f'{mae_std:.2f}'+"}}$" + "\t\t RMSE:" + "& $" + f"{rmse_mean:.2f}" + "\\textcolor{gray}{\\text{\scriptsize±" + f'{rmse_std:.2f}'+"}}$" + "\t\t MAPE:" + "& $" + f"{mape_mean:.2f}" + "\\textcolor{gray}{\\text{\scriptsize±" + f'{mape_std:.2f}'+"}}$")


def load_args():
    parser = argparse.ArgumentParser(description='ST Library')
    
    parser.add_argument('--dataset', type=str, default='pems03_flow', help='dataset name') # ['pems03_flow', 'pems04_flow', 'pems07_flow', 'pems08_flow', 'occpairs_occupancy', 'occhamburg_occupancy', 'pemsbay_speed', 'metrla_speed', 'trafficsh_speed', 'bikenyc_inflow', 'taxinyc_inflow', 'tdrive_inflow']
    parser.add_argument('--num_steps', type=int, default=12, help='number of steps')  # [12, 24]
    
    args = parser.parse_args()
    
    return args


def init_log(args):
    '''
    初始化日志记录对象
    '''
    logger = logging.getLogger(__name__)  # 创建一个logger对象
    logger.setLevel(logging.INFO)  # 设置logger的日志级别为INFO
    fh = logging.FileHandler(osp.join("./log/foundation/timesfm/", f"{args.dataset}_{args.num_steps}_record.log"))  # 创建一个文件处理器，将日志写入指定文件
    fh.setLevel(logging.INFO)  # 设置日志格式
    ch = logging.StreamHandler(sys.stdout)  # 创建一个流处理器，将日志输出到标准输出
    ch.setLevel(logging.INFO)  
    formatter = logging.Formatter("%(asctime)s - %(message)s")  # 设置日志格式
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)  # 将文件处理器和流处理器添加到logger中
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join("./log/foundation/timesfm/", f"{args.dataset}_{args.num_steps}_record.log"))  # 记录日志初始化的信息
    vars(args)["logger"] = logger  # 将logger对象添加到args参数中


if __name__ == '__main__':
    
    args = load_args()
    init_log(args)
    
    main(args.dataset, args.num_steps, args.logger)
    