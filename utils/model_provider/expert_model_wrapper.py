import torch
import torch.nn as nn
import torch.optim as optim

from utils.common_tools.graph_algo import normalize_adj_mx
from utils.metrics_calculator.metrics import masked_mae, masked_rmse, masked_mape, masked_mse, masked_mre

from models.expert.ts_model.LSTM import LSTMNet
from models.expert.ts_model.DLinear import DLinear
from models.expert.ts_model.PatchTST import PatchTST

from models.expert.st_graph.GWNet import GWNet
from models.expert.st_graph.D2STGNN import D2STGNN

from models.expert.st_raster.PredRNNv2 import PredRNNv2
from models.expert.st_raster.TAU import TAU

from models.expert.st_raster.STResNet import STResNet
from models.expert.st_raster.DSAN import DSAN

from models.expert.st_model.STNorm import STNorm
from models.expert.st_model.STID import STID
from models.expert.st_model.STAEformer import STAEformer
from models.expert.st_model.NexuSQN import NexuSQN

from models.foundation.vision_model.VisionTS import VisionTS

from models.foundation.denoise_model.STFormer import STFormer

EXPERT_MODEL_REGISTRY = {
    # TS Model
    'DLinear': DLinear,
    'LSTMNet': LSTMNet,
    'PatchTST': PatchTST,
    
    # ST Model
    'STNorm': STNorm,
    'STID': STID,
    'STAEformer': STAEformer,
    'NexuSQN': NexuSQN,
    
    # ST Graph Model
    "GWNet": GWNet,
    "D2STGNN": D2STGNN,
    
    # ST Raster Model
    "PredRNNv2": PredRNNv2,
    "TAU": TAU,
    "STResNet": STResNet,
    "DSAN": DSAN,
    
    # Vision Model
    'VisionTS': VisionTS,
    
    # Pretrained Model
    'STFormer': STFormer,
}


def load_model(model_name, args):
    if model_name not in EXPERT_MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in MODEL_REGISTRY.")
    
    model_class = EXPERT_MODEL_REGISTRY[model_name]
    
    model_params = get_model_params(model_name, args)
    
    return model_class(**model_params)


def get_model_params(model_name, args):
    if model_name == 'GWNet' or model_name == 'D2STGNN':
        adjs = normalize_adj_mx(args.adj, args.adj_type)
        aux_adjacency = [torch.tensor(adj).to(args.device) for adj in adjs]
        
    if model_name == 'LSTMNet':
        return {
            'in_channel_dim': args.in_channel_dim,
            'output_dim': args.output_dim,
            'out_channel_dim': args.out_channel_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'end_dim': args.end_dim,
        }
    elif model_name == 'DLinear':
        return {
            'node_num': args.node_num,
            'in_channel_dim': args.in_channel_dim,
            'input_dim': args.input_dim,
            'output_dim': args.output_dim,
            'individual': args.individual,
        }
    elif model_name == 'PatchTST':
        return {
            'node_num': args.node_num,
            'input_dim': args.input_dim,
            'output_dim': args.output_dim,
            'e_layers': args.e_layers,
            'n_heads': args.n_heads,
            'd_model': args.d_model,
            'd_ff': args.d_ff,
            'dropout': args.dropout,
            'fc_dropout': args.fc_dropout,
            'head_dropout': args.head_dropout,
            'patch_len': args.patch_len,
            'model_stride': args.model_stride,
            'individual': args.individual,
            'padding_patch': args.padding_patch,
            'revin': args.revin,
            'affine': args.affine,
            'subtract_last': args.subtract_last,
            'decomposition': args.decomposition,
            'kernel_size': args.kernel_size,
        }
    elif model_name == 'STNorm':
        return {
            'node_num': args.node_num,
            'in_channel_dim': args.in_channel_dim,
            'output_dim': args.output_dim,
            'tnorm_bool': args.tnorm_bool,
            'snorm_bool': args.snorm_bool,
            'out_channel_dim': args.out_channel_dim,
            'kernel_size': args.kernel_size,
            'blocks': args.blocks,
            'layers': args.layers,
        }
    elif model_name == 'STID':
        return {
            'node_num': args.node_num,
            'in_channel_dim': args.in_channel_dim,
            'input_dim': args.input_dim,
            'output_dim': args.output_dim,
            'if_spatial': args.if_spatial,
            'if_time_in_day': args.if_time_in_day,
            'if_day_in_week': args.if_day_in_week,
            'time_of_day_size': args.time_of_day_size,
            'day_of_week_size': args.day_of_week_size,
            'spatial_dim': args.spatial_dim,
            'temporal_dim_tod': args.temporal_dim_tod,
            'temporal_dim_dow': args.temporal_dim_dow,
            'out_channel_dim': args.out_channel_dim,
            'num_layer': args.num_layer,
        }
    elif model_name == 'STAEformer':
        return {
            'node_num': args.node_num,
            'in_channel_dim': args.in_channel_dim,
            'input_dim': args.input_dim,
            'output_dim': args.output_dim,
            'if_spatial': args.if_spatial,
            'if_time_in_day': args.if_time_in_day,
            'if_day_in_week': args.if_day_in_week,
            'if_adaptive': args.if_adaptive,
            'time_of_day_size': args.time_of_day_size,
            'day_of_week_size': args.day_of_week_size,
            'spatial_dim': args.spatial_dim,
            'temporal_dim_tod': args.temporal_dim_tod,
            'temporal_dim_dow': args.temporal_dim_dow,
            'adaptive_dim': args.adaptive_dim,
            'out_channel_dim': args.out_channel_dim,
            'if_use_mixed_proj': args.if_use_mixed_proj,
            'feed_forward_dim': args.feed_forward_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
        }
    elif model_name == 'NexuSQN':
        return {
            'node_num': args.node_num,
            'in_channel_dim': args.in_channel_dim,
            'input_dim': args.input_dim,
            'output_dim': args.output_dim,
            'out_channel_dim': args.out_channel_dim,
            'node_dim': args.node_dim,
            'num_layer': args.num_layer,
        }
    elif model_name == 'GWNet':
        return {
            'node_num': args.node_num,
            'supports': aux_adjacency,
            'in_channel_dim': args.in_channel_dim,
            'output_dim': args.output_dim,
            'dropout': args.dropout,
            'residual_channels': args.residual_channels,
            'dilation_channels': args.dilation_channels,
            'skip_channels': args.skip_channels,
            'end_channels': args.end_channels,
            'kernel_size': args.kernel_size,
            'blocks': args.blocks,
            'layers': args.layers,
        }
    elif model_name == 'D2STGNN':
        return {
            'node_num': args.node_num,
            'adjs': aux_adjacency,
            'input_dim': args.input_dim,
            'num_feat': args.num_feat,
            'num_hidden': args.num_hidden,
            'node_hidden': args.node_hidden,
            'time_emb_dim': args.time_emb_dim,
            'k_s': args.k_s,
            'k_t': args.k_t,
            'layer': args.layer,
            'gap': args.gap,
            'dropout': args.dropout,
        }
    elif model_name == 'PredRNNv2':
        return {
            'in_channel_dim': args.in_channel_dim,
            'out_channel_dim': args.out_channel_dim,
            'width': args.width,
            'height': args.height,
            'num_hidden': args.num_hidden,
            'num_layers': args.num_layers,
            'filter_size': args.filter_size,
            'layer_norm': args.layer_norm,
        }
    elif model_name == 'TAU':
        return {
            'input_dim': args.input_dim,
            'in_channel_dim': args.in_channel_dim,
            'out_channel_dim': args.out_channel_dim,
            'width': args.width,
            'height': args.height,
            'hid_S': args.hid_S,
            'hid_T': args.hid_T,
            'N_S': args.N_S,
            'N_T': args.N_T,
            'mlp_ratio': args.mlp_ratio,
            'drop': args.drop,
        }
    elif model_name == 'STResNet':
        return {
            'input_dim': args.input_dim,
            'width': args.width,
            'height': args.height,
            'in_channel_dim': args.in_channel_dim,
            'out_channel_dim': args.out_channel_dim,
            'nb_residual_unit': args.nb_residual_unit,
            'batch_norm': args.batch_norm,
        }
    elif model_name == 'DSAN':
        return {
            'input_dim': args.input_dim,
            'output_dim': args.output_dim,
            'width': args.width,
            'height': args.height,
            'in_channel_dim': args.in_channel_dim,
            'd': args.d,
            'l_d': args.l_d,
        }
    elif model_name == 'VisionTS':
        return {
            'arch': args.arch,
            'finetune_type': args.finetune_type,
            'ckpt_dir': args.ckpt_dir,
            'load_ckpt': args.load_ckpt,
        }
    elif model_name == 'STFormer':
        return {
            'd_model_spatial': args.d_model_spatial,
            'n_heads_spatial': args.n_heads_spatial,
            'num_layers_spatial': args.num_layers_spatial,
            'd_model_temporal': args.d_model_temporal,
            'n_heads_temporal': args.n_heads_temporal,
            'num_layers_temporal': args.num_layers_temporal,
            'dropout': args.dropout,
        }
    else:
        raise ValueError(f"Parameters for model {model_name} not defined.")


# 动态加载 optimizer
def load_optimizer(model, optimizer_name, learning_rate, weight_decay=0):
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")


# 动态加载 scheduler
def load_scheduler(optimizer, scheduler_name, args):
    if scheduler_name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif scheduler_name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    elif scheduler_name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience)
    else:
        return None


# 动态加载 criterion
def load_criterion(criterion_name):
    if criterion_name == 'masked_mse':
        return masked_mse
    elif criterion_name == 'masked_mae':
        return masked_mae
    elif criterion_name == 'MSELoss':
        return nn.MSELoss()
    elif criterion_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'L1Loss':
        return nn.L1Loss()
    else:
        raise ValueError(f"Criterion {criterion_name} not supported.")


# 动态加载 多个 metric的函数
def load_metrics(metric_names):
    metrics = []
    for metric_name in metric_names:
        if metric_name == 'masked_mae':
            metrics.append(masked_mae)
        elif metric_name == 'masked_rmse':
            metrics.append(masked_rmse)
        elif metric_name == 'masked_mape':
            metrics.append(masked_mape)
        elif metric_name == 'masked_mre':
            metrics.append(masked_mre)
        else:
            raise ValueError(f"Metric {metric_name} not supported.")
    return metrics

