import sys
import signal
import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.common_tools import initialize
from experiments.exp_st_pretrain import Exp_ST_Pretrain
from utils.data_provider.foundation_data_process import generate_foundation_pretrain_datasets, generate_foundation_downstream_forecasting_datasets, generate_foundation_downstream_imputation_datasets

from utils.data_provider.dataloader_wrapper import FoundationTrainDataset, ForecastingTestDataset, ImputationTestDataset


def cleanup_and_exit(signum):  # use ctrl + c to kill signal
    print(f"Received signal {signum}. Cleaning up...")
    torch.cuda.empty_cache()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)


def load_args():
    parser = argparse.ArgumentParser(description='ST Library')
    
    # task / model / data related arguments
    parser.add_argument('--config', type=str, default='configs/foundation_model/MaskedAutoencoderST/pretrain_small.json', help='configuration file path')
    
    # experiments and machine related arguments
    parser.add_argument('--is_training', type=str, default='development_pipeline', help='training status')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--log_name', type=str, default='log', help='log name')
    parser.add_argument('--use_multi_gpu', action='store_true', default=True, help='use multiple gpus')
    parser.add_argument('--device_ids', type=list, default=[1, 2, 3], help='device ids of multile gpus')
    parser.add_argument('--pin_memory', type=bool, default=False, help='pin memory')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--patience', type=int, default=5, help='patience')
    parser.add_argument('--distributed', action='store_true', default=True, help='use distributed training')
    
    args = parser.parse_args()
    
    return args


def main(args):
    # seed initialize
    initialize.seed_anything(args.seed)
    
    # experiment task define
    Exp_task = Exp_ST_Pretrain
    # dataset process and load
    Exp_pretrain_data = generate_foundation_pretrain_datasets(args.pretrain_data_path, args.spatial_window, args.pretrain_temporal_window, args.pretrain_stride, args.scaling_ratio)
    Exp_downstream_forecasting_data = generate_foundation_downstream_forecasting_datasets(args.downstream_forecasting_data_path, args.spatial_window, args.downstream_forecasting_temporal_window, args.downstream_forecasting_stride, args.train_val_test_ratio, args.few_shot_ratio)
    Exp_downstream_imputation_data = generate_foundation_downstream_imputation_datasets(args.downstream_imputation_data_path, args.spatial_window, args.pretrain_temporal_window, args.downstream_imputation_type, args.train_val_test_ratio)
    
    # 创建数据集
    pretrain_dataset = FoundationTrainDataset(Exp_pretrain_data['train_x'])
    
    # 如果是分布式训练，使用DistributedSampler
    if args.distributed:
        train_sampler = DistributedSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(
            pretrain_dataset, 
            batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers
        )
    else:
        pretrain_dataloader = DataLoader(
            pretrain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers
        )
    
    downstream_forecasting_train_dataloader_list, downstream_forecasting_valid_dataloader_list, downstream_forecasting_test_dataloader_list = {}, {}, {}
    for key in Exp_downstream_forecasting_data.keys():
        train_dataset = FoundationTrainDataset(Exp_downstream_forecasting_data[key]['train_x'])
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset)
            downstream_forecasting_train_dataloader_list[key] = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers
            )
        else:
            downstream_forecasting_train_dataloader_list[key] = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers
            )
            
        downstream_forecasting_valid_dataloader_list[key] = DataLoader(
            FoundationTrainDataset(Exp_downstream_forecasting_data[key]['valid_x']),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers
        )
        downstream_forecasting_test_dataloader_list[key] = DataLoader(
            ForecastingTestDataset(Exp_downstream_forecasting_data[key]['test_x'], Exp_downstream_forecasting_data[key]['test_mask']),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers
        )
    
    downstream_imputation_test_dataloader_list = {}
    for key in Exp_downstream_imputation_data.keys():
        downstream_imputation_test_dataloader_list[key] = DataLoader(
            ImputationTestDataset(Exp_downstream_imputation_data[key]['test_x'], Exp_downstream_imputation_data[key]['test_eval'], Exp_downstream_imputation_data[key]['test_mask']),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers
        )
    
    # The following is a development version for saving time:
    if args.is_training == 'development_pipeline':
        exp = Exp_task(args)
        # exp.dev_only_pretrain(pretrain_dataloader)
        exp.dev_pretrain(pretrain_dataloader, downstream_forecasting_test_dataloader_list, downstream_imputation_test_dataloader_list)
        
        torch.cuda.empty_cache()
    
    
    # # The following is the official version:
    
    # if args.is_training == 'pretrain':
    #     exp = Exp_task(args)
    #     exp.pretrain(pretrain_dataloader)
    #     exp.test(downstream_forecasting_test_dataloader)
    
    #     torch.cuda.empty_cache()
    # elif args.is_training == 'finetune':
    #     exp = Exp_task(args)
    #     exp.finetune(downstream_forecasting_train_dataloader, downstream_forecasting_valid_dataloader)
    #     exp.test(downstream_forecasting_test_dataloader)
        
    #     torch.cuda.empty_cache()
    # elif args.is_training == 'evaluate':
    #     exp = Exp_task(args)
    #     exp.test(downstream_forecasting_test_dataloader)
        
    #     torch.cuda.empty_cache()
    


if __name__ == '__main__':
    args = load_args()
    initialize.init_device(args)
    initialize.init_config(args)
    initialize.init_log(args)
    
    main(args)



