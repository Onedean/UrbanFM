import sys
import signal
import torch
import argparse
from torch.utils.data import DataLoader
from utils.common_tools import initialize
from experiments.exp_forecasting import Exp_Forecasting
from experiments.exp_classical_imputation import Exp_Classical_Imputation
from utils.data_provider.expert_forecasting_process import generate_expert_forecasting_datasets
from utils.data_provider.expert_imputation_process import generate_classical_expert_imputation_datasets
from utils.data_provider.dataloader_wrapper import ExpertForecastingDataset


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
    # parser.add_argument('--config', type=str, default='configs/expert_model/LSTMNet/long_forecasting.json', help='configuration file path')
    parser.add_argument('--config', type=str, default='configs/expert_model/classical_imputation_model/Mean/metrla_block_imputation.json', help='configuration file path')
    
    # experiments and machine related arguments
    parser.add_argument('--is_training', type=bool, default=False, help='training status')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log_name', type=str, default='log', help='log name')
    parser.add_argument('--use_multi_gpu', action='store_true', default=True, help='use multiple gpus')
    parser.add_argument('--device_ids', nargs='+', type=int, default=0, help='device ids of multile gpus')
    parser.add_argument('--pin_memory', type=bool, default=False, help='pin memory')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    
    args = parser.parse_args()
    
    return args


def main(args):
    # seed initialize
    initialize.seed_anything(args.seed)
    
    # experiment task define
    if args.task_name == 'short_forecasting' or args.task_name == 'long_forecasting':
        Exp_task = Exp_Forecasting
        
        # dataset process and load
        Exp_data = generate_expert_forecasting_datasets(args.dataset_path, args.dataset_name, args.dataset_type, args.train_val_test_ratio, args.past_steps, args.future_steps, args.stride, args.few_shot_ratio)
        
        train_dataloader = DataLoader(ExpertForecastingDataset(Exp_data['train_x'], Exp_data['train_y']), batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)
        valid_dataloader = DataLoader(ExpertForecastingDataset(Exp_data['val_x'], Exp_data['val_y']), batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)
        test_dataloader = DataLoader(ExpertForecastingDataset(Exp_data['test_x'], Exp_data['test_y']), batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)
        mean, std = Exp_data['mean'], Exp_data['std']
        vars(args)["adj"] = Exp_data['adj']
        
        # training and testing
        if args.is_training:
            exp = Exp_task(args, mean, std)
            exp.train(train_dataloader, valid_dataloader)
            exp.test(test_dataloader)
            
            torch.cuda.empty_cache()
        else:
            exp = Exp_task(args, mean, std)
            exp.test(test_dataloader)
            
            torch.cuda.empty_cache()
        
        
    elif args.task_name == 'point_imputation' or args.task_name == 'block_imputation':
        
        if args.model_name == 'mean' or args.model_name == 'knn' or args.model_name == 'svd' or args.model_name == 'mice':
            Exp_task = Exp_Classical_Imputation
            # dataset process and load
            Exp_data = generate_classical_expert_imputation_datasets(args.dataset_path, args.dataset_name, args.dataset_type, args.train_val_test_ratio, args.p_fault, args.p_noise)
            adj = Exp_data['adj']
            
            exp = Exp_task(args, adj)
            exp.run(Exp_data['x_train'], Exp_data['mask_train'], Exp_data['x_test'], Exp_data['mask_test'], Exp_data['test_eval_mask'])


if __name__ == '__main__':
    args = load_args()
    initialize.init_config(args)
    initialize.init_log(args)
    initialize.init_device(args)
    
    main(args)
