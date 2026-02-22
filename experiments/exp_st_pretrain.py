import os
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .exp_basic import Exp_Basic
from utils.common_tools.earlystop import EarlyStopping
from utils.model_provider.expert_model_wrapper import load_model, load_optimizer, load_scheduler, load_criterion, load_metrics
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.common_tools.initialize import is_main_process


class Exp_ST_Pretrain(Exp_Basic):
    def __init__(self, args):
        super(Exp_ST_Pretrain, self).__init__(args)
        self.args = args
        self.mask_ratio = args.mask_ratio
        self.model = self._build_model()

    def _build_model(self):
        model = load_model(self.model_name, self.args)
        model = model.to(self.device).to(torch.bfloat16)
        
        if self.args.distributed:
            model = DDP(model, device_ids=[self.args.local_rank])
        elif self.use_multi_gpu and self.device != torch.device('cpu') and len(self.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.args.device_ids, output_device=self.device)
            
        return model
    
    
    def _print_model_parameters(self, model, verbose=False):
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            if verbose:
                print(f"Parameter Name: {name}, Shape: {tuple(param.shape)}, Total: {num_params}")
        
        total_m = total_params / 1e6
        trainable_m = trainable_params / 1e6
        total_b = total_params / 1e9
        trainable_b = trainable_params / 1e9
        
        self.logger.info("\n" + "=" * 30 + " Model Parameters " + "=" * 30)
        self.logger.info(f"Total Parameters: {total_params:,} (≈ {total_m:.2f} M or ≈ {total_b:.2f} B)")
        self.logger.info(f"Trainable Parameters: {trainable_params:,} (≈ {trainable_m:.2f} M or ≈ {trainable_b:.2f} B)")
        self.logger.info("=" * 72 + "\n")
    
    def _load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.saving_path, 'best_checkpoint.pth')))
    
    def _select_optimizer(self):
        optimizer = load_optimizer(self.model, self.args.optimizer_name, self.args.learning_rate, self.args.weight_decay)
        return optimizer
    
    def _select_scheduler(self, optimizer):
        scheduler = load_scheduler(optimizer, self.args.scheduler_name, self.args)
        return scheduler
    
    def _select_criterion(self):
        criterion = load_criterion(self.args.criterion_name)
        return criterion
    
    def _select_forecasting_metrics(self):
        metric = load_metrics(self.args.forecasting_metric_name)
        return metric
    
    def _select_imputation_metrics(self):
        metric = load_metrics(self.args.imputation_metric_name)
        return metric
    
    def instance_normalize(self, x):
        """
        x: [B, C, N, T]
        - 对每个 (b,c) 在 (N,T) 上计算 mean, std
        - x_norm = (x - mean_bc) / std_bc
        - mean_bc, std_bc shape => [B,C,1,1]

        返回:
        - x_norm: [B, C, N, T], 每个通道都已归一化
        - (flow_means, flow_stds): 只包含流量通道(c=0)的 mean/std => shape [B,1,1,1]
        """
        B, C, N, T = x.shape

        # 计算所有通道的 mean, std => [B,C,1,1]
        means = x.mean(dim=(2,3), keepdim=True)
        stds  = x.std(dim=(2,3), keepdim=True) + 1e-5

        # 所有通道一起归一化
        x_norm = (x - means) / stds

        # 只保存流量通道 (c=0) 的 mean, std
        # => flow_means, flow_stds 形状 [B,1,1,1]
        means = means[:, 0:1, :, :]  # [B,1,1,1]
        stds  = stds[:, 0:1, :, :]   # [B,1,1,1]

        return x_norm, means, stds
    
    
    def instance_denormalize_flow(self, pred_norm, means, stds):
        """
        反归一化仅针对流量通道:
        - flow_pred_norm: [B,1,N,T], 模型输出 (归一化空间)
        - flow_mean_std: (flow_means, flow_stds), 均是 [B,1,1,1]

        返回: [B,1,N,T], 原始流量尺度
        """
        # broadcast到 [B,1,N,T]
        B, _, N, T = pred_norm.shape
        means_4d = means.expand(B, 1, N, T)  # [B,1,N,T]
        stds_4d  = stds.expand(B, 1, N, T)   # [B,1,N,T]

        flow_pred = pred_norm * stds_4d + means_4d
        return flow_pred
    
    
    def select_mask_strategy(self):
        """
        多个可选掩码策略: random_point, random_node, random_time, left_half
        """
        strategies = ["random_point", "random_node", "random_time", "right_half"]
        return random.choice(strategies)
    
    
    def mask_data(self, x_norm, strategy=None):
        """
        对数据 x_norm [B, C, N, T] 进行掩码处理 (同一位置的3个通道同时掩码 )
        
        参数:
            x_norm: 输入张量, 形状为 [B, C, N, T]
            strategy: 掩码策略, 支持：
                - "random_point": 每个样本在 [N, T] 中精确掩码 ratio 比例的点
                - "random_node": 每个样本在 N 个节点中精确掩码 ratio 比例的节点, 被掩码节点在所有 T 上均被置 0
                - "random_time": 每个样本在 T 个时间步中精确掩码 ratio 比例的时间步, 被掩码时间步在所有节点上均被置 0
                - "right_half": 对每个样本, 将时间维度后半部分全部置 0
            ratio: 掩码比例 (0~1之间的浮点数 )
            
        返回:
            x_masked: 掩码后的张量, 形状为 [B, C, N, T]
            mask_pos: 掩码位置, 布尔类型, 形状为 [B, 1, N, T], True 表示该位置被掩码
        """
        B, C, N, T = x_norm.shape    
        ratio = self.mask_ratio
        
        if strategy is None:
            strategy = self.select_mask_strategy()
        
        # 根据策略构造形状 [B, N, T] 的掩码布尔张量
        if strategy == "random_time":
            # 每个样本：在 T 个时间步中精确掩码 ratio 比例的时间步
            num_mask = int(T * ratio)
            # 创建随机分数, shape: [B, T]
            scores = torch.rand(B, T, device=x_norm.device)
            # 针对极端情况做处理
            if num_mask <= 0:
                mask_time = torch.zeros_like(scores, dtype=torch.bool)
            elif num_mask >= T:
                mask_time = torch.ones_like(scores, dtype=torch.bool)
            else:
                kth = torch.kthvalue(scores, num_mask, dim=1, keepdim=True).values  # shape: [B,1]
                mask_time = scores <= kth  # [B,T], 每个样本中恰好有 num_mask 个 True
            # 将时间步维度扩展到节点维度：最终 mask_pos 形状 [B,N,T]
            mask_pos = mask_time.unsqueeze(1).expand(B, N, T)

        elif strategy == "random_node":
            # 每个样本：在 N 个节点中精确掩码 ratio 比例的节点
            num_mask = int(N * ratio)
            scores = torch.rand(B, N, device=x_norm.device)
            if num_mask <= 0:
                mask_node = torch.zeros_like(scores, dtype=torch.bool)
            elif num_mask >= N:
                mask_node = torch.ones_like(scores, dtype=torch.bool)
            else:
                kth = torch.kthvalue(scores, num_mask, dim=1, keepdim=True).values  # shape: [B,1]
                mask_node = scores <= kth  # [B,N]
            # 将节点维度扩展到时间步维度：最终 mask_pos 形状 [B,N,T]
            mask_pos = mask_node.unsqueeze(-1).expand(B, N, T)

        elif strategy == "random_point":
            # 每个样本：在 [N, T] 个点中精确掩码 ratio 比例的点
            total_points = N * T
            num_mask = int(total_points * ratio)
            # 生成随机分数, shape: [B, total_points]
            scores = torch.rand(B, total_points, device=x_norm.device)
            if num_mask <= 0:
                mask_flat = torch.zeros_like(scores, dtype=torch.bool)
            elif num_mask >= total_points:
                mask_flat = torch.ones_like(scores, dtype=torch.bool)
            else:
                kth = torch.kthvalue(scores, num_mask, dim=1, keepdim=True).values  # shape: [B,1]
                mask_flat = scores <= kth  # [B, total_points]
            # reshape 成 [B, N, T]
            mask_pos = mask_flat.view(B, N, T)

        elif strategy == "right_half":
            # 将时间维度后半部分全部置 True
            mask_pos = torch.zeros(B, N, T, dtype=torch.bool, device=x_norm.device)
            mask_pos[..., T // 2:] = True

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 构造通道级别掩码, shape: [B, C, N, T]
        mask_4d = mask_pos.unsqueeze(1).expand(B, C, N, T)
        x_masked = x_norm.clone()
        x_masked[mask_4d] = 0.0

        # 返回的 mask_pos 扩展一维为 [B, 1, N, T]
        return x_masked, mask_pos.unsqueeze(1)
    
    
    def dev_only_pretrain(self, pretrain_dataloader):
        """
        在 pretrain_dataloader 上进行多epoch自监督预训练: instance normalize  / mask / forward / instance denormalize / 计算在mask位置的MSE
        
        每个 epoch 结束后, 对 downstream_test_dataloader_list 里的不同测试集做 zero-shot 测试. 
        """
        
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer)
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0.0
            n_batch = 0
            
            if is_main_process():
                iterator = tqdm(pretrain_dataloader, desc=f"Pretrain Epoch {epoch+1}")
            else:
                iterator = pretrain_dataloader
            
            for x in iterator:

                x = x.to(self.device).to(torch.bfloat16)  # [B, C, N, T]
                
                # (1) instance normalize(所有通道), 仅保留 flow channel mean/std
                x_norm, means, stds = self.instance_normalize(x)
                
                # 2) 掩码
                x_masked, mask_pos = self.mask_data(x_norm)
                
                # print(mask_pos.sum())
                
                # 3) 前向 (预计输出 [B, 1, N, T], 只重建流量通道)
                pred_norm = self.model(x_masked)

                # (4) 反归一化流量
                y_pred = self.instance_denormalize_flow(pred_norm, means, stds)  # [B,1,N,T]

                # (5) 计算 loss (只对被 mask 的流量位置)
                y_truth = x[:, 0:1, :, :]  # [B,1,N,T], 原始流量
                
                loss = criterion(y_pred[mask_pos], y_truth[mask_pos])

                # 6) backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batch += 1
            
            avg_loss = total_loss / max(n_batch, 1)
            self.logger.info(f"[Epoch {epoch+1}] Pretrain Loss: {avg_loss:.4f}")
            
            # ----- 保存模型 ---- #
            early_stopping(avg_loss, self.model, self.saving_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                if scheduler is not None:
                    scheduler.step()
    
    
    def dev_pretrain(self, pretrain_dataloader, downstream_forecasting_test_dataloader_list, downstream_imputation_test_dataloader_list):
        """
        在 pretrain_dataloader 上进行多epoch自监督预训练: instance normalize  / mask / forward / instance denormalize / 计算在mask位置的MSE
        
        每个 epoch 结束后, 对 downstream 里的不同测试集做 zero-shot 测试. 
        """
        
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer)
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        self._print_model_parameters(self.model)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0.0
            n_batch = 0
            
            if is_main_process():
                iterator = tqdm(pretrain_dataloader, desc=f"Pretrain Epoch {epoch+1}")
            else:
                iterator = pretrain_dataloader
            
            for x in iterator:

                x = x.to(self.device).to(torch.bfloat16)  # [B, C, N, T]
                
                # (1) instance normalize(所有通道), 仅保留 flow channel mean/std
                x_norm, means, stds = self.instance_normalize(x)
                
                # 2) 掩码
                x_masked, mask_pos = self.mask_data(x_norm, strategy="right_half") # "right_half", None
                
                # 3) 前向 (预计输出 [B, 1, N, T], 只重建流量通道)
                pred_norm = self.model(x_masked)

                # (4) 反归一化流量
                y_pred = self.instance_denormalize_flow(pred_norm, means, stds)  # [B,1,N,T]

                # (5) 计算 loss (只对被 mask 的流量位置)
                y_truth = x[:, 0:1, :, :]  # [B,1,N,T], 原始流量
                
                loss = criterion(y_pred[mask_pos], y_truth[mask_pos])
                
                # 6) backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batch += 1
            
            avg_loss = total_loss / max(n_batch, 1)
            self.logger.info(f"[Epoch {epoch+1}] Pretrain Loss: {avg_loss:.4f}")
            
            
            # ----- 每个 epoch 后, 对下游数据集做 zero-shot 测试 -----
            if epoch>=4 and (epoch+1) % 1 == 0:
                self.model.eval()
                
                self.logger.info("*"*60 + "   Forecasting Task   " + "*"*60)
                for ds_name, ds_loader in downstream_forecasting_test_dataloader_list.items():
                    self.logger.info(f'Evaluate on {ds_name}:')
                    self.dev_zero_shot_forecasting(ds_loader, mask_strategy="right_half")
                
                self.logger.info("*"*60 + "   Imputation Task   " + "*"*60)
                for ds_name, ds_loader in downstream_imputation_test_dataloader_list.items():
                    self.logger.info(f'Evaluate on {ds_name}:')
                    self.dev_zero_shot_imputation(ds_loader)
            
            
            # ----- 保存模型 ---- #
            early_stopping(avg_loss, self.model, self.saving_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                if scheduler is not None:
                    scheduler.step()
    
    
    def dev_zero_shot_forecasting(self, test_dataloader, mask_strategy="right_half"):
        """
        不微调模型, 直接对下游数据集进行"掩码 + 重建"评估. 
        mask_strategy 可指定, 如 'right_half' 用后半段掩码评估 (预测后半段) 等. 
        返回该策略下的Metrics. 
        """
        metrics = self._select_forecasting_metrics()
        
        self.model.eval()
        
        predict = []
        truth = []

        with torch.no_grad():
            # for x, mask in tqdm(test_dataloader, desc=f"Test"):
            for x, mask in test_dataloader:
                
                x = x.to(self.device).to(torch.bfloat16)
                mask = mask.to(self.device).to(torch.bfloat16)

                # 1) normalize all channels
                x_norm, means, stds = self.instance_normalize(x)

                # 2) 
                x_masked, mask_pos = self.mask_data(x_norm, strategy=mask_strategy)

                # 3) forward => [B, 1, N, T] (flow)
                y_pred = self.model(x_masked)

                # 4) 反归一化流量
                y_pred = self.instance_denormalize_flow(y_pred, means, stds) * mask
                
                # 
                y_truth = x[:, 0:1, :, :] * mask
                
                # 计算目标形状
                select_shape = list(mask_pos.shape)
                select_shape[-1] = select_shape[-1] // 2  # 第四个维度变为 预测一半
                
                predict.append(y_pred[mask_pos].view(select_shape))
                truth.append(y_truth[mask_pos].view(select_shape))
            
            pred = torch.concatenate(predict, axis=0).to(torch.float32)
            true = torch.concatenate(truth, axis=0).to(torch.float32)
            
            for metric in metrics:
                res = []
                for idx in range(pred.shape[-1]):
                    res.append(metric(pred[..., :idx+1], true[..., :idx+1]).detach().cpu())
                self.logger.info(f'{metric.__name__} : {np.mean(res)}')
    
    
    def dev_zero_shot_imputation(self, test_dataloader):
        metrics = self._select_imputation_metrics()
        
        self.model.eval()
        
        predict = []
        truth = []

        with torch.no_grad():
            # for x, eval, mask in tqdm(test_dataloader, desc=f"Test:"):
            for x, eval, mask in test_dataloader:
                
                x = x.to(self.device).to(torch.bfloat16)
                eval = eval.to(self.device).to(torch.bfloat16)
                eval = 1 - eval
                mask = mask.to(self.device).to(torch.bfloat16)

                # 1) normalize all channels
                x_norm, means, stds = self.instance_normalize(x)
                
                # 2) forward -> [B, 1, N, T]
                y_pred = self.model(x_norm)
                
                # 3) 反归一化流量
                y_pred = self.instance_denormalize_flow(y_pred, means, stds) * mask
                
                # 
                y_truth = x[:, 0:1, :, :] * mask
                
                predict.append(y_pred[eval.bool()].cpu().to(torch.float32).numpy())
                truth.append(y_truth[eval.bool()].cpu().to(torch.float32).numpy())
            
            pred = np.concatenate(predict, axis=0)
            true = np.concatenate(truth, axis=0)
            
            for metric in metrics:
                self.logger.info(f'{metric.__name__} : {metric(torch.tensor(pred), torch.tensor(true))}')
