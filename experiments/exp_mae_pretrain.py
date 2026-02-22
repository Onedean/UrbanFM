import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .exp_basic import Exp_Basic
from utils.data_provider.expert_forecasting_process import StandardScaler
from utils.common_tools.earlystop import EarlyStopping
from utils.model_provider.expert_model_wrapper import load_optimizer, load_scheduler, load_metrics
from models.foundation.denoise_model.MaskedAutoencoderST import MaskedAutoencoderST


class Exp_MAE_Pretrain(Exp_Basic):
    """
    Exp_MAE_Pretrain 实验类, 包含:
    - dev_pretrain: 使用随机 Mask 的策略进行预训练 (mask_strategy='random') 
    每个 epoch 结束后再用 mask_strategy='right_half' 在下游测试集上做一次 quick test
    - finetune: 在下游数据集上使用右半边 Mask (mask_strategy='right_half') 进行微调
    - test: 同样在右半边 Mask 下做最终评估
    """

    def __init__(self, args, mean, std):
        super(Exp_MAE_Pretrain, self).__init__(args)
        self.args = args
        self.scaler = StandardScaler(torch.tensor(mean), torch.tensor(std))
        self.model = self._build_model()

    def _build_model(self):
        """
        直接构造我们改好的 MaskedAutoencoderViT:
        - 初始的 img_size=(64, 288), patch_size=(8, 12)
        - 保留可学习位置编码
        - 其它超参数可以根据需求自行修改
        """
        model = MaskedAutoencoderST(
            img_size=(64, 288),
            patch_size=(8, 12),
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.,
            norm_pix_loss=False
        )
        # 多卡支持 (若需要) 
        if self.use_multi_gpu and self.device != torch.device('cpu') and len(self.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.args.device_ids, output_device=self.device)
        return model.to(self.device)
    
    def _load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.saving_path, 'best_checkpoint.pth')))
    
    def _select_optimizer(self):
        return load_optimizer(self.model, self.args.optimizer_name, self.args.learning_rate, self.args.weight_decay)
    
    def _select_scheduler(self, optimizer):
        return load_scheduler(optimizer, self.args.scheduler_name, self.args)
    
    def _select_metrics(self):
        return load_metrics(self.args.metric_name)
    
    
    # =========== 1) 预训练: 随机 Mask + 下游 quick test (右半边 Mask)  ===========
    def dev_pretrain(self, pretrain_dataloader, downstream_test_dataloader):
        """
        在预训练数据集上使用随机 Mask (mask_strategy='random'), 
        每个 epoch 后对下游测试集调用 _quick_test(mask_strategy='right_half')
        """
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch_idx in range(self.epochs):
            self.model.train()
            loss_train_list = []

            # === 随机 Mask 训练 ===
            for x, y in tqdm(pretrain_dataloader, desc=f'[Pretrain Epoch {epoch_idx}]'):
                x, y = x.float().to(self.device), y.float().to(self.device)

                # forward 时指定 mask_strategy='random'
                # mask_ratio=0.75 可自定义
                loss, pred, mask = self.model(x, mask_ratio=0.75, mask_strategy='random')
                # 这里 pred 是重建的 patch 像素；loss 已经在 forward_loss 中计算完毕
                optimizer.zero_grad()
                loss.backward()

                if str(self.clip_grad_value) != "None":
                    nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
                optimizer.step()

                loss_train_list.append(loss.item())

            train_loss = np.mean(loss_train_list)

            # === 下游测试集 quick test (mask_strategy='right_half') ===
            test_loss = self._quick_test(downstream_test_dataloader)

            early_stopping(test_loss, self.model, self.saving_path)
            if early_stopping.early_stop:
                print("Early stopping in dev_pretrain.")
                break
            else:
                if scheduler is not None:
                    scheduler.step()

            self.logger.info(f'[Pretrain] Epoch {epoch_idx}, train loss: {train_loss}, test loss: {test_loss}')\
    
    
    def _quick_test(self, dataloader):
        """
        快速测试: 在下游测试集上, 用右半边 Mask (mask_strategy='right_half') 看 loss
        """
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.float().to(self.device), y.float().to(self.device)

                # 不使用随机 Mask, 而是只 Mask 图像右半边
                loss, pred, mask = self.model(x, mask_ratio=0.0, mask_strategy='right_half')
                losses.append(loss.item())
        return np.mean(losses)
    
    
    # =========== 2) 微调: 右半边 Mask ===========
    def finetune(self, train_dataloader, valid_dataloader):
        """
        在下游训练集和验证集上进行右半边 Mask 的训练
        """
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch_idx in range(self.epochs):
            self.model.train()
            loss_train_list = []

            for x, y in tqdm(train_dataloader, desc=f'[Finetune Epoch {epoch_idx}]'):
                x, y = x.float().to(self.device), y.float().to(self.device)

                # 右半边 Mask
                loss, pred, mask = self.model(x, mask_ratio=0.0, mask_strategy='right_half')

                optimizer.zero_grad()
                loss.backward()
                if str(self.clip_grad_value) != "None":
                    nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
                optimizer.step()

                loss_train_list.append(loss.item())

            train_loss = np.mean(loss_train_list)

            # 验证
            self.model.eval()
            loss_val_list = []
            with torch.no_grad():
                for x, y in tqdm(valid_dataloader, desc='[Finetune Valid]'):
                    x, y = x.float().to(self.device), y.float().to(self.device)
                    loss, pred, mask = self.model(x, mask_ratio=0.0, mask_strategy='right_half')
                    loss_val_list.append(loss.item())

            valid_loss = np.mean(loss_val_list)

            early_stopping(valid_loss, self.model, self.saving_path)
            if early_stopping.early_stop:
                print("Early stopping in finetune.")
                break
            else:
                if scheduler is not None:
                    scheduler.step()

            self.logger.info(f'[Finetune] Epoch {epoch_idx}, train loss: {train_loss}, valid loss: {valid_loss}')
    
    
    # =========== 3) 测试: 右半边 Mask ===========
    def test(self, test_dataloader):
        """
        最终测试: 右半边 Mask, 计算下游任务的指标
        """
        metrics = self._select_metrics()
        self._load_model()
        self.model.eval()

        predict = []
        truth = []

        with torch.no_grad():
            for x, y in tqdm(test_dataloader, desc='[Testing]'):
                x, y = x.float().to(self.device), y.float().to(self.device)
                
                # 测试也采用右半边 Mask
                loss, pred, mask = self.model(x, mask_ratio=0.0, mask_strategy='right_half')
                predict.append(pred.cpu().numpy())
                truth.append(y.cpu().numpy())

        pred = np.concatenate(predict, axis=0)
        true = np.concatenate(truth, axis=0)

        # 根据你的下游任务 & 数据维度, 做相应指标计算
        for metric in metrics:
            val = metric(torch.tensor(pred), torch.tensor(true))
            self.logger.info(f'{metric.__name__} : {val.item()}')
