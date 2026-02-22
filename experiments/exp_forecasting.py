import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .exp_basic import Exp_Basic
from utils.data_provider.expert_forecasting_process import StandardScaler
from utils.common_tools.earlystop import EarlyStopping
from utils.model_provider.expert_model_wrapper import load_model, load_optimizer, load_scheduler, load_criterion, load_metrics


class Exp_Forecasting(Exp_Basic):
    def __init__(self, args, mean, std):
        super(Exp_Forecasting, self).__init__(args)
        self.args = args
        self.scaler = StandardScaler(torch.tensor(mean), torch.tensor(std))
        self.model = self._build_model()

    def _build_model(self):
        model = load_model(self.model_name, self.args)
        if self.use_multi_gpu and self.device != torch.device('cpu') and len(self.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.args.device_ids, output_device=self.device)
        return model.to(self.device)
    
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
    
    def _select_metrics(self):
        metric = load_metrics(self.args.metric_name)
        return metric
    
    
    def train(self, train_dataloader, valid_dataloader):
        
        checkpoint_path = os.path.join(self.saving_path, 'best_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            self.logger.info("can find the best checkpoint file, loading the model from the checkpoint file.")
            self._load_model()
        else:
            self.logger.info("can not find the best checkpoint file, training from scratch.")
        
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer)
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        for epoch_idx in range(self.epochs):
            
            self.model.train()
            loss_train_list = []
            
            for x, y in tqdm(train_dataloader):
                x, y = x.float().to(self.device), y.float().to(self.device)
                y_pred = self.model(x)
                y_pred = self.scaler.inverse_transform(y_pred)
                
                loss = criterion(y_pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                if str(self.clip_grad_value) != "None":
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
                
                optimizer.step()
                
                loss_train_list.append(loss.item())
            
            self.model.eval()
            
            loss_val_list = []
            
            with torch.no_grad():
                for x, y in tqdm(valid_dataloader):    
                    x, y = x.float().to(self.device), y.float().to(self.device)
                    
                    y_pred = self.model(x)
                    y_pred = self.scaler.inverse_transform(y_pred)
                    loss = criterion(y_pred, y)
                    
                    loss_val_list.append(loss.item())
            
            train_loss = np.mean(loss_train_list)
            valid_loss = np.mean(loss_val_list)
            
            early_stopping(valid_loss, self.model, self.saving_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                if scheduler is not None:
                    scheduler.step()
            
            self.logger.info(f'Epoch {epoch_idx} train loss: {train_loss}, valid loss: {valid_loss}')
    
    
    def test(self, test_dataloader):
        metrics = self._select_metrics()
        
        self._load_model()
        self.model.eval()
        
        predict = []
        truth = []
        
        with torch.no_grad():
            for x, y in tqdm(test_dataloader):
                x, y = x.float().to(self.device), y.float().to(self.device)
                
                y_pred = self.model(x)
                
                y_pred = self.scaler.inverse_transform(y_pred)
                
                predict.append(y_pred.cpu().numpy())
                truth.append(y.cpu().numpy())
            
            pred = np.concatenate(predict, axis=0)
            true = np.concatenate(truth, axis=0)
            
            # 分别保存pred和true到exp_vis文件夹下
            import os
            save_dir = "exp_vis"
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "pred.npy"), pred)
            np.save(os.path.join(save_dir, "true.npy"), true)
            
            for metric in metrics:
                res = []
                for idx in range(pred.shape[-1]):
                    res.append(metric(torch.tensor(pred[..., idx]), torch.tensor(true[..., idx])))
                self.logger.info(f'{metric.__name__} : {np.mean(res)}')
