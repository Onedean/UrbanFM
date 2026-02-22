import os


class Exp_Basic(object):
    def __init__(self, args):
        self.saving_path = args.saving_path
        
        self.use_multi_gpu = args.use_multi_gpu
        self.device_ids = args.device_ids
        self.device = args.device
        
        self.logger = args.logger
        self.epochs = args.epochs
        
        self.model_name = args.model_name
        
        self.clip_grad_value = args.clip_grad_value
        
    
    def _build_model(self):
        pass
    
    def _load_model(self):
        pass
    
    def _select_optimizer(self):
        pass
    
    def _select_scheduler(self):
        pass
    
    def _select_criterion(self):
        pass
    
    def _select_metric(self):
        pass
    
    def train(self):
        pass
    
    def test(self):
        pass
