import numpy as np
from models.expert.imputation.classical import MeanImputer, SpatialKNNImputer, SVDImputer, MICEImputer


def masked_mae(y_hat, y, mask):
    err = np.abs(y_hat - y) * mask
    return err.sum() / mask.sum()


def masked_mape(y_hat, y, mask):
    err = np.abs((y_hat - y) / (y + 1e-8)) * mask
    return err.sum() / mask.sum()


def masked_mse(y_hat, y, mask):
    err = np.square(y_hat - y) * mask
    return err.sum() / mask.sum()


def masked_rmse(y_hat, y, mask):
    err = np.square(y_hat - y) * mask
    return np.sqrt(err.sum() / mask.sum())


def masked_mre(y_hat, y, mask):
    err = np.abs(y_hat - y) * mask
    return err.sum() / ((y * mask).sum() + 1e-8)


metrics = {
    'mae': masked_mae,  # 平均绝对误差
    'rmse': masked_rmse,  # 均方根误差
    'mre': masked_mre  # 平均相对误差
}


class Exp_Classical_Imputation:
    def __init__(self, args, adj):
        super().__init__()
        self.in_sample = args.in_sample
        self.dataset_name = args.dataset_name
        self.task_name = args.task_name
        self.logger = args.logger
        self.imputer = self.get_imputer(args.model_name, args, adj)  # 获取插补方法
    
    
    def get_imputer(self, imputer_name, args, adj):
        if imputer_name == 'mean':  # 如果是均值插补
            imputer = MeanImputer(in_sample=args.in_sample)
        elif imputer_name == 'knn':  # 如果是空间最近邻插补
            imputer = SpatialKNNImputer(adj=adj, k=args.k)
        elif imputer_name == 'svd':  # 如果是SVD分解插补
            imputer = SVDImputer(rank=args.rank)
        elif imputer_name == 'mice':  # 如果是多重插补
            imputer = MICEImputer(max_iter=args.mice_iterations, n_nearest_features=args.mice_n_features, in_sample=args.in_sample)
        else:
            raise ValueError(f"Imputer {imputer_name} not available in this setting.")  # 抛出异常
        return imputer
    
    
    def run(self, x_train, mask_train, x_test, mask_test, test_eval_mask):
        
        # if self.in_sample:  # 如果是样本内插补
        #     y_hat = self.imputer.predict(x_train, mask_train)[test_slice]  # 进行预测
        # else:  # 如果是样本外插补
        
        self.imputer.fit(x_train, mask_train)  # 拟合模型
        y_hat = self.imputer.predict(x_test, mask_test)  # 进行预测

        # 评估模型性能
        y_true = x_test
        # eval_mask = test_eval_mask
        eval_mask = test_eval_mask & (y_true != 0)  # 修改：将 y_true 为 0 的位置排除在掩码之外
        
        for metric, metric_fn in metrics.items():  # 遍历评估指标
            error = metric_fn(y_hat, y_true, eval_mask)  # 计算误差
            if metric == 'mre':
                error *= 100
            self.logger.info(f'{self.imputer.name} on {self.dataset_name}-{self.task_name} {metric}: {error:.2f}')  # 打印结果
    