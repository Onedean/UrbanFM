
import numpy as np
from fancyimpute import IterativeImputer, IterativeSVD
from sklearn.neighbors import kneighbors_graph  # 导入 sklearn 的 kneighbors_graph，用于计算最近邻图


'''定义一个基类，用于插补方法'''
class Imputer:
    short_name: str  # 插补方法的简短名称

    # 初始化方法
    def __init__(self, method=None, is_deterministic=True, in_sample=True):
        self.name = self.__class__.__name__  # 获取类名作为插补方法的名称
        self.method = method  # 插补方法的具体实现
        self.is_deterministic = is_deterministic  # 是否是确定性方法
        self.in_sample = in_sample  # 是否是样本内插补

    # 拟合方法
    def fit(self, x, mask):
        if not self.in_sample:  # 如果不是样本内插补
            x_hat = np.where(mask, x, np.nan)  # 将缺失值替换为 NaN
            return self.method.fit(x_hat)  # 拟合模型

    # 预测方法
    def predict(self, x, mask):
        x_hat = np.where(mask, x, np.nan)  # 将缺失值替换为 NaN
        if self.in_sample:  # 如果是样本内插补
            return self.method.fit_transform(x_hat)  # 拟合并转换数据
        else:
            return self.method.transform(x_hat)  # 只转换数据

    # 获取参数
    def params(self):
        return dict()  # 返回一个空字典


'''定义一个基于空间最近邻的插补方法'''
class SpatialKNNImputer(Imputer):
    short_name = 'knn'  # 简短名称为 'knn'

    # 初始化方法
    def __init__(self, adj, k=20):
        super(SpatialKNNImputer, self).__init__()  # 调用基类的初始化方法
        self.k = 50 # 最近邻的数量
        sim = (adj + adj.min()) / (adj.max() + adj.min()) # 归一化相似度矩阵
        self.knns = kneighbors_graph(1 - sim, n_neighbors=self.k, include_self=False, metric='precomputed').toarray() # 计算最近邻图

    # 拟合方法（空实现）
    def fit(self, x, mask):
        pass

    # 预测方法
    def predict(self, x, mask):
        x = np.where(mask, x, 0)  # 将缺失值替换为 0
        # 计算加权平均值
        with np.errstate(divide='ignore', invalid='ignore'):
            y_hat = (x @ self.knns.T) / (mask @ self.knns.T)
        y_hat[~np.isfinite(y_hat)] = x.mean()  # 处理无效值
        return np.where(mask, x, y_hat)  # 返回预测结果

    # 获取参数
    def params(self):
        return dict(k=self.k)  # 返回最近邻数量作为参数


'''定义一个均值插补方法'''
class MeanImputer(Imputer):
    short_name = 'mean'  # 简短名称为 'mean'

    # 拟合方法
    def fit(self, x, mask):
        d = np.where(mask, x, np.nan)  # 将缺失值替换为 NaN
        self.means = np.nanmean(d, axis=0, keepdims=True)  # 计算均值

    # 预测方法
    def predict(self, x, mask):
        if self.in_sample:  # 如果是样本内插补
            d = np.where(mask, x, np.nan)  # 将缺失值替换为 NaN
            means = np.nanmean(d, axis=0, keepdims=True)  # 计算每列的均值
        else:
            means = self.means  # 使用拟合时计算的均值
        return np.where(mask, x, means)  # 返回预测结果


'''定义一个基于SVD分解的插补方法'''
class SVDImputer(Imputer):
    short_name = 'svd'  # 简短名称为 'mf'

    # 初始化方法
    def __init__(self, rank=10, loss='mae', verbose=0):
        method = IterativeSVD(rank=rank, verbose=verbose) # 初始化SVD分解方法
        super(SVDImputer, self).__init__(method, is_deterministic=False, in_sample=True)  # 调用基类的初始化方法

    # 获取参数
    def params(self):
        return dict(rank=self.method.rank)  # 返回矩阵分解的秩作为参数


'''定义一个基于多重插补的插补方法'''
class MICEImputer(Imputer):
    short_name = 'mice'  # 简短名称为 'mice'

    # 初始化方法
    def __init__(self, max_iter=100, n_nearest_features=None, in_sample=True, verbose=False):
        method = IterativeImputer(max_iter=max_iter, n_nearest_features=n_nearest_features, verbose=verbose)  # 初始化多重插补方法
        is_deterministic = n_nearest_features is None  # 判断是否是确定性方法
        super(MICEImputer, self).__init__(method, is_deterministic=is_deterministic, in_sample=in_sample)  # 调用基类的初始化方法

    # 获取参数
    def params(self):
        return dict(max_iter=self.method.max_iter, k=self.method.n_nearest_features or -1)  # 返回参数
