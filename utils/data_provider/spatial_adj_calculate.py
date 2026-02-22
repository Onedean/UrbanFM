import numpy as np
from scipy.spatial.distance import cdist


def haversine_np(lon1, lat1, lon2, lat2):
    """向量化Haversine距离计算"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def thresholded_gaussian_kernel(distance_matrix, r=0.5, sigma=None):
    """
    阈值高斯核实现
    
    参数：
    distance_matrix : np.ndarray [N x N]
        节点间距离矩阵
    r : float
        相似度阈值 (0~1), 默认0.5
    sigma : float, optional
        自定义标准差，默认自动计算
    
    返回：
    adjacency_matrix : np.ndarray [N x N]
        邻接矩阵（带阈值的高斯核）
    """
    # 有效距离过滤（排除对角线和无效值）
    valid_distances = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool) & (distance_matrix > 0)]
    
    # 自动计算sigma（全局距离标准差）
    if sigma is None:
        sigma = np.std(valid_distances)
        print(f"Automatically computed sigma: {sigma:.2f} km")
    
    # 高斯核计算
    kernel_matrix = np.exp(-(distance_matrix ** 2) / (sigma ** 2))
    
    # 应用阈值
    mask = (kernel_matrix >= r) & (~np.eye(distance_matrix.shape[0], dtype=bool))
    adjacency_matrix = np.where(mask, kernel_matrix, 0)
    
    # 显式设置对角线为0（即使i=j时计算结果满足条件）
    np.fill_diagonal(adjacency_matrix, 0)
    
    # 确保对称性
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    
    return adjacency_matrix.astype(np.float32)


def generate_graph_adjacency(df_spatial, r=0.5, sigma=None):
    """
    基于经纬度生成邻接矩阵
    
    参数：
    df_spatial : DataFrame
        包含传感器ID、纬度、经度的数据框
    r : float
        相似度阈值 (默认0.5)
    sigma : float, optional
        自定义标准差（默认自动计算）
    
    返回：
    adj_matrix : np.ndarray [N x N]
        阈值高斯核邻接矩阵
    distance_matrix : np.ndarray [N x N]
        原始距离矩阵（单位：千米）
    """
    # 坐标提取
    coords = df_spatial[['Longitude', 'Latitude']].values
    
    # 计算全距离矩阵
    distance_matrix = cdist(coords, coords, lambda u, v: haversine_np(u[0], u[1], v[0], v[1]))
    
    # 生成阈值高斯核矩阵
    adj_matrix = thresholded_gaussian_kernel(distance_matrix, r=r, sigma=sigma)
    
    return adj_matrix


def generate_raster_adjacency(df_spatial):
    """
    基于网格坐标生成一阶邻接矩阵 (八邻域)
    
    参数：
    df_spatial : DataFrame
        包含传感器ID、经度、纬度的数据框, 假设经纬度为网格整数坐标
    
    返回：
    adj_matrix : np.ndarray [N x N]
        邻接矩阵, 1表示相邻, 0表示不相邻
    distance_matrix : np.ndarray [N x N]
        欧式距离矩阵（基于网格坐标）
    """
    # 提取经纬度坐标
    lons = df_spatial['Longitude'].values
    lats = df_spatial['Latitude'].values
    
    # 计算坐标差矩阵
    dx = np.abs(lons[:, None] - lons)
    dy = np.abs(lats[:, None] - lats)
    
    # 生成邻接条件：dx和dy均<=1且不同时为0
    adj_matrix = (dx <= 1) & (dy <= 1) & ((dx + dy) > 0)
    
    # 转换为浮点型并确保对称性
    adj_matrix = adj_matrix.astype(np.float32)
    
    return adj_matrix
