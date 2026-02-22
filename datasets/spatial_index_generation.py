import numpy as np
import math
from matplotlib import colormaps
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle
import os
import pickle
import argparse
import concurrent.futures
import warnings

warnings.filterwarnings("ignore")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Haversine公式计算两点之间的大圆距离 (单位: 公里 )
    """
    R = 6371.0  # 地球半径 (单位: km )
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def cluster_points_kdtree(points, d=16):
    """
    使用 KDTree 对经纬度点进行聚类，每个簇包含 d 个点。
    """
    n = len(points)
    tree = cKDTree(points)
    assigned = [False] * n
    clusters = []
    clusters_idx = []
    if d <= n:
        for i in range(n):
            if not assigned[i]:
                # 查找最近的 d 个点
                distances, indices = tree.query(points[i], k=d)
                # 确保 indices 是一个列表
                if not isinstance(indices, (list, np.ndarray)):
                    indices = [indices]
                # 过滤未分配的点
                cluster = []
                for idx in indices:
                    if not assigned[idx]:
                        cluster.append(idx)
                    if len(cluster) == d:
                        break
                # 如果不足 d 个点，继续搜索更多点
                if len(cluster) < d:
                    distances_extra, indices_extra = tree.query(points[i], k=n)
                    for idx in indices_extra:
                        if not assigned[idx] and idx not in cluster:
                            cluster.append(idx)
                        if len(cluster) == d:
                            break
                # 标记为已分配
                for idx in cluster:
                    assigned[idx] = True
                # 添加到簇
                clusters.append([points[idx] for idx in cluster])
                clusters_idx.append(cluster)
        
        # 处理可能不足 d 个点的最后一个簇
        if len(clusters) > 0 and len(clusters[-1]) < d:
            last_cluster = clusters.pop()
            last_cluster_idx = clusters_idx.pop()
            needed = d - len(last_cluster)
            # 查找最接近的点
            for point in last_cluster:
                distances, indices = tree.query(point, k=n)
                for idx in indices:
                    if points[idx] not in last_cluster:
                        last_cluster.append(points[idx])
                        last_cluster_idx.append(idx)
                        needed -= 1
                        if needed == 0:
                            break
                if needed == 0:
                    break
            clusters.append(last_cluster)
            clusters_idx.append(last_cluster_idx)
    else:
        # 当 d > n 时，将所有点分配到一个簇中, 并重复直到簇的大小达到 d
        full_clusters = d // n  # 完整簇的数量
        remaining = d % n       # 剩余点的数量

        # 创建完整簇
        for _ in range(full_clusters):
            clusters.extend(points)
            clusters_idx.extend(list(range(n)))

        # 创建小簇（如有需要）
        if remaining > 0:
            small_cluster = points[:remaining]
            small_cluster_idx = list(range(remaining))
            clusters.extend(small_cluster)
            clusters_idx.extend(small_cluster_idx)
        clusters = [clusters]
        clusters_idx = [clusters_idx]
    
    return clusters, clusters_idx


def plot_clusters(clusters, title="Clusters", save_path="./"):
    """
    可视化聚类结果并保存图像。

    参数:
    - clusters: 聚类结果列表，每个聚类包含一组点。
    - title: 图像标题。
    - save_path: 图像保存路径。
    """
    plt.figure(figsize=(10, 8))
    cmap_name = 'tab20' if len(clusters) <= 20 else 'tab20b'  # 'tab20b' 或 'tab20c' 可用于更多颜色
    colors = plt.get_cmap(cmap_name, len(clusters))
    
    
    ax = plt.gca()
    
    for i, cluster in enumerate(clusters):
        lat_list = [p[0] for p in cluster]
        lon_list = [p[1] for p in cluster]
        plt.scatter(lon_list, lat_list, color=colors(i), label=f"C{i+1}", s=100)
        
        # 计算簇的边界
        min_lat, max_lat = min(lat_list), max(lat_list)
        min_lon, max_lon = min(lon_list), max(lon_list)
        width = max_lon - min_lon
        height = max_lat - min_lat
        
        # 创建一个透明度较低的矩形
        rect = Rectangle((min_lon, min_lat), width, height, linewidth=3, edgecolor=colors(i), facecolor='none')
        ax.add_patch(rect)
    
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{title}.png"))
    plt.close()


def generate_mask(clusters_idx, n, d):
    """
    生成对应的mask。

    参数:
    - clusters_idx: 聚类索引列表，每个子列表包含一个簇中点的索引。
    - n: 数据集中点的总数。
    - d: 每个簇的大小。

    返回:
    - mask: 二维列表，表示每个簇中点的掩码。
    """
    
    if d < n:
        num_clusters = n // d
        remaining = n % d
        
        # 创建 num_clusters 个全1的列表
        mask = [[1] * d for _ in range(num_clusters)]

        # 仅在有剩余点时，添加最后一个部分为1和0的列表
        if remaining > 0:
            mask.append([1] * remaining + [0] * (d - remaining))
    else:
        mask = [[1] * n + [0] * (d - n)]
    return mask


def process_dataset(dataset_path, d):
    """
    处理单个数据集: 读取 location.pkl，进行聚类，生成可视化图像，并保存 clusters_idx。

    参数:
    - dataset_path: 数据集文件夹路径。
    - d: 聚类参数。
    """
    dataset_name = dataset_path.split("/")[-1]
    location_pkl = os.path.join(dataset_path, f"{dataset_name}_spatial.pkl")
    if not os.path.exists(location_pkl):
        print(f"{dataset_name}_spatial.pkl not found in {dataset_path}, skipping.")
        return

    with open(location_pkl, "rb") as f:
        locations = pickle.load(f)
        lat = locations['Latitude'].values
        lng = locations['Longitude'].values
        points = list(zip(lat, lng))
    
    # 调用聚类函数
    clusters, clusters_idx = cluster_points_kdtree(points, d)
    
    # 生成并保存聚类图像
    title = f"clustering_with_{d}"
    plot_clusters(clusters, title=title, save_path=dataset_path)
    
    # 保存 clusters_idx
    idx_filename = f"index_{d}.pkl"
    idx_path = os.path.join(dataset_path, idx_filename)
    with open(idx_path, "wb") as f_idx:
        pickle.dump(clusters_idx, f_idx)
    
    # 生成并保存 mask
    mask = generate_mask(clusters_idx, len(points), d)
    
    mask_filename = f"mask_{d}.pkl"
    mask_path = os.path.join(dataset_path, mask_filename)
    with open(mask_path, "wb") as f_mask:
        pickle.dump(mask, f_mask)
    
    print(f"Processed dataset at {dataset_path}")
    

def main():
    parser = argparse.ArgumentParser(description="Cluster location data with greedy kdtree.")
    parser.add_argument('--d', type=int, default=32, help='Clustering parameter d (default: 32)')
    parser.add_argument('--type', type=str, default="pretrain_datasets", help='preprocess type (default: downstream)')
    args = parser.parse_args()
    d = args.d
    
    folder_path = os.path.join(args.type)
    if not os.path.exists(folder_path):
        print(f"{folder_path} does not exist.")
        return
    
    # 获取 downstream 文件夹下的所有数据集文件夹
    datasets = [os.path.join(folder_path, name) for name in os.listdir(folder_path) 
                if os.path.isdir(os.path.join(folder_path, name))]
    
    
    if not datasets:
        print(f"No datasets found in {folder_path}.")
        return

    # 并行处理所有数据集
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_dataset, dataset, d) for dataset in datasets]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing dataset: {e}")


if __name__ == "__main__":
    main()