import os, sys, json, re, random, torch, subprocess, logging
import numpy as np
import os.path as osp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_config(args):
    '''
    初始化配置参数
    '''
    def _update(src, info):
        # 仅更新终端没有提供的参数
        for key, value in info.items():
            if not hasattr(args, key) or getattr(args, key) is None:  # 检查是否已提供
                src[key] = value
    
    def _load_json_file(file_path):
        with open(file_path, "r") as f:
            s = f.read()
            s = re.sub('\s',"", s)
        return json.loads(s)
    
    conf_path = osp.join(args.config)  # 拼接配置文件路径
    info = _load_json_file(conf_path)  # 加载 JSON 配置文件
    _update(vars(args), info)  # 更新 args 参数
    
    if args.task_name == 'short_forecasting' or args.task_name == 'long_forecasting':
        if args.few_shot_ratio == 1.0:
            vars(args)["eval_type"] = "full_shot"
        elif args.few_shot_ratio == 0.0:
            vars(args)["eval_type"] = "zero_shot"
        else:
            vars(args)["eval_type"] = "few_shot"
    elif args.task_name == 'point_imputation' or args.task_name == 'block_imputation':
        vars(args)["eval_type"] = "."
    
    if args.model_type =='expert':
        vars(args)["saving_path"] = osp.join(args.log_name, f'{args.model_type}/{args.task_name}/{args.eval_type}/{args.dataset_name}/{args.model_name}/{args.seed}')  # 创建模型保存路径
    elif args.model_type == 'foundation':
        vars(args)["saving_path"] = osp.join(args.log_name, f'{args.model_type}/{args.task_name}/dataset_ratio_{args.scaling_ratio*100}%/model_layer_{args.num_layers_spatial}_{args.num_layers_temporal}/{args.seed}')  # 创建模型保存路径
    
    _mkdirs(args.saving_path)  # 创建对应目录
    
    del info  # 删除配置信息字典

def is_main_process():
    # 判断是否为主进程
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def init_log(args):
    '''
    初始化日志记录对象
    '''
    logger = logging.getLogger(__name__)  # 创建一个logger对象
    
    if is_main_process():
        logger.setLevel(logging.INFO)  # 设置logger的日志级别为INFO
        fh = logging.FileHandler(osp.join(args.saving_path, "record.log"))  # 创建一个文件处理器，将日志写入指定文件
        fh.setLevel(logging.INFO)  # 设置日志格式
        ch = logging.StreamHandler(sys.stdout)  # 创建一个流处理器，将日志输出到标准输出
        ch.setLevel(logging.INFO)  
        formatter = logging.Formatter("%(asctime)s - %(message)s")  # 设置日志格式
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)  # 将文件处理器和流处理器添加到logger中
        logger.addHandler(ch)
        logger.info("logger name:%s", osp.join(args.saving_path, "record.log"))  # 记录日志初始化的信息
    else:
        # 非主进程，将日志级别设置为CRITICAL，而INFO等常见级别会被忽略
        logger.setLevel(logging.CRITICAL) 

    vars(args)["logger"] = logger  # 确保所有进程的args都有logger属性
    


def _get_free_gpus(memory_threshold=1024*48):
    """
    获取当前服务器中显存占用小于指定阈值的 空闲GPU 的 ID 列表。
    
    参数：
    - memory_threshold (int): 认为 GPU 空闲的显存使用阈值，单位为 MiB。
    
    返回：
    - list: 空闲 GPU 的 ID 列表。
    """
    try:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
            shell=True
        ).decode('utf-8')
        gpu_info = [line.split(',') for line in result.strip().split('\n')]    
        free_gpus = [int(gpu[0]) for gpu in gpu_info if int(gpu[1]) <= memory_threshold]
        return free_gpus
    except subprocess.CalledProcessError as e:
        print(f"Error while trying to get GPU info: {e}")
        return []


def init_distributed():
    """
    初始化分布式训练环境
    """
    if 'LOCAL_RANK' not in os.environ:
        return None
    
    # 设置当前设备
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    
    return local_rank


def init_device(args):
    """
    设置设备, 支持CPU、单GPU、多GPU(DP)和分布式训练(DDP)
    """
    # 尝试初始化分布式环境
    local_rank = init_distributed()
    
    if local_rank is not None:
        vars(args)["device"] = torch.device(f'cuda:{local_rank}')
        vars(args)["local_rank"] = local_rank
        vars(args)["distributed"] = True
        return

    if not args.use_multi_gpu or not torch.cuda.is_available():
        print("No GPUs found. Using CPU.")
        vars(args)["device"] = torch.device('cpu')
        vars(args)["distributed"] = False
        return
    
    free_gpus = _get_free_gpus()
    
    if not free_gpus or (args.device_ids and not set(args.device_ids).issubset(free_gpus)):
        print("No free GPUs or No free target GPUs found. Using CPU.")
        vars(args)["device"] = torch.device('cpu')
        vars(args)["distributed"] = False
        return
    
    if args.use_multi_gpu:
        if args.device_ids:
            vars(args)["device"] = torch.device('cuda', args.device_ids[0])
        else:
            vars(args)["device"], vars(args)["device_ids"] = torch.device('cuda', free_gpus[0]), free_gpus
    else:
        if args.device_ids:
            vars(args)["device"], vars(args)["device_ids"] = torch.device(f'cuda:{args.device_ids[0]}'), args.device_ids[0]
        else:
            vars(args)["device"], vars(args)["device_ids"] = torch.device(f'cuda:{free_gpus[0]}'), free_gpus[0]
    vars(args)["distributed"] = False


def seed_anything(seed):
    '''
    初始化随机种子
    '''
    random.seed(seed)  # 设置 Python 内置的随机数生成器的种子
    np.random.seed(seed)  # 设置 NumPy 随机数生成器的种子
    torch.manual_seed(seed)  # 设置 PyTorch CPU 随机数生成器的种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch GPU 随机数生成器的种子（单个GPU）
    torch.cuda.manual_seed_all(seed)  # 设置 PyTorch GPU 随机数生成器的种子（所有GPU）
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False  # 关闭 CuDNN 的 benchmark 功能
    torch.backends.cudnn.deterministic = True  # 设置 CuDNN 的 deterministic 模式

