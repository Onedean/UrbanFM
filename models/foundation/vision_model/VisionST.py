import torch  # 导入 PyTorch 库，用于张量操作和深度学习模型
import os  # 导入 os 模块，用于文件和路径操作
from . import models_mae  # 导入本地模块 `models_mae`，定义了 MAE 模型
import einops  # 导入 einops 库，用于高效的张量重排操作
import torch.nn.functional as F  # 导入 PyTorch 的功能模块，提供常用函数
from torch import nn  # 从 PyTorch 导入神经网络模块
from PIL import Image  # 导入 PIL 库，用于图像处理
from . import util  # 导入本地模块 `util`，提供辅助功能


# 定义 MAE 模型架构和对应权重文件
MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}


# 定义 MAE 模型权重的下载地址
MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"


# 定义 VisionTS 类，继承自 nn.Module，用于时间序列预测
class VisionST(nn.Module):

    def __init__(self, arch='mae_base', finetune_type='ln', ckpt_dir='./ckpt/', load_ckpt=True):
        super(VisionST, self).__init__()

        # 检查指定的架构是否存在
        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")

        # 初始化指定的 MAE 模型
        self.vision_model = MAE_ARCH[arch][0]()

        # 加载模型权重
        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])  # 构建本地权重文件路径
            if not os.path.isfile(ckpt_path):  # 如果文件不存在，从远程下载
                remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[arch][1]
                util.download_file(remote_url, ckpt_path)
            try:
                # 加载权重并应用到模型
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                self.vision_model.load_state_dict(checkpoint['model'], strict=True)
            except Exception:
                print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")

        # 根据 finetune_type 调整模型的可训练参数
        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if finetune_type == 'ln':
                    param.requires_grad = 'norm' in n
                elif finetune_type == 'bias':
                    param.requires_grad = 'bias' in n
                elif finetune_type == 'none':
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n

    # 更新配置，根据输入时间序列的特性初始化相关参数
    def update_config(self, context_len, pred_len, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear'):
        self.image_size = self.vision_model.patch_embed.img_size[0]  # 图像尺寸，例如 224
        self.patch_size = self.vision_model.patch_embed.patch_size[0]  # Patch 尺寸，例如 16
        self.num_patch = self.image_size // self.patch_size  # Patch 数量，例如 14

        self.context_len = context_len  # 输入时间序列长度
        self.pred_len = pred_len  # 预测时间序列长度
        self.periodicity = 224  # 数据周期性，例如 24

        # 计算填充长度
        self.pad_left = 0
        self.pad_right = 0
        # if self.context_len % self.periodicity != 0:
        #     self.pad_left = self.periodicity - self.context_len % self.periodicity
        # if self.pred_len % self.periodicity != 0:
        #     self.pad_right = self.periodicity - self.pred_len % self.periodicity

        # 计算输入与输出的比例
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        # 配置插值方法
        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        # 初始化输入和输出的尺寸调整参数
        self.input_resize = util.safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const

        # 初始化 Mask
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()

    # 定义前向传播过程
    def forward(self, x, export_image=False, fp64=False):
        # 规范化
        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.norm_const
        x_enc /= stdev
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

        # 分块
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate')
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)

        # 图像输入
        x_resize = self.input_resize(x_2d)
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([x_resize, masked], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)

        # 模型推理
        _, y, mask = self.vision_model(image_input, mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0]))
        image_reconstructed = self.vision_model.unpatchify(y)

        # 预测结果
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True)
        y_segmentations = self.output_resize(y_grey)
        y_flatten = einops.rearrange(y_segmentations, '(b n) 1 f p -> b (p f) n', b=x_enc.shape[0], f=self.periodicity)
        y = y_flatten[:, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :]

        # 反规范化
        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        if export_image:
            # 返回预测图像
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.vision_model.patch_embed.patch_size[0]**2 *3)
            mask = self.vision_model.unpatchify(mask)
            image_reconstructed = image_input * (1 - mask) + image_reconstructed * mask
            green_bg = -torch.ones_like(image_reconstructed) * 2
            image_input = image_input * (1 - mask) + green_bg * mask
            image_input = einops.rearrange(image_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            image_reconstructed = einops.rearrange(image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            return y, image_input, image_reconstructed

        return y
