import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

# ---------------------------
# 辅助函数：空间位置编码
def get_angles(pos, l, d):
    """
    计算位置编码中的角度
    pos: 标量位置
    l: 维度索引数组，形状 (1, d)
    d: 总维度
    """
    angle_rates = 1 / np.power(10000, (2 * (l // 2)) / np.float32(d))
    return torch.tensor(pos * angle_rates, dtype=torch.float32)

def spatial_posenc(r, c, d, device):
    """
    根据行、列和维度 d 计算空间位置编码
    返回形状为 (1, 1, d)
    """
    angle_rads_r = get_angles(pos=r, l=np.arange(d)[np.newaxis, :], d=d)
    angle_rads_c = get_angles(pos=c, l=np.arange(d)[np.newaxis, :], d=d)
    pos_encoding = torch.zeros((1, d), dtype=torch.float32, device=device)
    pos_encoding[:, 0::2] = torch.sin(angle_rads_r[:, 0::2])
    pos_encoding[:, 1::2] = torch.cos(angle_rads_c[:, 1::2])
    return pos_encoding.unsqueeze(0)  # (1, 1, d)

# ---------------------------
# 简化的 DSAN 模型
# 模型仅依赖原始输入 X，内部自动生成局部邻域、空间位置编码等

class DSAN(nn.Module):
    def __init__(
        self, 
        input_dim: int=12, 
        output_dim: int=12, 
        width: int=32,
        height: int=32, 
        in_channel_dim: int=3, 
        d: int=64, 
        l_d: int=1
    ) -> None:
        """
        参数：
        input_window  : 输入时间步数 T
        output_window : 预测时间步数（这里为简化，我们使其与 T 相同）
        row, column   : 网格空间尺寸（N = width * height
        feature_dim   : 输入每个节点的通道数（例如 1 或更多，此处我们只使用预测目标通道）
        d             : 空间位置编码的维度
        l_d           : 局部块半径（实际块尺寸 L_d = 2*l_d+1）
        """
        super().__init__()
        self.input_window = input_dim
        self.output_window = output_dim
        self.row = width
        self.column = height
        self.feature_dim = in_channel_dim
        self.output_dim = 1  # 我们预测的目标通道固定为1
        self.l_d = l_d
        self.L_d = 2 * l_d + 1
        self.d = d
        # 定义一个非常简化的 dsan 模块，这里仅用一层 3x3 卷积 + Tanh 作为示意
        # 输入的通道数设为 input_window，输出通道数设为 output_window（通常 output_window= input_window）
        self.dsan = nn.Sequential(
            nn.Conv2d(in_channels=self.input_window, out_channels=self.output_window, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def generate_x(self, x):
        """
        输入 x 的形状为 [B, T, row, column, feature_dim]
        我们只取 x 的前 output_dim 个通道作为预测目标（这里假设 feature_dim >= output_dim）
        生成：
        - dae_inp_g: 原始目标数据，形状 [B, T, row, column, 1]
        - dae_inp: 对每个网格位置提取局部 patch（大小 L_d x L_d），形状 [B, T, row, column, L_d, L_d, 1]
        - sad_inp: 取最后 output_window 时间步，reshape 为 [B, output_window, row*column, 1]
        - cors: 局部块的空间位置编码，形状 [1, 1, L_d*L_d, d]
        - cors_g: 全局网格的空间位置编码，形状 [1, row*column, d]
        """
        B, T, r, c, F = x.shape
        device = x.device
        # 取预测目标通道（假设为第1个通道）
        X_target = x[..., :self.output_dim]  # [B, T, r, c, 1]
        dae_inp_g = X_target  # 原始数据，形状 [B, T, r, c, 1]
        # 局部邻域：先对空间维度进行 padding
        X_pad = nn.functional.pad(X_target.permute(0,1,4,2,3), (self.l_d, self.l_d, self.l_d, self.l_d))  
        # 形状变为 [B, T, 1, r+2*l_d, c+2*l_d]
        X_pad = X_pad.permute(0,1,3,4,2)  # [B, T, r+2*l_d, c+2*l_d, 1]
        # 对每个位置提取局部 patch
        patches = []
        for i in range(r):
            row_list = []
            for j in range(c):
                patch = X_pad[:, :, i:i+self.L_d, j:j+self.L_d, :]  # [B, T, L_d, L_d, 1]
                row_list.append(patch)
            row_stack = torch.stack(row_list, dim=2)  # [B, T, c, L_d, L_d, 1]
            patches.append(row_stack)
        dae_inp = torch.stack(patches, dim=2)  # [B, T, r, c, L_d, L_d, 1]
        # sad_inp: 取最后 output_window 时间步
        sad_inp = X_target[:, -self.output_window:, ...]  # [B, output_window, r, c, 1]
        sad_inp = sad_inp.view(B, self.output_window, r*c, 1)
        # cors: 为局部块的每个位置计算空间位置编码
        cors_list = []
        for i in range(self.L_d):
            for j in range(self.L_d):
                pe = spatial_posenc(i - self.l_d, j - self.l_d, self.d, device=device)  # [1, 1, d]
                cors_list.append(pe.squeeze(0).squeeze(0))  # [d]
        cors = torch.stack(cors_list, dim=0)  # [L_d*L_d, d]
        cors = cors.unsqueeze(0).unsqueeze(0)   # [1, 1, L_d*L_d, d]
        # cors_g: 为全局网格中每个位置计算空间位置编码
        cors_g_list = []
        for i in range(r):
            for j in range(c):
                pe = spatial_posenc(i - r//2, j - c//2, self.d, device=device)  # [1, 1, d]
                cors_g_list.append(pe.squeeze(0).squeeze(0))  # [d]
        cors_g = torch.stack(cors_g_list, dim=0)  # [r*c, d]
        cors_g = cors_g.unsqueeze(0)  # [1, r*c, d]
        return dae_inp_g, dae_inp, sad_inp, cors, cors_g

    def forward(self, x):
        """
        输入 x 的形状为 [B, T, row, column, feature_dim]
        生成各分支输入后，采用简化的 dsan 模块对原始数据进行融合，输出预测结果。
        本示例中仅采用原始数据（dae_inp_g）进行简单融合。
        """
        # 将 x 从 [B, F, N, T] 转换为 [B, T, N, F]
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, T, N, F]
        # 重构 N 为 (len_row, len_column)：[B, T, len_row, len_column, F]
        x = x.view(x.shape[0], self.input_window, self.row, self.column, self.feature_dim)
        
        B, T, r, c, _ = x.shape
        # 生成各分支输入
        dae_inp_g, dae_inp, sad_inp, cors, cors_g = self.generate_x(x)
        # 此处我们仅采用 dae_inp_g（形状 [B, T, r, c, 1]）进行处理
        # 将 dae_inp_g squeeze最后一维，得到 [B, T, r, c]
        X_fuse = dae_inp_g.squeeze(-1)
        # 为了利用 2D 卷积，我们将时间步 T 视为通道
        # 即将 X_fuse 重构为 [B, T, r, c]
        # 直接送入 dsan 模块（其输入要求形状为 [B, C, H, W]）
        out = self.dsan(X_fuse)  # 输出形状 [B, output_window, r, c]
        # 这里假设 output_window == T；将输出 reshape 为 [B, 1, N, T]，其中 N = r * c
        out = out.view(B, self.output_window, r*c)
        # 将通道维放到第二个维度，最终输出 [B, 1, N, T]
        out = out.unsqueeze(1)  # [B, 1, output_window, N]
        # 交换最后两个维度： [B, 1, N, output_window]
        out = out.permute(0, 1, 3, 2).contiguous()
        return out

    def predict(self, x):
        return self.forward(x)