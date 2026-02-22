import torch
from torch import nn
from collections import OrderedDict

# 3x3 卷积，padding=1
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=True)

# BN-ReLU-Conv 模块 (可选 BatchNorm) 
class BnReluConv(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(BnReluConv, self).__init__()
        self.has_bn = bn
        if self.has_bn:
            self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(nb_filter, nb_filter)
    def forward(self, x):
        if self.has_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x

# 残差单元，由两个 BN-ReLU-Conv 组成
class ResidualUnit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(ResidualUnit, self).__init__()
        self.bn_relu_conv1 = BnReluConv(nb_filter, bn)
        self.bn_relu_conv2 = BnReluConv(nb_filter, bn)
    def forward(self, x):
        residual = x
        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)
        out += residual  # short cut
        return out

# 多个残差单元堆叠
class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1, bn=False):
        super(ResUnits, self).__init__()
        layers = []
        for i in range(repetations):
            layers.append(residual_unit(nb_filter, bn))
        self.stacked_resunits = nn.Sequential(*layers)
    def forward(self, x):
        return self.stacked_resunits(x)

# 可训练的逐元素融合层 (矩阵融合) ，输入尺寸 [B, n, H, W]
class TrainableEltwiseLayer(nn.Module):
    def __init__(self, n, h, w):
        super(TrainableEltwiseLayer, self).__init__()
        # 不指定 device，后续可通过 model.to(device) 移动
        self.weights = nn.Parameter(torch.randn(1, n, h, w), requires_grad=True)
    def forward(self, x):
        return x * self.weights

# 改造后的 ST‐ResNet 模型 (仅使用 closeness 分支，无外部输入) 
class STResNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        width, 
        height, 
        in_channel_dim=3, 
        out_channel_dim=1,
        nb_residual_unit=12, 
        batch_norm=False
    ) -> None:
        """
        参数说明：
        T              : 时间步数 (既作为输入时间步数，也作为输出时间步数) 
        width, height  : 网格空间尺寸 (N = width * height) 
        in_channel     : 输入特征数 (F) 
        out_channel    : 输出特征数 (固定为1) 
        nb_residual_unit: 残差单元堆叠次数
        batch_norm     : 是否使用 BatchNorm
        """
        super().__init__()
        self.T = input_dim
        self.width = width
        self.height = height
        self.in_channel = in_channel_dim
        self.output_dim = out_channel_dim  # 固定输出为1
        self.len_row = height
        self.len_column = width
        self.feature_dim = in_channel_dim  # 每节点原始特征维度
        self.nb_residual_unit = nb_residual_unit
        self.bn = batch_norm
        # 输入到 closeness 分支的通道数为 T * in_channel
        in_channels = self.T * self.feature_dim
        # 构造 closeness 分支：conv1 -> 残差单元堆叠 -> ReLU -> conv2 -> 融合层
        # conv2 输出通道设为 T (每个预测时刻产生1个通道) 
        self.c_way = nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels, 64)),
            ('ResUnits', ResUnits(ResidualUnit, nb_filter=64, repetations=self.nb_residual_unit, bn=self.bn)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(64, self.T)),
            ('FusionLayer', TrainableEltwiseLayer(n=self.T, h=self.len_row, w=self.len_column))
        ]))
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        """
        输入 x 的形状为 [B, F, N, T]，其中 N = width * height
        1. 将 x 重构为 [B, T, len_row, len_column, F] (其中 F = in_channel) 
        2. 将各时间步的特征沿通道拼接，得到 [B, T * F, len_row, len_column]
        3. 送入 closeness 分支得到输出 (形状 [B, T, len_row, len_column]) 
        4. 最后重构输出为 [B, 1, N, T]
        """
        B, F, N, T_in = x.shape
        assert F == self.in_channel, "输入特征数不匹配"
        assert N == self.len_row * self.len_column, "N 必须等于 width * height"
        assert T_in == self.T, "输入时间步数必须等于 T"
        # 将 x 从 [B, F, N, T] 转换为 [B, T, N, F]
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, T, N, F]
        # 重构 N 为 (len_row, len_column)：[B, T, len_row, len_column, F]
        x = x.view(B, self.T, self.len_row, self.len_column, F)
        # 将各时间步的特征沿通道拼接：转换为 [B, T * F, len_row, len_column]
        x = x.view(B, self.T * F, self.len_row, self.len_column)
        # 送入 closeness 分支
        out = self.c_way(x)  # 输出形状 [B, T, len_row, len_column]
        out = self.tanh(out)
        # 将输出调整为 [B, 1, len_row, len_column, T]
        out = out.view(B, self.T, self.len_row, self.len_column)
        out = out.unsqueeze(1)  # [B, 1, T, len_row, len_column]
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # [B, 1, len_row, len_column, T]
        # reshape为 [B, 1, N, T]
        out = out.view(B, 1, self.len_row * self.len_column, self.T)
        return out
        