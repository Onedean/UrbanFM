import torch
from torch import nn

# ---------------------------
# 辅助函数与简单模块

def sampling_generator(N, reverse=False):
    # 生成交替 False 与 True 的列表，长度为 N
    samplings = [False, True] * (N // 2)
    if N % 2:
        samplings.append(False)
    return list(reversed(samplings)) if reverse else samplings

class ConvSC(nn.Module):
    """
    简单卷积块，可选择下采样或上采样。
    - 若 downsample 为 True，则采用 stride=2 的卷积；
    - 若 upsample 为 True，则先卷积后用 PixelShuffle 实现上采样。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False, upsample=False):
        super().__init__()
        padding = kernel_size // 2
        if downsample:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding)
        elif upsample:
            # 上采样时先将通道扩展为目标通道的 4 倍，再用 PixelShuffle 实现 2 倍上采样
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size, stride=1, padding=padding),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
    def forward(self, x):
        return self.conv(x)

# ---------------------------
# Encoder 与 Decoder

class Encoder(nn.Module):
    """
    Encoder：利用若干 ConvSC 层对输入进行下采样。
    """
    def __init__(self, C_in, C_hid, N_S, spatio_kernel=3):
        super().__init__()
        samplings = sampling_generator(N_S)
        layers = []
        # 第一层（保留用于 skip 连接）
        layers.append(ConvSC(C_in, C_hid, kernel_size=spatio_kernel, downsample=samplings[0]))
        self.first_layer = layers[0]
        for s in samplings[1:]:
            layers.append(ConvSC(C_hid, C_hid, kernel_size=spatio_kernel, downsample=s))
        self.enc = nn.Sequential(*layers)
    def forward(self, x):
        enc1 = self.first_layer(x)
        out = enc1
        for layer in self.enc[1:]:
            out = layer(out)
        return out, enc1

class Decoder(nn.Module):
    """
    Decoder：利用若干 ConvSC 层对隐藏特征上采样，并与 Encoder 的 skip 连接融合，
    最后用 1×1 卷积映射回目标通道数。
    """
    def __init__(self, C_hid, C_out, N_S, spatio_kernel=3):
        super().__init__()
        samplings = sampling_generator(N_S, reverse=True)
        layers = []
        for s in samplings[:-1]:
            layers.append(ConvSC(C_hid, C_hid, kernel_size=spatio_kernel, upsample=s))
        layers.append(ConvSC(C_hid, C_hid, kernel_size=spatio_kernel, upsample=samplings[-1]))
        self.dec = nn.Sequential(*layers)
        self.readout = nn.Conv2d(C_hid, C_out, kernel_size=1)
    def forward(self, x, skip):
        out = x
        for layer in self.dec[:-1]:
            out = layer(out)
        out = self.dec[-1](out + skip)
        out = self.readout(out)
        return out

# ---------------------------
# TAU 模块相关

class TAUSubBlock(nn.Module):
    """
    一个简化的 TAU 模块，用于计算时序注意力并调制输入。
    采用大核卷积（kernel_size=21）计算注意力特征，再用 1×1 卷积映射，
    最后经过 Sigmoid 将注意力权重归一化，与原输入逐元素相乘。
    """
    def __init__(self, in_channels, kernel_size=21, mlp_ratio=4., drop=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        attn = self.conv1(x)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn

class MetaBlock(nn.Module):
    """
    MetaBlock 将 TAUSubBlock 包装为基本模块。
    若 in_channels 与 out_channels 不一致，则用 1×1 卷积调整。
    """
    def __init__(self, in_channels, out_channels, model_type='tau', mlp_ratio=4., drop=0.0):
        super().__init__()
        if model_type.lower() == 'tau':
            self.block = TAUSubBlock(in_channels, kernel_size=21, mlp_ratio=mlp_ratio, drop=drop)
        else:
            self.block = nn.Identity()
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.reduction = None
    def forward(self, x):
        out = self.block(x)
        if self.reduction is not None:
            out = self.reduction(out)
        return out

class MidMetaNet(nn.Module):
    """
    隐藏转换器：先将 [B, T, C, H, W] 拼接为 [B, T*C, H, W]，
    再经过若干个 MetaBlock（此处采用 TAUSubBlock），最后恢复为原始时序格式。
    """
    def __init__(self, channel_in, channel_hid, N2, input_resolution, model_type='tau', mlp_ratio=4., drop=0.0):
        super().__init__()
        self.N2 = N2
        layers = []
        # 第1层：将 channel_in 映射到 channel_hid
        layers.append(MetaBlock(channel_in, channel_hid, model_type=model_type, mlp_ratio=mlp_ratio, drop=drop))
        for i in range(1, N2 - 1):
            layers.append(MetaBlock(channel_hid, channel_hid, model_type=model_type, mlp_ratio=mlp_ratio, drop=drop))
        # 最后一层：从 channel_hid 恢复到 channel_in
        layers.append(MetaBlock(channel_hid, channel_in, model_type=model_type, mlp_ratio=mlp_ratio, drop=drop))
        self.enc = nn.Sequential(*layers)
    def forward(self, x):
        # x: [B, T, C, H, W] → [B, T*C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        x = self.enc(x)
        x = x.view(B, T, C, H, W)
        return x


class TAU(nn.Module):
    def __init__(
        self, 
        input_dim = 12,
        in_channel_dim=3, 
        out_channel_dim=1,
        width=32, 
        height=32, 
        hid_S=16, 
        hid_T=256, 
        N_S=4, 
        N_T=4, 
        mlp_ratio=8., 
        drop=0.0
    ) -> None:
        """
        参数说明：
        T         : 时间步数
        width, height : 输入栅格的空间尺寸（N = width×height）
        in_channel: 输入特征数 F（例如 3）
        out_channel: 输出特征数（固定为 1）
        hid_S     : Encoder 的中间通道数
        N_S       : Encoder/Decoder 层数
        N_T       : 隐藏转换器（MidMetaNet）层数
        """
        super().__init__()
        # 根据 Encoder 中哪些层下采样，计算下采样因子
        samplings = sampling_generator(N_S)
        factor = 1
        for s in samplings:
            if s:
                factor *= 2
        width_d = width // factor
        height_d = height // factor

        self.encoder = Encoder(in_channel_dim, hid_S, N_S, spatio_kernel=3)
        self.decoder = Decoder(hid_S, out_channel_dim, N_S, spatio_kernel=3)
        # 隐藏转换器输入通道为 T * hid_S
        self.hid = MidMetaNet(input_dim * hid_S, hid_T, N_T, input_resolution=(height_d, width_d),
                            model_type='tau', mlp_ratio=mlp_ratio, drop=drop)
        self.T = input_dim
        self.width = width
        self.height = height
        self.in_channel = in_channel_dim
        self.out_channel = out_channel_dim

    def forward(self, x):
        """
        输入 x: [B, F, N, T]  其中 N = width * height, F = in_channel.
        流程：
        1. reshape 为 (B, F, width, height, T) → permute 成 [B, T, F, width, height]
        2. 合并 B 与 T 维度送入 Encoder 得到 embed 和 skip（embed: [B*T, hid_S, H_d, W_d]）
        3. 重塑为 [B, T, hid_S, H_d, W_d] 后送入隐藏转换器（MidMetaNet）
        4. 恢复为 [B*T, hid_S, H_d, W_d] 后送入 Decoder 得到 Y: [B*T, out_channel, width, height]
        5. 重塑 Y 为 [B, T, out_channel, width, height]，再转换为 [B, out_channel, N, T]
        """
        B, F, N, T = x.shape
        assert F == self.in_channel, "输入特征数不匹配"
        assert N == self.width * self.height, "N 必须等于 width×height"
        assert T == self.T, "时间步数不匹配"
        # reshape: [B, F, N, T] → [B, F, width, height, T]
        x = x.view(B, F, self.width, self.height, T)
        # permute: [B, F, width, height, T] → [B, T, F, width, height]
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        # 合并 B 与 T 维度: [B*T, F, width, height]
        x = x.view(B * T, F, self.width, self.height)
        embed, skip = self.encoder(x)  # embed: [B*T, hid_S, H_d, W_d]
        _, C_enc, H_d, W_d = embed.shape
        # 重塑为 [B, T, C_enc, H_d, W_d]
        z = embed.view(B, T, C_enc, H_d, W_d)
        # 隐藏转换器对时序特征进行建模，输出形状仍为 [B, T, C_enc, H_d, W_d]
        z = self.hid(z)
        # 合并 B 与 T: [B*T, C_enc, H_d, W_d]
        z = z.view(B * T, C_enc, H_d, W_d)
        # Decoder 得到 Y: [B*T, out_channel, width, height]
        Y = self.decoder(z, skip)
        # 重塑为 [B, T, out_channel, width, height]
        Y = Y.view(B, T, self.out_channel, self.width, self.height)
        # 调整为 [B, out_channel, width, height, T] 再 reshape 为 [B, out_channel, N, T]
        Y = Y.permute(0, 2, 3, 4, 1).contiguous()  # [B, out_channel, width, height, T]
        Y = Y.reshape(B, self.out_channel, self.width * self.height, T)
        return Y