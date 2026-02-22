import torch
from torch import nn

class SpatioTemporalLSTMCellv2(nn.Module):
    """
    Adapted from OpenSTL's SpatioTemporalLSTMCellv2
    Note: The code here is consistent with OpenSTL for capturing spatiotemporal features.
    """
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCellv2, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                        stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        # 对当前输入 x_t 做卷积
        x_concat = self.conv_x(x_t)
        # 对上一时刻隐状态 h_t 做卷积
        h_concat = self.conv_h(h_t)
        # 对辅助记忆 m_t 做卷积
        m_concat = self.conv_m(m_t)
        # 按通道数拆分得到各个门的信息
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = \
            torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        # 第一组门控：更新细胞状态 c_t
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        # 第二组门控：更新辅助记忆 m_t
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        # 拼接更新后的 c_new 与 m_new，并计算输出门
        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m


class PredRNNv2(nn.Module):
    """
    Modified PredRNNv2 model
    - Input: (b, f, w, h, t), where f=3 (the first channel is the target value, and the rest are time features)
    - Output: (b, f, w, h, t), where f=1 (only predict the target value)

    Core idea:
    1. Send the input frames (note that they are adjusted to the shape required by the convolution) to the multi-layer SpatioTemporalLSTMCellv2 to update the state in the order of time steps;
    2. At each time step, the last hidden state is used as the feature output of the time step;
    3. After being mapped to the output channel by 1×1 convolution, the output of (b, 1, w, h, t) is finally obtained.
    """
    def __init__(
        self,
        in_channel_dim=3,
        out_channel_dim=1, 
        width=28,
        height=32,
        num_hidden=64, 
        num_layers=2,
        filter_size=3, 
        layer_norm=True
    ) -> None:
        """
        Parameter description:
        - in_channel_dim: number of input channels (3)
        - out_channel_dim: number of output channels (1)
        - num_hidden: number of hidden state channels per cell
        - num_layers: number of stacked cell layers
        - height, width: spatial dimensions (Note: this requires corresponding to the input h and w;
        Since the input shape is (b, f, w, h, t), we will swap w and h when processing)
        """
        super().__init__()
        self.width = width
        self.height = height
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            # 第一层输入通道为 in_channel_dim，其它层为 num_hidden
            cell_in_channel = in_channel_dim if i == 0 else num_hidden
            self.cells.append(
                SpatioTemporalLSTMCellv2(in_channel=cell_in_channel,
                                        num_hidden=num_hidden,
                                        height=height,
                                        width=width,
                                        filter_size=filter_size,
                                        stride=1,
                                        layer_norm=layer_norm)
            )
        # 最后将最后一层隐状态映射到目标输出通道（1）
        self.conv_last = nn.Conv2d(num_hidden, out_channel_dim, kernel_size=1, stride=1, padding=0, bias=False)

    # def forward(self, input):
    #     """
    #     The shape of input is converted to (b, f, w, h, t)
    #     Processing flow:
    #     1. For each time step, extract the corresponding frame from input and convert (b, f, w, h) to (b, f, h, w) (adapt conv2d).
    #     2. Call SpatioTemporalLSTMCellv2 layer by layer to update the state of each layer (h_t, c_t, m_t).
    #     3. The hidden state of the last layer is used as the output of the current time step.
    #     4. Use conv_last to do 1×1 mapping for each frame output, and finally stack them to get (b, 1, h, w, t), and then adjust to (b, 1, w, h, t).
    #     """
        
    #     B, F, N, T = input.shape
    #     assert N == self.width * self.height, "input shape error"
    #     # 将 N 展开为 (width, height) 得到 (B, F, width, height, T)
    #     input = input.view(B, F, self.width, self.height, T)
        
    #     # 注意：输入 shape 为 (b, f, w, h, t)
    #     b, f, w, h, t = input.shape
    #     device = input.device
    #     # 注意 conv2d 要求 (b, channel, height, width)
    #     # 因此我们用输入的 h（第4维）作为 height，w（第3维）作为 width

    #     # 初始化各层的隐状态和细胞状态，尺寸均为 (b, num_hidden, height, width)
    #     h_t = [torch.zeros(b, self.cells[0].num_hidden, h, w, device=device) for _ in range(self.num_layers)]
    #     c_t = [torch.zeros(b, self.cells[0].num_hidden, h, w, device=device) for _ in range(self.num_layers)]
    #     m_t = torch.zeros(b, self.cells[0].num_hidden, h, w, device=device)

    #     outputs = []
    #     for t_idx in range(t):
    #         # 从输入中提取第 t_idx 帧，原 shape (b, f, w, h)
    #         x = input[..., t_idx]
    #         # 调整为 (b, f, h, w) 以符合 conv2d 要求
    #         x = x.permute(0, 1, 3, 2)
    #         for i, cell in enumerate(self.cells):
    #             if i == 0:
    #                 h_t[i], c_t[i], m_t, _, _ = cell(x, h_t[i], c_t[i], m_t)
    #             else:
    #                 h_t[i], c_t[i], m_t, _, _ = cell(h_t[i - 1], h_t[i], c_t[i], m_t)
    #         # 记录最后一层的隐状态作为当前时间步的输出
    #         outputs.append(h_t[-1])
    #     # 将所有时间步输出堆叠，shape 为 (b, num_hidden, h, w, t)
    #     outputs = torch.stack(outputs, dim=-1)

    #     # 对每个时间步应用 conv_last 映射到输出通道
    #     pred_frames = []
    #     for t_idx in range(t):
    #         frame = outputs[..., t_idx]  # (b, num_hidden, h, w)
    #         frame_pred = self.conv_last(frame)  # (b, out_channel_dim, h, w)
    #         pred_frames.append(frame_pred)
    #     # 堆叠得到 (b, out_channel_dim, h, w, t)
    #     pred = torch.stack(pred_frames, dim=-1)
    #     # 如果需要输出 (b, f, w, h, t) 且要求空间维顺序为 (w, h)，则将 h 与 w 交换
    #     pred = pred.permute(0, 1, 3, 2, 4)
        
    #     # 输出转换：将 (B, out_channel_dim, w, h, t) reshape 为 [B, out_channel_dim, N, t]
    #     pred = pred.reshape(b, -1, self.width * self.height, t)
        
    #     return pred

    def forward(self, input):
        """
        Parameters:
        - input: input of the past T_in steps, shape [B, F, N, T_in], where F=3, N=width*height

        Process:
        1. Reshape the input to (B, F, w, h, T_in)
        2. Encoder: Process the past T_in frames by time step and update the hidden state
        3. Decoder: Autoregressively generate the future T_f frames, and the input of each step consists of the target predicted at the previous moment and the corresponding time feature
        4. Reshape the generated future frames to [B, 1, N, T_f]
        """
        # --- 输入转换 ---
        # 如果输入为 4 维 [B, F, N, T_in]，则转换为 (B, F, w, h, T_in)
        B, F, N, T_in = input.shape
        assert N == self.width * self.height, "输入节点数与 width*height 不匹配"
        input = input.view(B, F, self.width, self.height, T_in)

        device = input.device

        # --- Encoder: 处理过去 T_in 步 ---
        # 初始化各层状态，尺寸均为 (B, num_hidden, height, width)
        h_t = [torch.zeros(B, self.cells[0].num_hidden, self.height, self.width, device=device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(B, self.cells[0].num_hidden, self.height, self.width, device=device) for _ in range(self.num_layers)]
        m_t = torch.zeros(B, self.cells[0].num_hidden, self.height, self.width, device=device)

        for t_idx in range(T_in):
            # 从输入中提取第 t_idx 帧，形状 (B, F, w, h)
            x = input[..., t_idx]
            # 为适应 conv2d，将 (B, F, w, h) 转为 (B, F, h, w)
            x = x.permute(0, 1, 3, 2)
            for i, cell in enumerate(self.cells):
                if i == 0:
                    h_t[i], c_t[i], m_t, _, _ = cell(x, h_t[i], c_t[i], m_t)
                else:
                    h_t[i], c_t[i], m_t, _, _ = cell(h_t[i - 1], h_t[i], c_t[i], m_t)

        # --- Decoder: 自回归预测未来 T_f 步 ---
        future_preds = []
        # 初始未来时刻的输入：使用 encoder 最后时刻的预测结果作为目标值部分
        # 注意 conv_last 计算预测目标，shape 为 (B, 1, h, w)
        pred_target = self.conv_last(h_t[-1])
        # 由于没有提供未来的时间特征，则从最后一帧输入中提取时间特征（通道 1 和 2）
        # 注意：原始输入的时间特征在输入时位于通道 2 和 3，原始 shape 为 (B, 3, w, h)
        last_input = input[..., -1]  # shape (B, 3, w, h)
        
        for t_idx in range(T_in):
            # 使用最后一帧的时间特征（通道 1 和 2），并转换维度
            ft = last_input[:, 1:3, :, :].permute(0, 1, 3, 2)
            # 构造未来步输入：拼接预测目标（B,1, h, w）和时间特征（B,2, h, w），得到 (B,3, h, w)
            x_future = torch.cat([pred_target, ft], dim=1)
            # 用 x_future 更新各层状态（与 Encoder 中类似）
            for i, cell in enumerate(self.cells):
                if i == 0:
                    h_t[i], c_t[i], m_t, _, _ = cell(x_future, h_t[i], c_t[i], m_t)
                else:
                    h_t[i], c_t[i], m_t, _, _ = cell(h_t[i - 1], h_t[i], c_t[i], m_t)
            # 计算当前步的预测目标
            pred_target = self.conv_last(h_t[-1])
            future_preds.append(pred_target)

        # 将未来预测帧堆叠，得到 shape (B, 1, h, w, T_f)
        future_preds = torch.stack(future_preds, dim=-1)
        # 由于在 conv2d 中我们用 (B, *, h, w)，需要先将 h 与 w 交换回 (B, 1, w, h, T_f)
        future_preds = future_preds.permute(0, 1, 3, 2, -1)
        # 最后 reshape 为 [B, 1, N, T_f]，其中 N = w * h
        B, out_ch, w, h, T_f = future_preds.shape
        future_preds = future_preds.reshape(B, out_ch, w * h, T_f)
        return future_preds
    