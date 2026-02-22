import torch
import torch.nn as nn


# class SpatialTransformer(nn.Module):
#     """
#     在空间维度进行注意力:
#     输入: [B, C, N, T]
#     - 其中 N 作为序列长度 (sequence length)
#     - T*C 作为 embedding dim (d_model)

#     输出: 同尺寸 [B, C, N, T]
#     """
#     def __init__(self, d_model=64, n_heads=4, num_layers=1, dropout=0.1):
#         super(SpatialTransformer, self).__init__()
        
#         self.temporal_input_proj = nn.Linear(288 * 3, d_model)
#         self.temporal_output_proj = nn.Linear(d_model, 288 * 3)
        
#         # 每层TransformerEncoderLayer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=n_heads, 
#             dim_feedforward=d_model*4,
#             dropout=dropout,
#             activation='relu',
#             batch_first=True  # 使输入形如 [B, seq_len, d_model]
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.d_model = d_model

#     def forward(self, x):
#         """
#         x: [B, C, N, T]
#         先 reshape => [B, N, T*C], 其中 N=seq_len, T*C=d_model
#         -> TransformerEncoder -> reshape回原状
#         """
#         B, C, N, T = x.shape
        
#         # (1) reshape到 [B, N, T*C]
#         #    确保 (T*C) == d_model
#         x_spatial = x.view(B, N, T*C)  # [B, seq_len=N, embed_dim=T*C]
        
#         x_spatial = self.temporal_input_proj(x_spatial)

#         # (2) 过 transformer encoder: [B, N, d_model] -> [B, N, d_model]
#         x_enc = self.encoder(x_spatial)  # [B, N, d_model]
        
#         x_enc = self.temporal_output_proj(x_enc)
        
#         # (3) reshape回 [B, C, N, T]
#         x_out = x_enc.view(B, C, N, T)
#         return x_out


# class TemporalTransformer(nn.Module):
#     """
#     在时间维度进行注意力:
#     输入: [B, C, N, T]
#     - 其中 T 作为序列长度
#     - N*C 作为 embedding dim (d_model)

#     输出: 同尺寸 [B, C, N, T]
#     """
#     def __init__(self, d_model=64, n_heads=4, num_layers=1, dropout=0.1):
#         super(TemporalTransformer, self).__init__()
        
#         self.spatial_input_proj = nn.Linear(64 * 3, d_model)
#         self.spatial_output_proj = nn.Linear(d_model, 64 * 3)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=n_heads,
#             dim_feedforward=d_model*4,
#             dropout=dropout,
#             activation='relu',
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.d_model = d_model

#     def forward(self, x):
#         """
#         x: [B, C, N, T]
#         先 reshape => [B, T, N*C], 其中 T=seq_len, N*C=d_model
#         -> TransformerEncoder -> reshape回原状
#         """
#         B, C, N, T = x.shape
        
#         # (1) reshape到 [B, T, N*C]
#         x_temporal = x.permute(0, 3, 2, 1).contiguous()  # => [B, T, N, C]
#         x_temporal = x_temporal.view(B, T, N*C)          # => [B, T, d_model]
        
#         x_temporal = self.spatial_input_proj(x_temporal)
        
#         # (2) transformer encoder
#         x_enc = self.encoder(x_temporal)  # [B, T, d_model]
        
#         x_enc = self.spatial_output_proj(x_enc)

#         # (3) reshape回 [B, C, N, T]
#         x_out = x_enc.view(B, T, N, C)    # => [B, T, N, C]
#         x_out = x_out.permute(0, 3, 2, 1).contiguous()  # => [B, C, N, T]
#         return x_out


# class STFormer(nn.Module):
#     """
#     先空间后时间的时空Transformer.
#     (B, C, N, T) -> Spatial -> (B, C, N, T) -> Temporal -> (B, C, N, T)
#     最终输出: [B, 1, N, T] (只针对流量通道)
#     """
#     def __init__(self, 
#                 d_model_spatial=64, n_heads_spatial=4, num_layers_spatial=1, 
#                 d_model_temporal=64, n_heads_temporal=4, num_layers_temporal=1, 
#                 dropout=0.1
#     ):
#         """
#         若想让 T*C == d_model_spatial, 需自己保证 T*C=64 或加线性映射
#         同理对时间维度 N*C==d_model_temporal
#         """
#         super(STFormer, self).__init__()

#         # 1) 空间Transformer
#         self.spatial_tf = nn.ModuleList([
#             SpatialTransformer(d_model=d_model_spatial, 
#                                 n_heads=n_heads_spatial, 
#                                 num_layers=num_layers_spatial,
#                                 dropout=dropout)
#         ])

#         # 2) 时间Transformer
#         self.temporal_tf = nn.ModuleList([
#             TemporalTransformer(d_model=d_model_temporal, 
#                                 n_heads=n_heads_temporal, 
#                                 num_layers=num_layers_temporal,
#                                 dropout=dropout)
#         ])

#         # 3) 输出投影: 将 C=3 -> C=1
#         #    可用 1x1 卷积 or Linear(不改变空间/时间形状)
#         self.output_proj = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

#     def forward(self, x):
#         """
#         x: [B, C=3, N, T]
#         - 假设 x 已归一化 & 掩码处理完
        
#         返回: [B, 1, N, T]  (只流量通道)
#         """
#         # 先经过若干层空间Transformer
#         for layer in self.spatial_tf:
#             x = layer(x)

#         # 再经过若干层时间Transformer
#         for layer in self.temporal_tf:
#             x = layer(x)

#         # 最后输出只留流量通道 => 3->1
#         x_out = self.output_proj(x)  # => [B,1,N,T]
#         return x_out








# class AttentionLayer(nn.Module):
#     """Perform attention across the -2 dim (the -1 dim is `model_dim`).

#     Make sure the tensor is permuted to correct shape before attention.

#     E.g.
#     - Input shape (batch_size, in_steps, num_nodes, model_dim).
#     - Then the attention will be performed across the nodes.

#     Also, it supports different src and tgt length.

#     But must `src length == K length == V length`.

#     """

#     def __init__(self, model_dim, num_heads=8, mask=False):
#         super().__init__()

#         self.model_dim = model_dim
#         self.num_heads = num_heads
#         self.mask = mask

#         self.head_dim = model_dim // num_heads

#         self.FC_Q = nn.Linear(model_dim, model_dim)
#         self.FC_K = nn.Linear(model_dim, model_dim)
#         self.FC_V = nn.Linear(model_dim, model_dim)

#         self.out_proj = nn.Linear(model_dim, model_dim)

#     def forward(self, query, key, value):
#         batch_size = query.shape[0]
#         tgt_length = query.shape[-2]
#         src_length = key.shape[-2]

#         query = self.FC_Q(query)
#         key = self.FC_K(key)
#         value = self.FC_V(value)
        
#         query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
#         key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
#         value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

#         key = key.transpose(-1, -2)

#         attn_score = (query @ key) / self.head_dim**0.5

#         if self.mask:
#             mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
#             attn_score.masked_fill_(~mask, -torch.inf)

#         attn_score = torch.softmax(attn_score, dim=-1)
#         out = attn_score @ value
#         out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
#         out = self.out_proj(out)

#         return out


# class SelfAttentionLayer(nn.Module):
#     def __init__(
#         self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
#     ):
#         super().__init__()

#         self.attn = AttentionLayer(model_dim, num_heads, mask)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(model_dim, feed_forward_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(feed_forward_dim, model_dim),
#         )
#         self.ln1 = nn.LayerNorm(model_dim)
#         self.ln2 = nn.LayerNorm(model_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x, dim=-2):
#         x = x.transpose(dim, -2)
#         residual = x
#         out = self.attn(x, x, x)
#         out = self.dropout1(out)
#         out = self.ln1(residual + out)

#         residual = out
#         out = self.feed_forward(out)
#         out = self.dropout2(out)
#         out = self.ln2(residual + out)

#         out = out.transpose(dim, -2)
#         return out



# class STFormer(nn.Module):
#     """
#     时空Transformer. (B, C, N, T) 
#     最终输出: [B, 1, N, T] (只针对流量通道)
#     """
#     def __init__(self, 
#                 d_model_spatial=64, n_heads_spatial=4, num_layers_spatial=2, 
#                 d_model_temporal=64, n_heads_temporal=4, num_layers_temporal=2, 
#                 dropout=0.1
#     ):
#         super(STFormer, self).__init__()
        
#         self.input_proj = nn.Linear(3, 64)
        
#         self.attn_layers_t = nn.ModuleList(
#             [
#                 SelfAttentionLayer(64, d_model_temporal * n_heads_temporal, n_heads_temporal, dropout)
#                 for _ in range(num_layers_temporal)
#             ]
#         )

#         self.attn_layers_s = nn.ModuleList(
#             [
#                 SelfAttentionLayer(64, d_model_spatial * n_heads_spatial, n_heads_spatial, dropout)
#                 for _ in range(num_layers_spatial)
#             ]
#         )
        
#         self.output_proj = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)


#     def forward(self, x):
#         """
#         x: [B, C=3, N, T]
#         - 假设 x 已归一化 & 掩码处理完
        
#         返回: [B, 1, N, T]  (只流量通道)
#         """
        
#         x = x.permute(0, 3, 2, 1) # [b, f, n, t] -> [b, t, n, f]
        
#         x = self.input_proj(x)
        
        
#         for attn in self.attn_layers_t:
#             x = attn(x, dim=1)
        
#         for attn in self.attn_layers_s:
#             x = attn(x, dim=2)
        
#         # 最后输出只留流量通道 => 3->1
#         x_out = self.output_proj(x.transpose(1, 3))  # => [B,1,N,T]
        
#         return x_out













# #  stable version




# import math
# import torch
# import torch.nn as nn

# def rotate_every_two(x):
#     """
#     将最后一个维度以每2个元素为一组进行旋转。
#     即对于 (..., x1, x2) 计算 (-x2, x1)
#     """
#     x1 = x[..., ::2]  # 取偶数位
#     x2 = x[..., 1::2]  # 取奇数位
#     # 返回拼接后的张量，注意这里先构成两两一组，再还原为原始最后一维大小
#     x_rotated = torch.stack((-x2, x1), dim=-1)
#     return x_rotated.flatten(-2)

# def apply_rotary_pos_emb(x, sin, cos):
#     """
#     对张量 x 应用 RoPE。
#     x: [batch_heads, seq_len, head_dim]
#     sin, cos: [1, seq_len, head_dim]（可以广播到 x 的形状）
#     """
#     return (x * cos) + (rotate_every_two(x) * sin)

# def get_sin_cos_emb(seq_len, head_dim, device):
#     """
#     根据序列长度和 head 维度生成正弦和余弦位置编码。
#     这里的编码与论文中给出的方法类似，保证 head_dim 为偶数。
#     返回 shape 均为 [1, seq_len, head_dim]
#     """
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim))
#     positions = torch.arange(seq_len, device=device, dtype=torch.float)
#     # 计算外积得到 shape [seq_len, head_dim/2]
#     sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
#     sin = torch.sin(sinusoid_inp)
#     cos = torch.cos(sinusoid_inp)
#     # 将最后一个维度重复两次以适配 head_dim（[seq_len, head_dim]），再增加 batch 维度 1
#     sin = torch.repeat_interleave(sin.unsqueeze(0), repeats=1, dim=0)  # [1, seq_len, head_dim/2]
#     cos = torch.repeat_interleave(cos.unsqueeze(0), repeats=1, dim=0)
#     # 将最后一维复制（交替），以便与 x 中相邻的两个维度配对
#     sin = torch.stack([sin, sin], dim=-1).flatten(-2)
#     cos = torch.stack([cos, cos], dim=-1).flatten(-2)
#     return sin, cos


# class AttentionLayer(nn.Module):
#     """对倒数第二个维度进行注意力计算（最后一个维度为 model_dim）"""

#     def __init__(self, model_dim, num_heads=8, mask=False):
#         super().__init__()
#         self.model_dim = model_dim
#         self.num_heads = num_heads
#         self.mask = mask

#         self.head_dim = model_dim // num_heads

#         self.FC_Q = nn.Linear(model_dim, model_dim)
#         self.FC_K = nn.Linear(model_dim, model_dim)
#         self.FC_V = nn.Linear(model_dim, model_dim)

#         self.out_proj = nn.Linear(model_dim, model_dim)

#     def forward(self, query, key, value, pos_emb=None):
#         """
#         pos_emb: 可选的 (sin, cos) tuple，用于 RoPE 编码
#                 要求 shape 为 [1, seq_len, head_dim]，
#                 其中 seq_len 为当前注意力计算的序列长度 
#                 （时域自注意力时为时间步数，空域自注意力时为节点数）
#         """
#         batch_size = query.shape[0]
#         tgt_length = query.shape[-2]
#         src_length = key.shape[-2]

#         # 线性变换
#         query = self.FC_Q(query)
#         key = self.FC_K(key)
#         value = self.FC_V(value)
        
#         # 按最后一个维度进行切分，并将 head 维度移到 batch 维度上
#         query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
#         key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
#         value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

#         # 如果传入了位置编码，则对 query 和 key 应用 RoPE
#         if pos_emb is not None:
#             sin, cos = pos_emb  # sin, cos 的 shape 应均为 [1, seq_len, head_dim]
#             query = apply_rotary_pos_emb(query, sin, cos)
#             key = apply_rotary_pos_emb(key, sin, cos)

#         key = key.transpose(-1, -2)
#         attn_score = (query @ key) / math.sqrt(self.head_dim)

#         if self.mask:
#             mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
#             attn_score.masked_fill_(~mask, -torch.inf)

#         attn_score = torch.softmax(attn_score, dim=-1)
#         out = attn_score @ value
#         out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
#         out = self.out_proj(out)

#         return out


# class SelfAttentionLayer(nn.Module):
#     def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
#         super().__init__()
#         self.attn = AttentionLayer(model_dim, num_heads, mask)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(model_dim, feed_forward_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(feed_forward_dim, model_dim),
#         )
#         self.ln1 = nn.LayerNorm(model_dim)
#         self.ln2 = nn.LayerNorm(model_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x, dim=-2):
#         # 将指定维度置换到倒数第二个位置，便于后续注意力计算
#         x = x.transpose(dim, -2)
#         residual = x
        
#         # 获得当前序列长度
#         seq_len = x.shape[-2]
#         # 根据序列长度和每个 head 的维度生成 RoPE 编码
#         sin, cos = get_sin_cos_emb(seq_len, self.attn.head_dim, x.device)
#         # 这里将位置编码传入到 AttentionLayer
#         out = self.attn(x, x, x, pos_emb=(sin, cos))
#         out = self.dropout1(out)
#         out = self.ln1(residual + out)

#         residual = out
#         out = self.feed_forward(out)
#         out = self.dropout2(out)
#         out = self.ln2(residual + out)

#         # 恢复维度顺序
#         out = out.transpose(dim, -2)
#         return out


# class STFormer(nn.Module):
#     """
#     时空Transformer. (B, C, N, T)
#     最终输出: [B, 1, N, T] (只针对流量通道)
#     """
#     def __init__(self, 
#                 d_model_spatial=64, n_heads_spatial=4, num_layers_spatial=2, 
#                 d_model_temporal=64, n_heads_temporal=4, num_layers_temporal=2, 
#                 dropout=0.1):
#         super(STFormer, self).__init__()
        
#         self.input_proj = nn.Linear(3, 64)
        
#         self.attn_layers_t = nn.ModuleList(
#             [
#                 SelfAttentionLayer(64, d_model_temporal * n_heads_temporal, n_heads_temporal, dropout)
#                 for _ in range(num_layers_temporal)
#             ]
#         )

#         self.attn_layers_s = nn.ModuleList(
#             [
#                 SelfAttentionLayer(64, d_model_spatial * n_heads_spatial, n_heads_spatial, dropout)
#                 for _ in range(num_layers_spatial)
#             ]
#         )
        
#         self.output_proj = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

#     def forward(self, x):
#         """
#         x: [B, C=3, N, T]
#         - 假设 x 已归一化 & 掩码处理完
        
#         返回: [B, 1, N, T]  (只流量通道)
#         """
#         x = x.permute(0, 3, 2, 1)  # [B, T, N, C]
#         x = self.input_proj(x)
        
#         # 先进行时间维度的自注意力（T 维度）
#         for attn in self.attn_layers_t:
#             x = attn(x, dim=1)
        
#         # 再进行空间维度的自注意力（N 维度）
#         for attn in self.attn_layers_s:
#             x = attn(x, dim=2)
        
#         # 输出层，注意 Conv2d 要求通道位于第二个维度，因此转换 x 的维度
#         x_out = self.output_proj(x.transpose(1, 3))  # => [B, 1, N, T]
        
#         return x_out
















#  FA Version


import math
import torch
import torch.nn as nn

# ==============================================================================
# 1. 輔助函式 (無變動)
# ==============================================================================

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rotated = torch.stack((-x2, x1), dim=-1)
    return x_rotated.flatten(-2)

def apply_rotary_pos_emb(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def get_sin_cos_emb(seq_len, head_dim, device, dtype):
    # 為保證精度，位置編碼的計算過程使用 float32
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float)
    sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    # 在計算完成後，再轉換為模型所需的低精度類型
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0)
    return sin.to(dtype), cos.to(dtype)


# ==============================================================================
# 2. 優化後的注意力層 (已加入最終修正)
# ==============================================================================

try:
    from flash_attn import flash_attn_func
except ImportError:
    print("FlashAttention 未安裝。請執行: pip install flash-attn --no-build-isolation")
    flash_attn_func = None

class AttentionLayerFlash(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == self.model_dim, "model_dim must be divisible by num_heads"
        self.Wqkv = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x, pos_emb=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.Wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        if pos_emb is not None:
            sin, cos = pos_emb
            q_rope = q.transpose(1, 2).reshape(-1, seq_len, self.head_dim)
            k_rope = k.transpose(1, 2).reshape(-1, seq_len, self.head_dim)
            q_rope = apply_rotary_pos_emb(q_rope, sin, cos)
            k_rope = apply_rotary_pos_emb(k_rope, sin, cos)
            q = q_rope.view(batch_size, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
            k = k_rope.view(batch_size, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        
        # *** 最終修正 ***
        # 在呼叫 flash_attn_func 前，確保 q, k, v 都是正確的 bf16/fp16 類型
        # 我們可以從 v 張量獲取目標 dtype，因為 v 沒有經過 RoPE 的複雜計算
        target_dtype = v.dtype
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        # v 自身類型已經正確，無需轉換，但寫出來更清晰
        v = v.to(target_dtype)

        if flash_attn_func and q.is_cuda:
            out = flash_attn_func(q, k, v, causal=self.mask, softmax_scale=None)
        else: # Fallback
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.mask:
                mask_val = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
                attn_score.masked_fill_(~mask_val, -torch.inf)
            attn_score = torch.softmax(attn_score, dim=-1)
            out = torch.matmul(attn_score, v).transpose(1, 2).contiguous()

        out = out.reshape(batch_size, seq_len, self.model_dim)
        out = self.out_proj(out)
        return out


class SelfAttentionLayerFlash(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0.1, mask=False):
        super().__init__()
        self.attn = AttentionLayerFlash(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim), nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1, self.ln2 = nn.LayerNorm(model_dim), nn.LayerNorm(model_dim)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        batch_size, other_dim, seq_len, model_dim = x.shape
        x_reshaped = x.reshape(batch_size * other_dim, seq_len, model_dim)
        
        sin, cos = get_sin_cos_emb(seq_len, self.attn.head_dim, x.device, x.dtype)
        out_reshaped = self.attn(x_reshaped, pos_emb=(sin, cos))
        
        out = out_reshaped.reshape(batch_size, other_dim, seq_len, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        out = out.transpose(dim, -2)
        return out


class STFormer(nn.Module):
    def __init__(self, 
                d_model_spatial=64, n_heads_spatial=4, num_layers_spatial=2, 
                d_model_temporal=64, n_heads_temporal=4, num_layers_temporal=2, 
                dropout=0.1):
        super(STFormer, self).__init__()
        
        self.input_proj = nn.Linear(3, 64)
        
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayerFlash(64, d_model_temporal * n_heads_temporal, n_heads_temporal, dropout)
                for _ in range(num_layers_temporal)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayerFlash(64, d_model_spatial * n_heads_spatial, n_heads_spatial, dropout)
                for _ in range(num_layers_spatial)
            ]
        )
        
        self.output_proj = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
    
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.input_proj(x)
        for attn in self.attn_layers_t: x = attn(x, dim=1)
        for attn in self.attn_layers_s: x = attn(x, dim=2)
        x_out = self.output_proj(x.permute(0, 3, 2, 1))
        return x_out