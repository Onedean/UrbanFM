import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):  # [16, 170, 12, 152] / [16, 170, 12, 152]
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]  # [16, 170, 12, 152] -> 16 / [16, 12, 170, 152] -> 16
        tgt_length = query.shape[-2]  # [16, 170, 12, 152] -> 12 / [16, 12, 170, 152] -> 170
        src_length = key.shape[-2]  # [16, 170, 12, 152] -> 170 / [16, 12, 170, 152] -> 170

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)  # [16, 170, 12, 152] -> [64, 170, 12, 38] / [16, 12, 170, 152] -> [64, 12, 170, 38]
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)  # [16, 170, 12, 152] -> [64, 170, 12, 38] / [16, 12, 170, 152] -> [64, 12, 170, 38]
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)  # [16, 170, 12, 152] -> [64, 170, 12, 38] / [16, 12, 170, 152] -> [64, 12, 170, 38]

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out) # [64, 12, 170, 152]

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)  # (batch_size, time_steps, node_num, model_dim) -> (batch_size, node_num, time_steps, model_dim) / (batch_size, time_steps, node_num, model_dim)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    def __init__(
        self,
        node_num: int,
        in_channel_dim: int = 3,
        input_dim: int = 12,
        output_dim: int = 12,
        if_spatial: bool = False, 
        if_time_in_day: bool = True, 
        if_day_in_week: bool = True,
        if_adaptive: bool = True,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        spatial_dim: int = 0,
        temporal_dim_tod: int = 24,
        temporal_dim_dow: int = 24,
        adaptive_dim: int = 80,
        out_channel_dim: int = 24,
        if_use_mixed_proj: bool = True,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        
        super().__init__()
        
        self.node_num = node_num
        
        self.if_spatial = if_spatial
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_adaptive = if_adaptive
        self.if_use_mixed_proj = if_use_mixed_proj
        
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        
        
        self.in_channel_dim = in_channel_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.input_proj = nn.Linear(in_channel_dim, out_channel_dim)
        
        if if_time_in_day:
            self.tod_embedding = nn.Embedding(time_of_day_size, temporal_dim_tod)
        
        if if_day_in_week:
            self.dow_embedding = nn.Embedding(day_of_week_size, temporal_dim_dow)
        
        if if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(node_num, spatial_dim)
            )
            nn.init.xavier_uniform_(node_num)
        
        if if_adaptive:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(input_dim, node_num, adaptive_dim))
            )
        
        hidden_dim = out_channel_dim + spatial_dim + temporal_dim_tod + temporal_dim_dow + adaptive_dim
        self.hidden_dim = hidden_dim
        
        if if_use_mixed_proj:
            self.output_proj = nn.Linear(
                input_dim * hidden_dim, output_dim * 1
            )
        else:
            self.temporal_proj = nn.Linear(input_dim, output_dim)
            self.output_proj = nn.Linear(hidden_dim, 1)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(hidden_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(hidden_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        
        x = x.permute(0, 3, 2, 1) # [b, f, n, t] -> [b, t, n, f]
        # x: (batch_size, time_steps, node_num, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.if_time_in_day:
            tod = x[..., 1]
        
        if self.if_day_in_week:
            dow = x[..., 2]
        
        x = x[..., : self.in_channel_dim]

        x = self.input_proj(x)  # (batch_size, time_steps, node_num, input_embedding_dim)
        features = [x]
        
        if self.if_time_in_day:
            tod_emb = self.tod_embedding(
                (tod * self.time_of_day_size).long()
            )  # (batch_size, time_steps, node_num, tod_embedding_dim)
            features.append(tod_emb)
        
        if self.if_day_in_week:
            dow_emb = self.dow_embedding(
                (dow * self.day_of_week_size).long()
            )  # (batch_size, time_steps, node_num, dow_embedding_dim)
            features.append(dow_emb)
        
        if self.if_spatial:
            spatial_emb = self.node_emb.expand(
                batch_size, self.input_dim, *self.node_emb.shape
            )
            features.append(spatial_emb)
        
        if self.if_adaptive:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        
        x = torch.cat(features, dim=-1)  # (batch_size, time_steps, node_num, model_dim)
        
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        
        if self.if_use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, node_num, time_steps, model_dim)
            out = out.reshape(
                batch_size, self.node_num, self.input_dim * self.hidden_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.node_num, self.output_dim, 1
            )
            out = out.transpose(1, 2)  # (batch_size, horizon, node_num, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, node_num, time_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, node_num, horizon)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, horizon, node_num, output_dim)
        
        out = out.permute(0, 3, 2, 1)  # (B, F, N, T)
        
        return out
