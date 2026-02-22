import torch
import torch.nn as nn


class STID(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(
        self,
        node_num: int,
        in_channel_dim: int = 3,
        input_dim: int = 12,
        output_dim: int = 12,
        if_spatial: bool = True, 
        if_time_in_day: bool = True, 
        if_day_in_week: bool = True,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
        spatial_dim: int = 32,
        temporal_dim_tod: int = 32,
        temporal_dim_dow: int = 32,
        out_channel_dim: int = 32,
        num_layer: int = 3,
    ) -> None:
        super().__init__()
        
        self.if_spatial = if_spatial
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        
        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(node_num, spatial_dim))
            nn.init.xavier_uniform_(self.node_emb)
        
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(time_of_day_size, temporal_dim_tod))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(day_of_week_size, temporal_dim_dow))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(in_channels=in_channel_dim * input_dim, out_channels=out_channel_dim, kernel_size=(1, 1), bias=True)

        # encoding
        hidden_dim = out_channel_dim + spatial_dim * int(self.if_spatial) + temporal_dim_tod * int(self.if_day_in_week) + temporal_dim_dow * int(self.if_time_in_day)
        self.encoder = nn.Sequential(*[MultiLayerPerceptron(hidden_dim, hidden_dim) for _ in range(num_layer)])
        
        # regression
        self.regression_layer = nn.Conv2d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=(1, 1), bias=True)
    
    
    def forward(self, input):
        
        input = input.permute(0, 3, 2, 1)  # (b, f, n, t) -> (b, t, n, f)
        

        if self.if_time_in_day:
            t_i_d_data = input[..., 1]  # [b, t, n, f] -> [b, t, n]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]  # [b, t, n] -> [b, n] -> [b, n, day_emb]
        else:
            time_in_day_emb = None
        
        if self.if_day_in_week:
            d_i_w_data = input[..., 2]  # [b, t, n, f] -> [b, t, n]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]  # [b, t, n] -> [b, n] -> [b, n, week_emb]
        else:
            day_in_week_emb = None
        
        # time series embedding
        B, _, N, _ = input.shape
        input = input.transpose(1, 2).contiguous()  # [b, t, n, f] -> [b, n, t, f]
        input = input.view(B, N, -1).transpose(1, 2).unsqueeze(-1)  # [b, n, t, f] -> [b, n, t * f] -> [b, t * f, n] -> [b, t * f, n, 1]
        time_series_emb = self.time_series_emb_layer(input)  # [b, t * f, n, 1] -> [b, emb_dim, n, 1]
        
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1))  # [n, node_dim] -> [1, n, node_dim] -> [b, n, node_dim] -> [b, node_dim, n, 1]
        
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))  # [b, n, day_emb] -> [b, day_emb, n, 1]
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))  # [b, n, week_emb] -> [b, week_emb, n, 1]

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)  # [b, emb_dim, n, 1] + [b, node_dim, n, 1] + [b, day_emb, n, 1] + [b, week_emb, n, 1] -> [b, emb_dim *  4, n, 1]

        # encoding
        hidden = self.encoder(hidden)  # [b, emb_dim *  4, n, 1] -> [b, emb_dim *  4, n, 1]

        # regression
        prediction = self.regression_layer(hidden)  # [b, emb_dim *  4, n, 1] -> [b, t, n, 1]
        
        prediction = prediction.permute(0, 3, 2, 1) # [b, t, n, 1] -> [b, 1, n, t]

        return prediction


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden
