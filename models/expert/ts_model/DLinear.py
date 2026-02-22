import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block"""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
        Paper: Are Transformers Effective for Time Series Forecasting?
        Link: https://arxiv.org/abs/2205.13504
        Official Code: https://github.com/cure-lab/DLinear
        Venue: AAAI 2023
        Task: Long-term Time Series Forecasting
    """
    def __init__(
        self,
        node_num: int,
        in_channel_dim: int = 3,
        input_dim: int = 12,
        output_dim: int = 12,
        individual: bool = False,
    ) -> None:
        super(DLinear, self).__init__()
        self.seq_len = input_dim
        self.pred_len = output_dim

        # Decompsition Kernel Size
        kernel_size = 25 # according to my tests, 25 is the relevant best value, smaller values will even lead to worse results
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = node_num

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, input):
        """Feed forward of DLinear.

        Args:
            input (torch.Tensor): input data with shape [B, F, N, T]

        Returns:
            torch.Tensor: prediction with shape [B, F, N, T]
        """
        input = input.permute(0, 3, 2, 1) # (b, f, n, t) -> (b, t, n, f)
        
        x = input[..., 0]     # B, T, N
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(
                1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(
                1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        prediction = seasonal_output + trend_output
        return prediction.permute(0, 2, 1).unsqueeze(-1).permute(0, 3, 2, 1)  # [B, T, N, 1] -> [B, 1, N, T]
