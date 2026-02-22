import torch
from torch import nn


class STNorm(nn.Module):
    """
    Paper: ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting
    Link: https://dl.acm.org/doi/10.1145/3447548.3467330
    Ref Official Code: https://github.com/JLDeng/ST-Norm/blob/master/models/Wavenet.py
    """
    def __init__(
        self, 
        node_num: int,
        in_channel_dim: int = 3,
        output_dim: int = 12,
        tnorm_bool: bool = True, 
        snorm_bool: bool = True,
        out_channel_dim: int = 32,
        kernel_size: int = 2, 
        blocks: int = 4, 
        layers: int = 2
    ) -> None:
        
        super().__init__()
        
        self.blocks = blocks
        self.layers = layers
        self.tnorm_bool = tnorm_bool
        self.snorm_bool = snorm_bool
        num = int(self.tnorm_bool) + int(self.snorm_bool) + 1
        receptive_field = 1

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        if self.snorm_bool:
            self.sn = nn.ModuleList()
        if self.tnorm_bool:
            self.tn = nn.ModuleList()
        
        self.dilation = []
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.dilation.append(new_dilation)
                if self.tnorm_bool:
                    self.tn.append(TNorm(node_num, out_channel_dim))
                if self.snorm_bool:
                    self.sn.append(SNorm(out_channel_dim))
                
                self.filter_convs.append(nn.Conv2d(in_channels=num * out_channel_dim, out_channels=out_channel_dim, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=num * out_channel_dim, out_channels=out_channel_dim, kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=out_channel_dim, out_channels=out_channel_dim, kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=out_channel_dim, out_channels=out_channel_dim, kernel_size=(1, 1)))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        
        self.start_conv = nn.Conv2d(in_channels=in_channel_dim, out_channels=out_channel_dim, kernel_size=(1, 1))
        self.end_conv_1 = nn.Conv2d(in_channels=out_channel_dim, out_channels=out_channel_dim, kernel_size=(1, 1), bias=True)
        self.activation = nn.ReLU()
        self.end_conv_2 = nn.Conv2d(in_channels=out_channel_dim, out_channels=output_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field
    
    
    def forward(self, input):
        
        in_len = input.size(3)
        x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0)) if in_len < self.receptive_field else input
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            x_list = [x]

            if self.tnorm_bool:
                x_tnorm = self.tn[i](x)
                x_list.append(x_tnorm)
            if self.snorm_bool:
                x_snorm = self.sn[i](x)
                x_list.append(x_snorm)
            
            # dilated convolution
            x = torch.cat(x_list, dim=1)
            filter = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filter * gate

            # parametrized skip connection
            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
        
        out = self.end_conv_2(self.activation(self.end_conv_1(self.activation(skip))))
        return out


class SNorm(nn.Module):
    def __init__(self,  channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))
    
    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    def __init__(self,  num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum
    
    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out
    