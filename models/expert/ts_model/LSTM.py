from torch import nn


class LSTMNet(nn.Module):
    """A simple LSTM-based neural network for regression tasks."""
    def __init__(
        self, 
        in_channel_dim: int = 3,
        output_dim: int = 12,
        out_channel_dim: int = 32, 
        hidden_dim: int = 64, 
        num_layers: int = 2,
        dropout: float = 0.1,
        end_dim: int = 512,
    ) -> None:
        
        super().__init__()
        
        self.start_conv = nn.Conv2d(in_channels=in_channel_dim, out_channels=out_channel_dim, kernel_size=(1,1))
        self.lstm = nn.LSTM(input_size=out_channel_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.activation = nn.ReLU()
        self.end_linear1 = nn.Linear(hidden_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, output_dim)
    
    def forward(self, input):
        b, f, n, t = input.shape
        
        x = input.transpose(1,2).reshape(b*n, f, 1, t)  # (b, f, n, t) -> (b, n, f, t) -> (b * n, f, 1, t)
        
        x = self.start_conv(x).squeeze().transpose(1, 2)  # (b * n, f, 1, t) -> (b * n, init_dim, 1, t) -> (b * n, init_dim, t) -> (b * n, t, init_dim)
        
        out, _ = self.lstm(x)  # (b * n, t, hidden_dim) -> (b * n, t, hidden_dim)
        x = out[:, -1, :] # (b * n, t, hidden_dim) -> (b * n, hidden_dim)

        x = self.activation(self.end_linear1(x)) # (b * n, hidden_dim) -> (b * n, end_dim)
        x = self.end_linear2(x) # (b * n, end_dim) -> (b * n, output_dim)
        x = x.reshape(b, n, 1, t).transpose(1, 2) # (b * n, output_dim) -> (b, n, 1, t) -> (b, 1, n, t)
        return x
