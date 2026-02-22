import math
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from torch import nn, Tensor
from torch_geometric.nn import inits
from torch_geometric.typing import OptTensor
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F


class NexuSQN(nn.Module):
    """
    Paper: Contextualizing MLP-Mixers Spatiotemporally for Urban Data Forecast at Scale
    Link: https://arxiv.org/abs/2307.01482
    Official Code: https://github.com/tongnie/NexuSQN
    Note: we have made a little modifications to the original code (replace sinusoidal positional encoding with hard encoding feature).
    
    Args:
        input_size (int): Size of the input.
        input_window_size (int): Size of the input window.
        input_embedding_dim (int): Size of the input projection.
        output_size (int): Size of the output.
        horizon (int): Forecasting steps.
        node_dim (int): Size of node embedding.
        exog_size (int): Size of the optional exogenous variables.
        num_layer (int): Number of dense layers in the TimeMixer.
        st_embd (int): Whether to use a spatiotemporal embedding.
    """

    def __init__(self,
                node_num: int,
                in_channel_dim=3,
                input_dim=12,
                output_dim=12,
                out_channel_dim=128,
                node_dim=96,
                num_layer=2
    ) -> None:
        
        super().__init__()
        
        self.input_encoder = nn.Linear(in_channel_dim * input_dim, out_channel_dim)
        
        # Spatiotemporal embeddings
        self.emb = nn.Parameter(
            torch.empty(node_num, node_dim))
        nn.init.xavier_uniform_(self.emb)
        
        self.u_enc = PositionalEncoder(in_channels=in_channel_dim-1 , out_channels=node_dim, n_layers=2, steps=input_dim, n_nodes=node_num)
        
        
        hidden_size = out_channel_dim + node_dim
        
        # TimeMixer blocks
        self.TimeMixer = nn.Sequential(
            *[MultiLayerPerceptron(hidden_size, hidden_size) for _ in range(num_layer)]
        )

        # SpaceMixer blocks
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )
        
        # Readout blocks
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_size, output_dim * 1),
            Rearrange('b n (h f) -> b h n f', h=output_dim, f=1)
        )


    def forward(self, x):
        B, _, N, _ = x.size()
        
        u = x[:, 1:, 0, :]
        
        # flat time dimension
        x = rearrange(x, 'b f n t -> b n (t f)')
        x = self.input_encoder(x)  # [b n c]

        # TimeMixer with spatial context
        
        # add encoding
        # q = self.u_enc(u, node_emb=self.emb)
                
        q = self.u_enc(u, node_index=torch.arange(N).repeat(B, 1))
        
        x = torch.cat([x, q], dim=-1)  # [b n c]
        
        x = self.TimeMixer(x) + x
        
        # SpaceMixer, single layer implementation
        # softmax kernel method
        e = torch.softmax(self.emb, dim=-1)  # [n c]
        et = torch.softmax(self.emb.T, dim=-1)  # [c n]

        # x: (batch_size, ..., length, model_dim)
        residual = x
        # space mxing
        x = et@x  # [c n] * [n c] -> [c c]
        x = e@x  # [n c] * [c c] -> [n c]
        x = self.linear(x)  # [b n c]
        x = F.gelu(x)
        x = residual + x

        residual = x
        x = self.feed_forward(x)  # (batch_size, ..., length, model_dim)
        x = residual + x

        # MultiStep Readout
        x = self.decoder(x) + x
        x = self.readout(x)

        return x




def expand_then_cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]],
                    dim: int = -1) -> Tensor:
    """Match the dimensions of tensors in the input list and then concatenate.

    Args:
        tensors (list): Tensors to concatenate.
        dim (int): Dimension along which to concatenate.
            (default: -1)
    """
    shapes = [t.shape for t in tensors]
    expand_dims = torch.max(torch.tensor(shapes), 0).values
    expand_dims[dim] = -1
    tensors = [t.expand(*expand_dims) for t in tensors]
    return torch.cat(tensors, dim=dim)


def maybe_cat_exog(x, u, dim=-1):
    r"""
    Concatenate `x` and `u` if `u` is not `None`.

    We assume `x` to be a 4-dimensional tensor, if `u` has only 3 dimensions we
    assume it to be a global exog variable.

    Args:
        x: Input 4-d tensor.
        u: Optional exogenous variable.
        dim (int): Concatenation dimension.

    Returns:
        Concatenated `x` and `u`.
    """
    if u is not None:
        if u.dim() == 3:
            u = rearrange(u, 'b s f -> b s 1 f')
        x = expand_then_cat([x, u], dim)
    return x


def get_layer_activation(activation: Optional[str] = None):
    
    _torch_activations_dict = {
    'elu': 'ELU',
    'leaky_relu': 'LeakyReLU',
    'prelu': 'PReLU',
    'relu': 'ReLU',
    'rrelu': 'RReLU',
    'selu': 'SELU',
    'celu': 'CELU',
    'gelu': 'GELU',
    'glu': 'GLU',
    'mish': 'Mish',
    'sigmoid': 'Sigmoid',
    'softplus': 'Softplus',
    'tanh': 'Tanh',
    'silu': 'SiLU',
    'swish': 'SiLU',
    'linear': 'Identity'
    }
    
    if activation is None:
        return nn.Identity
    activation = activation.lower()
    if activation in _torch_activations_dict:
        return getattr(nn, _torch_activations_dict[activation])
    raise ValueError(f"Activation '{activation}' not valid.")


class Dense(nn.Module):
    r"""A simple fully-connected layer implementing

    .. math::

        \mathbf{x}^{\prime} = \sigma\left(\boldsymbol{\Theta}\mathbf{x} +
        \mathbf{b}\right)

    where :math:`\mathbf{x} \in \mathbb{R}^{d_{in}}, \mathbf{x}^{\prime} \in
    \mathbb{R}^{d_{out}}` are the input and output features, respectively,
    :math:`\boldsymbol{\Theta} \in \mathbb{R}^{d_{out} \times d_{in}} \mathbf{b}
    \in \mathbb{R}^{d_{out}}` are trainable parameters, and :math:`\sigma` is
    an activation function.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        activation (str, optional): Activation function to be used.
            (default: :obj:`'relu'`)
        dropout (float, optional): The dropout rate.
            (default: :obj:`0`)
        bias (bool, optional): If :obj:`True`, then the bias vector is used.
            (default: :obj:`True`)
    """

    def __init__(self,
                input_size: int,
                output_size: int,
                activation: str = 'relu',
                dropout: float = 0.,
                bias: bool = True):
        super(Dense, self).__init__()
        self.affinity = nn.Linear(input_size, output_size, bias=bias)
        self.activation = get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def reset_parameters(self) -> None:
        """"""
        self.affinity.reset_parameters()

    def forward(self, x):
        """"""
        out = self.activation(self.affinity(x))
        return self.dropout(out)


class MLP(nn.Module):
    """Simple Multi-layer Perceptron encoder with optional linear readout.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int, optional): Size of the optional readout.
        exog_size (int, optional): Size of the optional exogenous variables.
        n_layers (int, optional): Number of hidden layers. (default: 1)
        activation (str, optional): Activation function. (default: `relu`)
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                input_size,
                hidden_size,
                output_size=None,
                exog_size=None,
                n_layers=1,
                activation='relu',
                dropout=0.):
        super(MLP, self).__init__()

        if exog_size is not None:
            input_size += exog_size
        layers = [
            Dense(input_size=input_size if i == 0 else hidden_size, output_size=hidden_size, activation=activation, dropout=dropout) for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def reset_parameters(self) -> None:
        """"""
        for module in self.mlp._modules.values():
            module.reset_parameters()
        if self.readout is not None:
            self.readout.reset_parameters()

    def forward(self, x, u=None):
        """"""
        x = maybe_cat_exog(x, u)
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out


class StaticGraphEmbedding(nn.Module):
    r"""Creates a table of embeddings with the specified size.

    Args:
        n_tokens (int): Number of elements for which to store an embedding.
        emb_size (int): Size of the embedding.
        initializer (str or Tensor): Initialization methods.
            (default :obj:`'uniform'`)
        requires_grad (bool): Whether to compute gradients for the embeddings.
            (default :obj:`True`)
        bind_to (nn.Module, optional): Bind the embedding to a nn.Module for
            lazy init. (default :obj:`None`)
        infer_tokens_from_pos (int): Index of the element of input data from
            which to infer the number of embeddings for lazy init.
            (default :obj:`0`)
        dim (int): Token dimension. (default :obj:`-2`)
    """

    def __init__(self, n_tokens: int, emb_size: int,
                initializer: Union[str, Tensor] = 'uniform',
                requires_grad: bool = True,
                bind_to: Optional[nn.Module] = None,
                infer_tokens_from_pos: int = 0,
                dim: int = -2):
        super(StaticGraphEmbedding, self).__init__()
        assert emb_size > 0
        self.n_tokens = int(n_tokens)
        self.emb_size = int(emb_size)
        self.dim = int(dim)
        self.infer_tokens_from_pos = infer_tokens_from_pos

        if isinstance(initializer, Tensor):
            self.initializer = "from_values"
            self.register_buffer('_default_values', initializer.float())
        else:
            self.initializer = initializer
            self.register_buffer('_default_values', None)

        if self.n_tokens > 0:
            self.emb = nn.Parameter(Tensor(self.n_tokens, self.emb_size),
                                    requires_grad=requires_grad)
        else:
            assert isinstance(bind_to, nn.Module)
            self.emb = nn.parameter.UninitializedParameter(
                requires_grad=requires_grad)
            bind_to._hook = bind_to.register_forward_pre_hook(
                self.initialize_parameters)

        self.reset_parameters()

    def reset_parameters(self):
        if self.n_tokens > 0:
            if self.initializer == 'from_values':
                self.emb.data = self._default_values.data
            if self.initializer == 'glorot':
                inits.glorot(self.emb)
            elif self.initializer == 'uniform' or self.initializer is None:
                inits.uniform(self.emb_size, self.emb)
            elif self.initializer == 'kaiming_normal':
                nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
            elif self.initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.emb, fan=self.emb_size, a=math.sqrt(5))
            else:
                raise RuntimeError(f"Embedding initializer '{self.initializer}' is not supported")

    def extra_repr(self) -> str:
        return f"n_tokens={self.n_tokens}, embedding_size={self.emb_size}"

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.emb, torch.nn.parameter.UninitializedParameter):
            self.n_tokens = input[self.infer_tokens_from_pos].size(self.dim)
            self.emb.materialize((self.n_tokens, self.emb_size))
            self.reset_parameters()
        module._hook.remove()
        delattr(module, '_hook')

    def forward(self, expand: Optional[List] = None,
                token_index: OptTensor = None,
                tokens_first: bool = True):
        """"""
        emb = self.emb if token_index is None else self.emb[token_index]
        if not tokens_first:
            emb = emb.T
        if expand is None:
            return emb
        shape = [*emb.size()]
        view = [1 if d > 0 else shape.pop(0 if tokens_first else -1)
                for d in expand]
        return emb.view(*view).expand(*expand)


class PositionalEncoder(nn.Module):
    """
    Spatiotemporal node embedding by integrating the learnable dictionary and sinusoidal encodings.
    """
    def __init__(self, in_channels, out_channels, n_layers: int = 1, steps: int = None, n_nodes: Optional[int] = None):
        super().__init__()
        self.lin = nn.Linear(steps*in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.mlp = MLP(out_channels, out_channels, out_channels, n_layers=n_layers, activation='relu')
        if n_nodes is not None:
            self.node_emb = StaticGraphEmbedding(n_nodes, out_channels)
        else:
            self.register_parameter('node_emb', None)

    def forward(self, u, node_index=None, node_emb=None):
        if node_emb is None:
            node_emb = self.node_emb(token_index=node_index)
        # u: [b s c], node_emb: [n c] -> [b n c]
        u = rearrange(u, 'b s c -> b (s c)')
        u = self.lin(u)  # [b, n_hid]
        out = self.activation(u.unsqueeze(-2) + node_emb) # [b, n_hid]->[b, 1, n_hid]
        out = self.mlp(out)  # [b n c]

        return out


class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron with residual connections
    """
    def __init__(self, input_dim, hidden_dim, p=0.15) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=p)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.
        Args:
            input_data (torch.Tensor): input data with shape [B, N, C]
        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual

        return hidden

