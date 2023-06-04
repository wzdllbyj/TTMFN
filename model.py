"""
Model definition of TTMFN

"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import init, Parameter
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MHAP(nn.Module):

    def __init__(self,channel,dim,heads):
        super(MHAP, self).__init__()
        self.dim = dim
        self.num_heads = heads
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        self.conv1 = nn.Sequential(
            nn.Conv1d(channel,24,1),
            nn.ReLU(),
            nn.Conv1d(24,1,1)
        )

    def forward(self, x, k):

        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)
        Q = self.conv1(Q)

        dim_split = self.dim // self.num_heads
        Q_ = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K_ = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V_ = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)
        A = torch.softmax(torch.matmul(Q_, K_.transpose(-1, -2)) / math.sqrt(dim_split), -1)

        #top rank
        _, index = torch.topk(A, int(A.size(-1) * k))
        area = torch.zeros(A.shape).cuda()
        for i in range(index.size(1)):
            for j in range(index.size(-1)):
                area[0,i,0, index[0,i,0,j]] = 1
        area = area
        A = torch.mul(A, area)
        A_sum = torch.sum(A, dim=-1).unsqueeze(-1)
        A = A / A_sum

        O = torch.matmul(A, V_)
        O = rearrange(O, 'b h n d -> b n (h d)')

        return O



def Gene_MLP_Block(dim1, dim2, dropout=0.1):

    return nn.Sequential(
            nn.Tanh(),
            nn.Linear(dim1, dim2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim2,dim2),
            nn.GELU(),
            nn.Dropout(p=dropout)
    )


from torch.nn.modules.activation import MultiheadAttention

class TSMCATlayer(nn.Module):
    def __init__(self,dim, heads, mlp_dim, dropout = 0.1, activation='gelu'):
        super().__init__()
        self.trans1 = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation=activation)
        self.trans2 = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation=activation)


        #MCAT

        self.self_attn_x = MultiheadAttention(dim, heads,dropout=dropout)
        self.self_attn_y = MultiheadAttention(dim, heads,dropout=dropout)

        self.linear1_x = nn.Linear(dim, mlp_dim)
        self.dropout_x = nn.Dropout(dropout)
        self.linear2_x = nn.Linear(mlp_dim, dim)

        self.norm1_x = nn.LayerNorm(dim, eps=1e-5)
        self.norm2_x = nn.LayerNorm(dim, eps=1e-5)

        self.dropout1_x = nn.Dropout(dropout)
        self.dropout2_x = nn.Dropout(dropout)

        self.linear1_y = nn.Linear(dim, mlp_dim)
        self.dropout_y = nn.Dropout(dropout)
        self.linear2_y = nn.Linear(mlp_dim, dim)

        self.norm1_y = nn.LayerNorm(dim, eps=1e-5)
        self.norm2_y = nn.LayerNorm(dim, eps=1e-5)

        self.dropout1_y = nn.Dropout(dropout)
        self.dropout2_y = nn.Dropout(dropout)

        self.activation_x = _get_activation_fn(activation)
        self.activation_y = _get_activation_fn(activation)

    def forward(self,x,y):

        X_c = self.trans1(x)
        Y_c = self.trans2(x)

        X_ = self.self_attn_x(x, y, y)[0]
        Y_ = self.self_attn_y(y, x, x)[0]

        x = x + self.dropout1_x(X_)
        x = self.norm1_x(x)
        X_ = self.linear2_x(self.dropout_x(self.activation_x(self.linear1_x(x))))
        x = x + self.dropout2_x(X_)
        x = self.norm2_x(x)

        y = y + self.dropout1_y(Y_)
        y = self.norm1_y(y)
        Y_ = self.linear2_y(self.dropout_y(self.activation_y(self.linear1_y(y))))
        y = y + self.dropout2_y(Y_)
        y = self.norm2_y(y)

        x = torch.cat([x,X_c],axis=1)

        y = torch.cat([Y_c,y],axis=1)
        return x,y

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x,y):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output_x,output_y = x,y

        for mod in self.layers:
            output_x,output_y = mod(output_x, output_y)

        return output_x,output_y




class TTMFN(nn.Module):
    """
    Deep AttnMISL Model definition
    """

    def __init__(self, cluster_num,omic_sizes):
        super(TTMFN, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.cluster_num = cluster_num
        self.omic_sizes=omic_sizes
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [Gene_MLP_Block(dim1=input_dim, dim2=128,dropout=0.1)]
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        dim = 128
        depth = 6
        heads = 4
        mlp_dim = 256
        dropout=0.1

        TSMCAT = TSMCATlayer(dim,heads,mlp_dim,dropout,activation='gelu')
        self.transformer = TransformerEncoder(TSMCAT , num_layers=depth)
        cnum = self.cluster_num * (2 ** depth)
        self.xpool = MHAP(cnum,dim,4)
        self.ypool = MHAP(cnum,dim,4)

        self.mlp_head = nn.Sequential(

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,32),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )


        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)


    def forward(self, x, mask,omic,k):

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)

        " x is a tensor list"
        res = []
        for i in range(self.cluster_num):
            hh = x[i]
            output = self.embedding_net(hh)
            output = output.view(output.size()[0], -1)
            res.append(output)

        h = torch.cat(res)
        b = h.size(0)
        c = h.size(1)
        h = h.view(b, c)
        h = rearrange(h, 'n d -> 1 n d')
        x=h

        y = rearrange(h_omic_bag, 'n b m p -> n (b m) p')
        x,y = self.transformer(x,y)
        x =self.xpool(x,k)
        y = self.ypool(y,k)
        fusion = torch.cat([x,y],axis=2)

        Y_pred = self.mlp_head(fusion)*self.output_range+self.output_shift

        return Y_pred



