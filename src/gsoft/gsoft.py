import torch
import torch.nn as nn
import torch.nn.functional as F

from gs_orthogonal import GSOrthogonal
from peft.tuners.tuners_utils import BaseTunerLayer


class GSOFTLinear(nn.Module, BaseTunerLayer):
    def __init__(
            self,
            pre_layer: nn.Module,
            in_features: int,
            out_features: int,
            nblocks: int,
            orthogonal: bool = True,
            method: str = 'cayley',
            block_size = None,
            scale: bool = True,
            is_left: bool = True,
            bias: bool = False,
        ):

        super().__init__()

        self.pre_layer = pre_layer
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks = nblocks
        self.scale = scale
        self.bias = bias
        self.is_left = is_left

        self.gsoft_s = None
        self.gsoft_bias = None

        base_tensor = pre_layer.weight
        gs_features = in_features if is_left else out_features
        self.gs_ort = GSOrthogonal(gs_features, nblocks, orthogonal, method, block_size, base_tensor=base_tensor)
        
        if self.scale:
            self.gsoft_s = nn.Parameter(base_tensor.new_ones(out_features, dtype=torch.float32))
        
        if self.bias:
            self.gsoft_bias = nn.Parameter(base_tensor.new_zeros(out_features, dtype=torch.float32))

        self._enabled = True

    def enable_adapters(self, enable: bool = True):
        self._enabled = enable

    def forward(self, x: torch.Tensor):
        if not self._enabled:
            return self.pre_layer(x)

        if self.is_left:
            x = self.gs_ort(x)
            x = F.linear(x, self.pre_layer.weight)
        else:
            x = F.linear(x, self.pre_layer.weight)
            x = self.gs_ort(x)
        
        if self.scale:
            x = self.gsoft_s * x
        
        if self.pre_layer.bias is not None:
            x = x + self.pre_layer.bias
        
        if self.bias:
            x = x + self.gsoft_bias

        return x

    def merge(self) -> nn.Linear:
        """
        merge may destruct GSOFTLinear structure. Do not use this layer after merging 
        """
        in_shape, out_shape = self.in_features, self.out_features
        W_0: torch.Tensor = self.pre_layer.weight.data

        if self.bias:
            raise NotImplementedError

        if self.is_left:
            I = torch.eye(in_shape, dtype=W_0.dtype, device=W_0.device)
            Q = self.gs_ort(I).transpose(0, 1)
            W = torch.mm(W_0, Q)
        else:
            I = torch.eye(out_shape, dtype=W_0.dtype, device=W_0.device)
            Q = self.gs_ort(I).transpose(0, 1) 
            W = torch.mm(Q, W_0)
        
        if self.scale is not None:
            W.mul_(self.gsoft_s.unsqueeze(1))
        
        self.pre_layer.weight.data = W
        del W_0

        return self.pre_layer
