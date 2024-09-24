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
            ):

        super().__init__()

        self.pre_layer = pre_layer
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks = nblocks
        self.scale = scale
        self.is_left = is_left

        base_tensor = pre_layer.weight
        gs_features = in_features if is_left else out_features
        self.gs_ort = GSOrthogonal(gs_features, nblocks, orthogonal, method, block_size, base_tensor=base_tensor)
        
        if self.scale:
            self.gsoft_s = nn.Parameter(base_tensor.new_ones(out_features, dtype=torch.float32))
        

    def forward(self, x: torch.Tensor):
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
        
        return x
