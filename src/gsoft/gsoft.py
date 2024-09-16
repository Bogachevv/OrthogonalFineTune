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
            scale: bool = True
            ):

        super().__init__()

        self.pre_layer = pre_layer
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks = nblocks
        self.scale = scale

        base_tensor = pre_layer.weight
        self.gs_ort = GSOrthogonal(in_features, nblocks, orthogonal, method, block_size, base_tensor=base_tensor)
        
        if self.scale:
            self.gsoft_s = nn.Parameter(base_tensor.new_ones(out_features))
        

    def forward(self, x: torch.Tensor):
        
        x = self.gs_ort(x)
        x = F.linear(x, self.pre_layer.weight)
        
        if self.scale:
            x = self.gsoft_s * x
        
        if self.pre_layer.bias is not None:
            x = x + self.pre_layer.bias
        
        return x
