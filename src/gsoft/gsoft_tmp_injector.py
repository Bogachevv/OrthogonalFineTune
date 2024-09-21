import torch
from torch import nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer

from gsoft.gsoft import GSOFTLinear


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)


def inject_gsoft(gsoft_config, model):
    model_adapter = model
    print("WARNING: inject_gsoft modify original model")

    for name, module in model_adapter.named_modules():
        if not check_target_module_exists(gsoft_config, name):
            continue

        if not isinstance(module, nn.Linear):
            continue

        out_f, in_f = module.weight.shape
        kwargs = {
            'nblocks': gsoft_config.nblocks,
            'orthogonal': gsoft_config.orthogonal,
            'method': gsoft_config.method,
            'block_size': gsoft_config.block_size,
            'scale': gsoft_config.scale,
        }

        gsoft_layer = GSOFTLinear(
            module, 
            in_features=in_f, 
            out_features=out_f, 
            **kwargs
        )

#         print(gsoft_layer, "\n", module, "\n\n")
        print(f"Setting adapter at {name:20} layer")
        set_layer(model_adapter, name, gsoft_layer)
    
    return model_adapter