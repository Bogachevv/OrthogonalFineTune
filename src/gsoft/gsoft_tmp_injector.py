import torch
from torch import nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer

from gsoft import GSOFTLinear


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


def _get_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def _get_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_num_trainable(model):
    total_params = _get_total_parameters(model)
    trainable_params = _get_trainable_parameters(model)
    frac = (float(trainable_params) / total_params) * 100
    
    print(f"trainable: {trainable_params}  |  total: {total_params}  |  trainable(%): {frac:.6f}")


def inject_gsoft(gsoft_config, model):
    model_adapter = model
    print("WARNING: inject_gsoft modify original model")

    disable_model_grads = gsoft_config.get('disable_model_grads', True)
    if disable_model_grads:
        for param_name, param in model_adapter.named_parameters():
            param.requires_grad = False

    for name, module in model_adapter.named_modules():
        if not check_target_module_exists(gsoft_config, name):
            continue

        if not isinstance(module, nn.Linear):
            continue

        out_f, in_f = module.weight.shape
        kwargs = {
            'nblocks': gsoft_config.get('nblocks', None),
            'orthogonal': gsoft_config.orthogonal,
            'method': gsoft_config.method,
            'block_size': gsoft_config.get('block_size', None),
            'scale': gsoft_config.scale,
        }

        gs_side = gsoft_config.get('side', 'left')
        if gs_side == 'min':
            gs_side = 'left' if (in_f <= out_f) else 'right'
        
        is_left = (gs_side == 'left')
        
        gsoft_layer = GSOFTLinear(
            module, 
            in_features=in_f, 
            out_features=out_f,
            is_left=is_left, 
            **kwargs
        )

        print(f"Setting adapter at {name:20} layer")
        set_layer(model_adapter, name, gsoft_layer)
    
    print_num_trainable(model_adapter)
    return model_adapter
