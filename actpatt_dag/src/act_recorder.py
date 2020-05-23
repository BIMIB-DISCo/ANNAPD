import torch.nn as nn

import numpy as np

from itertools import count
from collections import defaultdict


class ActivationRecorder:
    
    def __init__(self, layer_id):

        self._activations = np.asarray([])
        self._layer_id = layer_id
        self._unique_map = defaultdict(count().__next__)

    def __call__(self, m, i, out):

        if type(m) == nn.LogSoftmax:
            s_activation_pattern = self._out_layer(out)

        elif type(m) == nn.ReLU:
            s_activation_pattern = self._relu_layer(out)

        self._activations = s_activation_pattern

    def _relu_layer(self, out):
        """Computation for relu layers
        """
        layer_output = out.clone().detach().numpy()
        act_patt = np.apply_along_axis(
            lambda x: np.where(x == 0, 0, 1), 0, layer_output)
        act_patt_str = np.apply_along_axis(                                  
            lambda x: ''.join(str(a) for a in x), 1, act_patt)

        return act_patt_str

    def _out_layer(self, out):

        layer_output = out.clone().detach().numpy()
        activation_pattern = layer_output.argmax(axis=1)

        return activation_pattern

    def get_act(self):
        return self._activations

    def get_layer_id(self):
        return self._layer_id


def register_hooks(model):
    """Add hooks to model.
    """

    layer_id = 0

    hooks = []
    
    for name, m in model.named_modules():
        
        if "relus" in name and type(m) == nn.ReLU:
            
            hook = ActivationRecorder(layer_id)
            m.register_forward_hook(hook)
            hooks.append(hook)
            layer_id += 1

    return hooks
