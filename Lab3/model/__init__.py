#!/usr/bin/env python3

from .DeepConvNet import DeepConvNet
from .EEGNet import EEGNet

def get_model(name, activate_function):
    models = {
        'EEGNet' : EEGNet(activate_function),
        'DeepConvNet' : DeepConvNet(activate_function)
    }

    return models[name]