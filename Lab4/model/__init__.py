#!/usr/bin/env python3

from .ResNet import ResNet18, ResNet50 

def get_model(name, num_class, feature_extract, use_pretrained):
    
    models = {
        'ResNet18' : ResNet18(num_class, feature_extract, use_pretrained),
        'ResNet50' : ResNet50(num_class, feature_extract, use_pretrained)
    }

    return models[name]
