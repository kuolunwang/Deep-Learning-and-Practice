#!/usr/bin/env python3

from .iCLEVR.iCLEVR import iCLEVRDataset
from .CelebAHQ.CelebAHQ import CelebAHQDataset

def get_dataset(name, mode, model=None):
    dataset = {
        'iCLEVR' : iCLEVRDataset(mode=mode, model=model),
        'CelebAHQ' : CelebAHQDataset()
    }

    return dataset[name]