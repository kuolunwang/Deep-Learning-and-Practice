#!/usr/bin/env python3

from .iCLEVR.iCLEVR import iCLEVRDataset
from .CelebAHQ.CelebAHQ import CelebAHQDataset

def get_dataset(name, mode):
    dataset = {
        'iCLEVR' : iCLEVRDataset(mode=mode),
        'CelebAHQ' : CelebAHQDataset()
    }

    return dataset[name]