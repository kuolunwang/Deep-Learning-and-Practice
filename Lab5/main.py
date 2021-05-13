#!/usr/bin/env python3

from option import Option
from trainer import Trainer
import time

if __name__ == '__main__':

    # paramters
    args = Option().create()

    time_start = time.time()
    # trainer
    trainer = Trainer(args)

    trainer.Gaussian_score()

    time_end = time.time()

    print("Use {0}s finished.".format(time_end - time_start))

