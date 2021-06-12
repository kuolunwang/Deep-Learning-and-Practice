#!/usr/bin/env python3

from option_CNF import Option
from trainer_CNF import Trainer
import time

if __name__ == '__main__':

    # paramters
    args = Option().create()

    time_start = time.time()
    # trainer
    trainer = Trainer(args)

    time_end = time.time()

    print("Use {0}s finished.".format(time_end - time_start))
