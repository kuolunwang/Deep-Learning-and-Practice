#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib.pyplot as plt

def read(file):
    train_accurancy = []
    test_accurancy = []
    with open(file) as f:
        Lines = f.readlines()
        for line in Lines:
            _, _, tr, te = line.split(",")
            train_accurancy.append((float)(tr.split(":")[1]))
            test_accurancy.append((float)(te.split(":")[1]))

    return np.array(train_accurancy), np.array(test_accurancy)        
    
def plt_result(ReLU, LeakyReLU, ELU, network_name, save_file):

    plt.title("Activation function comparision({0})".format(network_name), fontsize = 18)

    Re_train, Re_test = read(ReLU)
    Le_train, Le_test = read(LeakyReLU)
    ELU_train, ELU_test = read(ELU)

    e = [x for x in range(1,301)]

    plt.plot(e, Re_train, label="ReLU_train")
    plt.plot(e, Re_test, label="ReLU_test")
    plt.plot(e, Le_train, label="LeakyReLU_train")
    plt.plot(e, Le_test, label="LeakyReLU_test")
    plt.plot(e, ELU_train, label="ELU_train")
    plt.plot(e, ELU_test, label="ELU_test")

    plt.legend(loc='lower right')

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")

    plt.savefig('{0}.png'.format(save_file))
    plt.show()

def create_argparse():
    parser = argparse.ArgumentParser(prog="DLP homework 3", description='This code will show activation function result with matpltlib')

    parser.add_argument("ReLU_file", type=str, default="none", help="input ReLU file path, default is none")
    parser.add_argument("LeakyReLU_file", type=str, default="none", help="input LeakyReLU file path, default is none")
    parser.add_argument("ELU_file", type=str, default="none", help="input ELU file path, default is none")
    parser.add_argument("network", type=str, default="none", help="input network name(EEGNet or DeepConvNet), default is none")
    parser.add_argument("save_file", type=str, default="none", help="save result with this save_file name, default is none")

    return parser

if __name__ == "__main__":

    parser = create_argparse()
    args = parser.parse_args()

    plt_result(args.ReLU_file, args.LeakyReLU_file, args.ELU_file, args.network, args.save_file)