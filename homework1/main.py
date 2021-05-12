#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from two_layer_network import Twolayernetwork

def generate_linear(n=100):
  
  point = np.random.uniform(0, 1, (n, 2))
  input = []
  label = []
  for p in point:
    input.append([p[0], p[1]])
    if p[0] > p[1]:
      label.append(0)
    else:
      label.append(1)
  return np.array(input), np.array(label).reshape(n, 1)

def generate_XOR_easy():
  input = []
  label = []

  for i in range(11):
    input.append([0.1 * i, 0.1 * i])
    label.append(0)

    if 0.1 * i == 0.5:
      continue

    input.append([0.1 * i, 1 - 0.1 * i])
    label.append(1)

  return np.array(input), np.array(label).reshape(21, 1)

def show_result(x, y, pre_y, file_name):
  plt.subplot(1, 2, 1)
  plt.title("Ground truth", fontsize = 18)
  for i in range(x.shape[0]):
    if y[i] == 0:
      plt.plot(x[i][0], x[i][1], "ro")
    else:
      plt.plot(x[i][0], x[i][1], "bo")
  
  plt.subplot(1, 2, 2)
  plt.title("Predict result", fontsize = 18)
  for i in range(x.shape[0]):
    if pre_y[i] <= 0.1:
      plt.plot(x[i][0], x[i][1], "ro")
    else:
      plt.plot(x[i][0], x[i][1], "bo")
  
  plt.savefig('{0}.png'.format(file_name))
  plt.show()
  
def create_argparse():

  parser = argparse.ArgumentParser(prog="DLP homework 1", description='This lab implement two layer network with numpy')

  # model parameters
  parser.add_argument("--input_size", type=int, default=2, help="An integer giving the size of the input, default is 2")
  parser.add_argument("--hidden_layer1", type=int, default=5, help="An integer giving the size of hidden layer 1, default is 5")
  parser.add_argument("--hidden_layer2", type=int, default=5, help="An integer giving the size of hidden layer 2, default is 5")
  parser.add_argument("--output_size", type=int, default=1, help="An integer giving the size of output, dafault is 1")
  parser.add_argument("--activate_function", type=str, default="sigmoid", help="activation function (sigmoid, ReLU, Leaky ReLU, tanh)for the network, default is sigmoid")
    
  # training hyper parameters
  parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the trainer, default is 1e-2")
  parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer(SGD, momentum, adam) for the trainer, default is SGD")
  parser.add_argument("--epochs", type=int, default=10000, help="The number of the training, default is 10000")
  parser.add_argument("--batch_size", type=int, default=10, help="Input batch size for testing, default is 10")

  # other parameters
  parser.add_argument("file_name", type=str, default="linear", help="save result picture with file name, default is linear")
  parser.add_argument("data", type=str, default="linear", help="select input data(linear or XOR), default is linear")
  
  return parser


if __name__ == "__main__":
  
  # model parameters
  parser = create_argparse()
  args = parser.parse_args()

  # network 
  net = Twolayernetwork(args)

  # dataset
  if(args.data == "linear"):

    # train linear data  
    linear_x, linear_y = generate_linear()
    net.train(linear_x, linear_y)
    linear_prediction = net.test(linear_x, linear_y)
    show_result(linear_x, linear_y, linear_prediction, args.file_name)

  elif(args.data == "XOR"):

    # train xor data
    XOR_x, XOR_y = generate_XOR_easy()
    net.train(XOR_x, XOR_y)
    XOR_prediction = net.test(XOR_x,XOR_y)
    show_result(XOR_x, XOR_y, XOR_prediction, args.file_name)

  else:
    raise("data error! no data {0}".format(args.data))
  