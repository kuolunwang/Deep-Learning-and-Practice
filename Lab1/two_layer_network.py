#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class Twolayernetwork(object):
  
  def __init__(self, args):

    # Sets the input and output sizes
    self.input = args.input_size
    self.output = args.output_size

    # Sets size of hidden layers
    self.hidden_layer1 = args.hidden_layer1
    self.hidden_layer2 = args.hidden_layer2

    # Initializes weights
    self.W1 = np.random.rand(self.input, self.hidden_layer1)
    self.W2 = np.random.rand(self.hidden_layer1, self.hidden_layer2)
    self.W3 = np.random.rand(self.hidden_layer2, self.output)

    # activate function
    self.activate = args.activate_function

    # hypreparameter
    self.learning_rate = args.learning_rate
    self.epochs = args.epochs
    self.batch = args.batch_size

    # optimizer
    self.optimizer = args.optimizer

    # momentum optimizer parameter
    self.momentum = 0.9
    self.vec_W1 = 0
    self.vec_W2 = 0
    self.vec_W3 = 0

    # adam optimizer parameter
    self.momentum_decay = 0.9
    self.scale_decay = 0.95
    self.epsilon = 10 ** -8
    self.sca_W1 = 0
    self.sca_W2 = 0
    self.sca_W3 = 0

    # other 
    self.file_name = args.file_name

  # activate forward and back propagation
  def activate_forward(self):

    if(self.activate == "sigmoid"):
      return lambda x : np.exp(x) / (1.0 + np.exp(x))
    elif(self.activate == "tanh"):
      return lambda x : (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    elif(self.activate == "ReLU"):
      def relu(x):
        y = np.copy(x)
        y[y<0] = 0
        return y
      return relu
    elif(self.activate == "Leaky_ReLU"):
      def leaky_relu(x):
        y = np.copy(x)
        y[y<0] = 0.01 * y[y<0]
        return y
      return leaky_relu
    else:
      return lambda x : x 

  def activate_back(self):
    
    if(self.activate == "sigmoid"):
      return lambda x : np.multiply(x, 1.0 - x)
    elif(self.activate == "tanh"):
      return lambda x : 1 -  ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))** 2
    elif(self.activate == "ReLU"):
      def der_relu(x):
        y = np.copy(x)
        y[y>=0] = 1
        y[y<0] = 0
        return y
      return der_relu
    elif(self.activate == "Leaky_ReLU"):
      def der_leaky_relu(x):
        y = np.copy(x)
        y[y>=0] = 1
        y[y<0] = 0.01
        return y
      return der_leaky_relu
    else:
      def one(x):
        y = np.zeros(x.shape)
        y[y==0] = 1
        return y
      return one

  # Feed forward
  def feed_forward(self, input):

    self.input_data = input
    act_fuc = self.activate_forward()

    # calculate layer 1
    self.layer1 = act_fuc(np.dot(input, self.W1))

    # calculate layer 2
    self.layer2 = act_fuc(np.dot(self.layer1, self.W2))

    # calculate end result
    result = act_fuc(np.dot(self.layer2, self.W3))

    return result

  # loss function
  def loss(self, pre, actual):
    return np.mean((actual - pre) ** 2)

  # back propagation
  def back_propagation(self, pre, actual):

    der_act = self.activate_back()

    # calculate gradient
    grad_y_pred3 = 2 * (pre - actual)
    grad_3 =  der_act(pre) * grad_y_pred3
    self.grad_W3 = np.dot(self.layer2.T, grad_3)
 
    grad_y_pred2 = np.dot(grad_3, self.W3.T)
    grad_2 = der_act(self.layer2) * grad_y_pred2
    self.grad_W2 = np.dot(self.layer1.T, grad_2)

    grad_y_pred1 = np.dot(grad_2, self.W2.T) 
    self.grad_W1 = np.dot(self.input_data.T, der_act(self.layer1) * grad_y_pred1)
  
  # update weight
  def update_weight(self, epochs):

    if(self.optimizer == "SGD"):

      self.W1 -= self.learning_rate * self.grad_W1
      self.W2 -= self.learning_rate * self.grad_W2
      self.W3 -= self.learning_rate * self.grad_W3

    elif(self.optimizer == "momentum"):

      self.vec_W1 = self.momentum * self.vec_W1 + self.learning_rate * self.grad_W1
      self.vec_W2 = self.momentum * self.vec_W2 + self.learning_rate * self.grad_W2
      self.vec_W3 = self.momentum * self.vec_W3 + self.learning_rate * self.grad_W3

      self.W1 -= self.vec_W1
      self.W2 -= self.vec_W2
      self.W3 -= self.vec_W3

    elif(self.optimizer == "adam"):

      self.vec_W1 = self.momentum_decay * self.vec_W1 + (1 - self.momentum_decay) * self.grad_W1
      self.vec_W2 = self.momentum_decay * self.vec_W2 + (1 - self.momentum_decay) * self.grad_W2
      self.vec_W3 = self.momentum_decay * self.vec_W3 + (1 - self.momentum_decay) * self.grad_W3

      self.vec_W1 = self.vec_W1 / (1 - self.momentum_decay ** epochs)
      self.vec_W2 = self.vec_W2 / (1 - self.momentum_decay ** epochs)
      self.vec_W3 = self.vec_W3 / (1 - self.momentum_decay ** epochs)  

      self.sca_W1 = self.scale_decay * self.sca_W1 + (1 - self.scale_decay) * self.grad_W1 ** 2 
      self.sca_W2 = self.scale_decay * self.sca_W2 + (1 - self.scale_decay) * self.grad_W2 ** 2
      self.sca_W3 = self.scale_decay * self.sca_W3 + (1 - self.scale_decay) * self.grad_W3 ** 2

      self.sca_W1 = self.sca_W1 / (1 - self.scale_decay ** epochs)
      self.sca_W2 = self.sca_W2 / (1 - self.scale_decay ** epochs)
      self.sca_W3 = self.sca_W3 / (1 - self.scale_decay ** epochs)      

      self.W1 -= self.learning_rate * self.vec_W1 / (np.sqrt(self.sca_W1) + self.epsilon)
      self.W2 -= self.learning_rate * self.vec_W2 / (np.sqrt(self.sca_W2) + self.epsilon)
      self.W3 -= self.learning_rate * self.vec_W3 / (np.sqrt(self.sca_W3) + self.epsilon)
  
  # split dataset into training and validation
  def split_data(self, x, y):

    size = (int)(x.shape[0] * 0.8)
    index = np.array([t for t in range(x.shape[0])])
    np.random.shuffle(index)

    tra_x, val_x = x[index[:size]], x[index[size:]]
    tra_y, val_y = y[index[:size]], y[index[size:]]

    return tra_x, val_x, tra_y, val_y

  # train
  def train(self, x, y):

    tra_x, val_x, tra_y, val_y = self.split_data(x,y)
    loss = []

    for i in range(self.epochs):
      bsize = 0
      while(bsize < len(tra_x)):
        x_batch = tra_x[bsize:bsize+self.batch]
        y_batch = tra_y[bsize:bsize+self.batch]
        
        pre_y = self.feed_forward(x_batch)
        self.back_propagation(pre_y, y_batch)
        self.update_weight(i+1)
        bsize += self.batch

      prediction = self.feed_forward(val_x)
      
      if (i+1) % 500 == 0:
        print("epoch : {0}".format(i+1))
        self.evaluate(prediction,val_y)
      
      loss.append(self.loss(prediction, val_y))
    
    plt.title("learning curve", fontsize = 18)
    ep = [x for x in range(1, self.epochs + 1)]
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(ep, loss)
    plt.savefig('{0}_learning_curve.png'.format(self.file_name))
    plt.show()

  # evalation
  def evaluate(self, prediction, actual):
    
    diff = np.abs(prediction - actual)
    result = np.zeros(prediction.shape)
    result[diff<=0.1] = 1
    accurancy = np.sum(result == 1) / (float)(actual.shape[0]) * 100

    print("accurancy : {0}%, loss : {1:.6f}".format(accurancy,self.loss(prediction, actual)))

  # test all data
  def test(self, x, y):

    prediction = self.feed_forward(x)
    print(prediction)

    self.evaluate(prediction, y)

    return prediction