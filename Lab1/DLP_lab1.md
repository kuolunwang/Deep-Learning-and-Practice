# DLP Homework 1

電控碩 0860077 王國倫

# 1. Introduction

This lab will implement a two hidden layers neural network shown as Fig 1, and only use Numpy and other standard libraries, the deep learning framework is not allowed to use in this homework.

![](https://i.imgur.com/oJQioVi.png) \
**Fig. 1 Two layer network**


Iuput data can divided into two parts, one is linear dataset, and the other one is nonlinear dataset, see the Fig. 2.

![](https://i.imgur.com/89qKjnv.png) \
**Fig.2 The left-hand side is linear, the right-hand side is nonlinear.**

> linear data

```javascript
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
```

> nonlinear data

```javascript
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
```

# 2. Experiment setups

## A. Sigmoid functions 

The sigmoid functions as activate function on neural network, this function can import generaliza-tion of model to conquer nonlinear problems, likes XOR. The derivation formula are shown as Fig 3, and the Graph of Sigmoid and the derivative of the Sigmoid function can see Fig 4.
 

The sigmoid function : $\sigma \left ( x \right ) = \dfrac{1}{1+e^{-x}}$ 

![](https://i.imgur.com/30VEBQu.png) \
**Fig. 3 Sigmoid derivative**

![](https://i.imgur.com/zRNR9Vw.png)
**Fig. 4 Sigmoid function and it derivation**


## B. Neural network 

This lab neural network architecture can follow Fig 5.

* $x_{1}$ and $x_{2}$ are input data
* $X$ is a matrix $m*2$
* $W_{1}$, $W_{2}$ and $W_{2}$ are model weigtht
* $y$ is predict result, matrix $m*1$
* $\hat{y}$ is ground truth
* $L(\theta )$ is loss function, this lab I use **MSE** 

The hidden layer and predict computations.
$Z_{1} = \sigma(XW_{1})$, $Z_{2} = \sigma(Z_{1}W_{2})$, $y = \sigma(Z_{2}W_{3})$
![](https://i.imgur.com/Qg7V75X.png) \
**Fig. 5 Forward**

## C. Backpropagation

In order to derivate the back-propagation, I will take Fig 5 for example.

$\dfrac{\partial L}{\partial W^{3}} = \dfrac{\partial L}{\partial \hat{y}}\dfrac{\partial \hat{y}}{\partial y^{'}}\dfrac{\partial y^{'}}{\partial W^{3}} = Z_{2}^{T} \cdot  \sigma {}'\left ( y \right )\circ  2(y - \hat{y})$

$y^{'} = Z_{2}W_{3}$

$\dfrac{\partial L}{\partial W^{2}} = \dfrac{\partial L}{\partial \hat{y}}\dfrac{\partial \hat{y}}{\partial y^{'}}\dfrac{\partial y^{'}}{\partial Z_{2}}\dfrac{\partial Z_{2}}{\partial Z_{2}^{'}}\dfrac{\partial Z_{2}^{'}}{\partial W^{2}} =Z_{1}^{T} \cdot \sigma {}'\left ( Z_{2} \right ) \circ(\sigma {}'\left ( y \right )\circ 2(y - \hat{y}) \cdot W_{3}^{T})$

$Z_{2}^{'} = Z_{1}W_{2}$

$\frac{\partial L}{\partial W^{1}} = \dfrac{\partial L}{\partial \hat{y}}\dfrac{\partial \hat{y}}{\partial y^{'}}\dfrac{\partial y^{'}}{\partial Z_{2}}\dfrac{\partial Z_{2}}{\partial Z_{2}^{'}}\dfrac{\partial Z_{2}^{'}}{\partial Z_{1}}\dfrac{\partial Z_{1}}{\partial Z_{1}^{'}}\dfrac{\partial Z_{1}^{'}}{\partial W^{1}} =X^{T} \cdot \sigma {}'\left ( Z_{1} \right )\circ ((\sigma {}'\left ( Z_{2} \right ) \circ(\sigma {}'\left ( y \right )\circ 2(y - \hat{y}) \cdot W_{3}^{T}))\cdot W_{2}^{T})$

$Z_{1}^{'} = XW_{1}$

operator priority: $\circ > \cdot$

The $\circ$ operation is element-wise product (Hadamard product) in the same dimension.
For example, the Hadamard product for a 3 × 3 matrix A with a 3 × 3 matrix B is

${\displaystyle {\begin{bmatrix}a_{11}&a_{12}&a_{13}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{bmatrix}}\circ {\begin{bmatrix}b_{11}&b_{12}&b_{13}\\b_{21}&b_{22}&b_{23}\\b_{31}&b_{32}&b_{33}\end{bmatrix}}={\begin{bmatrix}a_{11}\,b_{11}&a_{12}\,b_{12}&a_{13}\,b_{13}\\a_{21}\,b_{21}&a_{22}\,b_{22}&a_{23}\,b_{23}\\a_{31}\,b_{31}&a_{32}\,b_{32}&a_{33}\,b_{33}\end{bmatrix}}}$

# 3. Results of your testing

## A. Screenshot and comparison figure

1. linear data
![](https://i.imgur.com/RRbpf1d.png)

2. nonlinear data
![](https://i.imgur.com/1nfmEhb.png)

## B. Show the accuracy of your prediction

1. linear data
![](https://i.imgur.com/TEJ6VY4.png)

2. nonlinear data\
![](https://i.imgur.com/C74oX7i.png)

## C. Learning curve (loss, epoch curve)

1. linear data
![](https://i.imgur.com/ZGGu0Mh.png)\

2. nonlinear data\
![](https://i.imgur.com/FPos1Aa.png)\

## D. anything you want to present

(1) The network and training parameter:

1. linear data
    * hidden layer1 : 5
    * hidden layer2 : 5
    * activate function : sigmoid
    * learning rate : 0.01
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10
2. nonlinear data
    * hidden layer1 : 6
    * hidden layer2 : 8
    * activate function : sigmoid
    * learning rate : 0.1
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10

(2) Training, validation, test:
I split data into two parts, one is training data, and the other one is validation data, the ratio is 8:2. The whole data will be tested after training model. The training data in charge of train and update parameter, the validation is calculate loss, the test data is predict final result and accu-rancy.
![](https://i.imgur.com/2a7FJXr.png) \
**Fig. 6 Training, validation, test dataset**
```javascript
  def split_data(self, x, y):

    size = (int)(x.shape[0] * 0.8)
    index = np.array([t for t in range(x.shape[0])])
    np.random.shuffle(index)

    tra_x, val_x = x[index[:size]], x[index[size:]]
    tra_y, val_y = y[index[:size]], y[index[size:]]

    return tra_x, val_x, tra_y, val_y
```

(3) Accurancy:
The accurancy use difference of prediction and actual, and it error  must less than 0.1, then are considered a successful prediction.
```javascript
def evaluate(self, prediction, actual):
    
    diff = np.abs(prediction - actual)
    result = np.zeros(prediction.shape)
    result[diff<=0.1] = 1
    accurancy = np.sum(result == 1) / (float)(actual.shape[0]) * 100

    print("accurancy : {0}%, loss : {1:.6f}".format(accurancy,self.loss(prediction, actual)))
```
# 4. Discussion

## A. Try different learning rates

1. linear data
    I fixed other parameters, adjust learning rate $1,0.1,0.01$.
    * hidden layer1 : 5
    * hidden layer2 : 5
    * activate function : sigmoid
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10

    | learning rate | 1 | 0.1 | 0.01 |
    |:-------------:|:--------:|:--------:|:--------:|
    | accurancy     | 100%     | 90.48%     | 97%     |

2. nonlinear data
    I fixed other parameters, adjust learning rate $0.2,0.1,0.05$.
    * hidden layer1 : 6
    * hidden layer2 : 8
    * activate function : sigmoid
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10

    | learning rate | 0.2 | 0.1 | 0.05 |
    |:-------------:|:--------:|:--------:|:--------:|
    | accurancy     | 90.48%    | 90.48%     | 95.24%     |
    
## B. Try different numbers of hidden units
1. linear data
    I fixed other parameters, adjust hidden layer1 and layer2.
    * learning rate : 1
    * activate function : sigmoid
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10

    | hidden layer  | (5,5) | (3,4) | (3,3) |
    |:-------------:|:--------:|:--------:|:--------:|
    | accurancy     | 100%     | 99%     | 49%     |

## C. Try without activation functions
1. linear data
    I fixed other parameters, and without activate function.
    * hidden layer1 : 5
    * hidden layer2 : 3
    * learning rate :　0.01
    * activate function : none
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10\
![](https://i.imgur.com/QqiclR7.png)

2. nonlinear data
    I fixed other parameters, and without activate function.
    * hidden layer1 : 3
    * hidden layer2 : 4
    * learning rate :　0.01
    * activate function : none
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10\
![](https://i.imgur.com/P1b8jNG.png)

## D. Anything you want to share
(1) argparse:
In order to easy adjust parameters, I use argparse library to control train and model parameters, then I can use terminal to modify instead of manual adjust code.

```javascript
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
```
![](https://i.imgur.com/oTP70RN.png)

# 5. Extra

## A. Implement different optimizers

In this lab, I implement momentum and adam optimizer, the momentum formula shown as following, where $\beta$ is hyperparameter, Typically this value is set to $0.9$, $\eta$ is learning rate, $\theta$ is model weight.
The adam optimizer formular are list below, where $\beta_{1}$ is the exponential decay of the rate for the first moment estimates, it value set 0.9, $\beta_{2}$ is the exponential decay rate for the second-moment estimates, and its literature value is 0.95, small value $\varepsilon$ to prevent zero-division.

> Momentum

$m \leftarrow \beta m + \eta \bigtriangledown MSE$
$\theta \leftarrow \theta - m$

> Adam

$m_{t} = \beta _{1}m_{t-1}+(1-\beta _{1})\bigtriangledown MSE$
$v_{t} = \beta _{2}v_{t-1}+(1-\beta _{2})\bigtriangledown MSE^{2}$

$\hat{m}_{t} = \dfrac{m_{t}}{1-\beta _{1}^{t}}$
$\hat{v}_{t} = \dfrac{v_{t}}{1-\beta _{2}^{t}}$

$\theta \leftarrow \theta - \eta \dfrac{\hat{m}_{t}}{\sqrt{\hat{v}_{t}}+\varepsilon }$

```javascript
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
```

```javascript
# momentum update weight
self.vec_W1 = self.momentum * self.vec_W1 + self.learning_rate * self.grad_W1
self.vec_W2 = self.momentum * self.vec_W2 + self.learning_rate * self.grad_W2
self.vec_W3 = self.momentum * self.vec_W3 + self.learning_rate * self.grad_W3

self.W1 -= self.vec_W1
self.W2 -= self.vec_W2
self.W3 -= self.vec_W3
```

```javascript
# adam update weight
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
```

1. linear data
    I use momentum to optimize linear data.
    * hidden layer1 : 4
    * hidden layer2 : 6
    * learning rate :　0.01
    * activate function : sigmoid
    * optimizer : momentum
    * epochs : 10000
    * batch_size : 10\
![](https://i.imgur.com/cumLRBp.png)

2. nonlinear data
    I use adam to optimize nonlinear data.
    * hidden layer1 : 4
    * hidden layer2 : 3
    * learning rate :　0.01
    * activate function : sigmoid
    * optimizer : adam
    * epochs : 10000
    * batch_size : 21\
![](https://i.imgur.com/1c0ZMaC.png)



## B. Implement different activation functions

In this lab, I use three activation functions, include ReLU, Leaky ReLU, Tanh. There are shown as below.

> Tanh

![](https://i.imgur.com/4mUzkbP.png)

![](https://i.imgur.com/Dhe2qsC.png)

> ReLU

![](https://i.imgur.com/626mdQH.png)

![](https://i.imgur.com/8NZdoBB.png)

![](https://i.imgur.com/VbKm8Fc.png)


> Leaky ReLU

![](https://i.imgur.com/b6hh2k3.png)

![](https://i.imgur.com/3wCAFkb.png)

![](https://i.imgur.com/fT2CdHo.png)

```javascript
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
```

```javascript
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
```

1. ReLU
    * dataset : linear
    * hidden layer1 : 4
    * hidden layer2 : 6
    * learning rate :　0.001
    * activate function : ReLU
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10\
![](https://i.imgur.com/4WPHsnM.png)


2. Leaky ReLU
    * dataset : linear
    * hidden layer1 : 4
    * hidden layer2 : 6
    * learning rate :　0.001
    * activate function : Leaky ReLU
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10\
![](https://i.imgur.com/5BozocB.png)


3. Tanh
    * dataset : linear
    * hidden layer1 : 5
    * hidden layer2 : 8
    * learning rate :　0.01
    * activate function : Tanh
    * optimizer : SGD
    * epochs : 10000
    * batch_size : 10\
    ![](https://i.imgur.com/d7zEKcm.png)

###### tags: `DLP2021`

