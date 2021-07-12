# DLP_Lab6

## How to use

Before you run this code, you must install required library first, then execute code, take DQN use example.

```
    $ source install.sh
    $ DLP_Lab6
    (DLP_Lab6)$ cd DQN && python3 dqn.py
```

If you want to use population Based Training (PBT) to training.

```
    (DLP_Lab6)$ cd DQN && python3 dqn_PBT.py
```

## Use others setting 

you can choose network parameters by yourself.

```
    (DLP_Lab6)$ python3 main.py -h
```

## Visual result

Use tensorboard visualize the reward.

```
    $ tensorboard --logdir=[your log path]
```
