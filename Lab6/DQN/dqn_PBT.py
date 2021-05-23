'''Implement PBT in DQN'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time
import os
import json
from functools import partial

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, h1, h2, state_dim=8, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

class DQN:
    def __init__(self, args, config, checkpoint_dir):
        self._behavior_net = Net(h1=config["hidden1"], h2=config["hidden2"]).to(args.device)
        self._target_net = Net(h1=config["hidden1"], h2=config["hidden2"]).to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        
        self._optimizer = optim.Adam(self._behavior_net.parameters(), lr=config["lr"])
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = config["batch_size"]
        self.gamma = args.gamma
        self.freq = config["fre"]
        self.target_freq = args.target_freq

        self.checkpoint_dir = checkpoint_dir

        if self.checkpoint_dir:
            self.load(self.checkpoint_dir)

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        
        if random.random() < epsilon:
            return action_space.sample()
        else:
            with torch.no_grad():
                return torch.argmax(self._behavior_net(torch.from_numpy(state).view(1, -1).to(self.device)), dim=1).item()

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        q_value = self._behavior_net(state).gather(dim=1,index=action.long())
        with torch.no_grad():
            q_next = torch.max(self._target_net(next_state), dim=1)[0].view(-1, 1)
            q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, episode):
        with tune.checkpoint_dir(episode) as checkpoint_dir:
            self.checkpoint_dir = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((
                self._behavior_net.state_dict(),
                self._target_net.state_dict(),
                self._optimizer.state_dict()
                ), self.checkpoint_dir)

    def save_best(self, path):
        torch.save((
                self._behavior_net.state_dict(),
                self._target_net.state_dict(),
                self._optimizer.state_dict()
                ), path)

    def load(self, path):
        behavior_net, target_net, optimizer = torch.load(path)
        self._behavior_net.load_state_dict(behavior_net)
        self._target_net.load_state_dict(target_net)
        self._optimizer.load_state_dict(optimizer)

def train(args, env, agent, writer, config):
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    for episode in range(config["episode"]):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                break

        if(((episode + 1) % 300 ) == 0) or ((episode + 1) == config["episode"]):
            re_mean = test(args, env, agent, writer)
            tune.report(reward_mean=re_mean, )
            agent.save(episode)
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        
        for t in itertools.count(start=1):

            env.render()

            #select action
            action = agent.select_action(state, epsilon, action_space)

            #excute action
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print("total reward : {0:.2f}".format(total_reward))
                rewards.append(total_reward)
                break

    print('Average Reward', np.mean(rewards))
    env.close()
    return np.mean(rewards)

def DQN_BPT(config, args, env, checkpoint_dir=None):

    agent = DQN(args, config, checkpoint_dir)  
    
    writer = SummaryWriter(args.logdir)
    train(args, env, agent, writer, config)
    
def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn_PBT.pth')
    parser.add_argument('--logdir', default='log/dqn_PBT')
    parser.add_argument('--name', default='dqn_best_config')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    # parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    # parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    # parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')

    if(args.test_only):
        with open(args.name + ".json") as fp:
            config = json.load(fp)
        writer = SummaryWriter(args.logdir)
        best_model = os.path.join(os.getcwd(), args.model)
        agent = DQN(args, config, best_model)
        test(args, env, agent, writer)
    else:
        # defined hyperparameter
        config = {
            "hidden1" : tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            "hidden2" : tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            "lr" : tune.loguniform(1e-4, 1e-1),
            "batch_size" : tune.sample_from(lambda _: 2 ** np.random.randint(1, 7)),
            "fre" : tune.choice([x for x in range(1,11)])
        }

        pbt = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="reward_mean",
            mode="max",
            perturbation_interval=50,
            hyperparam_mutations={
                "episode" : tune.choice([x for x in range(1000, 2001)])
            })

        analysis = tune.run(
            partial(DQN_BPT, args=args, env=env),
            name="dqn_PBT",
            scheduler=pbt,
            stop={
                "reward_mean": 290,
                "training_iteration": 30,
            },
            config=config,
            resources_per_trial={"cpu": 4, "gpu": 1},
            num_samples=10,
        )

        best_trial = analysis.get_best_trial("reward_mean", "max", "last-5-avg")

        print("Best config", best_trial.config)

        with open(args.name + ".json", 'w') as fp:
            json.dump(best_trial.config, fp)

        best_checkpoint_dir = os.path.join(best_trial.checkpoint.value, "checkpoint")

        agent = DQN(args, best_trial.config, best_checkpoint_dir)
        writer = SummaryWriter(args.logdir)
        test(args, env, agent, writer)

        agent.save_best(args.model)

if __name__ == '__main__':
    main()


