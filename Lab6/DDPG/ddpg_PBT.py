'''Implement PBT in DDPG'''
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

class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)

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

class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.tanh(self.fc3(x))
        return out

class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        x = self.critic_head(torch.cat([x, action], dim=1))
        return self.critic(x)

class DDPG:
    def __init__(self, args, config, checkpoint_dir):
        # behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net = CriticNet().to(args.device)
        # target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net = CriticNet().to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net.load_state_dict(self._critic_net.state_dict())
        # optimizer
        self._actor_opt = optim.Adam(self._actor_net.parameters(), lr=config["lr"])
        self._critic_opt = optim.Adam(self._critic_net.parameters(), lr=config["lr"])

        # action noise
        self._action_noise = GaussianNoise(dim=2)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        self.checkpoint_dir = checkpoint_dir

        if self.checkpoint_dir:
            self.load(self.checkpoint_dir)

        ## config ##
        self.device = args.device
        self.batch_size = config['batch_size']
        self.tau = args.tau
        self.gamma = args.gamma

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        with torch.no_grad():
            action = self._actor_net(torch.from_numpy(state).view(1, -1).to(self.device)).cpu().numpy().squeeze()
            if(noise):
                action += self._action_noise.sample()       
            return action

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state,
                            [int(done)])

    def update(self):
        # update the behavior networks
        self._update_behavior_network(self.gamma)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net,
                                    self.tau)
        self._update_target_network(self._target_critic_net, self._critic_net,
                                    self.tau)

    def _update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss

        q_value = critic_net(state, action)
        with torch.no_grad():
           a_next = target_actor_net(next_state)
           q_next = target_critic_net(next_state, a_next)
           q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        critic_loss = criterion(q_value, q_target)

        # optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ## update actor ##
        # actor loss

        action = actor_net(state)
        actor_loss = -critic_net(state, action).mean()

        # optimize actor
        actor_net.zero_grad()
        critic_net.zero_grad()
        actor_loss.backward()
        actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_(target.data * (1 - tau) + behavior.data * tau)

    def save(self, episode):
        with tune.checkpoint_dir(episode) as checkpoint_dir:
            self.checkpoint_dir = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((
                self._actor_net.state_dict(),
                self._critic_net.state_dict(),
                ), self.checkpoint_dir)

    def save_best(self, path):
        torch.save((
            self._actor_net.state_dict(),
            self._critic_net.state_dict(),
            ), self.checkpoint_dir)

    def load(self, path):
        actor, critic = torch.load(path)
        self._actor_net.load_state_dict(actor)
        self._critic_net.load_state_dict(critic)

def train(args, env, agent, writer, config):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    for episode in range(config["episode"]):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update()

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
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
        if(((episode + 1) % 100 ) == 0) or ((episode + 1) == config["episode"]):
            re_mean = test(args, env, agent, writer)
            tune.report(reward_mean=re_mean, )
            agent.save(episode)
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        
        for t in itertools.count(start=1):

            #env.render()
            # select action
            action = agent.select_action(state, noise=False)
            # execute action
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

            if(done):
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print("total reward : {0:.2f}".format(total_reward))
                rewards.append(total_reward)
                break

    print('Average Reward', np.mean(rewards))
    env.close()
    return np.mean(rewards)

def DDPG_BPT(config, args, env, checkpoint_dir=None):

    agent = DDPG(args, config, checkpoint_dir)  
    
    writer = SummaryWriter(args.logdir)
    train(args, env, agent, writer, config)
    
def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ddpg_PBT.pth')
    parser.add_argument('--logdir', default='log/ddpg')
    parser.add_argument('--name', default='ddpg_best_config')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    # parser.add_argument('--episode', default=1200, type=int)
    # parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=500000, type=int)
    # parser.add_argument('--lra', default=1e-3, type=float)
    # parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLanderContinuous-v2')

    if(args.test_only):
        with open(args.name + ".json") as fp:
            config = json.load(fp)
        writer = SummaryWriter(args.logdir)
        best_model = os.path.join(os.getcwd(), args.model)
        agent = DDPG(args, config, best_model)
        test(args, env, agent, writer)
    else:
        # defined hyperparameter
        config = {
            "lr" : tune.loguniform(1e-4, 1e-3),
            "batch_size" : tune.sample_from(lambda _: 2 ** np.random.randint(4, 7)),
        }

        pbt = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="reward_mean",
            mode="max",
            perturbation_interval=100,
            hyperparam_mutations={
                "episode" : tune.choice([x for x in range(1000, 2001)])
            })

        analysis = tune.run(
            partial(DDPG_BPT, args=args, env=env),
            name="ddpg_PBT",
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

        agent = DDPG(args, best_trial.config, best_checkpoint_dir)
        writer = SummaryWriter(args.logdir)
        test(args, env, agent, writer)

        agent.save_best(args.model)

if __name__ == '__main__':
    main()


