import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from itertools import count


# actor输出的是action的概率分布
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim=None):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x))
        return action_probs


class PolicyGradient:
    def __init__(self, state_dim, action_dim, action_lim=None, learning_rate=0.01, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.lr = learning_rate
        self.gamma = gamma
        self.ep_state, self.ep_action, self.ep_reward = [], [], []

        self.actor = Actor(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def choose_action(self, state):
        action_probs = self.actor(state).detach()
        return action_probs

    def store(self, s, a, r):
        self.ep_state.append(s)
        self.ep_action.append(a)
        self.ep_reward.append(r)

    def learn(self):
        discounted_ep_reward_norm = self.discount_and_norm_rewards_()

        for i in self.ep_state:
            i.unsqueeze_(0)

        ep_state = torch.cat(self.ep_state, 0)
        ep_action = torch.tensor(self.ep_action)

        action_probs = self.actor(ep_state)
        neg_log_prob = self.loss_func(action_probs, ep_action)
        print(neg_log_prob)
        print(discounted_ep_reward_norm)
        loss = neg_log_prob.double() * torch.tensor(discounted_ep_reward_norm)
        loss = torch.mean(loss)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_state, self.ep_action, self.ep_reward = [], [], []  # empty episode data
        return discounted_ep_reward_norm

    # 衰减回合的 reward
    def discount_and_norm_rewards_(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_reward)
        running_add = 0
        for t in reversed(range(0, len(self.ep_reward))):
            running_add = running_add * self.gamma + self.ep_reward[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


def train(env_name='CartPole-v0'):
    env = gym.make(env_name).unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    episode_duration = []

    pg = PolicyGradient(state_dim, action_dim)

    for n_episode in range(200000):
        state = env.reset()
        state = torch.from_numpy(state).float()

        for t in count():
            env.render()
            action_probs = pg.choose_action(state)
            c = torch.distributions.Categorical(action_probs)
            action = c.sample().data.numpy().astype('int32')

            state_, reward, done, info = env.step(action)
            # reward = 0 if done else reward  # correct the reward
            pg.store(state, action.item(), reward)
            state = torch.from_numpy(state_).float()

            if done:
                episode_duration.append(t)
                pg.learn()
                break


train()
