from rll import Agent
from utils import Logger
from rll import ContinuousStateActionMap

import gym
import numpy as np


class QLearner(Agent):
    def __init__(self,
                 env: gym.Env,
                 logger=Logger(),
                 boxes_resolution=10,
                 alpha=0.1,
                 gamma=0.95,
                 action_selection_mode='epsilon-greedy',
                 epsilon=0.1,
                 epsilon_decay=0.997):
        self.alpha = alpha
        self.gamma = gamma
        self.action_selection_mode = action_selection_mode
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.Q = ContinuousStateActionMap(env=env, boxes_resolution=boxes_resolution)
        super().__init__(env, logger)

    def act(self, state, explore=True):
        best_action = self.Q.argmax(state)
        if explore and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # random !
        return best_action

    def step_update(self, state, action, new_state, reward, done=False):
        expected_reward=0
        if new_state is not None:
            expected_reward = self.gamma * np.max(self.Q[new_state])
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
            + self.alpha * (reward + expected_reward)
        return None

    def episode_update(self):
        self.log = []
        self.epsilon *= self.epsilon_decay


class SarsaLearner(QLearner):
    def step_update(self, state, action, new_state, reward, done):
        f = self.Q[new_state, self.act(new_state)] if not done else 0
        d = reward + self.gamma * f - self.Q[state, action]
        self.Q[state, action] += self.alpha * d
