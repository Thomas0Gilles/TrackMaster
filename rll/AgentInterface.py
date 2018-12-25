import gym
from utils import Logger
import time


class Agent:
    def __init__(self, env: gym.Env, logger=Logger()):
        self.env = env
        self.exploration = True
        self.log = []  # episode log
        self.logger = logger  # external logger

    def act(self, state, explore):
        raise NotImplementedError

    def step_update(self, state, action, new_state, reward):
        # updates knowledge each time an action is taken
        pass

    def episode_update(self):
        # update inner knowledge based on log
        pass

    def episode(self, state, explore=True, visualize=False):
        total_reward = 0
        done = False
        while not done:
            action = self.act(state, explore)
            new_state, reward, done, info = self.env.step(action)
            if not done:
                self.step_update(state, action, new_state, reward, done)
            else:
                self.step_update(state, action, None, reward, done)
            self.log.append(dict(state=state, action=action, reward=reward))
            state = new_state
            total_reward += reward
            if visualize:
                self.env.render()
        return total_reward

    def train(self, nb_episodes=1000, visualize=False, verbose=None):
        t0 = time.time()
        for i in range(nb_episodes):
            t1 = time.time()
            reward = self.episode(self.env.reset(), visualize=visualize)
            self.episode_update()
            self.logger.log('Rewards', reward)
        print('Mean Episodic Time :{0}s'.format((time.time()-t0)/nb_episodes))


    def test(self, nb_episodes=100, visualize=True):
        rewards = []
        for _ in range(nb_episodes):
            rewards.append(self.episode(self.env.reset(), explore=False, visualize=visualize))
        print('All Rewards :', rewards)
        return rewards