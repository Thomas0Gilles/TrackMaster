import gym
from rll.ImplAgents import QLearner, SarsaLearner

q_learner = QLearner(env=gym.make('MountainCar-v0'),
                     boxes_resolution=20,
                     gamma=0.99,
                     alpha=1 / 10,
                     tau=20,
                     action_selection_mode='epsilon-greedy',
                     epsilon=1 / 100)


sarsa_learner = SarsaLearner(env=gym.make('MountainCar-v0'),
                             boxes_resolution=20,
                             gamma=0.99,
                             alpha=1 / 10,
                             tau=20,
                             action_selection_mode='epsilon-greedy',
                             epsilon=1 / 100)
