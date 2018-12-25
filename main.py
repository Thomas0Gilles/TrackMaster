from track_env import TrackEnv
from rll.ImplAgents import DDPG, QLearner, SarsaLearner
from utils import ShowVariableLogger
import numpy as np

simple_track = TrackEnv(track_file='tracks/track_0.npy')

complex_track = TrackEnv(track_file='tracks/complex_13_15step.npy', nb_sensors=9)

ENV=simple_track

agent = QLearner(env=ENV,
                 logger=ShowVariableLogger(average_window=100),
                 boxes_resolution=3)

agent = DDPG(env=ENV,
                    logger=ShowVariableLogger(average_window=1),
                     n_layers_actor=3,
                     n_units_actor=16,
                     n_layers_critic=3,
                     n_units_critic=128,
                    )

agent = SarsaLearner(env=ENV,
logger=ShowVariableLogger(average_window=1),
                             boxes_resolution=3,
                             gamma=0.99,
                             alpha=1 / 10,
                             action_selection_mode='epsilon-greedy',
                             epsilon=1 / 100)

# Set visualize = True to see training process (Much Slower)
hist_train = agent.train(nb_episodes=200, visualize=True, verbose=1) #, nb_max_episode_steps=200)


rewards = agent.test(nb_episodes=10, visualize=True)

print("Mean reward", np.mean(rewards.history['episode_rewards']))

