Project with [Guillaume Couairon](https://gitlab.com/PhazCode) on Reinforcement Learning for Race Cars.
See Project Page [here](https://phazcode.gitlab.io/trackmaster/)

![l](https://phazcode.gitlab.io/projects/trackmaster_images/animated.gif)

Quick Start

imports :

```python
from track_env import TrackEnv
from rll.ImplAgents import DDPG, QLearner, SarsaLearner
from utils import ShowVariableLogger
import numpy as np
```

Loading tracks:

```python
ENV = simple_track = TrackEnv(track_file='tracks/track_0.npy')
complex_track = TrackEnv(track_file='tracks/complex_13_15step.npy', nb_sensors=9)
```

Loading agents:

```python
qlearner = QLearner(env=ENV,
                    logger=ShowVariableLogger(average_window=100),
                    boxes_resolution=3)

sarsa_agent = SarsaLearner(env=ENV,
                           logger=ShowVariableLogger(average_window=1),
                           boxes_resolution=3,
                           gamma=0.99,
                           alpha=1 / 10,
                           action_selection_mode='epsilon-greedy',
                           epsilon=1 / 100)


agent = ddpg_agent = DDPG(env=ENV,
                    logger=ShowVariableLogger(average_window=1),
                     n_layers_actor=3,
                     n_units_actor=16,
                     n_layers_critic=3,
                     n_units_critic=128)
```

Traning Agent (Set visualize = True to see training process, Much Slower):

```python
hist_train = agent.train(nb_episodes=200, visualize=True, verbose=1)
```

Test and Visualize Results:

```python
rewards = agent.test(nb_episodes=10, visualize=True)
```

Access rewards history:

```python
print("Mean reward", np.mean(rewards.history['episode_rewards']))
```
