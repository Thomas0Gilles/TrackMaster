from rll import Agent
from utils import Logger

import gym
import numpy as np
import time

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

class DDPG(Agent):
    def __init__(self,
                 env : gym.Env,
                 logger=Logger(),
                 n_layers_actor=3,
                 n_units_actor=16,
                 n_layers_critic=3,
                 n_units_critic=32,
                 sigma_decay=1,
                 sigma=0.3
                 ):
        nb_actions = env.action_space.shape[0]
        
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        for i in range(n_layers_actor):
            actor.add(Dense(n_units_actor))
            #actor.add(BatchNormalization())
            actor.add(Activation('relu'))
            #actor.add(LeakyReLU())
        actor.add(Dense(nb_actions))
        actor.add(Activation('tanh'))
        
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        for i in range(n_layers_critic):
            x = Dense(n_units_critic)(x)
            #x = BatchNormalization()(x)
            x = Activation('relu')(x)
            #x = LeakyReLU()(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        xo = Dense(n_units_critic, activation='relu')(flattened_observation)
        #xo = Dense(n_units_critic, activation='relu')(xo)
        x = Concatenate()([xo, action_input])
        x = Dense(n_units_critic, activation='relu')(x)
        x = Dense(n_units_critic, activation='relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=sigma)
        agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=.99, target_model_update=1e-3, batch_size=64, train_interval=16)
        agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
        self.agent = agent
        self.env = env
        self.sigma_decay = sigma_decay
        super().__init__(env, logger)
        
    def train(self, nb_episodes=1000, visualize=False, verbose=1, nb_max_episode_steps=None):
        history = self.agent.fit(self.env, nb_steps=nb_episodes, visualize=visualize, verbose=verbose, nb_max_episode_steps=nb_max_episode_steps, log_interval=10000)
        return history
        
    def train_env(self, env,  nb_episodes=1000, visualize=False, verbose=1, nb_max_episode_steps=None):
        history = self.agent.fit(env, nb_steps=nb_episodes, visualize=visualize, verbose=verbose, nb_max_episode_steps=nb_max_episode_steps, log_interval=10000)
        return history
        
    def test(self, nb_episodes, visualize=True, verbose=1):
        history = self.agent.test(self.env, nb_episodes=nb_episodes, visualize=visualize, verbose=verbose)
        print('Mean', np.mean(history.history['episode_reward']))
        print('Std', np.std(history.history['episode_reward']))
        return history
        
    def test_env(self, env, nb_episodes, visualize=True, verbose=1):
        t=time.time()
        history = self.agent.test(env, nb_episodes = nb_episodes, visualize=visualize, verbose=verbose)
        print('Mean', np.mean(history.history['episode_reward']))
        print('Std', np.std(history.history['episode_reward']))
        print(time.time()-t)
        return history
    
    def act(self, state, explore=True):
        action = self.agent.forward(state)
        return action
    
    def step_update(self, state, action, new_state, reward, done=False):
        self.agent.backward(reward, done)
        if new_state is None:
            self.agent.forward(state)
            self.agent.backward(0., terminal=False)
        return None
    
    def episode_update(self):
        self.log = []
        self.agent.random_process.sigma *= self.sigma_decay
    

class NAF(Agent):
    def __init__(self, 
                 env : gym.Env,
                 logger=Logger()):
        nb_actions = env.action_space.shape[0]
        
        V_model = Sequential()
        V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(1))
        V_model.add(Activation('linear'))
        
        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(nb_actions))
        mu_model.add(Activation('linear'))
        
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        x = Concatenate()([action_input, Flatten()(observation_input)])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
        x = Activation('linear')(x)
        L_model = Model(inputs=[action_input, observation_input], outputs=x)
        
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
        agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, random_process=random_process)
        agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
        self.agent = agent
        self.env = env
        super().__init__(env, logger)
        
    def train(self, nb_episodes=1000, visualize=False, verbose=1):
        history = self.agent.fit(self.env, nb_steps=nb_episodes, visualize=visualize, verbose=verbose, nb_max_episode_steps=None, log_interval=10000)
        return history
        
    def train_env(self, env,  nb_episodes=1000, visualize=False, verbose=1):
        history = self.agent.fit(env, nb_steps=nb_episodes, visualize=visualize, verbose=verbose, nb_max_episode_steps=None, log_interval=10000)
        return history
        
    def test(self, nb_episodes, visualize=True):
        history = self.agent.test(self.env, nb_episodes = nb_episodes, visualize=visualize)
        print(np.mean(history.history['episode_reward']))
        return history
        
    def test_env(self, env, nb_episodes, visualize=True, verbose=1):
        t=time.time()
        history = self.agent.test(env, nb_episodes = nb_episodes, visualize=visualize, verbose=verbose)
        print('Mean', np.mean(history.history['episode_reward']))
        print('Std', np.std(history.history['episode_reward']))
        print(time.time()-t)
        return history