from abc import ABC, abstractmethod
import random, numpy as np
from numpy.random import default_rng

class Agent(ABC):
    @abstractmethod
    def __init__(self, env):
        self.env = env
        self.random_generator = default_rng()
        pass

    @abstractmethod
    def act(self, observation, epsilon):
        pass

    def evaluate(self, episodes):
        """ Starts an evaluation cycle for an agent """
        
        won = 0
        sum_episode_len = 0
        for episode in range(episodes):
            done = False
            observations = []
            observation, reward, done = self.env.field, 0, False
            observations.append(observation)
            while True:
                self.env.show()
                sum_episode_len += 1
                action = self.act(observation, 1.0)
                observation, reward, done = self.env.step(action)
                observations.append(observation)
                if done:
                    if observation == self.env.winning_field:
                        won += 1
                    break
                if len(observations) > 2 and observations[-1] == observations[-2] and observations[-2] == observations[-3]:
                    print(f"Agent stuck at field {observation} trying to do action {action}")
                    break
        avg_episode_len = sum_episode_len / episodes
        print(f"The current policy is able to win {won} of {episodes} episodes with an average episode length of {avg_episode_len}")

class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        pass
    
    def act(self, observation, epsilon=0):
        return random.choice(self.env.action_space)

class VAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.V = {env.start_field:0} # Initialize starting field V-Value

    def best_action(self, observation):
        best_action, max_V = None, None
        for action in self.env.action_space:
            field, field_violation = self.env.calculate_field(action)
            if field_violation:
                continue
            if field not in self.V:
                self.V[field] = 0.0
            V = self.V[field]
            if max_V == None or V > max_V:
                best_action = action
                max_V = V
        return best_action

    def act(self, observation, epsilon = 1.0):
        if epsilon > self.random_generator.uniform():
            return self.best_action(observation)
        else:
            return random.choice(self.env.action_space)

    def learn(self, data, learning_rate, gamma):
        """ Starts a learning cycle
        Params:
        exploitation: The opposite of exploration, meaning if 1, the model won't learn anything new """

        V_prime = 0.0 # End state does not have any future reward
        for observation, immediate_reward in reversed(data):
            if observation not in self.V:
                self.V[observation] = 0.0
            V = V_prime * gamma + immediate_reward
            self.V[observation] += (V - self.V[observation]) * learning_rate
            V_prime = V

    def train(self, episodes, learning_rate, epsilon, gamma):
        """ Starts a training cycle for an agent
        Params:
        epsilon: The opposite of exploration, meaning if 1, the model won't learn anything new
        gamma: How important is future reward compared to immediate reward? """
        
        for episode in range(episodes):
            done = False
            data = []
            observation, reward, done = self.env.field, 0, False
            data.append((observation, reward))
            while True:
                observation, reward, done = self.env.step(self.act(observation, epsilon))
                if observation not in self.V:
                    self.V[observation] = 0.0
                data.append((observation, reward))
                if done:
                    break
            self.learn(data, learning_rate, gamma)

class QAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.Q = {}
        # Initialize starting field Q-Values
        self._init_unknown_observation(self.env.start_field)

    def best_action(self, observation):
        best_action, max_Q = None, None
        for action in self.env.action_space:
            Q = self.Q[self.env.field][action]
            if max_Q == None or Q > max_Q:
                best_action = action
                max_Q = Q
        return best_action

    def act(self, observation, epsilon = 1.0):
        if epsilon >= self.random_generator.uniform():
            return self.best_action(observation)
        else:
            return random.choice(self.env.action_space)

    def learn(self, data, learning_rate, gamma):
        """ Starts a learning cycle
        Params:
        exploitation: The opposite of exploration, meaning if 1, the model won't learn anything new """

        V_prime = 0.0 # End state does not have any future reward
        for observation, action, immediate_reward in reversed(data):
            Q = V_prime * gamma + immediate_reward
            self.Q[observation][action] += (Q - self.Q[observation][action]) * learning_rate
            V_prime = Q

    def train(self, episodes, learning_rate, epsilon, gamma):
        """ Starts a training cycle for an agent
        Params:
        epsilon: The opposite of exploration, meaning if 1, the model won't learn anything new
        gamma: How important is future reward compared to immediate reward? """
        
        for episode in range(episodes):
            done = False
            data = []
            observation, done = self.env.field, False
            self._init_unknown_observation(observation)
            while True:
                action = self.act(observation, epsilon)
                result, reward, done = self.env.step(action)
                if result not in self.Q:
                    self._init_unknown_observation(result)
                data.append((observation, action, reward))
                observation = result
                if done:
                    break
            self.learn(data, learning_rate, gamma)

    def _init_unknown_observation(self, observation):
        self.Q[observation] = {}
        for action in self.env.action_space:
            self.Q[observation][action] = 0

