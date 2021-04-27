from abc import ABC, abstractmethod
import random, numpy as np
from numpy.random import default_rng

class Agent(ABC):
    @abstractmethod
    def __init__(self, env):
        self.env = env
        pass
    @abstractmethod
    def act(self, observation, reward, done):
        pass

class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        pass
    
    def act(self, observation, reward, done):
        return random.choice(self.action_space)

class VAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.V = {env.start_field:0} # Initialize starting field V-Value
        self.random_generator = default_rng()

    def best_move(self, observation):
        best_move, max_V = None, None
        for move in self.env.action_space:
            field, field_violation = self.env.calculate_field(move)
            if field_violation:
                continue
            if field not in self.V:
                self.V[field] = 0.0
            V = self.V[field]
            if max_V == None or V > max_V:
                best_move = move
                max_V = V
        return best_move

    def act(self, observation, epsilon = 1.0):
        if epsilon > self.random_generator.uniform():
            return self.best_move(observation)
        else:
            return random.choice(self.env.action_space)

    def learn(self, data, learning_rate, gamma):
        """ Starts a learning cycle
        Params:
        exploitation: The opposite of exploration, meaning if 1, the model won't learn anything new """

        V_prime = 0.0 # End state does not have any future reward
        for observation, immediate_reward in reversed(data):
            V = V_prime * gamma + immediate_reward
            self.V[observation] += (V - self.V[observation]) * learning_rate
            V_prime = V

            #self.V[observation] + learning_rate * (immediate_reward - self.V[observation])
            #self.V[observation] = reward

    def train(self, episodes, learning_rate, epsilon, gamma):
        """ Starts a training cycle for an agent
        Params:
        epsilon: The opposite of exploration, meaning if 1, the model won't learn anything new """
        
        for episode in range(episodes):
            done = False
            episode_len = 0
            data = []
            observation, reward, done = self.env.field, 0, False
            data.append((observation, reward))
            while True:
                observation, reward, done = self.env.step(self.act(observation, epsilon))
                if observation not in self.V:
                    self.V[observation] = 0.0
                data.append((observation, reward))
                episode_len += 1
                if done:
                    break
            print(f"Finished episode with a length of {episode_len}")
            self.learn(data, learning_rate, gamma)
