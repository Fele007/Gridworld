from abc import ABC, abstractmethod
import random, numpy as np
from numpy.random import default_rng

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Agent(ABC):
    @abstractmethod
    def __init__(self, env):
        """ Params:
        progress: Seen training data of the agent """

        self.env = env
        self.evaluation_data = []
        self.progress = 0

    @abstractmethod
    def act(self, observation, epsilon):
        pass

    def evaluate(self, episodes):
        """ Starts an evaluation cycle for an agent
        Params:
        episodes: Number of episodes for evaluation """

        won = 0
        sum_episode_length = 0
        max_episode_length = 0
        min_episode_length = 0
        for episode in range(episodes):
            done = False
            observations = []
            observation, reward, done = self.env.field, 0, False
            observations.append(observation)
            while True:
                self.env.show()
                action = self.act(observation, 1.0)
                observation, reward, done = self.env.step(action)
                observations.append(observation)
                if done or len(observations) > self.max_random_episode_length:
                    episode_length = len(observations) - 1
                    if episode_length > max_episode_length or max_episode_length == 0:
                        max_episode_length = episode_length
                    if episode_length < min_episode_length or min_episode_length == 0:
                        min_episode_length = episode_length
                    sum_episode_length += episode_length
                    if observation == self.env.winning_field:
                       won += 1
                    break
        avg_episode_length = sum_episode_length / episodes
        print(f"The current policy is able to win {won} of {episodes} episodes with an average episode length of {avg_episode_length}")
        return [self.progress, avg_episode_length, max_episode_length, min_episode_length]

    def plot_evaluation_data(self):
        import matplotlib as plt



class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.random_generator = default_rng()
    
    def act(self, observation, epsilon=0):
        return random.choice(self.env.action_space)

    def evaluate(self, episodes):
        """ Starts an evaluation cycle for a random agent agent
        Params:
        episodes: Number of episodes for evaluation """

        won = 0
        sum_episode_length = 0
        max_episode_length = 0
        min_episode_length = 0
        for episode in range(episodes):
            done = False
            observations = []
            observation, reward, done = self.env.field, 0, False
            observations.append(observation)
            while True:
                action = self.act(observation, 1.0)
                observation, reward, done = self.env.step(action)
                observations.append(observation)
                if done:
                    episode_length = len(observations) - 1
                    if episode_length > max_episode_length or max_episode_length == 0:
                        max_episode_length = episode_length
                    if episode_length < min_episode_length or min_episode_length == 0:
                        min_episode_length = episode_length
                    sum_episode_length += episode_length
                    if observation == self.env.winning_field:
                       won += 1
                    break
        avg_episode_length = sum_episode_length / episodes
        print(f"The random agent is able to win {won} of {episodes} episodes with an average episode length of {avg_episode_length}")
        return [self.progress, avg_episode_length, max_episode_length, min_episode_length]

class VAgent(RandomAgent):
    def __init__(self, env):
        super().__init__(env)
        self.V = {env.start_field:0} # Initialize starting field V-Value
        self.max_random_episode_length = RandomAgent(self.env).evaluate(10)[2]

    def best_action(self, observation):
        best_action, max_V = None, None
        for action in self.env.action_space:
            field, field_violation = self.env.calculate_field(observation, action)
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

    def train(self, episodes, learning_rate, epsilon, gamma, evaluate = None):
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
                if done or len(data) > self.max_random_episode_length:
                    episode_length = len(data)
                    self.progress += 1
                    won = None
                    if observation == self.env.winning_field:
                        won = True
                    elif observation == self.env.losing_field:
                        won = False
                    self.evaluation_data.append([self.progress, episode_length, won])
                    break
            self.learn(data, learning_rate, gamma)

class QAgent(RandomAgent):
    def __init__(self, env):
        super().__init__(env)
        self.Q = {}
        # Initialize starting field Q-Values
        self._init_unknown_observation(self.env.start_field)
        self.max_random_episode_length = RandomAgent(self.env).evaluate(10)[2]

    def best_action(self, observation):
        best_action, max_Q = None, None
        for action in self.env.action_space:
            Q = self.Q[observation][action]
            if max_Q == None or Q > max_Q:
                best_action = action
                max_Q = Q
        return best_action

    def softmax_action(self, observation):
        Q = []
        for action in self.env.action_space:
            Q.append(self.Q[observation][action])
        Q = softmax(Q)
        return random.choices(self.env.action_space, weights = Q)[0]


    def act(self, observation, epsilon = 1.0, softmax=False):
        """ Params:
        softmax: uses a softmax of possible actions as greedy behavior"""

        randn = self.random_generator.uniform()
        if softmax and (epsilon > randn):
            return self.softmax_action(observation)
        elif epsilon > randn:
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

    def train(self, episodes, learning_rate, epsilon, gamma, softmax = False, evaluate = True):
        """ Starts a training cycle for an agent
        Params:
        epsilon: The opposite of exploration, meaning if 1, the model won't learn anything new
        gamma: How important is future reward compared to immediate reward?
        evaluate: Episodes used for evaluation if this should take place """

        for episode in range(episodes):
            done = False
            data = []
            observation, done = self.env.field, False
            self._init_unknown_observation(observation)
            while True:
                action = self.act(observation, epsilon, softmax)
                result, reward, done = self.env.step(action)
                if result not in self.Q:
                    self._init_unknown_observation(result)
                data.append((observation, action, reward))
                observation = result
                if done or len(data) > self.max_random_episode_length:
                    episode_length = len(data)
                    self.progress += 1
                    won = None
                    if observation == self.env.winning_field:
                       won = True
                    elif observation == self.env.losing_field:
                        won = False
                    self.evaluation_data.append([self.progress, episode_length, won])
                    break
            self.learn(data, learning_rate, gamma)

    def _init_unknown_observation(self, observation):
        self.Q[observation] = {}
        for action in self.env.action_space:
            self.Q[observation][action] = 0

