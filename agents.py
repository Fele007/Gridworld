from abc import ABC, abstractmethod
import random, numpy as np
from numpy.random import default_rng
from functions import *
import pickle
import time
import torch

class Agent(ABC):
    @abstractmethod
    def __init__(self, env):
        self.env = env
        self.episode_data = []
        self.total_training_episodes = 0

    @abstractmethod
    def act(self, observation, epsilon):
        pass

    def evaluate(self, episodes, permissible_episode_length=float('inf')):
        """ Starts an evaluation cycle for an agent
        Params:
        episodes: Number of episodes for evaluation
        Returns:
        {total_trainig, episodes, won, avg_episode_length, max_episode_length, min_episode_length, inference}"""

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
                if done or len(observations) > permissible_episode_length:
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
        start = time.perf_counter()
        self.act(observation, 1.0)
        inference = time.perf_counter() - start
        return {'inference_time':inference, 'total_training_episodes':self.total_training_episodes, 'episodes':episodes, 'won':won, 'avg_episode_length':avg_episode_length, 'max_episode_length':max_episode_length, 'min_episode_length':min_episode_length}

    def plot_episode_data(self):
        import matplotlib.pyplot as plt      
        for row in self.episode_data:
            if row[2] == 'won':
                color = 'g'
            elif row[2] == 'stuck':
                color = 'b'
            elif row[2] == 'lost':
               color = 'r'
            else:
                color = 'k'
            plt.scatter(row[0], row[1], color=color, s=4)
        #plt.show()

    def store(self, location):
        pickle.dump(self, open(location, 'wb'))

    def load(self, location):
        """ Returns a new object from a pickled representation """

        return pickle.load(open(location, 'rb'))

class RandomAgent(Agent):

    def __init__(self, env):
        super().__init__(env)
        self.random_generator = default_rng()

    def set_random_seed(self, seed):
        self.random_generator = default_rng(seed)

    def act(self, observation, epsilon=0):
        return self.random_generator.choice(self.env.action_space)

class VAgent(RandomAgent):
    def __init__(self, env):
        super().__init__(env)
        self.V = {env.start_field:0} # Initialize starting field V-Value

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
            return self.random_generator.choice(self.env.action_space)

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

    def train(self, episodes, learning_rate, epsilon, gamma=1, max_episode_length=float('inf')):
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
                if done or len(data) > max_episode_length:
                    episode_length = len(data)
                    self.total_training_episodes += 1
                    result = 'stuck'
                    if observation == self.env.winning_field:
                        result = 'won'
                    elif observation == self.env.losing_field:
                        result = 'lost'
                    self.episode_data.append([self.total_training_episodes, episode_length, result])
                    break
            self.learn(data, learning_rate, gamma)

class QAgent(RandomAgent):
    def __init__(self, env):
        super().__init__(env)
        self.Q = {}
        # Initialize starting field Q-Values
        self._init_unknown_observation(self.env.start_field)

    def best_action(self, observation):
        best_action, max_Q = None, None
        for action in self.env.action_space:
            try:
                Q = self.Q[observation][action]
            except:
                self._init_unknown_observation(observation)
                Q = 0.0
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
            return self.random_generator.choice(self.env.action_space)

    def learn(self, data, learning_rate, gamma):
        """ Starts a learning cycle
        Params:
        exploitation: The opposite of exploration, meaning if 1, the model won't learn anything new """

        Q_prime = 0.0 # End state does not have any future reward
        for observation, action, immediate_reward in reversed(data):
            Q = Q_prime * gamma + immediate_reward
            self.Q[observation][action] += (Q - self.Q[observation][action]) * learning_rate
            Q_prime = Q

    def train(self, episodes, learning_rate, epsilon, gamma = 1, softmax = False, max_episode_length = float('inf')):
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
                action = self.act(observation, epsilon, softmax)
                result, reward, done = self.env.step(action)
                if result not in self.Q:
                    self._init_unknown_observation(result)
                data.append((observation, action, reward))
                observation = result
                if done or len(data) > max_episode_length:
                    episode_length = len(data)
                    self.total_training_episodes += 1
                    result = 'stuck'
                    if observation == self.env.winning_field:
                        result = 'won'
                    elif observation == self.env.losing_field:
                        result = 'lost'
                    self.episode_data.append([self.total_training_episodes, episode_length, result])
                    break
            self.learn(data, learning_rate, gamma)

    def _init_unknown_observation(self, observation):
        self.Q[observation] = {}
        for action in self.env.action_space:
            self.Q[observation][action] = 0

class DQNAgent(RandomAgent):
    def __init__(self, env, layers, nodes):
        super().__init__(env)
        self.model = torch.nn.Sequential(
            torch.nn.Linear)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        

    def best_action(self, observation):
        best_action, max_Q = None, None
        for action in self.env.action_space:
            try:
                Q = self.Q[observation][action]
            except:
                self._init_unknown_observation(observation)
                Q = 0.0
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
            return self.random_generator.choice(self.env.action_space)

    def learn(self, data, learning_rate, gamma):
        """ Starts a learning cycle
        Params:
        exploitation: The opposite of exploration, meaning if 1, the model won't learn anything new """

        learning_rate = 1e-6
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        for t in range(2000):

            # Forward pass: compute predicted y by passing x to the model. Module objects
            # override the __call__ operator so you can call them like functions. When
            # doing so you pass a Tensor of input data to the Module and it produces
            # a Tensor of output data.
            y_pred = model(xx)

            # Compute and print loss. We pass Tensors containing the predicted and true
            # values of y, and the loss function returns a Tensor containing the
            # loss.
            loss = loss_fn(y_pred, y)
            if t % 100 == 99:
                print(t, loss.item())

            # Zero the gradients before running the backward pass.
            model.zero_grad()
            #optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            loss.backward()
            # optimizer.step()

            # Update the weights using gradient descent. Each parameter is a Tensor, so
            # we can access its gradients like we did before.
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad



        Q_prime = 0.0 # End state does not have any future reward
        for observation, action, immediate_reward in reversed(data):
            Q = Q_prime * gamma + immediate_reward
            self.Q[observation][action] += (Q - self.Q[observation][action]) * learning_rate
            Q_prime = Q

    def train(self, episodes, learning_rate, epsilon, gamma = 1, softmax = False, max_episode_length = float('inf')):
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
                action = self.act(observation, epsilon, softmax)
                result, reward, done = self.env.step(action)
                if result not in self.Q:
                    self._init_unknown_observation(result)
                data.append((observation, action, reward))
                observation = result
                if done or len(data) > max_episode_length:
                    episode_length = len(data)
                    self.total_training_episodes += 1
                    result = 'stuck'
                    if observation == self.env.winning_field:
                        result = 'won'
                    elif observation == self.env.losing_field:
                        result = 'lost'
                    self.episode_data.append([self.total_training_episodes, episode_length, result])
                    break
            self.learn(data, learning_rate, gamma)