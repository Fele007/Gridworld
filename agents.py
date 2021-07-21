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
        random.seed(seed)

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
    # Initialize a network
    def initialize_network(self, n_inputs, n_hidden, n_outputs, n_hidden_layers):
        network = list()
        for i in range(n_hidden_layers):
            hidden_layer = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
            network.append(hidden_layer)
        output_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network

    # Calculate neuron activation for an input
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    # Forward propagate input to a network output
    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                neuron['output'] = transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
          return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(self, network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    # Make a prediction with a network
    def predict(self, network, row):
        outputs = forward_propagate(network, row)
        return outputs.index(max(outputs))

    def __init__(self, env, hidden_layers, nodes):
        super().__init__(env)
        self.model = self.initialize_network(2, 1, 2, 3)

    def best_action(self, observation):
        return self.predict(self.model, observation)

    def softmax_action(self, observation):
        # TODO: Implement for DQN
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
        for observation, action, outputs, immediate_reward in reversed(data):
            # TODO: Implement this for DQN Agents
            Q = Q_prime * gamma + immediate_reward
            error = Q - outputs[action]
            backward_propagate_error(self.model, error)
            self.Q[observation][action] += (Q - self.Q[observation][action]) * learning_rate
            Q_prime = Q

            expected = [0 for i in range(len(self.env.action_space))]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

    def train(self, episodes, learning_rate, epsilon, gamma = 1, softmax = False, max_episode_length = float('inf')):
        """ Starts a training cycle for an agent
        Params:
        epsilon: The opposite of exploration, meaning if 1, the model won't learn anything new
        gamma: How important is future reward compared to immediate reward? """

        for episode in range(episodes):
            done = False
            data = []
            observation, done = self.env.field, False
            while True:
                outputs = self.forward_propagate(self.model, observation)
                randn = self.random_generator.uniform()
                if softmax and (epsilon > randn):
                    raise NotImplementedError("Softmax not yet implemented for DQN Agent")
                    #action = self.softmax_action(observation)
                elif epsilon > randn:
                    action = outputs.index(max(outputs))
                else:
                    action = self.random_generator.choice(self.env.action_space)
                result, reward, done = self.env.step(action)
                data.append([observation, outputs, action, reward])
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