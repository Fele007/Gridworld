import numpy as np
import random
from gridworld import Gridworld
from agents import RandomAgent, VAgent, QAgent, softmax

if __name__ == "__main__":
    gridworld = Gridworld(4, 4, 4, 8, {7}, 13)
    #agent = RandomAgent(gridworld)
    agent = VAgent(gridworld)
    #agent = QAgent(gridworld)
    
    agent.train(100, 0.1, 0.5, 0.98,)
    #agent.train(10, 0.1, 0.9, 0.98, softmax = True)
    #agent.train(10, 0.1, 0.9, 0.98, softmax = True)
    print("Q after some learning:")
    print(sorted(agent.V.items()))
    print(agent.evaluation_data)

    agent.env = Gridworld(4, 4, 4, 8, {6}, 13)
    agent.evaluate(100)



