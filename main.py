import numpy as np
import random
from gridworld import Gridworld
from agents import RandomAgent, VAgent, QAgent

if __name__ == "__main__":
    gridworld = Gridworld(4, 4, 4, 8, {7}, 13)
    #agent = RandomAgent(gridworld)
    #agent = VAgent(gridworld)
    agent = QAgent(gridworld)
    
    agent.train(3000, 0.1, 0.5, 0.98)
    print("Q after some learning:")
    print(sorted(agent.Q.items()))

    print("Use best policy:")
    agent.evaluate(1)

    #agent.env = Gridworld(4, 4, 4, 8, {6}, 13)
    #agent.evaluate(100)



