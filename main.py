import numpy as np
import random
from gridworld import Gridworld
from agents import RandomAgent, VAgent, QAgent
import time
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    gridworld = Gridworld(4, 4, 4, 8, {7}, 13)
    r_agent = RandomAgent(gridworld)
    
    q_agent = QAgent(gridworld)
    
    # Evaluate Random Agent
    evaluation = r_agent.evaluate(1000)
    max_random_episode_length = 5000                                 # r_agent.evaluate(10)[4]
    print(f"Random episode length = {max_random_episode_length}")

    # Evaluate Value Agent
    for seed in range(5):
        v_agent = VAgent(gridworld)
        v_agent.set_random_seed(seed)
        start = time.perf_counter()
        for i in range(1000):
            v_agent.train(100, 0.1, 0.9, max_episode_length=max_random_episode_length)
            evaluation = v_agent.evaluate(100, max_random_episode_length)
            if evaluation['max_episode_length']==evaluation['min_episode_length'] and evaluation['min_episode_length'] != max_random_episode_length:
                print (f"V-Agent needs {evaluation['total_training_episodes']} episodes to converge in {time.perf_counter()-start} seconds to an episode length of {evaluation['avg_episode_length']}, winning {evaluation['won']} of {evaluation['episodes']} episodes.")
                v_agent.plot_episode_data()
                break
    plt.show()

    # Evaluate Q-Value Agent
    for seed in range(5):
        q_agent = QAgent(gridworld)
        q_agent.set_random_seed(seed)
        start = time.perf_counter()
        for i in range(1000):
            q_agent.train(100, 0.1, 0.9, max_episode_length=max_random_episode_length)
            evaluation = q_agent.evaluate(100, max_random_episode_length)
            if evaluation['max_episode_length']==evaluation['min_episode_length'] and evaluation['min_episode_length'] != max_random_episode_length:
                print (f"Q-Agent needs {evaluation['total_training_episodes']} episodes to converge in {time.perf_counter()-start} seconds to an episode length of {evaluation['avg_episode_length']}, winning {evaluation['won']} of {evaluation['episodes']} episodes.")
                q_agent.plot_episode_data()
                break
    plt.show()

    # Plot Data
    #v_agent.plot_evaluation_data()
    #q_agent.plot_evaluation_data()




