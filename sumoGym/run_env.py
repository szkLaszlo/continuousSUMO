"""
This script contains a basic usage of the environment.
"""
import gym
import sumoGym.environment  # This import must not be removed

if __name__ == "__main__":
    # Modify simulation_directory for your directory path
    env = gym.make('SUMOEnvironment-v0', simulation_directory='..\\sim_conf')
    while True:
        terminate = False
        while not terminate:
            action = 4  # int(input('next action'))
            state, reward, terminate, info = env.step(action)
            print(info)
        env.reset()
