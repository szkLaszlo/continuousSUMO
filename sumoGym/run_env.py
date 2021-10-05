"""
This script contains a basic usage of the environment.
"""
import random
import time

import gym
# This import must not be removed: import continuousSUMO.sumoGym.environment
import continuousSUMO.sumoGym.environment
import matplotlib.pyplot as plt


def main():
    # Modify simulation_directory for your directory path
    env = gym.make('SUMOEnvironment-v0',
                   simulation_directory='../basic_env',
                   type_os="structured",
                   type_as='discrete',
                   reward_type='positive',
                   mode='none',
                   change_speed_interval=100,
                   )
    while True:
        terminate = False
        while not terminate:
            # action = [float(input('next steering')), float(input('next vel_dif'))]
            action = int(input())# random.randint(0,8)
            state, reward, terminate, info = env.step(action)
            time.sleep(0.1)
            if terminate:
                print(info)
        env.reset()


if __name__ == "__main__":
    random.seed(100)
    main()
