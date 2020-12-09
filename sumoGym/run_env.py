"""
This script contains a basic usage of the environment.
"""
import time

import gym
# This import must not be removed: import continuousSUMO.sumoGym.environment
import continuousSUMO.sumoGym.environment
import matplotlib.pyplot as plt


def main():
    # Modify simulation_directory for your directory path
    env = gym.make('SUMOEnvironment-v0',
                   simulation_directory='../basic_env',
                   type_os="image",
                   type_as='discrete',
                   reward_type='speed',
                   mode='none',
                   change_speed_interval=100,
                   )
    while True:
        terminate = False
        while not terminate:
            # action = [float(input('next steering')), float(input('next vel_dif'))]
            action = [0.01, 0.0]
            state, reward, terminate, info = env.step(action)
            time.sleep(0.1)
            print(info)
        env.reset()


if __name__ == "__main__":
    main()
