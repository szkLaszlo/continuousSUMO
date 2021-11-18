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
                   simulation_directory='../../fastcl/sumo_simulations',
                   type_os="merge",
                   type_as='discrete_longitudinal',
                   reward_type='merge',
                   mode='human',
                   change_speed_interval=1000,
                   )
    while True:
        terminate = False
        while not terminate:
            action = random.randint(0, env.action_space.n-1)
            state, reward, terminate, info = env.step(action)
            time.sleep(0.1)
            if terminate:
                print(info)
        env.reset()


if __name__ == "__main__":
    random.seed(100)
    main()
