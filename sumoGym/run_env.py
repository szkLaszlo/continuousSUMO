"""
This script contains a basic usage of the environment.
"""
import gym

if __name__ == "__main__":
    env = gym.make('SUMOEnvironment-v0')
    while True:
        terminate = False
        while not terminate:
            action = 4  # int(input('next action'))
            state, reward, terminate, info = env.step(action)
            print(info)
        env.reset()
