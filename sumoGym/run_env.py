"""
This script contains a basic usage of the environment.
"""
import gym
# This import must not be removed: import sumoGym.environment
import sumoGym.environment


def main():
    # Modify simulation_directory for your directory path
    env = gym.make('SUMOEnvironment-v0',
                   simulation_directory='..\\sim_conf',
                   type_os="image",
                   type_as='continuous',
                   reward_type='speed',
                   mode='human',
                   change_speed_interval=100,
                   )
    while True:
        terminate = False
        while not terminate:
            # action = [float(input('next steering')), float(input('next vel_dif'))]
            action = [0.00, 0]
            state, reward, terminate, info = env.step(action)
            print(info)
        env.reset()


if __name__ == "__main__":
    main()
