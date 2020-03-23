"""
Script created by szoke.laszlo95@gmail.com and bencepalotas@gmail.com
as a project of TRAFFIC MODELLING, SIMULATION AND CONTROL subject
"""

import copy
import math
import os
import platform
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import traci
import traci.constants as tc
from gym import spaces

grid_per_meter = 4  # Defines the precision of the returned image
x_range = 50  # symmetrically for front and back
x_range_grid = x_range * grid_per_meter  # symmetrically for front and back
y_range = 9  # symmetrically for left and right
y_range_grid = y_range * grid_per_meter  # symmetrically for left and right


class SUMOEnvironment(gym.Env):
    """

    """

    def __init__(self):
        pass

    def setup_observation_space(self):
        """
        This function is responsible for creating the desired observation space.
        For now discrete and continuous states are supported.
        :return:
        """
        pass

    def setup_action_space(self):
        """
        This function is responsible for creating the desired action space.
        For now discrete and continuous actions are supported.
        :return:
        """
        pass

    def setup_reward_system(self):
        """

        :return:
        """
        pass

    def setup_basic_environment_values(self):
        """

        :return:
        """
        pass

    def stop(self):
        """

        :return:
        """
        pass

    def start(self):
        """

        :return:
        """
        pass

    def reset(self):
        """

        :return:
        """
        pass

    def render(self, mode='human'):
        """

        :param mode:
        :return:
        """
        pass

    def step(self, action):
        """

        :param action:
        :return:
        """
        pass

    def


class EPHighWayEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):

        self.max_punishment = -10
        self.steps_done = 0
        self.rendering = None

        high = np.array([1, 0.5])
        low = np.array([-1, -0.7])

        self.action_space = spaces.Box(low, high, dtype=np.float)  # spaces.Discrete(9)
        self.observation_space = np.zeros((2 * x_range_grid, 2 * y_range_grid, 3))
        self._max_episode_steps = 2500
        self.rewards = [0, 0, 0, 0]  # was used for reward calculation
        # setting basic environment variables
        self.lane_width = None
        self.lane_offset = None
        self.sumoBinary = None
        self.sumoCmd = None
        self.egoID = None
        self.state = None
        self.desired_speed = None
        self.dt = None
        self.ego_start_position = 100000  # for the search of the last vehicle on the highway
        self.reward_type = "simple"  # Using simple calculation for now
        self.environment_state = None
        self.lanechange_counter = 0
        self.wants_to_change = []  # variable to count how many times the agent wanted to change lane
        self.change_after = 0  # variable after how many trials the lane is changed
        self.time_to_change_des_speed = 100
        # variable defining how many vehicles must exist on the road before ego is chosen.
        self.min_departed_vehicles = 1
        self.rand_index = 0

    def set_reward_type(self, reward_type):
        self.reward_type = reward_type

    def stop(self):
        traci.close()

    def reset(self):
        if self.rendering is not None:
            # this is for memory minimization
            try:
                for vehs in traci.vehicle.getIDList():
                    del vehs
            except KeyError:
                pass
            except AttributeError:
                pass
            except TypeError:
                pass
            # Changing configuration
            self.choose_random_simulation()
            # Loads traci configuration
            traci.load(self.sumoCmd[1:])
            # Loading variables with real values from traci
            self.lane_width = traci.lane.getWidth('A_0')
            self.lane_offset = traci.junction.getPosition('J1')[1] - 2 * self.lane_width - self.lane_width / 2
            self.dt = traci.simulation.getDeltaT()
            self.rewards = [0, 0, 0, 0]
            self.egoID = None  # Resetting chosen ego vehicle id
            self.steps_done = 0  # resetting steps done

            self.desired_speed = random.randint(120, 160) / 3.6
            self.state = None
            self.ego_start_position = 100000
            self.lanechange_counter = 0
            self.wants_to_change = []
            self.change_after = 0
            self.min_departed_vehicles = 10 if "5" in self.sumoCmd[2] else np.random.randint(25, 80, 1).item()
            self.time_to_change_des_speed = np.random.randint(100, 250)
            # Running simulation until ego can be inserted
            while self.egoID is None:
                self.one_step()
            # Getting initial environment state
            self.environment_state = self.get_surroundings_env()
            # Setting a starting speed of the ego
            self.state['velocity'] = self.desired_speed
            return self.environment_state
        else:
            raise RuntimeError('Please run render before reset!')

    def set_random_desired_speed(self):
        self.desired_speed = random.randint(130, 160) / 3.6

    def calculate_action(self, action):
        """
        This function is used to select the actions for steering and velocity change.
        Parameters
        ----------
        action an int between 0 and 9

        Returns a steering and velocity change action
        -------

        """
        st = [-1, 0, 1]  # [right, nothing, left] lane change
        ac = [-0.7, 0.0, 0.3]  # are in m/s
        if isinstance(action, np.ndarray) and len(action.shape) > 1:
            steer = action[:, 0]
            acc = action[:, 1]
        elif isinstance(action, np.ndarray):
            steer = action[0]
            acc = action[1]
        else:  # isinstance(action, int):
            steer = st[action // len(st)]
            acc = ac[action % len(st)]
        if steer > 0.35:
            steer = 1
        elif steer < -0.35:
            steer = -1
        else:
            steer = 0
        ctrl = [steer, acc]
        return ctrl

    def step(self, action):
        #        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        new_x, last_x = 0, 0  # for reward calculation
        # Collecting the ids of online vehicles
        IDsOfVehicles = traci.vehicle.getIDList()

        reward = 1
        # Checking if ego is still alive
        if self.egoID in IDsOfVehicles:
            # Selecting action to do
            ctrl = self.calculate_action(action)
            # Setting vehicle speed according to selected action
            traci.vehicle.setSpeed(self.egoID,
                                   min(max(self.state['velocity'] + ctrl[1], 0), 50))  # todo hardcoded max speed
            if self.steps_done % self.time_to_change_des_speed == 0:
                self.set_random_desired_speed()
            self.wants_to_change.append(ctrl[0])  # Collecting change attempts
            # Checking if change attempts are enough to change lane
            if sum(self.wants_to_change) > self.change_after or sum(self.wants_to_change) < -self.change_after:
                self.lanechange_counter += 1  # Storing successful lane change
                last_lane = traci.vehicle.getLaneID(self.egoID)[:-1]
                lane_new = int(traci.vehicle.getLaneID(self.egoID)[-1]) + ctrl[0]
                # Checking if new lane is still on the road
                if lane_new not in [0, 1, 2]:
                    reward = self.max_punishment
                    new_x = \
                        traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                    return self.environment_state, reward, True, {'cause': 'Left Highway',
                                                                  'rewards': reward,
                                                                  'velocity': self.state['velocity'],
                                                                  'distance': new_x - self.ego_start_position,
                                                                  'lane_change': self.lanechange_counter}
                else:
                    self.wants_to_change = []
                    lane_new = last_lane + str(lane_new)
                    x = traci.vehicle.getLanePosition(self.egoID)
                    done = False
                    while not done:
                        try:
                            traci.vehicle.moveTo(self.egoID, lane_new, x)
                        except traci.exceptions.TraCIException:
                            x += 0.1
                        else:
                            done = True
            # Removing elements of lane change attempts to have always the same lenght
            if len(self.wants_to_change) > self.change_after:
                self.wants_to_change.pop(0)

            if self.egoID is not None:
                last_x = traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                is_ok, cause = self.one_step()
                if is_ok:
                    self.get_surroundings_env()
                    new_x = traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                else:
                    new_x = last_x + max(self.state['velocity'] + ctrl[1], 0) * self.dt
        else:
            is_ok = False
            cause = None

        # Setting ego to the middle of the screen if rendering is "human"
        if self.egoID is not None and self.rendering and is_ok:
            egoPos = traci.vehicle.getPosition(self.egoID)
            traci.gui.setOffset('View #0', egoPos[0], egoPos[1])

        terminated = not is_ok
        if terminated and cause is not None:
            # case for some bad event with termination
            reward = self.max_punishment
        elif not terminated:
            # Case for successful step
            if self.reward_type == 'simple':
                reward = reward - (abs(self.state['velocity'] - self.desired_speed)) / self.desired_speed
            self.steps_done += 1
        else:
            # Case for completing the highway without a problem
            reward = -1 * self.max_punishment

        return self.environment_state, reward, terminated, {'cause': cause, 'rewards': reward,
                                                            'velocity': self.state['velocity'],
                                                            'distance': new_x - self.ego_start_position,
                                                            'lane_change': self.lanechange_counter}

    def choose_random_simulation(self):
        self.rand_index = np.random.choice(np.arange(0, 6), p=[0.20, 0.15, 0.20, 0.20, 0.20, 0.05])
        # self.rand_index = 0
        # print(f"Simulation {self.rand_index} loaded.")
        if "jatek" in self.sumoCmd[2]:
            self.sumoCmd[2] = f"../envs/sim_conf/jatek_{self.rand_index}.sumocfg"
        elif "no_gui" in self.sumoCmd[2]:
            self.sumoCmd[2] = f"../envs/sim_conf/no_gui_{self.rand_index}.sumocfg"

    def render(self, mode='human'):
        """
        This function lets us choose between simulation with or without GUI
        Parameters
        ----------
        mode: "human" means rendering with GUI
        -------
        """
        if mode == 'human':
            self.rendering = True
        else:
            self.rendering = False

        if "Windows" in platform.system():
            # Case for windows execution
            if self.rendering:
                self.sumoBinary = "C:/Sumo/bin/sumo-gui"
                self.sumoCmd = [self.sumoBinary, "-c", "../envs/sim_conf/jatek.sumocfg", "--start", "--quit-on-end",
                                "--collision.mingap-factor", "2", "--collision.action", "remove", "--no-warnings", "1",
                                "--random"]
            else:
                self.sumoBinary = "C:/Sumo/bin/sumo"
                self.sumoCmd = [self.sumoBinary, "-c", "../envs/sim_conf/no_gui.sumocfg", "--start", "--quit-on-end",
                                "--collision.mingap-factor", "2", "--collision.action", "remove", "--no-warnings", "1",
                                "--random"]
        else:
            # Case for linux execution
            if self.rendering:
                self.sumoBinary = "/usr/share/sumo/bin/sumo-gui"
                self.sumoCmd = [self.sumoBinary, "-c", "../envs/sim_conf/jatek.sumocfg", "--start", "--quit-on-end",
                                "--collision.mingap-factor", "2", "--collision.action", "remove", "--no-warnings", "1",
                                "--random"]
            else:
                self.sumoBinary = "/usr/share/sumo/bin/sumo"
                self.sumoCmd = [self.sumoBinary, "-c", "../envs/sim_conf/no_gui.sumocfg", "--start", "--quit-on-end",
                                "--collision.mingap-factor", "2", "--collision.action", "remove", "--no-warnings", "1",
                                "--random"]

        traci.start(self.sumoCmd[:4])

    def get_surroundings_env(self):
        """
        This is used to call environment surroundings
        Returns: environment state in the shape of [1, range_x, range_y, 3]
        where the last dimension is the channels of speed, lane_id, and desired speed
        -------

        """
        self.environment_state, self.state = self.calculate_environment()
        return self.environment_state

    def one_step(self):
        """
        This function is used to step the SUMO.
        Returns
        -------

        """
        terminated = False
        try:
            w = traci.simulationStep()
        except traci.exceptions.FatalTraCIError:
            self.stop()
            raise RuntimeError

        # Collecting online vehicles
        IDsOfVehicles = traci.vehicle.getIDList()
        # Moving forward if ego can be inserted
        if len(IDsOfVehicles) > self.min_departed_vehicles and self.egoID is None:
            # Finding the last car on the highway
            for carID in IDsOfVehicles:
                if traci.vehicle.getPosition(carID)[0] < self.ego_start_position and \
                        traci.vehicle.getSpeed(carID) > (60 / 3.6):  # and "0" in traci.vehicle.getLaneID(carID):
                    # Saving ID and start position for ego vehicle
                    self.egoID = carID
                    self.ego_start_position = traci.vehicle.getPosition(self.egoID)[0]
            if self.egoID is None:
                self.egoID = IDsOfVehicles[0]
                self.ego_start_position = traci.vehicle.getPosition(self.egoID)[0]
            # Setting ego simulation variables to be controlled by us (not SUMO)
            lanes = [-2, -1, 0, 1, 2]
            traci.vehicle.setLaneChangeMode(self.egoID, 0x0)
            traci.vehicle.setSpeedMode(self.egoID, 0x0)
            traci.vehicle.setColor(self.egoID, (255, 0, 0))
            traci.vehicle.setType(self.egoID, 'ego')
            traci.vehicle.setMinGap(self.egoID, 0)
            traci.vehicle.setMinGapLat(self.egoID, 0)

            traci.vehicle.setSpeedFactor(self.egoID, 2)
            traci.vehicle.setSpeed(self.egoID, self.desired_speed)
            traci.vehicle.setMaxSpeed(self.egoID, 50)
            traci.vehicle.subscribeContext(self.egoID, tc.CMD_GET_VEHICLE_VARIABLE, 0.0,
                                           [tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION,
                                            tc.VAR_LENGTH, tc.VAR_WIDTH])
            traci.vehicle.addSubscriptionFilterLanes(lanes, noOpposite=True, downstreamDist=100.0, upstreamDist=100.0)
            # Since it is when we start the simulation it returns True for the reset function.
            return True, None

        cause = None
        # Checking abnormal cases for ego (if events happened which terminate the simulation)
        if self.egoID is not None:
            if self.egoID in traci.simulation.getCollidingVehiclesIDList():
                cause = "Collision"
            elif self.egoID in traci.vehicle.getIDList() and traci.vehicle.getSpeed(self.egoID) < (50 / 3.6):
                cause = 'Too Slow'
            elif self.egoID in traci.simulation.getArrivedIDList():
                # Case for finished route
                cause = None
                self.egoID = None
                terminated = True
            else:
                # Case for running simulation (no events yet)
                cause = None
            if cause is not None:
                terminated = True
                self.egoID = None
        return (not terminated), cause

    def calculate_environment(self):
        """

        Returns: environment state in the shape of [1, range_x, range_y, 3]
        where the last dimension is the channels of speed, lane_id, and desired speed
        and ego state as a dict: {'x_position', 'y_position', 'length', 'width', 'velocity', 'lane_id', 'heading'}
        -------
        """
        # Getting cars around ego vehicle
        cars_around = traci.vehicle.getContextSubscriptionResults(self.egoID)
        ego_state = {}
        # Collecting car details
        environment_collection = []
        for car_id, car in cars_around.items():
            car_state = {'x_position': car[tc.VAR_POSITION][0] - car[tc.VAR_LENGTH] / 2,
                         'y_position': car[tc.VAR_POSITION][1],
                         'length': car[tc.VAR_LENGTH],
                         'width': car[tc.VAR_WIDTH],
                         'velocity': car[tc.VAR_SPEED],
                         'lane_id': car[tc.VAR_LANE_INDEX],
                         'heading': car[tc.VAR_ANGLE]}
            # Saving ego state
            if car_id == self.egoID:
                ego_state = copy.copy(car_state)
            environment_collection.append(copy.copy(car_state))

        # Creating state representation as a matrix (image)
        state_matrix = np.zeros((2 * x_range_grid, 2 * y_range_grid, 3))
        # Drawing the image channels with actual data
        for element in environment_collection:
            dx = int(np.rint((element['x_position'] - ego_state["x_position"]) * grid_per_meter))
            dy = int(np.rint((ego_state["y_position"] - element['y_position']) * grid_per_meter))
            l = int(np.ceil(element['length'] / 2 * grid_per_meter))
            w = int(np.ceil(element['width'] / 2 * grid_per_meter))

            # Only if car is in the range
            if (abs(dx) < (x_range_grid - element['length'] / 2 * grid_per_meter)) and \
                    abs(dy) < (y_range_grid - element['width'] / 2 * grid_per_meter):

                # Drawing speed of the current car
                state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                y_range_grid + dy - w:y_range_grid + dy + w, 0] += np.ones_like(
                    state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                    y_range_grid + dy - w:y_range_grid + dy + w, 0]) * element['velocity'] / 50

                # Drawing lane of the current car
                state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                y_range_grid + dy - w:y_range_grid + dy + w, 1] += np.ones_like(
                    state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                    y_range_grid + dy - w:y_range_grid + dy + w, 1]) * element['lane_id'] / 2

                # If ego, drawing the desired speed
                if math.isclose(dx, 0) and math.isclose(dy, 0):
                    state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                    y_range_grid + dy - w:y_range_grid + dy + w,
                    2] += np.ones_like(
                        state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                        y_range_grid + dy - w:y_range_grid + dy + w,
                        2]) * self.desired_speed / 50
        filename = os.path.join(os.path.curdir, "scenarios", f"{self.steps_done}.jpg")
        plt.imsave(filename, state_matrix)
        return state_matrix, ego_state
