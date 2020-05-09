"""
Script created by szoke.laszlo95@gmail.com and bencepalotas@gmail.com
as a project of TRAFFIC MODELLING, SIMULATION AND CONTROL subject
"""

import copy
import os
import platform
import random

import gym
import numpy as np
import traci
import traci.constants as tc
from gym import spaces

from model import LateralModel


class SUMOEnvironment(gym.Env):
    """

    """

    def __init__(self,
                 simulation_directory='D:\\msc\\forth\\traffic\\continuousSUMOv1\\sim_conf',
                 type_os="image",
                 type_as="discrete",
                 reward_type='speed',
                 mode='human',
                 change_speed_interval=100):

        # Basic gym environment variables
        # Type action space
        self.type_as = type_as

        # Type observation
        self.type_os = type_os
        self.setup_observation_space()
        self.setup_action_space()
        self.setup_reward_system(reward_type=reward_type)
        self._max_episode_steps = 2500
        self.rendering = True if mode == 'human' else False

        # Simulation data and constants
        self.sumoBinary = None
        self.sumoCmd = None
        self.simulation_list = self.get_possible_simulations(simulation_directory)
        self.min_departed_vehicles = 3

        # variable for desired speed random change (after x time steps)
        self.time_to_change_des_speed = change_speed_interval

        self.start()
        self.reset()

    def setup_observation_space(self):
        """
        This function is responsible for creating the desired observation space.
        :type: describes what the observation space will be
        """
        if self.type_os == "image":
            self.grid_per_meter = 4  # Defines the precision of the returned image
            x_range = 50  # symmetrically for front and back
            self.x_range_grid = x_range * self.grid_per_meter  # symmetrically for front and back
            y_range = 9  # symmetrically for left and right
            self.y_range_grid = y_range * self.grid_per_meter  # symmetrically for left and right
            self.observation_space = np.zeros((2 * self.x_range_grid, 2 * self.y_range_grid, 3))
            # Assigning the environment call
            self.get_environment = self.calculate_image_environment

        # elif type_os == "structured": todo: create discrete observation space
        else:
            raise RuntimeError("This type of observation space is not yet implemented.")

    def setup_action_space(self):
        """
        This function is responsible for creating the desired action space.
        :number_of_actions: describes how many actions can the agent take.
        """
        if self.type_as == "continuous":
            # todo: hardcoded range of actions, first is steer 2. is acceleration command
            # todo: radian or degree???
            low = np.array([-1, -1])
            high = np.array([1, 1])
            self.action_space = spaces.Box(low, high, dtype=np.float)
            self.calculate_action = self.calculate_continuous_action

        elif self.type_as == "discrete":
            # todo: hardcoded steering and speed
            self.steering_constant = [-1, 0, 1]  # [right, nothing, left] lane change
            self.accel_constant = [-0.7, 0.0, 0.3]  # are in m/s
            self.action_space = spaces.Discrete(9)
            self.calculate_action = self.calculate_discrete_action
        else:
            raise RuntimeError("This type of action space is not yet implemented.")

    def setup_reward_system(self, reward_type='basic'):
        """
        This should set how the different events are handled.
        :reward_type: selects the reward system
        """
        # Bool shows if the event terminates, value shows how much it costs.
        if reward_type == "basic":
            self.reward_dict = {'collision': [True, 0],
                                'slow': [True, 0],
                                'left_highway': [True, 0],
                                'immediate': [False, 1],
                                'success': [True, 0],
                                'type': reward_type}
        elif reward_type == 'speed':
            self.reward_dict = {'collision': [True, 0],
                                'slow': [True, 0],
                                'left_highway': [True, 0],
                                'immediate': [False, 1],
                                'success': [True, 1],
                                'type': reward_type}
        else:
            raise RuntimeError("Reward system can not be found")

    def calculate_immediate_reward(self):
        """
        In this function the possible immediate reward calculation models are implemented.
        :return:
        """
        if self.reward_dict["type"] == "basic":
            reward = self.reward_dict['immediate'][1]
        elif self.reward_dict["type"] == 'speed':
            reward = self.reward_dict['immediate'][1] - (abs(self.state['velocity'] - self.desired_speed)) \
                     / self.desired_speed
        else:
            raise RuntimeError('Reward type is not implemented')

        if self.steps_done % self.time_to_change_des_speed == 0:
            self.set_random_desired_speed()
        return reward

    # noinspection PyAttributeOutsideInit
    def setup_basic_environment_values(self):
        """
        This is dedicated to reset the environment basic variables.
        todo: Check all necessary reset when the model is ready
        :return: None
        """
        # setting basic environment variables
        # Loading variables with real values from traci
        # todo: hardcoded places of the highway.. this should be handled from code.
        self.lane_width = traci.lane.getWidth('A_0')
        self.lane_offset = traci.junction.getPosition('J1')[1] - 2 * self.lane_width - self.lane_width / 2
        self.dt = traci.simulation.getDeltaT()
        self.egoID = None  # Resetting chosen ego vehicle id
        self.steps_done = 0  # resetting steps done
        self.desired_speed = random.randint(120, 160) / 3.6
        self.state = None
        self.lane_change_counter = 0
        self.time_to_change_des_speed = np.random.randint(100, 250)
        self.ego_start_position = 10000  # used for result display
        self.lanechange_counter = 0
        self.lateral_model = None

    def stop(self):
        """
        Stops the simulation.
        :return: None
        """
        traci.close()

    def start(self):
        """
        This function starts the SUMO connection and loads an initial simulation.
        :return: None
        """
        if "Windows" in platform.system():
            # Case for windows execution
            self.sumoBinary = "C:/Sumo/bin/sumo-gui" if self.rendering else "C:/Sumo/bin/sumo"
        else:
            # Case for linux execution
            self.sumoBinary = "/usr/share/sumo/bin/sumo-gui" if self.rendering else "/usr/share/sumo/bin/sumo"

        self.sumoCmd = [self.sumoBinary, "-c", self.simulation_list[0], "--start", "--quit-on-end",
                        "--collision.mingap-factor", "2", "--collision.action", "remove", "--no-warnings", "1",
                        "--random"]

        traci.start(self.sumoCmd[:4])

    def reset(self):
        """
        Resets the environment to initial state.
        :return: The initial state
        """
        # Changing configuration
        self.choose_random_simulation()
        # Loads traci configuration
        traci.load(self.sumoCmd[1:])
        # Resetting configuration
        self.setup_basic_environment_values()
        # Running simulation until ego can be inserted
        self.select_egos()
        # Getting initial environment state
        self.refresh_environment()
        # Init lateral model
        # TODO: Remove else when lateral model is ready
        if self.type_as == 'continuous':
            self.lateral_model = LateralModel(
                x_position=self.state['x_position'],
                y_position=self.state['y_position'],
                # TODO: can be refactored to this if self.state['velocity'] is initiated with self.desired_speed earlier
                # velocity=self.state['velocity'],
                velocity=self.desired_speed,
                heading=self.state['heading'],
                lane_id=self.state['lane_id'],
                lane_width=self.lane_width
            )
        else:
            self.lateral_model = self.state

        # Setting a starting speed of the ego
        self.state['velocity'] = self.desired_speed
        return self.environment_state

    def render(self, mode='human'):
        """
        Basic function of the OpenAI gym, this does not require anything
        :param mode:
        """
        pass

    def step(self, action):
        """

        :param action: int
        :return:
        """
        # Collecting the ids of online vehicles
        IDsOfVehicles = traci.vehicle.getIDList()

        # Checking if ego is still alive
        if self.egoID in IDsOfVehicles:
            # Selecting action to do
            ctrl = self.calculate_action(action)

            # Checking if current lane is the same with previous
            # also itt hívjuk ezeket az akiókat is át kell adni a laterálnak
            if self.type_as == 'continuous':
                # TODO: pass steering_angle and velocity to lateral model
                steering_angle = random.uniform(-0.5, 0.5)
                self.lateral_state = self.lateral_model.step(dt=self.dt,
                                                             steering_angle=steering_angle,
                                                             velocity=self.state['velocity'],
                                                             )
            else:
                # self.lateral_model.step(ctrl)  # gives None for lane if it left the highway
                self.lateral_state = copy.deepcopy(self.state)
            self.lane_ID = traci.vehicle.getLaneID(self.egoID)
            self.lane_width = traci.lane.getWidth(self.lane_ID)
            last_lane = self.lane_ID[:-1]
            last_lane_idx = traci.vehicle.getLaneID(self.egoID)[-1]

            # Setting vehicle speed according to selected action
            traci.vehicle.setSpeed(self.egoID, self.lateral_state['velocity'] + ctrl[1])
            new_lane = int(int(last_lane_idx) + ctrl[0])
            # TODO: Remove else when lateral model is ready
            # TODO: consider replacing condition with a more parametrized form for other road structures
            if self.type_as == 'continuous':
                if self.lateral_model.lateral_state['lane_id'] in [0, 1, 2]:
                    self.lateral_state['lane_id'] = self.lateral_model.lateral_state['lane_id']
                else:
                    self.lateral_state['lane_id'] = None
                # TODO: if none reset as vehicle potentially left the road?
            else:
                self.lateral_state['lane_id'] = new_lane if new_lane in [0, 1, 2] else None

            # Potentially update lane
            if self.lateral_state['lane_id'] != self.state['lane_id']:
                # Checking if new lane is still on the road
                if self.lateral_state['lane_id'] is None:
                    reward = self.reward_dict['left_highway'][1]
                    new_x = traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                    self.egoID = None
                    return self.environment_state, reward, True, {'cause': 'left_highway',
                                                                  'rewards': reward,
                                                                  'velocity': self.state['velocity'],
                                                                  'distance': new_x - self.ego_start_position,
                                                                  'lane_change': self.lanechange_counter}

                else:
                    self.lanechange_counter += 1  # Storing successful lane change
                    lane_new = last_lane + str(self.lateral_state['lane_id'])
                    x = traci.vehicle.getLanePosition(self.egoID)  # todo: use self.lateral_state['x_position']
                    # todo: try out this placing method
                    # lane = traci.vehicle.getLaneID(self.egoID)
                    # edgeID = traci.lane.getEdgeID(lane)
                    # traci.vehicle.moveToXY(self.egoID, edgeID, lane, self.lateral_state['x_position'],
                    # self.lateral_state['y_position'], angle=tc.INVALID_DOUBLE_VALUE, keepRoute=1)
                    # Perform lane change
                    traci.vehicle.moveTo(self.egoID, lane_new, x)
                    # sync new state with lateral state
                    self.state['lane_id'] = copy.deepcopy(self.lateral_state['lane_id'])

            # TODO: ezt meg kéne máshogy oldani ne legyen runtime error
            terminated = False
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                self.stop()
                raise RuntimeError

            # Checking abnormal cases for ego (if events happened which terminate the simulation)
            if self.egoID in traci.simulation.getCollidingVehiclesIDList():
                cause = "collision" if self.reward_dict['collision'][0] else None
                reward = self.reward_dict[cause][1]
                terminated = True
                self.egoID = None

            elif self.egoID in traci.vehicle.getIDList() and traci.vehicle.getSpeed(self.egoID) < (50 / 3.6):
                cause = 'slow' if self.reward_dict['slow'][0] else None
                reward = self.reward_dict[cause][1]
                terminated = True
                self.egoID = None

            elif self.egoID in traci.simulation.getArrivedIDList():
                # Case for completing the highway without a problem
                cause = None
                reward = self.reward_dict['success'][1]
                self.egoID = None
                terminated = True
            else:
                # Case for successful step
                cause = None
                reward = self.calculate_immediate_reward()
                terminated = False
                self.steps_done += 1
                self.refresh_environment()

            new_x = self.lateral_state['x_position']

            return self.environment_state, reward, terminated, {'cause': cause, 'rewards': reward,
                                                                'velocity': self.state['velocity'],
                                                                'distance': new_x - self.ego_start_position,
                                                                'lane_change': self.lanechange_counter}
        else:
            raise RuntimeError('After terminated episode, reset is needed. '
                               'Please run env.reset() before starting a new episode.')

    def select_egos(self, number_of_egos=1):
        """
        This selects the ego(s) from the environment. The ID of the cars are given back to the simulation
        :param number_of_egos: shows how many car to select from the simulation
        :return: ID list of ego(s)
        """
        while self.egoID is None:
            traci.simulationStep()
            # Collecting online vehicles
            IDsOfVehicles = traci.vehicle.getIDList()
            # Moving forward if ego can be inserted
            if traci.simulation.getArrivedNumber() - 2 * traci.simulation.getCollidingVehiclesNumber() > 0 \
                    and len(IDsOfVehicles) > self.min_departed_vehicles and self.egoID is None:
                # Finding the last car on the highway
                for carID in IDsOfVehicles:
                    # todo: this search only works in the straight simulation
                    if traci.vehicle.getPosition(carID)[0] < self.ego_start_position and \
                            traci.vehicle.getSpeed(carID) > (60 / 3.6):
                        # Saving ID and start position for ego vehicle
                        self.egoID = carID
                        self.ego_start_position = traci.vehicle.getPosition(self.egoID)[0]
                if self.egoID is None:
                    self.egoID = IDsOfVehicles[-1]
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
                traci.vehicle.addSubscriptionFilterLanes(lanes, noOpposite=True, downstreamDist=100.0,
                                                         upstreamDist=100.0)
                if self.rendering:
                    traci.gui.trackVehicle('View #0', self.egoID)

    def choose_random_simulation(self):
        """
        This chooses a new simulation randomly, so it will load a different one every time.
        :return: None
        """
        self.sumoCmd[2] = np.random.choice(self.simulation_list)

    def get_possible_simulations(self, simulation_directory):
        """
        From given directory gets the simulation files and creates the file paths.
        if rendering, simulations without 'no' will be loaded, otherwise them.
        :param simulation_directory: Absolute path to the simulation folder with ".sumocfg"
        :return: list of valid simulation paths
        """
        simulation_list = os.listdir(simulation_directory)
        for item in list(simulation_list):
            if not item.endswith('.sumocfg'):
                simulation_list.remove(item)
            else:
                if item.__contains__('no'):
                    if self.rendering:
                        simulation_list.remove(item)
                else:
                    if not self.rendering:
                        simulation_list.remove(item)
        return [os.path.join(simulation_directory, item) for item in simulation_list]

    def set_random_desired_speed(self):
        """
        Function to set random speed of ego(s)
        """
        # TODO: make this work for more ego
        self.desired_speed = random.randint(130, 160) / 3.6

    def calculate_discrete_action(self, action):
        """
        This is used to convert int action into steer and acceleration commands
        :param action: Int
        :return: [steering, acceleration] values
        """
        steer = self.steering_constant[action // len(self.steering_constant)]
        acc = self.accel_constant[action % len(self.accel_constant)]
        return [steer, acc]

    def calculate_continuous_action(self, action):
        """
        Calculate continuous action with or without batch
        :param action: actions to select [batch x [steering, acceleration]] or [[steering, acceleration]]
        :return: list of selected actions
        """
        if isinstance(action, np.ndarray) and len(action.shape) > 1:
            # this is for batch actions
            steer = action[:, 0]
            acc = action[:, 1]
            # todo: the disretization here is not solved. Find NICE solution
        else:
            # this is for single actions
            steer = action[0]
            acc = action[1]
            # todo: currently it is here because of the lack of the lateral model
            # It is a basic discretisation
            if steer > 0.35:
                steer = 1
            elif steer < -0.35:
                steer = -1
            else:
                steer = 0
        return [steer, acc]

    def calculate_image_environment(self, environment_collection):
        """
        :param environment_collection:
        :return environment_state: [1, range_x, range_y, 3]
        where the last dimension is the channels of speed, lane_id, and desired speed
        :return ego_state: dict of {'x_position', 'y_position', 'length', 'width', 'velocity', 'lane_id', 'heading'}
        """
        ego_state = environment_collection[self.egoID]
        # Creating state representation as a matrix (image)
        environment_state = np.zeros((2 * self.x_range_grid, 2 * self.y_range_grid, 3))
        # Drawing the image channels with actual data
        for car_id in environment_collection.keys():
            dx = int(np.rint((environment_collection[car_id]['x_position'] - ego_state[
                "x_position"]) * self.grid_per_meter))
            dy = int(np.rint((ego_state["y_position"] - environment_collection[car_id][
                'y_position']) * self.grid_per_meter))
            l = int(np.ceil(environment_collection[car_id]['length'] / 2 * self.grid_per_meter))
            w = int(np.ceil(environment_collection[car_id]['width'] / 2 * self.grid_per_meter))

            # Only if car is in the range
            if (abs(dx) < (self.x_range_grid - environment_collection[car_id]['length'] / 2 * self.grid_per_meter)) \
                    and abs(dy) < (
                    self.y_range_grid - environment_collection[car_id]['width'] / 2 * self.grid_per_meter):

                # Drawing speed of the current car
                environment_state[self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                self.y_range_grid + dy - w:self.y_range_grid + dy + w, 0] += np.ones_like(
                    environment_state[self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w, 0]) * environment_collection[car_id][
                                                                                 'velocity'] / 50

                # Drawing lane of the current car
                environment_state[self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                self.y_range_grid + dy - w:self.y_range_grid + dy + w, 1] += np.ones_like(
                    environment_state[self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w, 1]) * environment_collection[car_id][
                                                                                 'lane_id'] / 2

                # If ego, drawing the desired speed
                if car_id == self.egoID:
                    environment_state[self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w,
                    2] += np.ones_like(
                        environment_state[self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                        self.y_range_grid + dy - w:self.y_range_grid + dy + w,
                        2]) * self.desired_speed / 50
        # filename = os.path.join(os.path.curdir, "scenarios", f"{self.steps_done}.jpg")
        # plt.imsave(filename, environment_state)
        return environment_state, ego_state

    def get_simulation_environment(self):
        """
        Function for getting the cars and their attributes from SUMO.
        :return: A car_id dict with {'x_position', 'y_position', 'length', 'width', 'velocity', 'lane_id', 'heading'}
        """
        # Getting cars around ego vehicle
        cars_around = traci.vehicle.getContextSubscriptionResults(self.egoID)
        # Collecting car details
        environment_collection = {}
        for car_id, car in cars_around.items():
            car_state = {'x_position': car[tc.VAR_POSITION][0] - car[tc.VAR_LENGTH] / 2,
                         'y_position': car[tc.VAR_POSITION][1],
                         'length': car[tc.VAR_LENGTH],
                         'width': car[tc.VAR_WIDTH],
                         'velocity': car[tc.VAR_SPEED],
                         'lane_id': car[tc.VAR_LANE_INDEX],
                         'heading': car[tc.VAR_ANGLE]}
            environment_collection[car_id] = copy.copy(car_state)
        return environment_collection

    def refresh_environment(self):
        """
        This is used to refresh the environment
        Sets environment state in the shape of [1, range_x, range_y, 3]
        and state in {'x_position', 'y_position', 'length', 'width', 'velocity', 'lane_id', 'heading'}
        where the last dimension is the channels of speed, lane_id, and desired speed
        -------

        """

        environment_collection = self.get_simulation_environment()
        # todo: here will be the lateral calculation for state i guess
        self.environment_state, self.state = self.get_environment(environment_collection)
        self.lateral_state = copy.deepcopy(self.state)  ## todo: until the lateral is not working I use this

    def calculate_structured_environment(self, cars_around):
        """
        This fuction is deprecated, in the futute the implementation should be rethought based on the lateral control.
        :return:
        """
        ego_state = cars_around[self.egoID]
        environment_state = {}
        basic_vals = {'dx': 200, 'dv': 0}
        basic_keys = ['FL', 'FE', 'FR', 'RL', 'RE', 'RR', 'EL', 'ER']
        for state_key in basic_keys:
            if state_key in ['RL', 'RE', 'RR']:
                environment_state[state_key] = copy.copy(basic_vals)
                environment_state[state_key]['dv'] = 0
                environment_state[state_key]['dx'] = -200
            elif state_key in ['EL', 'ER']:
                environment_state[state_key] = 0
            else:
                environment_state[state_key] = copy.copy(basic_vals)
        lane = {0: [], 1: [], 2: []}
        ego_data = cars_around[self.egoID]

        for car_id in cars_around.keys():
            if car_id is not self.egoID:
                new_car = dict()
                new_car['dx'] = cars_around[car_id]['x_position'] - cars_around[self.egoID]['x_position']
                new_car['dy'] = abs(cars_around[car_id]["y_position"] - cars_around[self.egoID]['y_position'])
                new_car['dv'] = cars_around[car_id]['velocity'] - cars_around[self.egoID]['velocity']
                new_car['l'] = cars_around[car_id]["length"]
                lane[cars_around[car_id][tc.VAR_LANE_INDEX]].append(new_car)
        [lane[i].sort(key=lambda x: x['dx']) for i in lane.keys()]
        for lane_id in lane.keys():
            if lane_id == ego_data['lane_id']:
                for veh in lane[lane_id]:
                    if veh['dx'] - veh['l'] > 0:
                        if veh['dx'] - veh['l'] < environment_state['FE']['dx']:
                            environment_state['FE']['dx'] = veh['dx'] - veh['l']
                            environment_state['FE']['dv'] = veh['dv']
                    elif veh['dx'] + ego_data["length"] < 0:
                        if veh['dx'] + ego_data["length"] > environment_state['RE']['dx']:
                            environment_state['RE']['dx'] = veh['dx'] + ego_data["length"]
                            environment_state['RE']['dv'] = veh['dv']
            elif lane_id > ego_data['lane_id']:
                for veh in lane[lane_id]:
                    if veh['dx'] - veh['l'] > 0:
                        if veh['dx'] - veh['l'] < environment_state['FL']['dx']:
                            environment_state['FL']['dx'] = veh['dx'] - veh['l']
                            environment_state['FL']['dv'] = veh['dv']
                    elif veh['dx'] + ego_data["length"] < 0:
                        if veh['dx'] + ego_data["length"] > environment_state['RL']['dx']:
                            environment_state['RL']['dx'] = veh['dx'] + ego_data["length"]
                            environment_state['RL']['dv'] = veh['dv']
                    else:
                        environment_state['EL'] = 1

            elif lane_id < ego_data["lane_id"]:
                for veh in lane[lane_id]:
                    if veh['dx'] - veh['l'] > 0:
                        if veh['dx'] - veh['l'] < environment_state['FR']['dx']:
                            environment_state['FR']['dx'] = veh['dx'] - veh['l']
                            environment_state['FR']['dv'] = veh['dv']
                    elif veh['dx'] + ego_data["length"] < 0:
                        if veh['dx'] + ego_data["length"] > environment_state['RR']['dx']:
                            environment_state['RR']['dx'] = veh['dx'] + ego_data["length"]
                            environment_state['RR']['dv'] = veh['dv']
                    else:
                        environment_state['ER'] = 1

        environment_state['speed'] = ego_data[tc.VAR_SPEED]
        environment_state['lane_id'] = ego_data[tc.VAR_LANE_INDEX]  # todo: onehot vector
        # todo: here comes the lateral model , the lane and speed are calculated based on that also
        #  the others like dx dy should be...
        environment_state['des_speed'] = self.desired_speed
        return environment_state, ego_state
