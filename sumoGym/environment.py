"""
Script created by szoke.laszlo95@gmail.com and bencepalotas@gmail.com
as a project of TRAFFIC MODELLING, SIMULATION AND CONTROL subject
"""

import copy
import os
import platform
import random
from time import sleep

import gym
import matplotlib.pyplot as plt
import numpy as np
import traci
import traci.constants as tc
from gym import spaces
from traci import TraCIException, FatalTraCIError

from continuousSUMO.sumoGym.model import LateralModel


def makeContinuousSumoEnv(env_name='SUMOEnvironment-v0',
                          simulation_directory=None,
                          type_os="image",
                          type_as="discrete",
                          reward_type='speed',
                          mode='none',
                          change_speed_interval=100):
    """
    This function creates the gym environment. It is used for initiating the continuousSUMO package.
    """
    if simulation_directory is None:
        file_path = os.path.dirname(os.path.abspath(__file__))
        simulation_directory = os.path.join(os.path.split(file_path)[0], "sim_conf")

    return gym.make(env_name,
                    simulation_directory=simulation_directory,
                    type_os=type_os,
                    type_as=type_as,
                    reward_type=reward_type,
                    mode=mode,
                    change_speed_interval=change_speed_interval)


class SUMOEnvironment(gym.Env):
    """

    """

    def __init__(self,
                 simulation_directory='../sim_conf',
                 type_os="image",
                 type_as="discrete",
                 reward_type='speed',
                 mode='none',
                 radar_range=None,
                 change_speed_interval=100):

        super(SUMOEnvironment, self).__init__()
        np.random.seed(42)
        if radar_range is None:
            radar_range = [50, 9] # x and y
        self.radar_range = radar_range
        # Basic gym environment variables
        # Type action space


        self.type_as = type_as
        # self.dynamic_model = self.setup_dynamic_model(type_as)

        # Type observation
        self.type_os = type_os
        self._setup_observation_space(*radar_range)
        self._setup_action_space()
        self._setup_reward_system(reward_type=reward_type)
        self._max_episode_steps = 2500
        self.rendering = True if mode == 'human' else False

        # Simulation data and constants
        self.sumoBinary = None
        self.sumoCmd = None
        self.simulation_list = self._get_possible_simulations(simulation_directory)
        self.min_departed_vehicles = 3

        # variable for desired speed random change (after x time steps)
        self.time_to_change_des_speed = change_speed_interval

        self.start()
        self.reset()

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
                        # "--lateral-resolution", "0.8",
                        "--collision.mingap-factor", "0", "--collision.action", "warn", "--no-warnings", "1",
                        "--random"
                        ]

        traci.start(self.sumoCmd[:4])

    def reset(self):
        try:
            return self._inner_reset()
        except FatalTraCIError:
            self.stop()
            self.start()
            return self.reset()
        except TraCIException:
            self.stop()
            self.start()
            return self.reset()
        except RuntimeError:
            self.stop()
            self.start()
            return self.reset()

    def _inner_reset(self):
        """
        Resets the environment to initial state.
        :return: The initial state
        """
        # Changing configuration
        self._choose_random_simulation()
        # Loads traci configuration
        traci.load(self.sumoCmd[1:])
        # Resetting configuration
        self._setup_basic_environment_values()
        # Running simulation until ego can be inserted
        self._select_egos()
        environment_collection = self._get_simulation_environment()
        # Getting initial environment state
        self._refresh_environment(environment_collection)
        # Init lateral model
        # Setting a starting speed of the ego
        self.state['velocity'] = self.desired_speed

        if "continuous" in self.type_as:
            self.lateral_model = LateralModel(
                self.state,
                lane_width=self.lane_width,
                dt=self.dt
            )

        return self.observation

    def _setup_observation_space(self, x_range=50, y_range=50):
        """
        This function is responsible for creating the desired observation space.
        :param: x_range: defines the radar range symmetrically for front and back
        :param: y_range: defines the radar range symmetrically for the sides
        """
        if self.type_os == "image":
            self.grid_per_meter = 1  # Defines the precision of the returned image
            self.x_range_grid = x_range * self.grid_per_meter  # symmetrically for front and back
            self.y_range_grid = y_range * self.grid_per_meter  # symmetrically for left and right
            self.observation_space = np.zeros((4, 2 * self.x_range_grid, 2 * self.y_range_grid))
            # Assigning the environment call
            self.get_environment = self._calculate_image_environment

        # elif type_os == "structured": todo: create discrete observation space
        else:
            raise RuntimeError("This type of observation space is not yet implemented.")

    def _setup_action_space(self):
        """
        This function is responsible for creating the desired action space.
        :number_of_actions: describes how many actions can the agent take.
        """
        if self.type_as == "continuous":
            # todo: hardcoded range of actions, first is steer 2. is acceleration command
            low = np.array([-0.1, -1])  # radian, m/s2
            high = np.array([0.1, 1])  # radian, m/s2
            self.action_space = spaces.Box(low, high, dtype=np.float)
            self.calculate_action = self._calculate_continuous_action
            self.model_step = self._continuous_step

        elif self.type_as == "discrete":
            # todo: hardcoded steering and speed
            self.steering_constant = [-1, 0, 1]  # [right, nothing, left] change in radian
            self.accel_constant = [-0.5, 0.0, 0.3]  # are in m/s
            if "longitudinal" in self.type_as:
                self.action_space = spaces.Discrete(3)
                self.calculate_action = self._calculate_discrete_longitudinal_action
            elif "lateral" in self.type_as:
                self.action_space = spaces.Discrete(3)
                self.calculate_action = self._calculate_discrete_lateral_action
            else:
                self.action_space = spaces.Discrete(9)
                self.calculate_action = self._calculate_discrete_action
            self.model_step = self._discrete_step

        else:
            raise RuntimeError("This type of action space is not yet implemented.")

    def _setup_reward_system(self, reward_type='basic'):
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
            self.reward_dict = {'collision': [True, -2],
                                'slow': [True, -1],
                                'left_highway': [True, -0.5],
                                'immediate': [False, 1],
                                'success': [True, 1],
                                'type': reward_type}
        else:
            raise RuntimeError("Reward system can not be found")

    def _calculate_immediate_reward(self):
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
            self._set_random_desired_speed()
        return reward

    # noinspection PyAttributeOutsideInit
    def _setup_basic_environment_values(self):
        """
        This is dedicated to reset the environment basic variables.
        todo: Check all necessary reset when the model is ready
        :return: None
        """
        # setting basic environment variables
        # Loading variables with real values from traci
        # todo: hardcoded places of the highway.. this should be handled from code.
        self.lane_width = traci.lane.getWidth(f"{traci.route.getEdges(traci.route.getIDList()[0])[0]}_0")
        self.end_zone = traci.junction.getPosition(traci.junction.getIDList()[-1])[0]
        self.lane_offset = traci.junction.getPosition(traci.junction.getIDList()[0])[
                               1] - 2 * self.lane_width - self.lane_width / 2
        self.dt = traci.simulation.getDeltaT()
        self.egoID = None  # Resetting chosen ego vehicle id
        self.steps_done = 0  # resetting steps done
        self.desired_speed = random.randint(120, 160) / 3.6
        self.state = None
        self.lane_change_counter = 0
        self.time_to_change_des_speed = np.random.randint(100, 250)
        self.ego_start_position = 10000  # used for result display
        self.lanechange_counter = 0

    def render(self, mode='human'):
        """
        Basic function of the OpenAI gym, this does not require anything
        :param mode:
        """
        pass

    def step(self, action):
        try:
            return self._inner_step(action)
        except TraCIException:
            self.stop()
            self.start()
            return RuntimeError
        except FatalTraCIError:
            self.stop()
            self.start()
            raise RuntimeError

    def _continuous_step(self, action):
        # Selecting action to do
        steering_angle, velocity_dif = self.calculate_action(action)
        lateral_state = self.lateral_model.step(steering_angle=steering_angle,
                                                velocity_dif=velocity_dif)
        lane_ID = traci.vehicle.getLaneID(self.egoID)
        last_lane = lane_ID[:-1]

        # Setting vehicle speed according to selected action
        traci.vehicle.setSpeed(self.egoID, lateral_state['velocity'])
        # Potentially update lane
        # Checking if new lane is still on the road
        if lateral_state['lane_id'] != self.state['lane_id']:
            self.lanechange_counter += 1  # Storing successful lane change
            lane_new = last_lane + str(lateral_state['lane_id'])
            edgeID = traci.lane.getEdgeID(lane_new)
            traci.vehicle.moveToXY(self.egoID, edgeID, lateral_state['lane_id'],
                                   lateral_state['x_position'],
                                   lateral_state['y_position'],
                                   angle=tc.INVALID_DOUBLE_VALUE, keepRoute=1)
        self.state = lateral_state

    def _discrete_step(self, action):
        # Selecting action to do
        steering_angle, velocity_dif = self.calculate_action(action)
        # setting speed for the vehicle
        traci.vehicle.setSpeed(self.egoID, self.state['velocity'] + velocity_dif)
        if steering_angle != 0:
            traci.vehicle.changeLaneRelative(self.egoID, steering_angle, self.dt)
        # prev_lane = traci.vehicle.getLaneID(self.egoID)
        # lane_id = traci.vehicle.getLaneIndex(self.egoID)
        # if steering_angle != 0:
        #     # setting lane for the vehicle
        #     self.lanechange_counter += 1
        #     lane_id +=  steering_angle
        #     # edgeID = traci.lane.getEdgeID(prev_lane)
        #     # traci.vehicle.moveToXY(self.egoID, edgeID, lane_id,
        #     #                    self.state['x_position']+traci.vehicle.getSpeed(self.egoID)*self.dt,
        #     #                    self.state['y_position'],
        #     #                    angle=tc.INVALID_DOUBLE_VALUE, keepRoute=1)
        #     # traci.vehicle.changeTarget(self.egoID, traci.route.getEdges(traci.route.getIDList()[0])[-1])
        #
        # target_lane = prev_lane[:-1]+str(lane_id)
        # traci.vehicle.moveTo(self.egoID, laneID=target_lane, pos=self.state["x_position"])

    def _inner_step(self, action):
        """

        :param action: int
        :return:
        """
        # Collecting the ids of online vehicles
        IDsOfVehicles = traci.vehicle.getIDList()

        # Checking if ego is still alive
        if self.egoID in IDsOfVehicles:
            # try to place the vehicle to the required position
            try:
                self.model_step(action)
            # if the lane can not be selected, the model_step will throw an error
            except TraCIException as exc:
                if "lane" in exc.args[0]:
                    reward = self.reward_dict['left_highway'][1]
                    new_x = \
                        traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                    self.egoID = None
                    return self.observation * 0., reward, True, {'cause': 'left_highway',
                                                                 'rewards': reward,
                                                                 'velocity': self.state['velocity'],
                                                                 'distance': new_x - self.ego_start_position,
                                                                 'lane_change': self.lanechange_counter}
            if self.rendering:
                sleep(0.5)
            traci.simulationStep()
            # getting termination values
            cause, reward, terminated = self._get_terminating_events()

            return self.observation, reward, terminated, {'cause': cause, 'rewards': reward,
                                                          'velocity': self.state['velocity'],
                                                          'distance': self.state['x_position']
                                                                      - self.ego_start_position,
                                                          'lane_change': self.lanechange_counter}
        else:
            raise RuntimeError('After terminated episode, reset is needed. '
                               'Please run env.reset() before starting a new episode.')

    def _get_terminating_events(self):
        reward = 0
        terminated = False

        # Checking abnormal cases for ego (if events happened which terminate the simulation)
        if self.egoID in traci.simulation.getArrivedIDList() and self.state['x_position'] >= self.end_zone-10:
            # Case for completing the highway without a problem
            cause = None
            reward = self.reward_dict['success'][1]
            terminated = True
            self.egoID = None
            self.observation *= 0

        elif self.egoID in traci.simulation.getCollidingVehiclesIDList():# or self._check_collision( environment_collection):
            reward = self.reward_dict['collision'][1]
            cause = "collision"
            terminated = True
            self.egoID = None
            self.observation *= 0

        elif self.egoID in traci.vehicle.getIDList() and traci.vehicle.getSpeed(self.egoID) < (50 / 3.6):
            cause = 'slow' if self.reward_dict['slow'][0] else None
            reward = self.reward_dict[cause][1]
            terminated = True
            self.egoID = None
            self.observation *= 0

        else:
            # Case for successful step
            cause = None
            environment_collection = self._get_simulation_environment()
            if "continuous" in self.type_as:
                environment_collection[self.egoID] = copy.deepcopy(self.state)
            reward += self._calculate_immediate_reward()
            self.steps_done += 1
            self._refresh_environment(environment_collection)

        return cause, reward, terminated

    def _check_collision(self, env):
        """
        This function checks if there are any colliding vehicles with the ego
        """
        ego = env.get(self.egoID, None)
        if ego is None:
            return True
        for idx in env.keys():
            if idx == self.egoID:
                continue
            if abs(env[idx]['x_position'] - ego['x_position']) <= (env[idx]['length'] / 2 + ego['length'] / 2):
                if env[idx]['lane_id'] == ego['lane_id']:
                    return True #, 'front collision'
                elif abs(env[idx]['y_position'] - ego['y_position']) <= (env[idx]['width'] / 2 + ego['width'] / 2):
                    return True #, 'side collision'
        return False #, None

    def _select_egos(self, number_of_egos=1):
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
                traci.vehicle.setLaneChangeMode(self.egoID, 0x0)
                traci.vehicle.setSpeedMode(self.egoID, 0x0)
                traci.vehicle.setColor(self.egoID, (255, 0, 0))
                #traci.vehicle.setType(self.egoID, 'ego')

                traci.vehicle.setRouteID(self.egoID, "r1")

                traci.vehicle.setSpeedFactor(self.egoID, 2)
                traci.vehicle.setSpeed(self.egoID, self.desired_speed)
                traci.vehicle.setMaxSpeed(self.egoID, 50)

                traci.vehicle.subscribeContext(self.egoID,tc.CMD_GET_VEHICLE_VARIABLE, dist=self.radar_range[0],
                                               varIDs=[tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION,
                                                tc.VAR_LENGTH, tc.VAR_WIDTH])
                traci.junction.subscribeContext("C", tc.CMD_GET_VEHICLE_VARIABLE,dist=50)
                if self.rendering:
                    traci.gui.trackVehicle('View #0', self.egoID)
        else:
            traci.simulationStep()

    def _choose_random_simulation(self):
        """
        This chooses a new simulation randomly, so it will load a different one every time.
        :return: None
        """
        self.sumoCmd[2] = np.random.choice(self.simulation_list)

    def _get_possible_simulations(self, simulation_directory):
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
        if len(simulation_list) < 1:
            raise RuntimeError(f"There is no valid simulation in the given folder {simulation_directory}"
                               f"Please set rendering or change the directory.")
        return [os.path.join(simulation_directory, item) for item in simulation_list]

    def _set_random_desired_speed(self):
        """
        Function to set random speed of ego(s)
        """
        # TODO: make this work for more ego
        self.desired_speed = random.randint(130, 160) / 3.6

    def _calculate_discrete_action(self, action):
        """
        This is used to convert int action into steer and acceleration commands
        :param action: Int
        :return: [steering, acceleration] values
        """
        steer = self.steering_constant[action // len(self.steering_constant)]
        acc = self.accel_constant[action % len(self.accel_constant)]
        return [steer, acc]

    def _calculate_discrete_longitudinal_action(self, action):
        """
        This is used to convert int action into acceleration commands
        :param action: Int
        :return: [0, acceleration] values
        """
        acc = self.accel_constant[action]
        return [0.0, acc]

    def _calculate_discrete_lateral_action(self, action):
        """
        This is used to convert int action into lateral commands
        :param action: Int
        :return: [steering, 0] values
        """
        steer = self.steering_constant[action]
        return [steer, 0.0]

    @staticmethod
    def _calculate_continuous_action(action):
        """
        Calculate continuous action with or without batch
        :param action: actions to select [batch x [steering, acceleration]] or [[steering, acceleration]]
        :return: list of selected actions
        """
        if isinstance(action, np.ndarray):
            # this is for single actions
            steer = action[..., 0]
            acc = action[..., 1]
        elif isinstance(action, list):
            steer = action[0]
            acc = action[1]
        else:
            raise ValueError("this type of action is not supported")
        return [steer, acc]

    def _refresh_environment(self, environment_collection):
        """
        This is used to refresh the environment
        Sets environment state in the shape of [1, range_x, range_y, 3]
        and state in {'x_position', 'y_position', 'length', 'width', 'velocity', 'lane_id', 'heading'}
        where the last dimension is the channels of speed, lane_id, and desired speed
        -------

        """
        self.observation, self.state = self.get_environment(environment_collection)

    def _get_simulation_environment(self):
        """
        Function for getting the cars and their attributes from SUMO.
        :return: A car_id dict with {'x_position', 'y_position', 'length', 'width', 'velocity', 'lane_id', 'heading'}
        """
        # Getting cars around ego vehicle
        cars_around = traci.vehicle.getAllContextSubscriptionResults()
        # Collecting car details
        environment_collection = {}
        if len(cars_around):
            for car_id, car in cars_around[self.egoID].items():
                # Move from bumper to  vehicle center
                fi = car[tc.VAR_ANGLE] - 90
                x = car[tc.VAR_POSITION][0] - np.cos(fi) * car[tc.VAR_LENGTH] / 2
                y = car[tc.VAR_POSITION][1] - np.sin(fi) * car[tc.VAR_LENGTH] / 2

                car_state = {'x_position': x,
                             'y_position': y,
                             'length': car[tc.VAR_LENGTH],
                             'width': car[tc.VAR_WIDTH],
                             'velocity': car[tc.VAR_SPEED],
                             'lane_id': car[tc.VAR_LANE_INDEX],
                             'heading': fi}
                environment_collection[car_id] = copy.copy(car_state)
        else:
            raise TraCIException
        return environment_collection

    def _calculate_image_environment(self, env):
        """
        :param env: dict of environmental objects in scene
        :return observation: [4, range_x, range_y]
        where the last dimension is the channels of speed, lane_id, and desired speed
        :return ego_state: dict of {'x_position', 'y_position', 'length', 'width', 'velocity', 'lane_id', 'heading'}
        """
        ego_state = env.get(self.egoID, self.state)

        # Creating state representation as a matrix (image)
        observation = np.zeros((4, 2 * self.x_range_grid, 2 * self.y_range_grid))
        # Drawing the image channels with actual data
        for car_id in env.keys():
            dx = int(np.rint((env[car_id]['x_position'] - ego_state["x_position"]) * self.grid_per_meter))
            dy = int(np.rint((ego_state["y_position"] - env[car_id]['y_position']) * self.grid_per_meter))
            l = int(np.ceil(env[car_id]['length'] / 2 * self.grid_per_meter))
            w = int(np.ceil(env[car_id]['width'] / 2 * self.grid_per_meter))

            # Only if car is in the range
            if (abs(dx) < (self.x_range_grid - env[car_id]['length'] / 2 * self.grid_per_meter)) \
                    and abs(dy) < (
                    self.y_range_grid - env[car_id]['width'] / 2 * self.grid_per_meter):

                # Drawing speed of the current car
                velocity = env[car_id]['velocity'] / 50
                observation[0, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                self.y_range_grid + dy - w:self.y_range_grid + dy + w] += np.ones_like(
                    observation[0, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w]) * velocity

                # Drawing lane of the current car
                lane_id = env[car_id]['lane_id'] / 2
                observation[1, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                self.y_range_grid + dy - w:self.y_range_grid + dy + w] += np.ones_like(
                    observation[1, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w]) * lane_id

                heading = np.radians(env[car_id]["heading"]) / 2 / np.pi
                observation[2, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                self.y_range_grid + dy - w:self.y_range_grid + dy + w] += np.ones_like(
                    observation[2, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w]) * heading

                # If ego, drawing the desired speed
                if car_id == self.egoID:
                    observation[3, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w] += np.ones_like(
                        observation[3, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                        self.y_range_grid + dy - w:self.y_range_grid + dy + w]) * self.desired_speed / 50

        # plt.imshow(observation.transpose((1, 2, 0))[:, :, :3])
        # plt.show()
        return observation, ego_state

    def _calculate_structured_environment(self, cars_around):
        """
        DEPRECATED
        TODO: refactor if needed
        
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
