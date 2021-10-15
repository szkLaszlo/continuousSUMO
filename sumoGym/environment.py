"""
Script created by szoke.laszlo95@gmail.com and bencepalotas@gmail.com
as a project of TRAFFIC MODELLING, SIMULATION AND CONTROL subject
"""

import copy
import os
import platform
import random
from time import sleep

import cv2
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
                          reward_type='all',
                          mode='none',
                          save_log_path=None,
                          change_speed_interval=100,
                          default_w=None,
                          seed=None):
    """
    This function creates the gym environment. It is used for initiating the continuousSUMO package.
    """
    if simulation_directory is None:
        file_path = os.path.dirname(os.path.abspath(__file__))
        simulation_directory = os.path.join(os.path.split(file_path)[0], "basic_env")

    return gym.make(env_name,
                    simulation_directory=simulation_directory,
                    type_os=type_os,
                    type_as=type_as,
                    reward_type=reward_type,
                    mode=mode,
                    save_log_path=save_log_path,
                    change_speed_interval=change_speed_interval,
                    default_w=default_w,
                    seed=seed)


class SUMOEnvironment(gym.Env):
    """

    """

    def __init__(self,
                 simulation_directory='../sim_conf',
                 type_os="image",
                 type_as="discrete",
                 reward_type='all',
                 mode='none',
                 radar_range=None,
                 flatten=True,
                 save_log_path=None,
                 change_speed_interval=100,
                 default_w=None,
                 seed=None):

        super(SUMOEnvironment, self).__init__()
        self.render_mode = mode
        self.save_path = None
        self.name = "SuMoGyM"
        np.random.seed(seed if seed is not None else 42)
        self.seed_ = seed
        if radar_range is None:
            radar_range = [50, 9]  # x and y
        self.radar_range = radar_range
        # Basic gym environment variables
        # Type action space

        self.type_as = type_as
        # self.dynamic_model = self.setup_dynamic_model(type_as)

        # Type observation
        self.type_os = type_os
        self.flatten = flatten
        self._setup_observation_space(*radar_range)
        self._setup_action_space()
        self._setup_reward_system(reward_type=reward_type)
        self.max_num_steps = 2
        self.rendering = True if mode == 'human' else False

        # Simulation data and constants
        self.sumoBinary = None
        self.sumoCmd = None
        self.simulation_list = self._get_possible_simulations(simulation_directory)
        self.min_departed_vehicles = 3
        self.save_log_path = save_log_path if save_log_path is not None else "stdout"
        if self.save_log_path != "stdout" and not os.path.exists(self.save_log_path):
            os.makedirs(self.save_log_path)
        # variable for desired speed random change (after x time steps)
        self.time_to_change_des_speed = change_speed_interval
        self.default_w = np.asarray(default_w) if default_w is not None else np.ones_like(self.get_max_reward(1))

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
                        "--collision.mingap-factor", "0",
                        "--collision.action", "remove",
                        "--no-warnings", "1",
                        ]

        self.sumoCmd.append("--seed")
        self.sumoCmd.append(str(int(np.random.randint(0, 1000000))) if self.seed_ is None else f"{self.seed_}")

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
        if self.seed_ is None:
            self.seed()
        # Loads traci configuration
        traci.load(self.sumoCmd[1:])
        # Resetting configuration
        self._setup_basic_environment_values()
        # Running simulation until ego can be inserted
        self._select_egos()
        self._get_simulation_environment()
        # Getting initial environment state
        self._refresh_environment()
        # Init lateral model
        # Setting a starting speed of the ego
        self.state['speed'] = self.desired_speed

        if "continuous" in self.type_as:
            self.lateral_model = LateralModel(
                self.state,
                lane_width=self.lane_width,
                dt=self.dt
            )

        return self._get_observation()

    def seed(self, seed=None):
        """

        :return:
        """
        index = self.sumoCmd.index("--seed")
        if seed is None:
            self.sumoCmd[index + 1] = str(int(np.random.randint(0, 1000000)))
        else:
            self.sumoCmd[index + 1] = str(int(seed))
        self.seed_ = seed

    def _setup_observation_space(self, x_range=50, y_range=9):
        """
        This function is responsible for creating the desired observation space.
        :param: x_range: defines the radar range symmetrically for front and back
        :param: y_range: defines the radar range symmetrically for the sides
        """
        self.grid_per_meter = 1  # Defines the precision of the returned image
        self.x_range_grid = x_range * self.grid_per_meter  # symmetrically for front and back
        self.y_range_grid = y_range * self.grid_per_meter  # symmetrically for left and right

        if self.type_os == "image":
            self.observation_space = np.zeros((3, 2 * self.y_range_grid, 2 * self.x_range_grid))
            if self.flatten:
                self.observation_space = gym.spaces.Discrete(self.observation_space.flatten().shape[0])
            # Assigning the environment call
            self._get_observation = self._convert_image_state_space_to_vector

        elif self.type_os == "structured":
            self.observation_space = gym.spaces.Discrete(18)
            # Assigning the environment call
            self._get_observation = self._convert_structural_state_space_to_vector

        else:
            raise RuntimeError("This type of observation space is not yet implemented.")

    def _setup_action_space(self):
        """
        This function is responsible for creating the desired action space.
        :number_of_actions: describes how many actions can the agent take.
        """
        if self.type_as == "continuous":
            # todo: hardcoded range of actions, 1. is steer, 2. is acceleration command
            low = np.array([-0.1, -1])  # radian, m/s2
            high = np.array([0.1, 1])  # radian, m/s2
            self.action_space = spaces.Box(low, high, dtype=np.float)
            self.calculate_action = self._calculate_continuous_action
            self.model_step = self._continuous_step

        elif "discrete" in self.type_as:
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

    def _setup_reward_system(self, reward_type='features'):
        """
        This should set how the different events are handled.
        :reward_type: selects the reward system
        """
        # Bool shows if the event terminates, value shows how much it costs.
        if reward_type == "basic":
            raise NotImplementedError
        elif reward_type == 'features':
            self.reward_dict = {'success': [True, 0.0, True],  # if successful episode
                                'collision': [True, -10.0, False],  # when causing collision
                                'slow': [True, -10.0, False],  # when being too slow
                                'left_highway': [True, -10.0, False],  # when leaving highway
                                'speed': [False, 0.0, True],
                                # negative reward proportional to the difference from v_des
                                'lane_change': [False, 0.0, True],  # successful lane-change
                                'keep_right': [False, 0.0, True],  # whenever the available most right lane is used
                                'follow_distance': [False, 0.0, True],
                                # whenever closer than required follow distance,
                                # proportional negative
                                'cut_in_distance': [False, 0.0, True],  # whenever cuts in closer then should.
                                'type': reward_type}

        elif reward_type == 'positive':
            self.reward_dict = {'success': [True, 0.0, True],  # if successful episode
                                'collision': [True, -10.0, False],  # when causing collision
                                'slow': [True, -10.0, False],  # when being too slow
                                'left_highway': [True, -10.0, False],  # when leaving highway
                                'speed': [False, 1.0, True],
                                # negative reward proportional to the difference from v_des
                                'lane_change': [False, 1.0, True],  # successful lane-change
                                'keep_right': [False, 1.0, True],  # whenever the available most right lane is used
                                'follow_distance': [False, 1.0, True],
                                # whenever closer than required follow distance,
                                # proportional negative
                                'cut_in_distance': [False, 1.0, True],  # whenever cuts in closer then should.
                                'type': reward_type}
        else:
            raise RuntimeError("Reward system can not be found")

    # noinspection PyAttributeOutsideInit
    def _setup_basic_environment_values(self):
        """
        This is dedicated to reset the environment basic variables.
        todo: Check all necessary reset when the model is ready
        :return: None
        """
        # setting basic environment variables
        # Loading variables with real values from traci
        self.lane_width = traci.lane.getWidth(f"{traci.route.getEdges(traci.route.getIDList()[0])[0]}_0")
        self.end_zone = traci.junction.getPosition(traci.junction.getIDList()[-1])[0]
        self.lane_offset = traci.junction.getPosition(traci.junction.getIDList()[0])[
                               1] - 2 * self.lane_width - self.lane_width / 2
        self.dt = traci.simulation.getDeltaT()

        self.egoID = None  # Resetting chosen ego vehicle id
        self.steps_done = 0  # resetting steps done
        self.desired_speed = random.randint(110, 140) / 3.6
        self.state = None
        self.observation = None
        self.env_obs = None
        self.lane_change_counter = 0
        self.ego_start_position = 10000  # used for result display
        self.lanechange_counter = 0

    def render(self, mode="human"):

        if self.render_mode == 'human':
            img = self._calculate_image_environment(False)
            img = img.transpose((1, 2, 0))
            if self.save_path is not None:
                dir_name = os.path.split(self.save_path)[0]
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                im = np.zeros((90, 500, 3))
                scale_x = int(np.ceil(im.shape[0] // img.shape[0]))
                scale_y = int(np.ceil(im.shape[1] // img.shape[1]))
                for k in range(0, im.shape[0], scale_x):
                    for j in range(0, im.shape[1], scale_y):
                        value = img[k // scale_x, j // scale_y]

                        im[k:k + scale_x, j:j + scale_y] = value * 255 * np.ones_like(
                            (im[k:k + scale_x, j:j + scale_y]))
                cv2.imwrite(f"{self.save_path}_img{self.steps_done}.jpg", img=im)
            else:
                plt.imshow(img)
                plt.show()

    def set_render(self, mode, save_path=None):
        self.render_mode = mode
        self.save_path = save_path

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
        traci.vehicle.setSpeed(self.egoID, lateral_state['speed'])
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
            return True

        self.state = lateral_state
        return False

    def _discrete_step(self, action):
        # Selecting action to do
        steering_angle, velocity_dif = self.calculate_action(action)
        # setting speed for the vehicle
        traci.vehicle.setSpeed(self.egoID, min(self.state['speed'] + velocity_dif, 50))
        if steering_angle != 0:
            lane = traci.vehicle.getLaneIndex(self.egoID)
            lane += steering_angle
            # traci.vehicle.changeLane(self.egoID, lane, self.dt)
            if lane in [0, 1, 2]:
                traci.vehicle.changeLaneRelative(self.egoID, steering_angle, self.dt)
                self.lanechange_counter += 1
                return True

        return False

    def _inner_step(self, action):
        """
        This is the function for a step in the environment.
        :param action: int
        :return:
        """
        # Collecting the ids of online vehicles
        IDsOfVehicles = traci.vehicle.getIDList()
        # Checking if ego is still alive
        is_lane_change = False
        if self.egoID in IDsOfVehicles:
            # try to place the vehicle to the required position
            try:
                is_lane_change = self.model_step(action)
            # if the lane can not be selected, the model_step will throw an error
            except TraCIException as exc:
                if "lane" in exc.args[0]:
                    cause, reward, terminated = self._get_terminating_events(True, left_=True)
                    self.render()
                    return self._get_observation(), sum(reward), terminated, {'cause': cause,
                                                                              'cumulants': reward,
                                                                              'velocity': self.state['speed'],
                                                                              'distance': self.state['x_position']
                                                                                          - self.ego_start_position,
                                                                              'lane_change': self.lanechange_counter}
            if self.rendering:
                sleep(0.2)

            traci.simulationStep()
            # getting termination values
            cause, reward, terminated = self._get_terminating_events(is_lane_change)

            if self.steps_done % self.time_to_change_des_speed == 0:
                self._set_random_desired_speed()

            # creating the images if render is true.
            self.render()

            return self._get_observation(), sum(reward), terminated, {'cause': cause,
                                                                      'cumulants': reward,
                                                                      'velocity': self.state['speed'],
                                                                      'distance': self.state['x_position']
                                                                                  - self.ego_start_position,
                                                                      'lane_change': self.lanechange_counter}
        else:
            raise RuntimeError('After terminated episode, reset is needed. '
                               'Please run env.reset() before starting a new episode.')

    def _get_terminating_events(self, is_lane_change, left_=False):
        """
        This function returns the reward, terminated status and the couse.
        :param is_lane_change: if lane change, we calculate the rewards differently.
        :param left_: if the ego left the highway.
        :return: reward termination and its cause.
        """
        temp_reward = copy.deepcopy(self.reward_dict)
        terminated = False

        if left_:
            cause = "left_highway"
            temp_reward['success'] = self.reward_dict[cause][1]
            terminated = True
            self.egoID = None
            self.observation = None

        # Checking abnormal cases for ego (if events happened which terminate the simulation)
        elif self.egoID in traci.simulation.getArrivedIDList() and self.state['x_position'] >= self.end_zone - 10:
            # Case for completing the highway without a problem
            cause = None
            temp_reward['success'] = self.reward_dict["success"][1]
            terminated = True
            self.egoID = None
            self.observation = None

        elif self.egoID in traci.simulation.getCollidingVehiclesIDList():  # or self._check_collision( environment_collection):
            cause = "collision"
            temp_reward['success'] = self.reward_dict[cause][1]
            terminated = True
            self.egoID = None
            self.observation = None

        elif self.egoID in traci.vehicle.getIDList() and traci.vehicle.getSpeed(self.egoID) < (60 / 3.6):
            cause = 'slow'
            temp_reward['success'] = self.reward_dict[cause][1]
            terminated = True
            self.egoID = None
            self.observation = None

        else:
            # Case for successful step
            cause = None
            self._get_simulation_environment()
            if "continuous" in self.type_as:
                self.env_obs[self.egoID] = copy.deepcopy(self.state)
            self.steps_done += 1
            self._refresh_environment()

        if temp_reward.get("keep_right", [False, False, False])[2]:
            if self.observation is not None:
                temp_reward["keep_right"] = self.reward_dict["keep_right"][1] if self.observation["lane_id"] == 0 or (
                        self.observation["ER"] == 1 and self.observation["RE"]["dv"] < 1) else \
                    self.reward_dict["keep_right"][1] - 1
            else:
                if cause is not None:
                    temp_reward["keep_right"] = self.reward_dict[cause][1]
                else:
                    temp_reward["keep_right"] = 0

        if temp_reward.get("follow_distance", [False, False, False])[2]:
            if self.observation is not None:
                follow_time = (
                        (self.observation["FE"]["dx"] + self.observation["FE"]["dv"] * self.dt) / self.observation[
                    "speed"] * 2)
                if follow_time < 1:
                    temp_reward["follow_distance"] = max(self.reward_dict["follow_distance"][1] + follow_time - 1,
                                                         self.reward_dict["follow_distance"][1] - 1)
                else:
                    temp_reward["follow_distance"] = self.reward_dict["follow_distance"][1]
            elif cause is None:
                temp_reward["follow_distance"] = self.reward_dict["follow_distance"][1]
            else:
                temp_reward["follow_distance"] = self.reward_dict[cause][1]

        if temp_reward.get("cut_in_distance", [False, False, False])[2]:
            if self.observation is not None:
                follow_time = ((-1 * self.observation["RE"]["dx"] - self.observation["RE"]["dv"] * self.dt) /
                               self.observation["speed"] * 2)
                if follow_time < 0.5:
                    temp_reward["cut_in_distance"] = max(
                        self.reward_dict["cut_in_distance"][1] + 2 * (follow_time - 0.5),
                        self.reward_dict["cut_in_distance"][1] - 1)
                else:
                    temp_reward["cut_in_distance"] = self.reward_dict["cut_in_distance"][1]
            elif cause is None:
                temp_reward["cut_in_distance"] = self.reward_dict["cut_in_distance"][1]
            else:
                temp_reward["cut_in_distance"] = self.reward_dict[cause][1]

        # getting speed reward
        if cause is None:
            dv = abs(self.state['speed'] - self.desired_speed)
            temp_reward["speed"] = self.reward_dict["speed"][1] - dv / max(self.desired_speed, self.state["speed"])
        else:
            temp_reward["speed"] = self.reward_dict[cause][1]
        # getting lane change reward.
        if is_lane_change and cause is None:
            temp_reward["lane_change"] = self.reward_dict["lane_change"][1]
        elif cause is None:
            temp_reward["lane_change"] = self.reward_dict["lane_change"][1] - 1
        else:
            temp_reward["lane_change"] = self.reward_dict[cause][1]
        # constructing the reward vector
        reward = self.get_max_reward(temp_reward) * self.default_w

        return cause, reward, terminated

    def _check_collision(self, env):
        """
        Deprecated.
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
                    return True
                elif abs(env[idx]['y_position'] - ego['y_position']) <= (env[idx]['width'] / 2 + ego['width'] / 2):
                    return True
        return False

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
                            traci.vehicle.getSpeed(carID) > (62 / 3.6):
                        # Saving ID and start position for ego vehicle
                        self.egoID = carID
                        self.ego_start_position = traci.vehicle.getPosition(self.egoID)[0] - traci.vehicle.getLength(
                            self.egoID) / 2
                if self.egoID is None:
                    self.egoID = IDsOfVehicles[-1]
                    self.ego_start_position = traci.vehicle.getPosition(self.egoID)[0] - traci.vehicle.getLength(
                        self.egoID) / 2

                # Setting ego simulation variables to be controlled by us (not SUMO)
                traci.vehicle.setLaneChangeMode(self.egoID, 0x0)
                traci.vehicle.setSpeedMode(self.egoID, 0x0)
                traci.vehicle.setColor(self.egoID, (255, 0, 0))
                # traci.vehicle.setType(self.egoID, 'ego')

                traci.vehicle.setRouteID(self.egoID, "r1")

                traci.vehicle.setSpeedFactor(self.egoID, 2)
                traci.vehicle.setSpeed(self.egoID, self.desired_speed)
                traci.vehicle.setMaxSpeed(self.egoID, 50)

                traci.vehicle.subscribeContext(self.egoID, tc.CMD_GET_VEHICLE_VARIABLE, dist=self.radar_range[0],
                                               varIDs=[tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION,
                                                       tc.VAR_LENGTH, tc.VAR_WIDTH])
                traci.junction.subscribeContext("C", tc.CMD_GET_VEHICLE_VARIABLE, dist=50)
                self.free_ego_surroundings(IDsOfVehicles=IDsOfVehicles)
                if self.rendering:
                    traci.gui.trackVehicle('View #0', self.egoID)
        else:
            traci.simulationStep()

    def free_ego_surroundings(self, IDsOfVehicles):
        # Finding the last car on the highway
        for carID in IDsOfVehicles:
            if carID == self.egoID:
                continue
            distance = np.sqrt((traci.vehicle.getPosition(carID)[0] - traci.vehicle.getPosition(self.egoID)[0]) ** 2 \
                               + (traci.vehicle.getPosition(carID)[1] - traci.vehicle.getPosition(self.egoID)[1]) ** 2)

            if distance < 10:
                traci.vehicle.remove(carID)

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
        self.desired_speed = random.randint(110, 140) / 3.6

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

    def _refresh_environment(self):
        """
        This is used to refresh the environment
        Sets the current observation with respect to the representation.
        and state in {'x_position', 'y_position', 'length', 'width', 'speed', 'lane_id', 'heading'}
        where the last dimension is the channels of speed, lane_id, and desired speed
        -------

        """
        self.observation, self.state = self._calculate_structured_environment()

    def _get_simulation_environment(self):
        """
        Function for getting the cars and their attributes from SUMO.
        It also stores the env observation for visualisation processes.
        :return: A car_id dict with {'x_position', 'y_position', 'length', 'width', 'speed', 'lane_id', 'heading'}
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
                             'speed': car[tc.VAR_SPEED],
                             'lane_id': car[tc.VAR_LANE_INDEX],
                             'heading': fi}
                environment_collection[car_id] = copy.copy(car_state)
        else:
            raise TraCIException("Failed to restart, trying again...")

        self.env_obs = environment_collection

    def _calculate_image_environment(self, flatten=True):
        """
        :return observation: [3, range_x, range_y]
        where the last dimension is the channels of speed, lane_id, heading and desired speed
        """

        ego_state = self.env_obs.get(self.egoID, self.state)

        # Creating state representation as a matrix (image)
        observation = np.zeros((3, 2 * self.x_range_grid, 2 * self.y_range_grid))
        # Drawing the image channels with actual data
        for car_id in self.env_obs.keys():
            dx = int(np.rint((self.env_obs[car_id]['x_position'] - ego_state["x_position"]) * self.grid_per_meter))
            dy = int(np.rint((ego_state["y_position"] - self.env_obs[car_id]['y_position']) * self.grid_per_meter))
            l = int(np.ceil(self.env_obs[car_id]['length'] / 2 * self.grid_per_meter))
            w = int(np.ceil(self.env_obs[car_id]['width'] / 2 * self.grid_per_meter))

            # Only if car is in the range
            if (abs(dx) < (self.x_range_grid - self.env_obs[car_id]['length'] / 2 * self.grid_per_meter)) \
                    and abs(dy) < (
                    self.y_range_grid - self.env_obs[car_id]['width'] / 2 * self.grid_per_meter):

                # Drawing speed of the current car
                velocity = self.env_obs[car_id]['speed'] / 50
                if self.egoID == car_id:
                    velocity = 1 - abs(self.env_obs[car_id]['speed'] - self.desired_speed) / max(self.desired_speed,
                                                                                                 self.env_obs[car_id][
                                                                                                     "speed"])
                observation[0, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                self.y_range_grid + dy - w:self.y_range_grid + dy + w] += np.ones_like(
                    observation[0, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w]) * velocity

                # Drawing lane of the current car
                lane_id = self.env_obs[car_id]['lane_id'] / 2
                observation[1, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                self.y_range_grid + dy - w:self.y_range_grid + dy + w] += np.ones_like(
                    observation[1, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w]) * lane_id
                # Drawing heading of the car
                heading = (np.pi / 2 - np.radians(self.env_obs[car_id]["heading"])) / np.pi
                observation[2, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                self.y_range_grid + dy - w:self.y_range_grid + dy + w] += np.ones_like(
                    observation[2, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                    self.y_range_grid + dy - w:self.y_range_grid + dy + w]) * heading

                # # If ego, drawing the desired speed
                # if car_id == self.egoID:
                #     observation[3, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                #     self.y_range_grid + dy - w:self.y_range_grid + dy + w] += np.ones_like(
                #         observation[3, self.x_range_grid + dx - l:self.x_range_grid + dx + l,
                #         self.y_range_grid + dy - w:self.y_range_grid + dy + w]) * self.desired_speed / 50

        # Channel x width (y) x heigth (x) because it is an image.
        observation = observation.transpose((0, 2, 1))
        # plt.imshow(observation.transpose((1, 2, 0))[:, :, :3])
        # plt.show()
        # if we want to flatten the output, due to network feed.
        if flatten:
            observation = observation.flatten()

        return observation

    def _calculate_structured_environment(self):
        """
        This function calculates the structural observation of the environment.
        :return:
        """
        ego_state = self.env_obs.get(self.egoID, self.state)
        if self.egoID is not None:
            observation = {}
            basic_vals = {'dx': 50, 'dv': 0}
            basic_keys = ['FL', 'FE', 'FR', 'RL', 'RE', 'RR', 'EL', 'ER']
            # Creating a dict for all the present vehicles
            for state_key in basic_keys:
                if state_key in ['RL', 'RE', 'RR']:
                    observation[state_key] = copy.copy(basic_vals)
                    observation[state_key]['dv'] = 0
                    observation[state_key]['dx'] = -50
                elif state_key in ['EL', 'ER']:
                    observation[state_key] = 0
                else:
                    observation[state_key] = copy.copy(basic_vals)
            lane = {0: [], 1: [], 2: []}
            # Calculating all the vehicle data
            for car_id in self.env_obs.keys():
                if car_id != self.egoID:
                    new_car = dict()
                    new_car['dx'] = (self.env_obs[car_id]['x_position'] - self.env_obs[self.egoID]['x_position'])
                    new_car['dy'] = (abs(self.env_obs[car_id]["y_position"] - self.env_obs[self.egoID]['y_position']))
                    new_car['dv'] = (self.env_obs[car_id]['speed'] - self.env_obs[self.egoID]['speed'])
                    new_car['l'] = (self.env_obs[car_id]["length"])
                    lane[self.env_obs[car_id]["lane_id"]].append(new_car)

            [lane[i].sort(key=lambda x: x['dx']) for i in lane.keys()]
            # Going through the data and selecting the closest ones for the ego.
            for lane_id in lane.keys():

                if lane_id == ego_state['lane_id']:
                    for veh in lane[lane_id]:
                        common_length = (veh['l'] + ego_state["length"]) / 2
                        if veh['dx'] - common_length > 0:
                            if veh['dx'] - common_length < observation['FE']['dx']:
                                observation['FE']['dx'] = veh['dx'] - common_length
                                observation['FE']['dv'] = veh['dv']
                        elif veh['dx'] + common_length < 0:
                            if veh['dx'] + common_length > observation['RE']['dx']:
                                observation['RE']['dx'] = veh['dx'] + common_length
                                observation['RE']['dv'] = veh['dv']
                elif lane_id == ego_state['lane_id'] + 1:
                    for veh in lane[lane_id]:
                        common_length = (veh['l'] + ego_state["length"]) / 2
                        if veh['dx'] - common_length > 0:
                            if veh['dx'] - common_length < observation['FL']['dx']:
                                observation['FL']['dx'] = veh['dx'] - common_length
                                observation['FL']['dv'] = veh['dv']
                        elif veh['dx'] + common_length < 0:
                            if veh['dx'] + common_length > observation['RL']['dx']:
                                observation['RL']['dx'] = veh['dx'] + common_length
                                observation['RL']['dv'] = veh['dv']
                        else:
                            observation['EL'] = 1

                elif lane_id == ego_state["lane_id"] - 1:
                    for veh in lane[lane_id]:
                        common_length = (veh['l'] + ego_state["length"]) / 2
                        if veh['dx'] - common_length > 0:
                            if veh['dx'] - common_length < observation['FR']['dx']:
                                observation['FR']['dx'] = veh['dx'] - common_length
                                observation['FR']['dv'] = veh['dv']
                        elif veh['dx'] + common_length < 0:
                            if veh['dx'] + common_length > observation['RR']['dx']:
                                observation['RR']['dx'] = veh['dx'] + common_length
                                observation['RR']['dv'] = veh['dv']
                        else:
                            observation['ER'] = 1

            if ego_state['lane_id'] == 0:
                observation['ER'] = 1
            elif ego_state['lane_id'] == 2:
                observation['EL'] = 1

            observation['speed'] = ego_state['speed']
            observation['lane_id'] = ego_state['lane_id']  # todo: onehot vector
            observation['des_speed'] = self.desired_speed
            observation['heading'] = ego_state['heading']
        else:
            observation = None

        return observation, ego_state

    def _convert_image_state_space_to_vector(self):
        if self.observation is not None:
            return self._calculate_image_environment()
        else:
            return self.observation_space if not self.flatten \
                else np.asarray([0] * self.observation_space.n, dtype=np.float32)

    def _convert_structural_state_space_to_vector(self):
        obs_vector = []
        if self.observation is not None:
            # Normalizing the output
            for idx, value in self.observation.items():
                if isinstance(value, (int, float)):
                    if idx in ["speed", "des_speed"]:
                        obs_vector.append(value / 50)
                    elif idx in ["lane_id"]:
                        obs_vector.append(value / 2)
                    else:
                        obs_vector.append(value)
                elif isinstance(value, dict):
                    for iddx, item in value.items():
                        obs_vector.append(item / 50)
            assert max(obs_vector) <= 1 and min(obs_vector) >= -1

        else:
            obs_vector = [0] * 18

        obs_vector = np.asarray(obs_vector, dtype=np.float32)

        return obs_vector

    def calculate_good_objects_based_on_policy(self, policy):
        return self.get_max_reward(policy)

    def get_max_reward(self, temp_reward):
        """
        This function is to calculate the reward of the current reward system.
        :param temp_reward: can be dict, or ndarray or int. This will be multiplied with the current  reward vector
        :return: returns the rewards for the current policy
        """
        reward = []
        if isinstance(temp_reward, dict):
            for idx, value in temp_reward.items():
                if isinstance(value, list) and self.reward_dict[idx][2]:
                    reward.append(0.0)
                elif isinstance(value, (int, float)) and self.reward_dict[idx][2]:
                    reward.append(value)

        elif isinstance(temp_reward, (np.ndarray, int)):
            reward = []
            for idx, value in self.reward_dict.items():
                if isinstance(value, list) and value[2]:
                    reward.append(value[1])
            reward = np.asarray(reward, dtype=np.float32) * temp_reward
        return reward
