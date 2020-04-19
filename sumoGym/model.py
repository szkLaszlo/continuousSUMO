"""
Lateral Motion Model using the python-control package

The model uses a bicycle model that assumes zero wheel slip!

Script created by szoke.laszlo95@gmail.com and bencepalotas@gmail.com
as a project of TRAFFIC MODELLING, SIMULATION AND CONTROL subject
"""
import numpy as np
import control as ct

# Base configuration
ct.use_fbs_defaults()
ct.use_numpy_matrix(False)


class LateralModel():
    """
     Class for keeping track of the agent movement in case of continuous SUMO state space.
     """

    def __init__(self, position, speed, heading, lane_id, lane_width):
        """
         Function to initiate the lateral Model, it will set the parameters.
         :param position: [x, y] vector, initial position of vehicle
         :param speed: double, initial speed of the car in the direction of the heading
         :param heading: deg, initial heading of the vehicle
         :param lane_id: int, initial lane_id
         :param lane_width: float, width of initial lane in meters
         """
        # Initial lane position should be lane center
        self.position = position
        self.speed = speed
        self.heading = heading
        self.lane_id = lane_id
        self.lane_width = lane_width

        # Initial steering angle is 0
        self.steering_angle = 0

        # In lane position is the distance from the right tangent line of the lane
        # Initial lane position is middle of the lane
        self.in_lane_pos = self.lane_width / 2

        # Initial ego vehicle state
        self.state = {
            'heading': self.heading,
            'x_position': self.position[0],
            'y_position': self.postion[1],
            'velocity': self.speed,
            'lane_id': self.lane_id,
            # Vehicle parameters that are not supposed to update:
            'length': None,
            'width': None,
            # Rear wheel offset in meters
            'refoffset': 1.5,
            # Wheelbase in meters
            'wheelbase': 3,
            # Maximum steering angle in radians
            'maxsteer': 0.5,
        }

        # Define the vehicle steering dynamics as an input/output system
        self.vehicle = ct.NonlinearIOSystem(
            self.vehicle_update,
            self.vehicle_output,
            states=3,
            name='vehicle',
            inputs=('v', 'delta'),
            outputs=('x', 'y'),
            params=self.state,
        )

    def reset(self, position, speed, orientation, lane_id):
        """
         Function to reset the model to initial state. #it could be deleted and a new model created at every reset
         TODO: consider
         :param position: [x, y] vector position
         :param speed: double, in the direction of the orientation.
         :param orientation: orientation of the car, and initially at the road.
         :param lane_id: id of the occupied lane.
         :return:
         """

    def step(self, action, lane_direction_angle):
        """
         Function responsible for the action transform into the continuous state space.
         :return: new_state: containing the input vehicle's new position, orientation, speed, lane.
         """
        pass

    def calculate_speed(self, speed, orientation):
        """
        Function to be used for speed calculation based on orientation and previous things...
        :param speed:
        :param orientation:
        :return:
        """

    def convert_self_state_to_SUMO_state(self):
        """
         Function to calculate own state into SUMO state, because we need to give SUMO its own model based state
         :return:
         """

    def vehicle_update(self, t, x, u, params):
        """

        :param t: float, current time
        :param x: 1-D array with shape (nstates,)
        :param u: 1-D array with shape (ninputs,)
        :param params:
        :return:
        """
        # Get the parameters for the model
        a = params.get('refoffset', 1.5)  # offset to vehicle reference point
        b = params.get('wheelbase', 3.)  # vehicle wheelbase
        maxsteer = params.get('maxsteer', 0.5)  # max steering angle (rad)

        # Saturate the steering input
        delta = np.clip(u[1], -maxsteer, maxsteer)
        alpha = np.arctan2(a * np.tan(delta), b)

        # Return the derivative of the state
        return np.array([
            u[0] * np.cos(x[2] + alpha),    # xdot = cos(theta + alpha) v
            u[0] * np.sin(x[2] + alpha),    # ydot = sin(theta + alpha) v
            (u[0] / b) * np.tan(delta)      # thdot = v/l tan(phi)
        ])

    def vehicle_output(self, x, u, params):
        """
        Method truncating the recieved input
        :param x: 1-D array with shape (nstates)
        :param u:
        :param params:
        :return: 1-D array with shape (2)
        """
        return x[:2]

    def update_in_lane_position(self, dif_x, dif_y, road_curve=None):
        """
        Update in lane position based on new position and lane curvature
        NOTE: now only works on a road with straight segments
        """
        # TODO: implement road curvature based update
        self.in_lane_pos += dif_y

    def update_state(self):
        """
        Method to update self.state dictionary variables
        """
        self.state['x_position'] = self.position[0]
        self.state['y_position'] = self.position[1]
        self.state['velocity'] = self.speed
        self.state['heading'] = self.heading
        self.state['lane_id'] = self.lane_id
