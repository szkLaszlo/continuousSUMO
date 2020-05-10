"""
Lateral Motion Model using the python-control package

The model uses a bicycle model that assumes zero wheel slip!

Script created by szoke.laszlo95@gmail.com and bencepalotas@gmail.com
as a project of TRAFFIC MODELLING, SIMULATION AND CONTROL subject
"""
from cmath import sqrt

import control as ct
import numpy as np
from random import uniform

# Base configuration
ct.use_fbs_defaults()
ct.use_numpy_matrix(False)


class LateralModel:
    """
     Class for keeping track of the agent movement in case of continuous SUMO state space.
     """

    def __init__(self, state, lane_width, dt=0.1):
        """
         Function to initiate the lateral Model, it will set the parameters.

         :param state: dict,
         :param lane_width: float, width of initial lane in meters [m]
         :param dt: timestep increment of the simulation
        """

        # Initial lane position should be lane center
        self.lane_width = lane_width
        self.dt = dt

        # Initial steering angle is 0
        self.steering_angle = 0

        # In lane position is the distance from the right tangent line of the lane
        # Initial lane position is middle of the lane
        self.in_lane_pos = self.lane_width / 2

        # Initial ego vehicle state and parameters
        self.lateral_state = {
            'heading': np.radians(state['heading']) - np.pi / 2,
            'steering_angle': self.steering_angle,
            'x_position': state['x_position'],
            'y_position': state['y_position'],
            'velocity': state['velocity'],
            'lane_id': state['lane_id'],

            # Vehicle parameters that are not supposed to update:
            # Vehicle dimensions for collision detection
            'length': state['length'],
            'width': state['width'],
        }

        # Move from bumper to  vehicle center
        self.lateral_state['x_position'] -= np.cos(self.lateral_state['heading']) * self.lateral_state['length']
        self.lateral_state['y_position'] -= np.sin(self.lateral_state['heading']) * self.lateral_state['width']

        self.vehicle_params = {
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
            params=self.vehicle_params,
        )

        self.pos_log = {
            'x_position': [self.lateral_state['x_position']],
            'y_position': [self.lateral_state['y_position']],
            'heading': [self.lateral_state['heading']],
            'lane_id': [self.lateral_state['lane_id']]
        }

    def step(self, steering_angle, velocity_dif):
        """
         Function responsible for the action transform into the continuous state space.
         :param dt: float, time step [sec]
         :param steering_angle: float, [rad]
         :param velocity_dif: float, difference to previous velocity [m/s]
         :return: new_state: containing the input vehicle's new position, orientation, speed, lane.
         """
        self.lateral_state['steering_angle'] = steering_angle
        self.lateral_state['velocity'] += velocity_dif

        # Simulate the system + estimator
        # Resolution of the trajectory
        timesteps = np.array([0, self.dt])

        # Velocity array
        v_curvy = np.full(timesteps.shape, self.lateral_state['velocity'])

        # Steering angle array
        delta_curvy = np.full(timesteps.shape, self.lateral_state['steering_angle'])

        # Input array
        u_curvy = [v_curvy, delta_curvy]

        # Initial condition (x, y, heading [rad])
        x_0 = [self.lateral_state['x_position'], self.lateral_state['y_position'], self.lateral_state['heading']]

        _, _, x_curvy = ct.input_output_response(self.vehicle,
                                                 timesteps,
                                                 u_curvy,
                                                 x_0,
                                                 params=self.vehicle_params,
                                                 return_x=True,
                                                 )

        new_x, new_y, heading = x_curvy[:, -1]

        self.pos_log['x_position'].append(new_x)
        self.pos_log['y_position'].append(new_y)
        self.pos_log['heading'].append(heading)

        dif_x = new_x - self.lateral_state['x_position']
        dif_y = new_y - self.lateral_state['y_position']
        self.update_in_lane_position(dif_x, dif_y, road_curve=None)
        self.update_absolute_position(new_x, new_y)

        return self.lateral_state

    @staticmethod
    def vehicle_update(t, x, u, params):
        """
        Update vehicle location

        :param t: float, current time step, necessary for control library
        :param x: 1-D array with shape (nstates,)
        :param u: 1-D array with shape (ninputs,)
        :param params:
        :return:
        """
        # Get the parameters for the model
        wheelbase = params.get('wheelbase', 3.)  # vehicle wheelbase
        phi_max = params.get('maxsteer', 0.5)  # max steering angle (rad)

        # Saturate the steering input
        phi = np.clip(u[1], -phi_max, phi_max)

        # Return the derivative of the state
        return np.array([
            # x_dot = cos(theta) * v
            np.cos(x[2]) * u[0],
            # y_dot = sin(theta) * v
            np.sin(x[2]) * u[0],
            # th_dot = v/wheelbase * tan(phi)
            (u[0] / wheelbase) * np.tan(phi)
        ])

    @staticmethod
    def vehicle_output(t, x, u, params):
        """
        Method truncating the recieved input
        :param x: 1-D array with shape (nstates)
        :param u:
        :param params:
        :return: 1-D array with shape (2)
        """
        return x[0:2]

    @staticmethod
    def control_output(t, x, u, params):
        """

        :param t:
        :param x:
        :param u:
        :param params:
        :return:
        """
        # Get the controller parameters
        longpole = params.get('longpole', -2.)
        latpole1 = params.get('latpole1', -1 / 2 + sqrt(-7) / 2)
        latpole2 = params.get('latpole2', -1 / 2 - sqrt(-7) / 2)
        l = params.get('wheelbase', 3)

        # Extract the system inputs
        ex, ey, etheta, vd, phid = u

        # Determine the controller gains
        alpha1 = -np.real(latpole1 + latpole2)
        alpha2 = np.real(latpole1 * latpole2)

        # Compute and return the control law
        v = -longpole * ex  # Note: no feedforward (to make plot interesting)
        if vd != 0:
            phi = phid + (alpha1 * l) / vd * ey + (alpha2 * l) / vd * etheta
        else:
            # We aren't moving, so don't turn the steering wheel
            phi = phid

        return np.array([v, phi])

    def update_in_lane_position(self, dif_x, dif_y, road_curve=None):
        """
        Update in lane position based on new position and lane curvature
        NOTE: now only works on a road with straight segments
        :param road_curve:
        """
        # TODO: implement road curvature based update
        self.in_lane_pos += dif_y

        # Change to lane to the right
        if self.in_lane_pos < 0:
            self.lateral_state['lane_id'] -= 1
            self.in_lane_pos = self.lane_width + self.in_lane_pos

        # Change to lane to the left
        elif self.in_lane_pos > self.lane_width:
            self.lateral_state['lane_id'] += 1
            self.in_lane_pos = self.in_lane_pos - self.lane_width

        self.pos_log['lane_id'].append(self.lateral_state['lane_id'])

    def update_absolute_position(self, new_x, new_y):
        """
        Update state dict with new x and y positions
        :param new_x: float
        :param new_y: float
        """
        self.lateral_state['x_position'] = new_x
        self.lateral_state['y_position'] = new_y


def main():
    """
    Example usage of the lateral state
    """
    model = LateralModel(x_position=30,
                         y_position=0,
                         velocity=10,
                         heading=np.pi/2,
                         lane_id=1,
                         lane_width=3.2)
    terminate = False
    counter = 0
    while not terminate:
        if counter % 100 == 0:
            steering_angle = uniform(-np.pi/2, np.pi)
            steering_angle = 0

        state = model.step(dt=0.1,
                           steering_angle=steering_angle,
                           velocity_dif=15,
                           )

        if counter == 1000:
            terminate = True
        counter += 1
    pass


if __name__ == '__main__':
    main()
