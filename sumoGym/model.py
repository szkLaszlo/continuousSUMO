class LateralModel():
    """
     Class for keeping track of the agent movement in case of continuous SUMO state space.
     """

    def __init__(self, position, speed, orientation, lane_id):
        """
         Function to initiate the lateral Model, it will set the parameters.
         :param position: Initial position at step 0
         :param speed: initial speed of the car in the direction of the orientation
         :param orientation: initial orientation
         :param lane_id: initial lane_id
         """
        self.position = position
        # todo: must be calculated based on the orientation.
        self.speed_x = None
        self.speed_y = None
        self.orientation = None
        self.lane_id = None
        self.steering_angle = 0

    def reset(self, position, speed, orientation, lane_id):
        """
         Function to reset the model to initial state. #it could be deleted and a new model created at every reset
         todo: consider
         :param position: vector [x,y]
         :param speed: double in the direction of the orientation.
         :param orientation: orientation of the car, and at initially the road.
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