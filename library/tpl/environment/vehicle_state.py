import numpy as np


class VehicleState:

    def __init__(self):

        self.x: float = 0.0
        self.y: float = 0.0
        self.phi: float = 0.0
        self.phi_dot: float = 0.0
        self.k: float = 0.0
        self.v: float = 0.0
        self.a: float = 0.0
        self.delta: float = 0.0
        self.delta_dot: float = 0.0
        self.lat_acc: float = 0.0
        self.pitch: float = 0.0

        self.delta_max: float = np.radians(35.0)
        self.a_lat_max: float = 2.5

        self.width: float = 2.0
        self.track_width: float = 1.5
        self.wheel_base: float = 2.0
        self.rear_axis_to_rear: float = 2.0
        self.rear_axis_to_front: float = 3.0

        self.dead_time_steer: float = 0.0
        """
        The dead time of the steering actuators.
        """
        self.dead_time_acc: float = 0.0
        """
        This refers to the dead time of the drive train (acc > 0.0).
        Brakes are assumed to have neglegible dead time.
        """

        self.imu_state: int = 0
        """off: 0, gps: 1, rtk-float: 2, rtk-lock: 3"""

        self.turn_indicator: int = 0
        """off: 0, right: -1, left: 1, hazard: 2"""

        self.steering_wheel_button: bool = False

        self.automated: bool = True
