import numpy as np

from tpl.environment import TrafficLight


class TrafficLightDetection:

    def __init__(self):

        self.t = 0.0

        self.near_point = np.array([0.0, 0.0])
        self.far_point = np.array([0.0, 0.0])
        self.state = TrafficLight.UNKNOWN


class DynamicObject:

    def __init__(self):

        self.id = None

        self.t = 0.0

        self.object_class = None

        # object reference point (x,y) in UTM
        self.pos = np.zeros((2,))

        # object orientation in rad in UTM
        self.yaw = None

        # velocity in direction of yaw
        self.v = None

        # 2D points of the convex object hull in UTM
        self.hull = np.zeros((0, 2))

        # radius of the enclosing circle
        self.hull_radius = 0.0

        # if set to "left" or "right" will evade object on that side
        self.evade = ""

        self.cam_id = None

        # meta infos from CPM
        self.meta_info = []

        # list of multiple possible predictions 
        self.predictions = []


class Prediction:

    def __init__(self):

        self.proj_assoc_map = None
        self.uuid_assoc_map = None
        # array with dims: t, x, y, yaw, v
        self.states = np.array((0, 5))
