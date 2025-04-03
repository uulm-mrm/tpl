import numpy as np


class Trajectory:

    def __init__(self):

        self.time = np.zeros((1,))
        self.s = np.zeros((1,))
        self.x = np.zeros((1,))
        self.y = np.zeros((1,))
        self.orientation = np.zeros((1,))
        self.curvature = np.zeros((1,))
        self.velocity = np.zeros((1,))
        self.acceleration = np.zeros((1,))

        self.emergency = False
