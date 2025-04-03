import numba
import numpy as np

import tpl.util as util

from tpl.control import BaseController
from tpl.planning import Trajectory
from scipy.interpolate import interp1d


class Params:

    def __init__(self):

        self.a_max = 3.0
        self.a_min = -3.0

        self.steer_rate_max_abs = 1.0


class DirectController(BaseController):

    def __init__(self, shared, lock_shared):

        self.shared = shared
        self.lock_shared = lock_shared

        # shared state initialization
        with self.lock_shared():
            self.shared.params = Params()

        self.con_traj = Trajectory()

        self.last_update_time = 0.0

        self.acc = 0.0
        self.steering_angle = 0.0

    def update(self, con_input):

        t = con_input.t
        veh = con_input.vehicle
        traj = con_input.trajectory

        # controller has no real trajectory -> create dummy trajectory
        self.con_traj = Trajectory()
        self.con_traj.x = np.array([veh.x])
        self.con_traj.y = np.array([veh.y])

        # compute delta time
        dt = t - self.last_update_time

        self.last_update_time = t
        if dt == 0:
            return (self.acc, self.steering_angle), self.con_traj

        # update params
        with self.lock_shared():
            params = self.shared.params.deepcopy()

        # get target states

        t_clip = min(traj.time[-1], max(traj.time[0], t + veh.dead_time_steer))

        accs = interp1d(traj.time, traj.acceleration)
        curvatures = interp1d(traj.time, traj.curvature)

        acc = float(accs(t_clip))
        steering_angle = float(np.arctan(curvatures(t_clip) * veh.wheel_base))

        self.acc = min(params.a_max, max(params.a_min, acc))
        self.steering_angle = min(veh.delta_max, max(-veh.delta_max, steering_angle))

        return (self.acc, self.steering_angle), self.con_traj
