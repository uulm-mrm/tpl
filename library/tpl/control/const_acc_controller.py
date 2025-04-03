import numpy as np

from tpl.control import BaseController
from tpl.planning import Trajectory


class ConstAccController(BaseController):

    def __init__(self, shared, lock_shared):

        self.shared = shared
        self.lock_shared = lock_shared

        self.steering_angle = 0.0
        self.acceleration = -6.0
        self.con_traj = Trajectory()

        self.last_update_time = -1.0

    def update(self, con_input):

        t = con_input.t
        veh = con_input.vehicle
        traj = con_input.trajectory

        if t - self.last_update_time >= 1.0:
            # This prevents steering angle creeping in case there
            # is biased noise on the steering angle measurements.
            self.steering_angle = 0.0

        self.last_update_time = t

        # generate control trajectory by forward integration of vehicle model

        dt = 0.1
        ts = np.arange(0.0, 2.0, dt)
        ss = [0.0]
        xs = [veh.x]
        ys = [veh.y]
        phis = [veh.phi]
        vs = [veh.v]

        for t in ts:
            dx = vs[-1] * np.cos(phis[-1])
            dy = vs[-1] * np.sin(phis[-1])
            ds = np.sqrt(dx**2 + dy**2)
            dphi = vs[-1] * np.tan(self.steering_angle) / veh.wheel_base
            dv = self.acceleration
            xs.append(xs[-1] + dt * dx)
            ys.append(ys[-1] + dt * dy)
            ss.append(ss[-1] + dt * ds)
            phis.append(phis[-1] + dt * dphi)
            vs.append(max(0.0, vs[-1] + dt * dv))

        self.con_traj = Trajectory()
        self.con_traj.t = np.array(ts)
        self.con_traj.x = np.array(xs)
        self.con_traj.y = np.array(ys)
        self.con_traj.orientation = np.array(phis)
        self.con_traj.velocity = np.array(vs)
        self.con_traj.s = np.array(ss)

        return (self.acceleration, self.steering_angle), self.con_traj
