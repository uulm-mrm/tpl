import numba
import numpy as np

import tpl.util as util

from tpl.control import BaseController
from tpl.planning import Trajectory
from scipy.interpolate import interp1d


@numba.njit
def sim_veh_model(x0, accs, steer_angles, wheel_base, dt):

    for i in range(len(accs)):
        x0[0] = x0[0] + dt * x0[4] * np.cos(x0[2])
        x0[1] = x0[1] + dt * x0[4] * np.sin(x0[2])
        x0[2] = x0[2] + dt * x0[4] * np.tan(x0[3]) / wheel_base
        x0[3] = steer_angles[i]
        x0[4] = x0[4] + dt * accs[i]

    return x0


class Params:

    def __init__(self):

        self.k_pos = 10.0
        self.k_vel = 5.0

        self.ki_pos = 0.1

        self.a_max = 3.0
        self.a_min = -3.0

        self.steer_rate_max_abs = 1.0

        self.step_comp_dead_time = 0.005


class FlatController(BaseController):

    def __init__(self, shared, lock_shared):

        self.shared = shared
        self.lock_shared = lock_shared

        # shared state initialization
        with self.lock_shared():
            self.shared.params = Params()

        self.con_traj = Trajectory()

        self.ctrl_vars_history = []

        self.last_update_time = 0.0

        self.integrator_x = 0.0
        self.integrator_y = 0.0

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
        dt = min(0.1, t - self.last_update_time)

        # reset on time jump
        if dt < 0.0:
            dt = 0
            self.ctrl_vars_history = []

        self.last_update_time = t
        if dt == 0:
            return (self.acc, self.steering_angle), self.con_traj

        # update params
        with self.lock_shared():
            params = self.shared.params.deepcopy()

        # deadtime compensation

        x0 = np.array([
            veh.x,
            veh.y,
            veh.phi,
            veh.delta,
            veh.v
        ])

        if len(self.ctrl_vars_history) > 0:

            ctrl_vars_hist_np = np.array(self.ctrl_vars_history)

            times = np.arange(t - veh.dead_time_steer, t, params.step_comp_dead_time)
            accs = interp1d(
                    ctrl_vars_hist_np[:, 0],
                    ctrl_vars_hist_np[:, 1],
                    kind='zero',
                    fill_value='extrapolate')(times)
            steer_angles = interp1d(
                    ctrl_vars_hist_np[:, 0],
                    ctrl_vars_hist_np[:, 2],
                    kind='zero',
                    fill_value='extrapolate')(times)

            x0 = sim_veh_model(x0,
                               accs,
                               steer_angles,
                               veh.wheel_base,
                               params.step_comp_dead_time)

        # get target states

        arr_traj = np.zeros((len(traj.time), 6))
        arr_traj[:, 0] = traj.x
        arr_traj[:, 1] = traj.y
        arr_traj[:, 2] = traj.velocity
        arr_traj[:, 3] = traj.acceleration
        arr_traj[:, 4] = np.unwrap(traj.orientation, period=np.pi*2.0)
        arr_traj[:, 5] = traj.curvature

        t_clip = min(traj.time[-1], max(traj.time[0], t + veh.dead_time_steer))
        tp = interp1d(traj.time, arr_traj, axis=0)(t_clip)
        if np.any(np.isnan(tp)):
            return (self.acc, self.steering_angle), self.con_traj

        x_trg, y_trg, v_trg, a_trg, phi_trg, k_trg = tp

        beta = np.arcsin(max(-1.0, min(1.0, k_trg * veh.wheel_base * 0.5)))
        psi = phi_trg - beta

        x_trg -= veh.wheel_base * 0.5 * np.cos(psi)
        y_trg -= veh.wheel_base * 0.5 * np.sin(psi)
        xd_trg = v_trg * np.cos(psi) 
        yd_trg = v_trg * np.sin(psi)
        xdd_trg = a_trg * np.cos(psi)
        ydd_trg = a_trg * np.sin(psi)

        # compute control law

        stopping = False
        if x0[4] < 1.0:
            x0[4] = 1.0
            stopping = True

        xd = x0[4] * np.cos(x0[2])
        yd = x0[4] * np.sin(x0[2])

        self.integrator_x += x0[0] - x_trg
        self.integrator_y += x0[1] - y_trg
        self.integrator_x = max(-1.0, min(1.0, self.integrator_x))
        self.integrator_y = max(-1.0, min(1.0, self.integrator_y))

        v1 = xdd_trg - params.k_vel * (xd - xd_trg) - params.k_pos * (x0[0] - x_trg) - params.ki_pos * self.integrator_x
        v2 = ydd_trg - params.k_vel * (yd - yd_trg) - params.k_pos * (x0[1] - y_trg) - params.ki_pos * self.integrator_y

        with self.lock_shared():
            self.shared.x_trg = x_trg
            self.shared.y_trg = y_trg
            self.shared.xd_trg = xd_trg
            self.shared.yd_trg = yd_trg
            self.shared.ydd_trg = ydd_trg
            self.shared.xdd_trg = xdd_trg
            self.shared.xd_err = xd - xd_trg
            self.shared.yd_err = yd - yd_trg
            self.shared.x_err = veh.x - x_trg
            self.shared.y_err = veh.y - y_trg
            self.shared.int_x = self.integrator_x
            self.shared.int_y = self.integrator_y

        dir_sign = np.sign(x0[4])

        acc = (xd*v1 + yd*v2) / np.sqrt(xd**2 + yd**2)
        if stopping:
            steering_angle = self.steering_angle
        else:
            steering_angle = np.arctan(
                    dir_sign * (xd*v2 - yd*v1) * veh.wheel_base 
                    / ((xd**2 + yd**2)**(3/2)))

        steer_rate = (self.steering_angle - steering_angle) / dt
        steer_rate = min(params.steer_rate_max_abs, max(steer_rate, -params.steer_rate_max_abs))
        self.steering_angle += steer_rate * dt

        self.acc = min(params.a_max, max(params.a_min, acc))
        self.steering_angle = min(veh.delta_max, max(-veh.delta_max, steering_angle))

        # store controls for deadtime compensation 
        
        if dt > 0.0:
            self.ctrl_vars_history.append((t, self.acc, self.steering_angle))
        if len(self.ctrl_vars_history) > 500:
            self.ctrl_vars_history.pop(0)

        return (self.acc, self.steering_angle), self.con_traj
