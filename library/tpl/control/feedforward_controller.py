import numba
import numpy as np

import tpl.util as util

from tpl.control import BaseController
from tpl.planning import Trajectory
from scipy.interpolate import interp1d


@numba.njit
def sim_veh_model(x0, accs, steer_angles, wheel_base, v_ch, dt):

    for i in range(len(accs)):
        x0[0] = x0[0] + dt * x0[4] * np.cos(x0[2])
        x0[1] = x0[1] + dt * x0[4] * np.sin(x0[2])
        x0[2] = x0[2] + dt * x0[4] * np.tan(x0[3]) / (wheel_base * (1 + (x0[4] / v_ch)**2))
        x0[3] = steer_angles[i]
        x0[4] = x0[4] + dt * accs[i]

    return x0


class Params:

    def __init__(self):

        self.k_p_s_lon = 1.0
        self.k_i_s_lon = 0.01
        self.k_p_v_lon = 1.0
        self.k_i_v_lon = 0.02

        self.k_p_lat = 2.0
        self.k_p_heading = 10.0

        self.k_stan_lat = 2.0

        self.use_stanley_law = False

        self.a_max = 3.0
        self.a_min = -3.0

        self.steer_rate_max = 1.0

        self.err_lat_max = 0.2
        self.err_int_lon_max_s = 2.0
        self.err_int_lon_max_v = 2.0

        self.v_ch = 32.0
        self.step_comp_dead_time = 0.005


class FeedforwardController(BaseController):

    def __init__(self, shared, lock_shared):

        self.shared = shared
        self.lock_shared = lock_shared

        # shared state initialization
        with self.lock_shared():
            self.shared.params = Params()

        self.con_traj = Trajectory()

        self.ctrl_vars_history = []

        self.last_update_time = 0.0

        self.err_int_lon_s = 0.0
        self.err_int_lon_v = 0.0

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

        # reset on time jump
        if dt < 0.0 or dt >= 0.1:
            dt = max(0.0, min(0.1, dt))
            self.err_int_lon_s = 0.0
            self.err_int_lon_v = 0.0
            self.ctrl_vars_history = []

        self.last_update_time = t
        if dt == 0:
            return (self.acc, self.steering_angle), self.con_traj

        # update params
        with self.lock_shared():
            params = self.shared.params.deepcopy()
            self.shared.err_int_lon_s = self.err_int_lon_s
            self.shared.err_int_lon_v = self.err_int_lon_v

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
                               params.v_ch,
                               params.step_comp_dead_time)

        # get feed-forward inputs

        try:
            t_clip = min(traj.time[-1], max(traj.time[0], t))
            x_trg = float(interp1d(traj.time, traj.x)(t_clip + veh.dead_time_steer))
            y_trg = float(interp1d(traj.time, traj.y)(t_clip + veh.dead_time_steer))
            phi_trg = util.lerp(t_clip + veh.dead_time_steer, traj.time, traj.orientation, angle=True)
            v_trg = float(interp1d(traj.time, traj.velocity)(t_clip + veh.dead_time_steer))
            a_trg = float(interp1d(traj.time, traj.acceleration)(t_clip + veh.dead_time_steer))
            curv_trg = float(interp1d(traj.time, traj.curvature)(t_clip + veh.dead_time_steer))
        except Exception as e:
            return (self.acc, self.steering_angle), self.con_traj

        # compute acceleration controls

        acc_ff = a_trg - 9.81 * np.sin(veh.pitch)

        x0_x_cog = x0[0] + np.cos(x0[2]) * veh.wheel_base * 0.5
        x0_y_cog = x0[1] + np.sin(x0[2]) * veh.wheel_base * 0.5

        path = np.vstack((traj.x, traj.y)).T
        proj = util.project(path, (x0_x_cog, x0_y_cog))
        proj_trg = util.project(path, (x_trg, y_trg))

        err_s = proj_trg.arc_len - proj.arc_len
        err_v = v_trg - x0[4]

        self.err_int_lon_s = max(-params.err_int_lon_max_s,
                min(params.err_int_lon_max_s, self.err_int_lon_s + err_s * dt))
        self.err_int_lon_v = max(-params.err_int_lon_max_v,
                min(params.err_int_lon_max_v, self.err_int_lon_v + err_v * dt))
        self.acc = (acc_ff 
                    + params.k_p_s_lon * err_s 
                    + params.k_i_s_lon * self.err_int_lon_s
                    + params.k_p_v_lon * err_v 
                    + params.k_i_v_lon * self.err_int_lon_v)
        self.acc = max(params.a_min, min(params.a_max, self.acc))

        # compute lateral controls

        steering_angle_ff = np.arctan(veh.wheel_base
                                      * (1 + (x0[4]/params.v_ch)**2) * curv_trg)

        err_d_vec = np.array([x0_x_cog, x0_y_cog]) - np.array([x_trg, y_trg])
        err_d_lat = np.array([-np.sin(phi_trg), np.cos(phi_trg)]) @ err_d_vec
        err_heading = util.short_angle_dist(x0[2], phi_trg)

        if params.use_stanley_law:
            new_steering_angle = steering_angle_ff + err_heading + np.arctan(
                    params.k_stan_lat * -err_d_lat / max(1.0, x0[4]))
        else:
            new_steering_angle = (steering_angle_ff
                    + params.k_p_lat / max(1.0, veh.v) * err_d_lat
                    + params.k_p_heading * err_heading)

        steer_rate = min(params.steer_rate_max, max(-params.steer_rate_max,
                         (new_steering_angle - self.steering_angle) / dt))
        if veh.v > 0.5:
            self.steering_angle += steer_rate * dt
            self.steering_angle = max(-veh.delta_max, min(veh.delta_max, self.steering_angle))

        # store controls for deadtime compensation 

        if dt > 0.0:
            self.ctrl_vars_history.append((t, self.acc, self.steering_angle))
        if len(self.ctrl_vars_history) > 500:
            self.ctrl_vars_history.pop(0)

        return (self.acc, self.steering_angle), self.con_traj
