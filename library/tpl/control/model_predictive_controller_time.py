import numpy as np

from tpl import util
from tpl.util import runtime
from tpl.optim import optimizers as opts
from tpl.control import BaseController
from tpl.planning import Trajectory

import objtoolbox as otb


class IdleCompensationParams:

    def __init__(self):

        self.active = False
        self.min_acc = -2.0
        self.jerk = -1.0
        self.veh_thresh = 0.5
        self.traj_thresh = 0.1
        self.traj_look_ahead_steps = 5


class CostFunctionParams:

    def __init__(self):

        self.pd = 10.0
        self.pv = 5.0
        self.pdelta = 0.0
        self.min_pdelta_dot = 0.1
        self.pdelta_dot = 0.1
        self.min_p_phi_dot = 0.0
        self.p_phi_dot = 0.0
        self.p_phi = 0.0
        self.p_phi_ref_dot_diff = 0.0
        self.pa = 2.0
        self.pj = 0.5


class Params:

    def __init__(self):

        self.horizon = 40
        self.step = 0.05
        self.max_iterations = 20

        self.cycle_time = 0.01
        self.acc_min = -3.0
        self.acc_max = 3.0
        self.jerk_min = -3.0
        self.jerk_max = 1.5
        self.steer_rate_min = -1.0
        self.steer_rate_max = 1.0

        self.cog_pos = 0.5

        self.ref_dt = 0.1

        self.cost_function = CostFunctionParams()
        self.idle_comp = IdleCompensationParams()


class ModelPredictiveControllerTime(BaseController):

    def __init__(self, shared, lock_shared):

        self.shared = shared
        self.lock_shared = lock_shared

        self.opt = opts.trajectory_tracking_mpc_time()
        self.opt.integrator_type = self.opt.HEUN
        self.opt.lg_mult_limit = 0.0
        self.opt.barrier_weight[:] = 10000.0
        self.opt.lagrange_multiplier[:] = 0.0

        self.opt.params.v_ch = 32.0
        self.opt.params.max_delta = 0.7

        # internal state
        self.lat_dist_to_traj = 0.0
        self.jerk = 0.0
        self.dead_time_trajectory = np.zeros((0, 5))
        self.controls = (0.0, 0.0)
        self.con_traj = Trajectory()
        self.last_update_time = 0.0
        self.ctrl_vars_history = []
        self.idle_comp_acc = 0.0
        self.idle_comp_steer = 0.0

        # shared state initialization
        with self.lock_shared():
            self.shared.params = Params()

    @runtime
    def update(self, con_input):

        t = con_input.t
        veh = con_input.vehicle
        traj = con_input.trajectory

        opt = self.opt

        delta_time = t - self.last_update_time
        if delta_time < 0.0:
            self.ctrl_vars_history = []

        # get params
        with self.lock_shared():
            params = self.shared.params.deepcopy()

        if len(traj.time) < 2:
            return self.controls, self.con_traj

        if traj is None:
            return self.controls, self.con_traj

        # update constraints
        opt.u_min[:, 0] = params.jerk_min
        opt.u_max[:, 0] = params.jerk_max
        opt.u_min[:, 1] = params.steer_rate_min
        opt.u_max[:, 1] = params.steer_rate_max
        opt.params.min_acc = params.acc_min
        opt.params.max_acc = params.acc_max

        # update cost function params
        otb.merge(opt.params, params.cost_function)

        # update internal params
        opt.horizon = params.horizon
        opt.step = params.step
        opt.max_iterations = params.max_iterations
        opt.params.l = veh.wheel_base
        opt.params.ref_x = traj.x
        opt.params.ref_y = traj.y
        opt.params.ref_phi = traj.orientation
        opt.params.ref_v = traj.velocity
        opt.params.ref_dt = params.ref_dt
        opt.params.ref_t_offset = veh.dead_time_steer
        opt.params.a_offset = 9.81 * np.sin(veh.pitch)
        opt.params.cog_pos = params.cog_pos

        # for visualization
        self.lat_dist_to_traj = util.project(
                np.vstack((traj.x, traj.y)).T, (veh.x, veh.y)).distance

        # deadtime compensation

        x0 = np.array([
            veh.x + np.cos(veh.phi) * params.cog_pos * veh.wheel_base,
            veh.y + np.sin(veh.phi) * params.cog_pos * veh.wheel_base,
            veh.phi,
            veh.delta,
            veh.v,
            veh.a
        ])

        if veh.dead_time_steer > 0.0:
            x0s = []
            rt = t
            dead_time_index = int(veh.dead_time_steer / params.cycle_time + 1e-5)
            for acc, delta in self.ctrl_vars_history[-dead_time_index:]:
                x0s.append(np.array([rt, *x0]))
                u = np.array([0.0, 0.0])
                x0[3] = delta
                x0[5] = acc
                x0 = opt.dynamics(x0, u, 0, params.cycle_time)
                rt += params.cycle_time
            x0s.append(np.array([rt, *x0]))
            self.dead_time_trajectory = np.array(x0s)

        # call optimizer
        opt.x[0] = x0
        opt.update()

        # we interpolate the steering angle and acceleration 
        j = 1.0
        steering_angle = (1.0 - j) * opt.x[0][3] + j * opt.x[1][3]
        steering_angle = min(veh.delta_max, max(-veh.delta_max, steering_angle))
        acc = (1.0 - j) * opt.x[0][5] + j * opt.x[1][5] 
        acc = min(params.acc_max, max(params.acc_min, acc))

        # store values for deadtime compensation
        if delta_time > 0.0:
            self.ctrl_vars_history.append((acc, steering_angle))
        if len(self.ctrl_vars_history) > 100:
            self.ctrl_vars_history.pop(0)

        # store time for next update step
        self.last_update_time = t

        # idle acceleration compensation
        vel_idx = 5
        if params.idle_comp.active and (veh.v < params.idle_comp.veh_thresh
                and traj[params.idle_comp.traj_look_ahead_steps, vel_idx] < params.idle_comp.traj_thresh):
            self.idle_comp_acc += params.idle_comp.jerk * delta_time
            steering_angle = self.idle_comp_steer
        else:
            self.idle_comp_steer = steering_angle
            self.idle_comp_acc = 0.0
        self.idle_comp_acc = min(0.0, max(params.idle_comp.min_acc, self.idle_comp_acc))
        acc += self.idle_comp_acc

        self.jerk = opt.u[0][0]

        self.controls = (acc, steering_angle)

        self.con_traj = Trajectory()
        self.con_traj.time = t + np.arange(0, opt.T*opt.dt, opt.dt)
        self.con_traj.x = opt.x[:-1, 0]
        self.con_traj.y = opt.x[:-1, 1]
        self.con_traj.orientation = opt.x[:-1, 2]
        self.con_traj.velocity = opt.x[:-1, 4]
        self.con_traj.curvature = np.tan(opt.x[:-1, 3]) / veh.wheel_base
        self.con_traj.acceleration = opt.x[:, 5]

        return self.controls, self.con_traj
