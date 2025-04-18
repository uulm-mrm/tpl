import time
import copy
import numpy as np

from tpl.util import runtime
from tpl.planning import BasePlanner, PathSmoothing, PathOptim, VelocityOptim, Trajectory
from tpl.planning.path_vel_decomp.path_smoothing import Params as PathSmoothingParams
from tpl.planning.path_vel_decomp.path_optim import Params as PathOptimParams
from tpl.planning.path_vel_decomp.velocity_optim import Params as VelocityOptimParams

from tpl.environment import EnvironmentState, map_module

import objtoolbox as otb


class Params:

    def __init__(self):

        self.horizon = 250

        # this deactivates path optimization (e.g. obstacle avoidance)
        # instead the path is only locally smoothed
        self.smooth_only = False

        # if enabled more debug data is written out
        self.write_debug_data = False

        self.path_smoothing = PathSmoothingParams()
        self.path_optim = PathOptimParams()
        self.velocity_optim = VelocityOptimParams()


class PathVelDecompPlanner(BasePlanner):

    def __init__(self, shared, lock_shared):

        self.shared = shared
        self.lock_shared = lock_shared

        self.path_smoothing = PathSmoothing()
        self.path_optim = PathOptim()
        self.velocity_optim = VelocityOptim()

        self.trajectory = Trajectory()

        with self.lock_shared():
            self.shared.params = Params()
            self.shared.debug = otb.bundle()

        self.env = EnvironmentState()

    def shift_path(self, path, offset):

        p = path[:, :2].copy()
        p[:, 0] -= np.sin(path[:, 2]) * offset
        p[:, 1] += np.cos(path[:, 2]) * offset

        return p

    def write_debug_data(self, env, params):

        with self.lock_shared():
            self.shared.debug = otb.bundle()
            dbg = self.shared.debug
            dbg.s_leader = self.velocity_optim.s_leader
            dbg.v_leader = self.velocity_optim.v_leader

            if params.write_debug_data:
                path_len = min(params.horizon, env.local_map.steps_ref)
                path = env.local_map.path[:path_len].copy()

                dbg.opt_path = self.path_optim.opt_path
                dbg.d_lower_ref = self.path_optim.opt.params.d_lower_constr
                dbg.path_lower_ref = self.shift_path(path, dbg.d_lower_ref)
                dbg.d_upper_ref = self.path_optim.opt.params.d_upper_constr
                dbg.path_upper_ref = self.shift_path(path, dbg.d_upper_ref)
                dbg.d_upper_constr = self.path_optim.d_upper_constr
                dbg.path_upper_constr = self.shift_path(path, dbg.d_upper_constr)
                dbg.d_lower_constr = self.path_optim.d_lower_constr
                dbg.path_lower_constr = self.shift_path(path, dbg.d_lower_constr)
                dbg.d_offset = self.path_optim.opt.params.d_offset
                dbg.d_opt = self.path_optim.opt.x[:-1, 0]
                dbg.v_ref_weight = self.velocity_optim.opt.params.ref_v_weight
                dbg.man_min_time_cons = self.velocity_optim.man_min_time_cons
                dbg.man_max_time_cons = self.velocity_optim.man_max_time_cons

                dbg.v_lim = np.maximum(0.0, self.velocity_optim.v_lim)
                dbg.v_ref = np.maximum(0.0, np.minimum(
                    self.velocity_optim.v_lim,
                    self.velocity_optim.v_ref[:, 0]))

    @runtime
    def update(self, sh_env):

        with self.lock_shared():
            params = self.shared.params.deepcopy()

        params.path_optim.horizon = params.horizon
        params.velocity_optim.horizon = params.horizon

        env = self.env

        with sh_env.lock():
            if sh_env.local_map is None:
                return self.trajectory

            env.t = sh_env.t
            env.reset_counter = sh_env.reset_counter
            env.vehicle_state = copy.deepcopy(sh_env.vehicle_state)
            env.local_map = copy.deepcopy(sh_env.local_map)
            env.local_map_behind = copy.deepcopy(sh_env.local_map_behind)
            env.selected_map = copy.deepcopy(sh_env.selected_map)
            env.tracks = copy.deepcopy(sh_env.tracks)
            env.predicted = copy.deepcopy(sh_env.predicted)
            env.man_time_cons = copy.deepcopy(sh_env.man_time_cons)

            sh_env.local_map.update_inters_paths = True
            sh_env.local_map.step_shift_idx = 1
            sh_env.local_map.shift_vel_lim = -int(
                    (env.vehicle_state.rear_axis_to_front
                     + params.velocity_optim.min_d_safe)
                    / sh_env.local_map.step_size_ref)

        if params.smooth_only:
            self.path_optim.reset_required = True
            self.path_smoothing.update(
                    env,
                    params.path_smoothing)
            opt_path = self.path_smoothing.opt_path
        else:
            self.path_smoothing.reset_required = True
            self.path_optim.update(
                    env,
                    params.path_optim)
            opt_path = self.path_optim.opt_path

        self.velocity_optim.update(
                opt_path,
                env,
                params.velocity_optim)

        self.write_debug_data(env, params)

        # build and send trajectory

        traj = self.trajectory

        traj.time = env.t + self.velocity_optim.opt.x[:-1, 1].copy()
        traj.s = np.arange(
                0.0,
                params.velocity_optim.horizon * params.velocity_optim.step,
                params.velocity_optim.step)
        traj.x = opt_path[:, 0].copy()
        traj.y = opt_path[:, 1].copy()
        traj.orientation = opt_path[:, 2].copy()
        traj.curvature = opt_path[:, 4].copy()
        traj.velocity = self.velocity_optim.v_opt
        traj.acceleration = self.velocity_optim.opt.u.copy()

        return self.trajectory
