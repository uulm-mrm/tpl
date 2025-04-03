import time
import copy
import numba
import datetime

import numpy as np
import os.path as osp
import objtoolbox as otb

from tpl import util
from tplcpp import resample
from tplcpp import (
        DynProgLatLonPlanner,
        DynProgLatLonPlannerParams,
        LatLonState
    )

from tpl.environment import EnvironmentState

from tpl.planning import BasePlanner, Trajectory
from tpl.planning.dyn_prog.dp_env import DpEnv

from scipy.interpolate import interp1d, BSpline


class Params:

    def __init__(self):

        self.write_debug_data = True

        self.update_always = False

        self.replan_time_step = 0.1

        self.dead_time = 0.0

        self.d_reinit = 2.0

        self.cpp = DynProgLatLonPlannerParams()


class DpLatLonPlanner(BasePlanner):

    def __init__(self, shared, lock_shared):

        np.seterr(divide='ignore', invalid='ignore')

        self.shared = shared
        self.lock_shared = lock_shared

        self.emergency_counter = 0
        
        self.reset_counter = 0

        self.last_update_time = -1.0
        self.last_replan_time = -1.0
        self.dt_start = None

        self.state_reinit_msg = ""

        self.ref_line = None
        self.ref_line_shift = 0.0
        self.ref_line_step_size = 0.0
        self.ref_proj = None

        self.traj_dp = None

        self.trajectory = Trajectory()
        self.trajectory_np = None

        self.dp_planner = DynProgLatLonPlanner()

        self.debug = False
        self.runtime_dp = 0.0

        with self.lock_shared():
            self.shared.params = otb.bundle()
            self.shared.params.planner = otb.bundle()
            self.shared.params.planner = Params()

            self.shared.debug = otb.bundle()
            self.shared.debug.planner = otb.bundle()

        self.dp_env = DpEnv(shared, lock_shared)
        self.env = EnvironmentState()

    def is_traj_valid(self, traj):

        return all(s.constr == 0 for s in traj.states[1:])

    def update_params(self, env):

        veh = env.vehicle_state

        with self.lock_shared():
            params = self.shared.params.planner

            length_veh = veh.rear_axis_to_front + veh.rear_axis_to_rear

            params.cpp.length_veh = length_veh
            params.cpp.width_veh = env.vehicle_state.width

            dt_update = env.t - self.last_update_time
            if self.dt_start is None:
                self.dt_start = params.cpp.dt
            else:
                self.dt_start = (self.dt_start - dt_update) % params.cpp.dt
            params.cpp.dt_start = self.dt_start

            sh_params = params.deepcopy()

        params = Params()
        otb.merge(params, sh_params)

        return params

    def update_planner(self, env, params, replan):

        # set time constraints

        if len(env.man_time_cons) > 0:
            pos_st, t_st_min, t_st_max = env.man_time_cons[0]
            s_st = util.project(self.dp_env.ref_line[:, :2], pos_st).arc_len
            params.cpp.t_st_min = t_st_min - env.t - params.dead_time
            params.cpp.t_st_max = t_st_max - env.t - params.dead_time
            params.cpp.s_st = s_st
        else:
            params.cpp.t_st_min = 0.0
            params.cpp.t_st_max = 1000.0
            params.cpp.s_st = 0.0

        # update dp planner

        if replan:
            self.dp_planner.reinit_buffers(params.cpp)

            start = time.perf_counter()
            self.dp_planner.update_traj_dp(self.dp_env.cpp_env)
            self.runtime_dp = (time.perf_counter() - start) * 1000.0

            self.traj_dp = self.dp_planner.traj_dp.copy()
            self.last_replan_time = env.t

        # update smoothed trajectory

        self.dp_planner.update_traj_smooth(self.dp_env.cpp_env)
        self.dp_planner.update_traj_cart(self.dp_env.cpp_env.ref_line)

    def update_trajectory(self, env, params):

        traj_np = self.dp_planner.traj_smooth_cart.as_numpy()
        traj_np[:, 0] += env.t + params.dead_time

        if self.trajectory_np is None:
            self.trajectory_np = traj_np
        self.trajectory_np[:, 6] = np.unwrap(
                self.trajectory_np[:, 6], period=np.pi*2.0)

        interp_traj = interp1d(self.trajectory_np[:, 0],
                               self.trajectory_np,
                               axis=0,
                               fill_value='extrapolate')
        ts = np.arange(env.t,
                       env.t + params.dead_time,
                       params.cpp.dt_smooth_traj)
        traj_dead_time = interp_traj(ts)
        traj_np = np.concatenate((traj_dead_time, traj_np), axis=0)
        
        if self.is_traj_valid(self.traj_dp):
            self.emergency_counter = max(0, self.emergency_counter - 1)
        else:
            self.emergency_counter = 50

        traj = Trajectory()
        traj.emergency = self.emergency_counter > 0
        if not traj.emergency:
            traj.time = traj_np[:, 0]
            traj.s = traj_np[:, 1]
            traj.x = traj_np[:, 2]
            traj.y = traj_np[:, 3]
            traj.velocity = traj_np[:, 4]
            traj.acceleration = traj_np[:, 5]
            traj.orientation = traj_np[:, 6]
            traj.curvature = traj_np[:, 7]

        self.trajectory_np = traj_np
        self.trajectory = traj

    def reset_initial_state(self, veh, params):

        x_cog = veh.x + np.cos(veh.phi) * veh.wheel_base * 0.5
        y_cog = veh.y + np.sin(veh.phi) * veh.wheel_base * 0.5
        ref_line_proj = util.project(self.dp_env.ref_line[:, :2], [x_cog, y_cog])

        init_state = LatLonState()
        init_state.s = ref_line_proj.arc_len + veh.v * params.dead_time
        init_state.ds = veh.v
        init_state.l = self.ref_proj.distance
        self.dp_planner.traj_dp[0] = init_state
        self.dp_planner.traj_smooth[0] = init_state

        self.trajectory_np = None

    def shift_trajectory(self, env, params):
        
        if self.traj_dp is None:
            return
        
        dt_update = env.t - self.last_update_time

        # shift frenet trajectory
        
        for s in self.traj_dp.states:
            s.t -= dt_update
            s.s -= self.dp_env.ref_line_shift

        # throw away states in the past

        self.traj_dp.states = [self.traj_dp.state(0.0)] + [
                s for s in self.traj_dp.states if s.t > 0.0]

        self.dp_planner.traj_dp = self.traj_dp

        # shift smoothed trajectory start

        self.dp_planner.traj_smooth[0] = self.dp_planner.traj_smooth.lerp(dt_update)
        self.dp_planner.traj_smooth[0].t = 0.0
        self.dp_planner.traj_smooth[0].s -= self.dp_env.ref_line_shift

    def check_replan(self, env, params):

        veh = env.vehicle_state
        self.ref_proj = util.project(env.local_map.path[:, :2], [veh.x, veh.y])

        if self.emergency_counter > 0:
            self.state_reinit_msg = ""
            self.reset_initial_state(veh, params)
            return True

        # check if reset requested

        if not veh.automated:
            if env.t - self.last_replan_time >= 1.0:
                self.emergency_counter = 0
                self.state_reinit_msg = ""
                self.reset_initial_state(veh, params)
                return True

        reset_required = self.reset_counter != env.reset_counter
        self.reset_counter = env.reset_counter

        if self.traj_dp is None or reset_required:
            self.emergency_counter = 0
            self.state_reinit_msg = ""
            self.reset_initial_state(veh, params)
            return True

        # reset if too far from trajectory
        
        x_cog = veh.x + np.cos(veh.phi) * veh.wheel_base * 0.5
        y_cog = veh.y + np.sin(veh.phi) * veh.wheel_base * 0.5
        d_traj = np.linalg.norm([self.trajectory.x[0] - x_cog,
                                 self.trajectory.y[0] - y_cog])
        if abs(d_traj) > params.d_reinit:
            self.state_reinit_msg = ("Warning: Planner reinit, distance to trajectory too high"
                                     + f"#{int(time.time()/5)*5}")
            self.reset_initial_state(veh, params)
            return True

        # replan because traj is not long enough

        if len(self.traj_dp.states) < params.cpp.t_steps:
            return True

        # replan because timeout

        if env.t - self.last_replan_time > params.replan_time_step:
            return True

        # replan because trajectory invalid

        self.dp_planner.reinit_buffers(params.cpp)
        self.traj_dp = self.dp_planner.reeval_traj(
                self.traj_dp, self.dp_env.cpp_env)

        if not self.is_traj_valid(self.traj_dp):
            return True

        return False

    def write_debug_data(self, t, params, veh):

        with self.lock_shared():
            dbg = self.shared.debug.planner

            dbg.reinit_msg = self.state_reinit_msg
            dbg.runtime_dp = self.runtime_dp

        if not params.write_debug_data:
            return

        with self.lock_shared():
            dbg = self.shared.debug.planner

            dbg.traj_dp = self.dp_planner.traj_dp.as_numpy()
            dbg.traj_smooth = self.dp_planner.traj_smooth.as_numpy()
            dbg.traj_dp_cart = self.dp_planner.traj_dp_cart.as_numpy()
            dbg.traj_smooth_cart = self.dp_planner.traj_smooth_cart.as_numpy()

    def update(self, sh_env):

        env = self.env

        with sh_env.lock():
            env.t = sh_env.t
            env.reset_counter = sh_env.reset_counter
            env.vehicle_state = copy.deepcopy(sh_env.vehicle_state)
            env.local_map = copy.deepcopy(sh_env.local_map)
            env.local_map_behind = copy.deepcopy(sh_env.local_map_behind)
            env.selected_map = copy.deepcopy(sh_env.selected_map)
            env.tracks = copy.deepcopy(sh_env.tracks)
            env.predicted = copy.deepcopy(sh_env.predicted)
            env.man_time_cons = copy.deepcopy(sh_env.man_time_cons)

        params = self.update_params(env)

        update_needed = True

        if env.local_map is None:
            update_needed = False

        if env.t == self.last_update_time and not params.update_always:
            # only update if time changed
            time.sleep(0.001)
            update_needed = False

        if env.t < self.last_update_time:
            # time jumped backwards
            self.last_update_time = 0.0

        if update_needed:
            self.dp_env.update(env)

            self.shift_trajectory(env, params)
            replan = self.check_replan(env, params)
            self.update_planner(env, params, replan)
            self.update_trajectory(env, params)

            self.last_update_time = env.t

        self.write_debug_data(env.t, params, env.vehicle_state)

        return self.trajectory
