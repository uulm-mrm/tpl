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
        PolyLatPlanner,
        PolyLatPlannerParams,
        PolyLatTrajPoint,
        DynProgLonPlanner,
        DynProgLonPlannerParams,
        LonState
    )

from tpl.environment import EnvironmentState
from tpl.environment.map_module import curv_to_vel_profile

from tpl.planning import BasePlanner, Trajectory
from tpl.planning.dyn_prog.dp_env import DpEnv
from tpl.planning.utils import rampify_profile


class Params:

    def __init__(self):

        self.write_debug_data = True

        self.update_always = False

        self.replan_time_step = 0.1

        self.dead_time = 0.0

        self.dist_path_fix_min = 5.0
        self.dist_path_fix = 1.0

        self.d_reinit = 2.0

        self.cpp_lat = PolyLatPlannerParams()
        self.cpp_lon = DynProgLonPlannerParams()


class PolyLatDpLonPlanner(BasePlanner):

    def __init__(self, shared, lock_shared):

        np.seterr(divide='ignore', invalid='ignore')

        self.shared = shared
        self.lock_shared = lock_shared
        
        self.reset_counter = 0

        self.invalid_counter = 0
        self.emergency_counter = 0

        self.last_update_time = -1.0
        self.last_replan_time = -1.0
        self.dt_start = None

        self.state_reinit_msg = ""

        self.ref_proj = None

        self.traj_lat = None
        self.path = None
        self.traj_lon = None

        self.trajectory = Trajectory()
        self.trajectory_np = None

        self.poly_lat_start = PolyLatTrajPoint()
        self.dp_lon_start = LonState()

        self.poly_lat = PolyLatPlanner()
        self.dp_lon = DynProgLonPlanner()

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

        return all(s.constr < 0.1 for s in traj.states[1:-1])

    def update_params(self, env):

        veh = env.vehicle_state

        with self.lock_shared():
            params = self.shared.params.planner

            length_veh = veh.rear_axis_to_front + veh.rear_axis_to_rear

            params.cpp_lat.length_veh = length_veh
            params.cpp_lat.width_veh = env.vehicle_state.width
            params.cpp_lon.length_veh = length_veh
            params.cpp_lon.width_veh = env.vehicle_state.width

            dt_update = env.t - self.last_update_time
            if self.dt_start is None:
                self.dt_start = params.cpp_lon.dt
            else:
                self.dt_start = (self.dt_start - dt_update) % params.cpp_lon.dt
            if self.dt_start == 0.0:
                self.dt_start = params.cpp_lon.dt
            params.cpp_lon.dt_start = self.dt_start

            sh_params = params.deepcopy()

        params = Params()
        otb.merge(params, sh_params)

        return params

    def update_planner(self, env, params):

        start = time.perf_counter()

        veh = env.vehicle_state

        self.poly_lat.reinit_buffers(params.cpp_lat)
        new_traj_lat = self.poly_lat.update(
                self.poly_lat_start,
                self.dp_env.cpp_env)

        if self.traj_lat is None:
            self.traj_lat = new_traj_lat
        else:
            self.traj_lat.insert_after_station(self.poly_lat_start.s, new_traj_lat)

        # resample and reorganize lateral trajectory into path
                
        self.path = self.traj_lat.lerp(np.arange(
            0.0,
            params.cpp_lon.path_steps * params.cpp_lon.path_step_size, 
            params.cpp_lon.path_step_size))
        self.path = self.path[:, [7, 8, 5, 1, 11, 6, 10]]

        self.path[:, 5] = curv_to_vel_profile(self.path[:, 4],
                                              self.path[:, 5],
                                              params.cpp_lat.a_lat_abs_max)
        self.path[:, 5] = rampify_profile(None,
                                          None,
                                          self.path[:, 5],
                                          params.cpp_lon.a_min, params.cpp_lon.a_max,
                                          params.cpp_lon.j_min, params.cpp_lon.j_max,
                                          1.0,
                                          1.0)[:, 0]

        # update projection onto new path

        self.traj_point_prev = util.lerp(env.t + params.dead_time,
                                         self.trajectory_np[:, 0],
                                         self.trajectory_np[:, 2:4])
        path_proj = util.project(self.path[:, :2], self.traj_point_prev)
        self.dp_lon_start.s = path_proj.arc_len

        self.dp_lon.reinit_buffers(params.cpp_lon)
        self.traj_lon = self.dp_lon.update(
                self.dp_lon_start,
                self.path,
                self.dp_env.cpp_env)

        self.runtime_dp = (time.perf_counter() - start) 

        self.last_replan_time = env.t

    def update_trajectory(self, env, params):

        ts = np.arange(
            0.0,
            (params.cpp_lon.t_steps-1) * params.cpp_lon.dt,
            0.1)

        lon_states = self.traj_lon.state(ts)
        lat_states = self.traj_lat.lerp(lon_states[:, 0])

        traj_np = np.zeros((len(ts), 8))
        traj_np[:, 0] = ts + env.t + params.dead_time
        traj_np[:, 1] = lat_states[:, 5]
        traj_np[:, 2] = lat_states[:, 7]
        traj_np[:, 3] = lat_states[:, 8]
        traj_np[:, 4] = lon_states[:, 1]
        traj_np[:, 5] = lon_states[:, 2]
        traj_np[:, 6] = lat_states[:, 9]
        traj_np[:, 7] = lat_states[:, 11]

        ts = np.arange(env.t,
                       env.t + params.dead_time,
                       0.1)
        if len(ts) > 0:
            traj_dead_time = util.lerp(
                    ts,
                    self.trajectory_np[:, 0],
                    self.trajectory_np)
            traj_dead_time[:, 6] = util.lerp(
                    ts,
                    self.trajectory_np[:, 0],
                    self.trajectory_np[:, 6],
                    angle=True)
            traj_np = np.concatenate((traj_dead_time, traj_np), axis=0)

        if not self.is_traj_valid(self.traj_lon):
            self.invalid_counter += 1
        else:
            self.invalid_counter = 0

        if self.invalid_counter > 10:
            self.invalid_counter = 0
            self.emergency_counter = 50
        self.emergency_counter = max(0, self.emergency_counter - 1)

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

    def reset_initial_state(self, env, params):

        veh = env.vehicle_state

        # reset trajectories

        self.traj_lon = None
        self.traj_lat = None
        self.path = None
        self.trajectory_np = None
        self.trajectory = None

        # calculate constant velocity trajectory

        ts = np.arange(0.0, 10.0, 0.1)
        self.trajectory_np = np.zeros((len(ts), 8))
        self.trajectory_np[:, 0] = ts + env.t
        self.trajectory_np[:, 1] = ts * veh.v
        self.trajectory_np[:, 2] = veh.x + np.cos(veh.phi) * (ts * veh.v + veh.wheel_base*0.5)
        self.trajectory_np[:, 3] = veh.y + np.sin(veh.phi) * (ts * veh.v + veh.wheel_base*0.5)
        self.trajectory_np[:, 4] = veh.v
        self.trajectory_np[:, 5] = 0.0
        self.trajectory_np[:, 6] = veh.phi
        self.trajectory_np[:, 7] = 0.0

        # lateral start state

        ref_proj = util.project(self.dp_env.ref_line[:, :2], [veh.x, veh.y])

        self.poly_lat_start = PolyLatTrajPoint()
        self.poly_lat_start.t = 0.0
        self.poly_lat_start.s = 0.0
        self.poly_lat_start.l = ref_proj.distance
        self.poly_lat_start.dl = np.tan(veh.phi - ref_proj.angle)
        self.poly_lat_start.ddl = 0.0
        self.poly_lat_start.dddl = 0.0
        self.poly_lat_start.v = veh.v

        # longitudinal start state

        self.dp_lon_start.t = 0.0
        self.dp_lon_start.s = 0.0
        self.dp_lon_start.v = veh.v
        self.dp_lon_start.a = min(params.cpp_lon.a_max, max(0.0, veh.a))
        self.dp_lon_start.j = 0.0

    def shift_trajectory(self, env, params):
        
        if self.traj_lon is None:
            return

        # shift lateral trajectory and set lateral trajectory start

        shift = env.local_map.shift_idx_start_ref * env.local_map.step_size_ref

        for p in self.traj_lat.points:
            p.s -= shift
        self.traj_lat.points = [p for p in self.traj_lat.points if p.s >= 0.0]

        self.poly_lat_start = self.traj_lat.lerp(
                params.dist_path_fix_min + params.dist_path_fix * env.vehicle_state.v)

        self.poly_lat_start.v = env.vehicle_state.v

        # shift previous trajectory stations for new ref_line

        self.trajectory_np[:, 1] -= shift

        # longitudinal

        dt_update = env.t - self.last_update_time

        for s in self.traj_lon.states:
            s.t -= dt_update

        self.traj_lon.states = [self.traj_lon.state(0.0)] + [
                s for s in self.traj_lon.states if s.t > 0.0]

        self.dp_lon_start = self.traj_lon.states[0]
        self.dp_lon_start.a = min(params.cpp_lon.a_max, max(
            params.cpp_lon.a_min, self.dp_lon_start.a))

    def check_replan(self, env, params):

        veh = env.vehicle_state
        self.ref_proj = util.project(env.local_map.path[:, :2], [veh.x, veh.y])

        # check if reset requested

        if not veh.automated:
            self.state_reinit_msg = ""
            self.reset_initial_state(env, params)
            if env.t - self.last_replan_time >= 1.0:
                return True

        reset_required = self.reset_counter != env.reset_counter
        self.reset_counter = env.reset_counter

        if self.traj_lon is None or reset_required or self.trajectory.emergency:
            self.state_reinit_msg = ""
            self.reset_initial_state(env, params)
            return True

        # reset if too far from trajectory
        
        x_cog = veh.x + np.cos(veh.phi) * veh.wheel_base * 0.5
        y_cog = veh.y + np.sin(veh.phi) * veh.wheel_base * 0.5
        d_traj = np.linalg.norm([self.trajectory.x[0] - x_cog,
                                 self.trajectory.y[0] - y_cog])
        if abs(d_traj) > params.d_reinit:
            self.state_reinit_msg = ("Warning: Planner reinit, distance to trajectory too high"
                                     + f"#{int(time.time()/5)*5}")
            self.reset_initial_state(env, params)
            return True

        # replan because traj is not long enough

        if len(self.traj_lon.states) < params.cpp_lon.t_steps:
            return True

        # replan because timeout

        if env.t - self.last_replan_time > params.replan_time_step:
            return True

        # replan because trajectory invalid

        self.dp_lon.reinit_buffers(params.cpp_lon)
        self.traj_lon = self.dp_lon.reeval_traj(
                self.traj_lon,
                self.path,
                self.dp_env.cpp_env)

        if not self.is_traj_valid(self.traj_lon):
            return True

        return False

    def write_debug_data(self, t, params, veh):

        if not params.write_debug_data:
            return

        if self.traj_lon is None or self.traj_lat is None:
            # did not plan yet
            return

        with self.lock_shared():
            dbg = self.shared.debug.planner

            dbg.traj_point_prev = self.traj_point_prev

            dbg.traj_lon = self.traj_lon.as_numpy()
            dbg.traj_lat = self.traj_lat.as_numpy()
            dbg.path = self.path

            dbg.runtime_dp = self.runtime_dp

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
            if self.check_replan(env, params):
                self.update_planner(env, params)
            self.update_trajectory(env, params)

            self.last_update_time = env.t

        self.write_debug_data(env.t, params, env.vehicle_state)

        return self.trajectory
