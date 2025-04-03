import copy
import time
import numba
import numpy as np
import os.path as osp

import objtoolbox as otb

from tpl import util
from tpl.planning import BasePlanner, Trajectory
from tpl.planning.utils import rampify_profile
from tpl.environment import map_module, EnvironmentState

from tplcpp import (
        RefLine,
        IdmSamplingParams, 
        IdmSamplingState,
        IdmSamplingRefState,
        IdmSamplingPlanner as _IdmSamplingPlanner,
    )


class Params:

    def __init__(self):

        self.a_min = -2.5
        self.a_max = 2.5
        self.j_min = -1.5
        self.j_max = 1.5

        self.cpp = IdmSamplingParams()


class IdmSamplingPlanner(BasePlanner):

    def __init__(self, shared, lock_shared):

        self.shared = shared
        self.lock_shared = lock_shared

        self.reset_counter = 0
        self.invalid_counter = 0
        self.emergency_counter = 0
        self.reverse_counter = 0

        self.enable_reverse = False

        self.last_update_time = 0.0

        self.trajectory = Trajectory()

        self.planner = _IdmSamplingPlanner()
        self.traj_cpp = None

        self.env = EnvironmentState()

        self.v_ref = None

        with self.lock_shared():
            self.shared.params = Params()

    def update_closest_inters_point(self, cmap, veh, params):

        veh_pos = np.array([veh.x, veh.y])

        ip_close = None
        d_ip_close = 1.0e6

        for ip in cmap.intersection_paths:
            if not ip.stop_proj.in_bounds:
                continue
            if abs(ip.stop_proj.distance) > 1.0:
                continue

            d = ip.stop_proj.arc_len 
            if d < d_ip_close:
                ip_close = ip
                d_ip_close = d

        params.cpp.d_next_inters_point = d_ip_close

    def update(self, sh_env):

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

            if self.enable_reverse:
                sh_env.local_map.position_vehicle = 10.0
            else:
                sh_env.local_map.position_vehicle = 1.0

        veh = env.vehicle_state

        params = Params()
        with self.lock_shared():
            sh_params = self.shared.params
            sh_params.cpp.dead_time = veh.dead_time_steer
            sh_params.cpp.wheel_base = veh.wheel_base
            sh_params.cpp.width_veh = veh.width
            sh_params.cpp.length_veh = veh.rear_axis_to_rear + veh.rear_axis_to_front
            sh_params.cpp.radius_veh = np.sqrt(
                    (sh_params.cpp.width_veh * 0.5)**2 
                    + (sh_params.cpp.length_veh * 0.5)**2)
            sh_params.cpp.dist_front_veh = veh.rear_axis_to_front
            sh_params.cpp.dist_back_veh = veh.rear_axis_to_rear

            otb.merge(params, sh_params)

        cmap = env.local_map
        if cmap is None:
            return self.trajectory

        dt_replan = env.t - self.last_update_time
        if dt_replan == 0.0:
            return self.trajectory

        self.last_update_time = env.t

        for i in range(len(env.local_map.path)):
            if env.local_map.path[i, 5] == 0.0:
                break

        if self.reset_counter != env.reset_counter or dt_replan < 0.0 or not veh.automated:
            self.reset_counter = env.reset_counter
            self.invalid_counter = 0
            self.emergency_counter = 0
            self.traj_cpp = None
            self.planner.reset()

        dt_replan = max(0.0, dt_replan)

        self.update_closest_inters_point(cmap, veh, params)

        if self.v_ref is None:
            self.v_ref = np.array([[veh.v, veh.a]])
        else:
            self.v_ref[0, :] = self.v_ref[cmap.shift_idx_start_ref]

        ref_line = cmap.path
        self.v_ref = rampify_profile(self.v_ref[0, 0], self.v_ref[0, 1],
                                     ref_line[:, 5],
                                     params.a_min, params.a_max,
                                     params.j_min, params.j_max,
                                     1.0,
                                     cmap.step_size_ref)
        self.v_ref[cmap.path[:, 5] < 0.1, :] = 0.0
        ref_line[:, 5] = self.v_ref[:, 0]
        ref_proj = util.project(ref_line[:, :2], [veh.x, veh.y])

        ref_line_arr = np.zeros((len(env.local_map.path), 9))
        ref_line_arr[:, :6] = ref_line
        ref_line_arr[:, 6] = cmap.d_left
        ref_line_arr[:, 7] = cmap.d_right
        ref_line_cpp = RefLine.fromarray(ref_line_arr, cmap.step_size_ref, True)

        for obj in env.predicted:
            for pred in obj.predictions:
                on_local_map = pred.uuid_assoc_map == env.local_map_behind.uuid
                self.planner.insert_dyn_obj(pred.states, obj.hull, on_local_map)

        if self.traj_cpp is None or not veh.automated:
            init_state = IdmSamplingState()
            init_ref_state = IdmSamplingRefState()
            init_ref_state.t = 0.0
            init_ref_state.x = veh.x
            init_ref_state.y = veh.y
            init_ref_state.heading = veh.phi
            init_ref_state.v = veh.v
            init_ref_state.a = veh.a
            init_ref_state.s = ref_proj.arc_len
            init_ref_state.l = ref_proj.distance
            self.prev_init_ref_state = init_ref_state
        else:
            init_state = self.traj_cpp.lerp(dt_replan)
            init_ref_state = self.traj_cpp.lerp_ref(params.cpp.dead_time + dt_replan)
            init_ref_state.t = params.cpp.dead_time
            proj_init = util.project(ref_line[:, :2], (init_ref_state.x, init_ref_state.y))
            init_ref_state.s = proj_init.arc_len
            init_ref_state.l = proj_init.distance

        init_state.t = 0.0
        init_state.x = veh.x
        init_state.y = veh.y
        init_state.heading = veh.phi
        init_state.steer_angle = veh.delta
        init_state.v = veh.v
        init_state.a = veh.a
        init_state.s = ref_proj.arc_len
        init_state.l = ref_proj.distance

        if veh.v < 0.1:
            self.reverse_counter = min(100, self.reverse_counter + 1)
        if veh.v > 2.0:
            self.reverse_counter = 0
        self.enable_reverse = self.reverse_counter == 100
        if self.enable_reverse:
            params.cpp.d_safe_min = 5.0
        params.cpp.enable_reverse = self.enable_reverse

        t = time.perf_counter()
        self.traj_cpp = self.planner.update(
                init_state,
                init_ref_state,
                dt_replan,
                ref_line_cpp,
                params.cpp)

        traj_np = self.traj_cpp.as_numpy()
        traj_ref_np = self.traj_cpp.ref_as_numpy()
        planning_time = time.perf_counter() - t

        if self.traj_cpp.invalid:
            self.invalid_counter += 1
        else:
            self.invalid_counter = 0

        if self.invalid_counter > 50:
            self.invalid_counter = 0
            self.emergency_counter = 50
        self.emergency_counter = max(0, self.emergency_counter - 1)

        emergency = self.emergency_counter > 0
        if not emergency:
            self.trajectory = Trajectory()
            self.trajectory.time = env.t + traj_np[:, 0]
            self.trajectory.s = [0, *np.cumsum(np.linalg.norm(
                np.diff(traj_np[:, 1:3], axis=0), axis=1))]
            self.trajectory.x = traj_np[:, 1]
            self.trajectory.y = traj_np[:, 2]
            self.trajectory.orientation = traj_np[:, 3]
            self.trajectory.curvature = np.tan(traj_np[:, 4]) / veh.wheel_base
            self.trajectory.velocity = traj_np[:, 5]
            self.trajectory.acceleration = traj_np[:, 6]
        else:
            self.trajectory = Trajectory()
            self.trajectory.emergency = True
            self.traj_cpp = None
            self.planner.reset()

        all_traj_points = []
        for tr in self.planner.trajs:
            all_traj_points.append(tr.as_numpy()[:, 1:3])
        all_traj_points = np.vstack(all_traj_points).reshape((-1, 2))

        with self.lock_shared():
            dbg = otb.bundle()
            dbg.d_left = cmap.boundary_left
            dbg.d_right = cmap.boundary_right
            dbg.planning_time = t
            dbg.traj_np = traj_np
            dbg.traj_ref_np = traj_ref_np
            dbg.ref_line = ref_line
            dbg.all_traj_points = all_traj_points
            self.shared.debug = dbg

        return self.trajectory
