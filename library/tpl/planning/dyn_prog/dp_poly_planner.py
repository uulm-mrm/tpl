import time
import copy
import numba
import numpy as np
import os.path as osp

from tpl import util
from tplcpp import resample
from tplcpp import (
        DynProgEnvironment,
        DynProgEnvParams,
        DynProgPolyPlanner,
        DynProgPolyPlannerParams,
        DynProgPolyPoint,
        DynProgPolyTraj)

from tpl.planning import BasePlanner, Trajectory
from tpl.planning.utils import apply_velocity_limits, rampify_profile, curv_to_vel_profile

import objtoolbox as otb

from scipy.interpolate import interp1d, BSpline


class Params:

    def __init__(self):

        self.update_always = False

        self.a_lat_max = 2.5
        self.j_max_v_profile = 1.5
        self.j_min_v_profile = -1.5

        self.replan_time = 1.0
        self.replan_rate_max = 1.0

        self.dead_time = 0.0

        self.d_reinit_lat = 0.5

        self.write_debug_data = True

        self.dp_env = DynProgEnvParams()
        self.dp_planner = DynProgPolyPlannerParams()


class Behavior:

    def __init__(self):

        self.last_replan_time = 0.0

        self.ref_line = None
        self.traj_dp = None
        self.params = None

    def configure(self, params):
        pass

    def valid(self):
        return True


class FollowBehavior(Behavior):

    def configure(self, params):

        self.params = copy.deepcopy(params)
        self.params.dp_planner.w_l = 10.0

    def cost(self):

        if not self.valid():
            return float("inf")

        return 1000.0 - self.traj_dp.points[-1].s - self.params.dp_planner.length_veh


class EvasiveBehavior(Behavior):

    def configure(self, params):

        self.params = copy.deepcopy(params)
        self.params.dp_planner.w_l = 0.1

    def cost(self):

        if not self.valid():
            return float("inf")

        return 1000.0 - self.traj_dp.points[-1].s


class DpPolyPlanner(BasePlanner):

    def __init__(self, shared, lock_shared):

        np.seterr(divide='ignore', invalid='ignore')

        self.reset_counter = 0

        self.shared = shared
        self.lock_shared = lock_shared

        self.last_time = -1.0
        self.last_reinit_time = -1.0

        self.ref_line = None
        self.ref_line_shift = 0.0
        self.ref_line_step_size = 0.0
        self.ref_proj = None

        self.behavior_options = [
                FollowBehavior(),
                #EvasiveBehavior()
            ]
        self.behavior = self.behavior_options[0]

        self.emergency = False

        self.init_state = None
        self.traj_cart = None

        self.trajectory = Trajectory()
        self.trajectory_np = None

        self.dp_env = DynProgEnvironment()
        self.dp_planner = DynProgPolyPlanner()

        self.dt_start = 1.0
        self.t_shift = 0.0

        self.debug = False
        self.runtime_planning = 0.0
        self.runtime_environment = 0.0

        with self.lock_shared():
            self.shared.params = Params()
            self.shared.debug = otb.bundle()
            self.shared.debug.bwd_pass_query = np.array([1.0, 0.5, 0.0, 0.0])

    def update_params(self, env):

        veh = env.vehicle_state
        t_traj = env.t - self.last_reinit_time

        with self.lock_shared():
            params = self.shared.params

            length_veh = veh.rear_axis_to_front + veh.rear_axis_to_rear
            params.dp_planner.length_veh = length_veh
            params.dp_planner.width_veh = env.vehicle_state.width
            params.dp_env.dilation = np.sqrt(2.0) * env.vehicle_state.width * 0.5

            params.dp_env.dead_time = params.dead_time
            params.dp_planner.dead_time = params.dead_time

            if self.behavior.traj_dp is None:
                self.dt_start = params.dp_planner.dt
                self.t_shift = 0.0
            elif t_traj >= self.behavior.traj_dp.points[1].t:
                t_diff_tp_next = t_traj - min(params.dp_planner.dt, self.behavior.traj_dp.points[1].t)
                self.dt_start = params.dp_planner.dt - t_diff_tp_next % params.dp_planner.dt
                self.t_shift = params.dp_planner.dt
            else:
                self.dt_start = min(params.dp_planner.dt, self.behavior.traj_dp.points[1].t) - t_traj
                self.t_shift = 0.0

            params.dp_env.dt_start = 1.0 #self.dt_start
            params.dp_planner.dt_start = 1.0 #self.dt_start

            sh_params = params.deepcopy()

        params = Params()
        otb.merge(params, sh_params)

        params.dp_planner.update_step_sizes()

        return params

    def update_reference_line(self, env, params):

        # update current reference line

        if self.ref_line is not None:
            proj_prev_ref_start = util.project(
                    self.ref_line[:, :2], env.local_map.path[0, :2])
            self.ref_line_shift = round(
                    proj_prev_ref_start.arc_len 
                    / self.ref_line_step_size) * self.ref_line_step_size

        self.ref_line = np.zeros((len(env.local_map.path), 9))
        self.ref_line[:, :6] = env.local_map.path
        self.ref_line[:, 6] = env.local_map.d_left
        self.ref_line[:, 7] = env.local_map.d_right
        self.ref_line_step_size = env.local_map.step_size_ref

        safety_dist = (1.0 + params.dp_planner.length_veh) / self.ref_line_step_size
        self.ref_line[:, 5] = curv_to_vel_profile(self.ref_line[:, 4],
                                                  self.ref_line[:, 5],
                                                  params.a_lat_max)
        apply_velocity_limits(self.ref_line[:, 5],
                              env.local_map,
                              safety_dist)

        # compute profile with limited acceleration and jerk

        idxs_zero = self.ref_line[:, 5] < 1.0

        self.ref_line[:, 5] = rampify_profile(
                None,
                None,
                self.ref_line[:, 5],
                params.dp_planner.dds_min, params.dp_planner.dds_max,
                params.j_min_v_profile, params.j_max_v_profile,
                1.0,
                env.local_map.step_size_ref)[:, 0]

        self.ref_line[idxs_zero, 5] = 0.0

        # semantic info (e.g. intersections)

        for ip in env.local_map.intersection_paths:
            conflict_zone_np = np.array(ip.conflict_zone)[:2]
            conflict_zone_np -= env.local_map.idx_start_ref
            conflict_zone_np = np.clip(conflict_zone_np, 0, len(env.local_map.path) - 1)
            self.ref_line[conflict_zone_np[0]:conflict_zone_np[1], 8] = 1.0

    def update_environment(self, env, params):

        start = time.perf_counter()

        # write to environment and do actual update

        self.dp_env.reinit_buffers(params.dp_env)
        self.dp_env.set_ref_line(self.ref_line, self.ref_line_step_size)

        for obj in env.predicted:
            for pred in obj.predictions:
                pred.states[:, 0] -= params.dead_time
                self.dp_env.insert_dyn_obj(pred.states, obj.hull)

        self.dp_env.update()

        self.runtime_environment = (time.perf_counter() - start) * 1000.0

    def update_planner(self, env, params, replan):

        self.dp_planner.reinit_buffers(params.dp_planner)

        beh = None
        if replan:
            beh = self.behavior
        else:
            for b in self.behavior_options:
                if b == self.behavior:
                    continue
                if env.t - b.last_replan_time < params.replan_time:
                    beh = b
                    break

        if beh is not None:
            start = time.perf_counter()
            beh.traj_dp = self.dp_planner.update(self.init_state, self.dp_env)
            self.runtime_planning = (time.perf_counter() - start) * 1000.0

            for p in beh.traj_dp.points:
                print(p.t, p.s, p.ds, p.dds, p.l, p.dl, p.ddl, p.cost)
    
            beh.last_replan_time = env.t

    def update_trajectory(self, params):

        traj_resampled = self.behavior.traj_dp.point_at(np.arange(
            self.behavior.traj_dp.points[0].t,
            (params.dp_planner.t_steps-1) * params.dp_planner.dt,
            params.dp_planner.dt_cart))

        self.traj_cart = self.dp_planner.frenet_to_cartesian(
                traj_resampled, self.dp_env.ref_line)

        traj_np = self.traj_cart.as_numpy()
        traj_np[:, 0] += self.last_reinit_time + params.dead_time

        if self.trajectory_np is None:
            self.trajectory_np = traj_np
        self.trajectory_np[:, 6] = np.unwrap(
                self.trajectory_np[:, 6], period=np.pi*2.0)

        interp_traj = interp1d(self.trajectory_np[:, 0],
                               self.trajectory_np,
                               axis=0,
                               fill_value='extrapolate')
        ts = np.arange(self.last_reinit_time,
                       self.last_reinit_time + params.dead_time,
                       params.dp_planner.dt_cart)
        traj_dead_time = interp_traj(ts)
        traj_np = np.concatenate((traj_dead_time, traj_np), axis=0)

        traj = Trajectory()
        traj.time = traj_np[:, 0]
        traj.s = traj_np[:, 1]
        traj.x = traj_np[:, 2]
        traj.y = traj_np[:, 3]
        traj.velocity = traj_np[:, 4]
        traj.acceleration = traj_np[:, 5]
        traj.orientation = traj_np[:, 6]
        traj.curvature = traj_np[:, 7]
        traj.detail = otb.bundle()
        traj.detail.emergency = False

        self.trajectory_np = traj_np
        self.trajectory = traj

    def reset_initial_state(self, veh, params):

        x_cog = veh.x + np.cos(veh.phi) * veh.wheel_base * 0.5
        y_cog = veh.y + np.sin(veh.phi) * veh.wheel_base * 0.5
        ref_line_proj = util.project(self.ref_line[:, :2], [x_cog, y_cog])

        self.init_state = DynProgPolyPoint()
        self.init_state.t = 0.0
        self.init_state.s = ref_line_proj.arc_len + veh.v * params.dead_time
        self.init_state.ds = veh.v
        self.init_state.dds = 0.0
        self.init_state.l = self.ref_proj.distance
        self.init_state.dl = 0.0
        self.init_state.ddl = 0.0

        self.dt_start = params.dp_planner.dt
        self.t_shift = 0.0
        params.dp_env.dt_start = params.dp_env.dt
        params.dp_planner.dt_start = params.dp_planner.dt

        self.trajectory_np = None

    def update_initial_state(self, env, params):

        veh = env.vehicle_state
        self.ref_proj = util.project(env.local_map.path[:, :2], [veh.x, veh.y])

        t_traj = env.t - self.last_reinit_time
        self.last_reinit_time = env.t

        # check if reset requested

        if not veh.automated:
            self.reset_initial_state(veh, params)
            return True
        
        reset_required = self.reset_counter != env.reset_counter
        self.reset_counter = env.reset_counter

        if self.behavior.traj_dp is None or reset_required:
            self.reset_initial_state(veh, params)
            return True

        # reset if too far from trajectory
        
        pos_traj = np.vstack([self.trajectory.x, self.trajectory.y]).T
        x_cog = veh.x + np.cos(veh.phi) * veh.wheel_base * 0.5
        y_cog = veh.y + np.sin(veh.phi) * veh.wheel_base * 0.5
        d_lat_traj = util.project(pos_traj, [x_cog, y_cog]).distance
        if abs(d_lat_traj) > params.d_reinit_lat:
            self.reset_initial_state(veh, params)
            return True

        # shift trajectory

        self.init_state = self.behavior.traj_dp.point_at(t_traj)
        self.init_state.t = 0.0
        self.init_state.s -= self.ref_line_shift

        traj_dp_shifted = [self.init_state]
        for s in self.behavior.traj_dp.points[1:]:
            shifted_state = self.behavior.traj_dp.point_at(s.t)
            shifted_state.t -= t_traj
            shifted_state.s -= self.ref_line_shift
            if shifted_state.t > 0.0:
                traj_dp_shifted.append(shifted_state)

        for i, s in enumerate(traj_dp_shifted):
            self.behavior.traj_dp[i] = s

        #self.dp_planner.traj_smooth[0] = self.traj_smooth.lerp(t_traj)
        #self.dp_planner.traj_smooth[0].t = 0.0
        #self.dp_planner.traj_smooth[0].s -= self.ref_line_shift

        # replan because timeout

        if env.t - self.behavior.last_replan_time >= params.replan_time:
            return True

        #self.behavior.configure(params)
        #self.dp_planner.reinit_buffers(self.behavior.params.dp_planner)
        #self.behavior.traj_dp = self.dp_planner.reeval_traj(self.behavior.traj_dp, self.dp_env)

        #if not self.behavior.valid():
        #    return True

        if params.update_always:
            return True

        return False

    def write_debug_data(self, t, params, veh):

        with self.lock_shared():
            dbg = self.shared.debug

            dbg.ref_line = self.ref_line

            dist_map = self.dp_env.get_dist_map()
            dist_map_flat = np.zeros((dist_map.shape[0]*dist_map.shape[2] + dist_map.shape[2],
                                      dist_map.shape[1]))
            r = 0
            for i in range(dist_map.shape[2]):
                dist_map_flat[r:r+dist_map.shape[0], :] = dist_map[:, :, i]
                r += dist_map.shape[0]
                dist_map_flat[r, :] = 0.0
                r += 1
            dbg.dist_map_flat = dist_map_flat

            dbg.intersection_ref_line = self.ref_line[self.ref_line[:, 8] > 0.0]

            occ_map = self.dp_env.get_occ_map_cartesian()
            dbg.occ_cart_map = occ_map
            
            dbg.traj_dp = self.behavior.traj_dp.as_numpy()
            dbg.traj_cart = self.traj_cart.as_numpy()

            dbg.runtime_planning = self.runtime_planning
            dbg.runtime_environment = self.runtime_environment

    def update(self, sh_env):

        with sh_env.lock():
            # TODO: extract only the things we actually need
            env = copy.deepcopy(sh_env)

        if env.local_map is None:
            update_needed = False

        params = self.update_params(env)

        update_needed = True
        if self.last_time == env.t and not params.update_always:
            # only update if time changed
            time.sleep(0.001)
            update_needed = False

        if self.last_time > env.t:
            # time jumped backwards
            self.last_reinit_time = 0.0
            for b in self.behavior_options:
                b.last_replan_time = 0.0
        
        if update_needed:
            self.last_time = env.t
            self.update_reference_line(env, params)
            self.update_environment(env, params)
            replan = self.update_initial_state(env, params)
            self.update_planner(env, params, replan)
            self.update_trajectory(params)

        if params.write_debug_data:
            self.write_debug_data(env.t, params, env.vehicle_state)

        return self.trajectory
