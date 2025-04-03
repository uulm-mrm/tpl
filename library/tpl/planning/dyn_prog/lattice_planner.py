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
        DynProgLatPlanner,
        DynProgLonPlanner,
        DynProgLonPlannerParams,
        PolyLatPlanner,
        PolyLatPlannerParams,
        PolyLatTrajPoint,
        LatState,
        LonState)

from tpl.planning import BasePlanner, Trajectory
from tpl.planning.utils import rampify_profile

import objtoolbox as otb

from scipy.interpolate import interp1d, make_interp_spline


class Params:

    def __init__(self):

        self.update_always = False

        self.dead_time = 0.3

        self.use_lat_sampling_planner = False

        self.a_lat_max = 2.5

        self.d_reinit_lat = 0.2
        self.reinit_time = 1.0

        self.write_debug_data = True

        self.dyn_prog = DynProgLonPlannerParams()
        self.dyn_prog_env = DynProgEnvParams()
        self.lat_sampling = PolyLatPlannerParams()


class LatticePlanner(BasePlanner):

    def __init__(self, shared, lock_shared):

        np.seterr(divide='ignore', invalid='ignore')

        self.shared = shared
        self.lock_shared = lock_shared

        self.reset_counter = 0

        self.last_time = -1.0
        self.last_reinit_time = -1.0

        self.did_reinit_from_vehicle = False

        self.ref_line = None
        self.ref_proj = None

        self.traj_lat = None
        self.traj_lat_start = LatState()
        self.traj_lat_init_dt = 1.0
        self.path = None
        self.prev_path = None
        self.traj_states_lat = None
        self.prev_traj_states_lat = None

        self.cp_lat = None 
        self.prev_cp_lat = None 

        self.traj_lon = None
        self.traj_lon_start = LonState()

        self.trajectory = Trajectory()
        self.prev_trajectory = Trajectory()

        self.veh_state_emergency = None

        self.dp_lat_planner = DynProgLatPlanner()
        self.lat_sampling_planner = PolyLatPlanner()
        self.dp_lon_planner = DynProgLonPlanner()
        self.dp_env = DynProgEnvironment()

        self.debug = False

        with self.lock_shared():
            self.shared.params = Params()
            self.shared.debug = otb.bundle()

    def write_debug(self, key, value):

        if not self.debug:
            return

        with self.lock_shared():
            setattr(self.shared.debug, key, value)

    def update_environment(self, env, params):

        start = time.perf_counter()

        # write to environment and do actual update

        self.dp_env.reinit_buffers(params.dyn_prog_env)
        self.dp_env.set_ref_line(self.ref_line, self.ref_line_step_size)

        for obj in env.predicted:
            for pred in obj.predictions:
                pred.states[:, 0] -= params.dead_time
                self.dp_env.insert_dyn_obj(pred.states, obj.hull, obj.crossing)

        self.dp_env.update()

        self.runtime_environment = (time.perf_counter() - start) * 1000.0

    def convert_to_cart(self, traj_states_lat, dp_params):

        lat_s = traj_states_lat[:, 0]
        lat_l = traj_states_lat[:, 1]
        lat_dl = traj_states_lat[:, 2]
        lat_ddl = traj_states_lat[:, 3]
        lat_k = traj_states_lat[:, 6]

        ixs = interp1d(self.ref_line[:, 3], self.ref_line[:, 0])(lat_s)
        iys = interp1d(self.ref_line[:, 3], self.ref_line[:, 1])(lat_s)
        iphis = interp1d(self.ref_line[:, 3], self.ref_line[:, 2])(lat_s)
        ik = interp1d(self.ref_line[:, 3], self.ref_line[:, 4], kind="zero")(lat_s)
        iv = interp1d(self.ref_line[:, 3], self.ref_line[:, 5])(lat_s)
        
        lat_dl_ds = lat_dl / dp_params.s_lat_step_size
        lat_ddl_ds = lat_ddl / dp_params.s_lat_step_size**2
        
        rel_phis = np.arctan(lat_dl_ds)

        path = np.zeros((len(traj_states_lat), 6))

        path[:, 0] = ixs - np.sin(iphis) * lat_l
        path[:, 1] = iys + np.cos(iphis) * lat_l
        path[:, 2] = iphis + rel_phis
        path[:, 3] = lat_s
        path[:, 4] = lat_k
        path[:, 5] = iv

        return path

    def update_lat_sampling_planner(self, veh, cmap, params):

        self.lat_sampling_planner.reinit_buffers(params.lat_sampling)
        self.traj_lat = self.lat_sampling_planner.update(
                self.traj_lat_start,
                self.dp_env)
        end = time.time()

        traj_states_lat = []
        for tp in self.traj_lat.points:
            istate = np.array([
                    tp.s,
                    tp.l,
                    tp.dl,
                    tp.ddl,
                    tp.dddl,
                    tp.distance,
                    tp.k,
                    tp.t,
                ])
            traj_states_lat.append(istate)
        traj_states_lat = np.array(traj_states_lat)

        self.prev_traj_states_lat = self.traj_states_lat
        if self.prev_traj_states_lat is not None:
            self.prev_traj_states_lat[:, 7] -= 1.0

        self.traj_states_lat = traj_states_lat

        self.prev_path = copy.deepcopy(self.path)

        self.path = np.zeros((len(self.traj_lat.points), 6))
        self.path[:, 0] = [p.x for p in self.traj_lat.points]
        self.path[:, 1] = [p.y for p in self.traj_lat.points]
        self.path[:, 2] = [p.heading for p in self.traj_lat.points]
        self.path[:, 3] = [p.s for p in self.traj_lat.points]
        self.path[:, 4] = [p.k for p in self.traj_lat.points]
        self.path[:, 5] = [p.v for p in self.traj_lat.points]
        self.path[:, 5] = self.path[:, 5].round()

        self.path[:, 0] += self.dp_env.ref_line.x_offset
        self.path[:, 1] += self.dp_env.ref_line.y_offset

        # resample_path

        dp_params = params.dyn_prog

        dp_params.path_steps = int(np.ceil(dp_params.s_max/dp_params.path_step_size))

        rsi = resample(self.path[:, :2],
                       dp_params.path_step_size,
                       dp_params.path_steps)

        rs_path = util.interp_resampled_path(self.path,
                                             rsi,
                                             dp_params.path_step_size,
                                             dp_params.path_steps,
                                             False,
                                             False)

        rs_path[:, 5] = rampify_profile(
                None,
                None,
                rs_path[:, 5],
                -2.5, 2.5, -1.5, 1.5, 1.0,
                dp_params.path_step_size)[:, 0]

        # resample frenet coordinate info

        alpha = rsi[:, 2]
        prev_idx = rsi[:, 3].astype('int')
        next_idx = rsi[:, 4].astype('int')

        rs_frenet_s = self.path[prev_idx, 3] * (1.0 - alpha) + self.path[next_idx, 3] * alpha
        rs_path[:, 4] = self.path[prev_idx, 4] * (1.0 - alpha) + self.path[next_idx, 4] * alpha

        l_interp = interp1d(traj_states_lat[:, 0],
                            traj_states_lat[:, 1],
                            fill_value="extrapolate")
        dl_interp = interp1d(traj_states_lat[:, 0],
                             traj_states_lat[:, 2],
                             fill_value="extrapolate")

        ext_path = np.hstack([rs_path[:, :2],
                             rs_frenet_s.reshape((-1, 1)),
                             l_interp(rs_frenet_s).reshape((-1, 1)),
                             dl_interp(rs_frenet_s).reshape((-1, 1)),
                             rs_path[:, 5].reshape((-1, 1))])

        return ext_path

    def update_planner(self, veh, cmap, params):

        if params.use_lat_sampling_planner:
            self.ext_path = self.update_lat_sampling_planner(veh, cmap, params)
        else:
            self.dp_lat_planner.reinit_buffers(dp_params)
            self.traj_lat = self.dp_lat_planner.update(self.traj_lat_start, self.traj_lat_init_dt, self.dp_env)

            # resample lateral trajectory with finer temporal resolution

            traj_states_lat = []
            ts = np.arange(0, self.traj_lat.states[-1].t, 0.1)
            for t in ts:
                istate = self.traj_lat.state(t).as_numpy()
                traj_states_lat.append(istate)
            traj_states_lat = np.array(traj_states_lat)

            # compute cartesian coordinates

            self.prev_path = self.path
            self.path = self.convert_to_cart(traj_states_lat, dp_params)

            cp_lat = np.array([tp.as_numpy() for tp in self.traj_lat.states])
            self.prev_cp_lat = self.cp_lat
            self.cp_lat = self.convert_to_cart(cp_lat, dp_params)

            dp_params.path_steps = int(np.ceil(dp_params.s_max/dp_params.path_resample_step))

            # resample computed path in equidistant steps

            rsi = resample(self.path[:, :2],
                           dp_params.path_resample_step,
                           dp_params.path_steps)

            rs_path = util.interp_resampled_path(self.path,
                                                 rsi,
                                                 dp_params.path_resample_step,
                                                 dp_params.path_steps,
                                                 False)

            # additionally resample frenet coordinate info

            alpha = rsi[:, 2]
            prev_idx = rsi[:, 3].astype('int')
            next_idx = rsi[:, 4].astype('int')

            rs_frenet_s = self.path[prev_idx, 3] * (1.0 - alpha) + self.path[next_idx, 3] * alpha

            l_interp = interp1d(traj_states_lat[:, 0],
                                traj_states_lat[:, 1],
                                fill_value="extrapolate")
            dl_interp = interp1d(traj_states_lat[:, 0],
                                 traj_states_lat[:, 2],
                                 fill_value="extrapolate")

            self.ext_path = np.hstack([rs_path[:, :2],
                                       rs_frenet_s.reshape((-1, 1)),
                                       l_interp(rs_frenet_s).reshape((-1, 1)),
                                       dl_interp(rs_frenet_s).reshape((-1, 1))])

        # compute longitudinal trajectory / velocity profile

        self.dp_lon_planner.reinit_buffers(params.dyn_prog)
        self.traj_lon = self.dp_lon_planner.update(self.traj_lon_start, self.ext_path, self.dp_env)

        ts = np.arange(0, self.traj_lon.states[-1].t, 0.1)

        # sample intermediate states
        traj_states = []
        for t in ts:
            istate = self.traj_lon.state(t).as_numpy()
            traj_states.append(istate)
        traj_states = np.array(traj_states)

        interp_x = interp1d(self.path[:, 3], self.path[:, 0], fill_value="extrapolate")
        interp_y = interp1d(self.path[:, 3], self.path[:, 1], fill_value="extrapolate")
        interp_k = interp1d(self.path[:, 3], self.path[:, 4], fill_value="extrapolate")

        interp_phis = np.zeros((len(traj_states[:, 0]),))
        for i, s in enumerate(traj_states[:, 0]):
            interp_phis[i] = util.lerp(s, self.path[:, 3], self.path[:, 2], angle=True)

        traj = Trajectory()

        traj.time = ts + self.last_reinit_time
        traj.s = traj_states[:, 0]
        traj.x = interp_x(traj_states[:, 0])
        traj.y = interp_y(traj_states[:, 0])
        traj.orientation = interp_phis
        traj.curvature = interp_k(traj_states[:, 0])
        traj.velocity = traj_states[:, 1]
        traj.acceleration = traj_states[:, 2]

        return traj

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

        safety_dist = (1.0 + params.dyn_prog.length_veh * 0.5) / self.ref_line_step_size
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
                params.dyn_prog.a_min, params.dyn_prog.a_max,
                -1.5, 1.5,
                1.0,
                env.local_map.step_size_ref)[:, 0]

        self.ref_line[idxs_zero, 5] = 0.0

        # semantic info (e.g. intersections)

        for ip in env.local_map.intersection_paths:
            conflict_zone_np = np.array(ip.conflict_zone)[:2]
            conflict_zone_np -= env.local_map.idx_start_ref
            conflict_zone_np = np.clip(conflict_zone_np, 0, len(env.local_map.path) - 1)
            self.ref_line[conflict_zone_np[0]:conflict_zone_np[1], 8] = 1.0

        # make sure to cover the entire range of the ref_line

        #params.dyn_prog_env.ds_max = np.ceil(np.max(self.ref_line[:, 5]))
        #params.dyn_prog_env.l_min = np.floor(np.min(-self.ref_line[:, 7]))
        #params.dyn_prog_env.l_max = np.ceil(np.max(self.ref_line[:, 6]))
        #params.dyn_prog.v_max = np.ceil(np.max(self.ref_line[:, 5]))
        #params.dyn_prog.l_min = np.floor(np.min(-self.ref_line[:, 7]))
        #params.dyn_prog.l_max = np.ceil(np.max(self.ref_line[:, 6]))

    def update_initial_state(self, env, params, force_reinit=False):

        veh = env.vehicle_state
        self.ref_proj = util.project(env.local_map.path[:, :2], [veh.x, veh.y])

        self.did_reinit_from_vehicle = False

        t_traj = env.t - self.last_reinit_time

        # step 1: determine if we actually need to reinit

        reinit = force_reinit
        if force_reinit:
            reinit_from_traj = True
        else:
            reinit_from_traj = False

        reset_required = self.reset_counter != env.reset_counter
        self.reset_counter = env.reset_counter
        if reset_required:
            reinit = True
            reinit_from_traj = False

        if self.ref_line is None:
            reinit = True
        if self.traj_lon is None:
            reinit = True
        else:
            if t_traj >= params.reinit_time:
                reinit = True
                reinit_from_traj = True

            d_lat_traj = util.project(self.path[:, :2], [veh.x, veh.y]).distance
            if abs(d_lat_traj) > params.d_reinit_lat:
                reinit = True
                reinit_from_traj = False

        if not reinit:
            return False

        if reinit_from_traj:
            s_diff = util.project(self.ref_line[:, :2], [veh.x, veh.y]).arc_len - self.path[0, 3]
        else:
            s_diff = 0.0

        # step 2: reinitialize ref line

        self.update_reference_line(env, params)

        #self.ref_line = env.local_map.path
        #self.ref_line_step_size = env.local_map.step_size_ref

        #safety_dist = (1.0 + params.dyn_prog.length_veh) / self.ref_line_step_size
        #apply_velocity_limits(self.ref_line[:, 5], env.local_map, safety_dist)

        # step 3: reinitialize initial states

        if reinit_from_traj:

            ref_line_proj = util.project(self.ref_line[:, :2], [veh.x, veh.y])

            cmap_route_step_size = 0.5
            
            if params.use_lat_sampling_planner:
                reinit_pos = self.traj_lat.points[0].s + s_diff
                reinit_pos = max(self.traj_lat.poly.x0_, min(self.traj_lat.poly.x1_, reinit_pos))
                self.traj_lat_start = PolyLatTrajPoint()
                self.traj_lat_start.t = 0.0
                self.traj_lat_start.s = ref_line_proj.arc_len
                self.traj_lat_start.l = self.traj_lat.poly.f(reinit_pos)
                self.traj_lat_start.dl = self.traj_lat.poly.df(reinit_pos)
                self.traj_lat_start.ddl = self.traj_lat.poly.ddf(reinit_pos)
                self.traj_lat_start.dddl = self.traj_lat.poly.dddf(reinit_pos)
                self.traj_lat_start.v = veh.v
            else:
                path_time = s_diff / params.dyn_prog.s_lat_step_size

                lat_ss = np.array([tp.s for tp in self.traj_lat.states])
                lat_ts = np.array([tp.t for tp in self.traj_lat.states])
                path_time = interp1d(lat_ss, lat_ts, fill_value="extrapolate")(s_diff)

                if path_time >= lat_ts[1]:
                    self.traj_lat_init_dt = params.dyn_prog.dt - (path_time - lat_ts[1]) % params.dyn_prog.dt
                else:
                    self.traj_lat_init_dt = lat_ts[1] - path_time

                self.traj_lat_start = self.traj_lat.state(path_time)
                self.traj_lat_start.t = 0.0
                self.traj_lat_start.s = 0.0
            
            self.traj_lon_start = self.traj_lon.state(t_traj)
            self.traj_lon_start.t = 0.0
            self.traj_lon_start.s = 0.0

            if t_traj >= self.traj_lon.states[1].t:
                params.dyn_prog.dt_start = params.dyn_prog.dt - (t_traj - self.traj_lon.states[1].t) % params.dyn_prog.dt
            else:
                params.dyn_prog.dt_start = self.traj_lon.states[1].t - t_traj
        else:
            ref_line_proj = util.project(self.ref_line[:, :2], [veh.x, veh.y])

            if params.use_lat_sampling_planner:
                self.traj_lat_start = PolyLatTrajPoint()
                self.traj_lat_start.t = 0.0
                self.traj_lat_start.s = ref_line_proj.arc_len
                self.traj_lat_start.l = self.ref_proj.distance
                self.traj_lat_start.dl = np.tan(veh.phi - self.ref_proj.angle)
                self.traj_lat_start.ddl = 0.0
                self.traj_lat_start.dddl = 0.0
                self.traj_lat_start.v = veh.v
            else:
                self.traj_lat_start = LatState()
                self.traj_lat_start.t = 0.0
                self.traj_lat_start.s = 0.0
                self.traj_lat_start.l = self.ref_proj.distance
                self.traj_lat_start.dl = np.tan(veh.phi - self.ref_proj.angle) * params.dyn_prog.s_lat_step_size

            self.traj_lon_start = LonState()
            self.traj_lon_start.t = 0.0
            self.traj_lon_start.s = 0.0
            self.traj_lon_start.v = round(veh.v)
            self.traj_lon_start.a = round(veh.a)
            self.traj_lon_start.j = 0.0
            
            self.did_reinit_from_vehicle = True

        # store time for next reinit step

        self.last_reinit_proj = self.ref_proj
        self.last_reinit_time = env.t

        return True

    def current_traj_valid(self):

        eval_traj = self.dp_lon_planner.reeval_traj(self.traj_lon, self.ext_path, self.dp_env)
        valids = [tp.constr_valid for tp in eval_traj.states]

        print(valids)

        return all(valids)

    def write_debug_data(self, t, dp_params, veh):

        current_k = interp1d(
                self.trajectory.time,
                self.trajectory.curvature,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate")(t)

        current_phi = interp1d(
                self.trajectory.time,
                self.trajectory.orientation,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate")(t)

        current_a = interp1d(
                self.trajectory.time,
                self.trajectory.acceleration,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate")(t)
        current_v = interp1d(
                self.trajectory.time,
                self.trajectory.velocity,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate")(t)

        s_diff = util.project(self.ref_line[:, :2], [veh.x, veh.y]).arc_len
        current_k = interp1d(
                self.path[:, 3],
                self.path[:, 4],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate")(s_diff)

        #current_phi = interp1d(
        #        self.path[:, 3],
        #        self.path[:, 2],
        #        kind="linear",
        #        bounds_error=False,
        #        fill_value="extrapolate")(s_diff)

        with self.lock_shared():
            dbg = self.shared.debug

            dbg.current_t = float(t)
            dbg.current_a = float(current_a)
            dbg.current_k = float(current_k)
            dbg.current_v = float(current_v)

            dbg.traj_states_lat = self.traj_states_lat
            dbg.prev_traj_states_lat = self.prev_traj_states_lat

            dbg.current_phi = float(current_phi)

            #dbg.distance_map = otb.bundle()
            #occ = self.dp_env.get_dist_map()
            #for i in range(occ.shape[2]):
            #    setattr(dbg.distance_map, f'time_step_{i}', occ[:, :, i, 0])

            #dbg.path_distance_map = otb.bundle()
            #occ = self.dp_env.get_dist_map_path(dp_params.path_steps)
            #for i in range(occ.shape[1]):
            #    setattr(dbg.path_distance_map, f'time_step_{i}', occ[:, i, 0].reshape((-1, 1)))

            dbg.ref_line = self.ref_line

            dbg.path = self.path
            dbg.prev_path = self.prev_path

            dbg.ext_path = self.ext_path

            #dbg.cp_lat = self.cp_lat
            #dbg.prev_cp_lat = self.prev_cp_lat

            #traj_states_lon = []
            #for t in np.arange(0.0, self.traj_lon.states[-1].t, 0.1):
            #    traj_states_lon.append(self.traj_lon.state(t).as_numpy())
            #traj_states_lon = np.array(traj_states_lon)

            #dbg.traj_lon = traj_states_lon

            #dbg.prev_trajectory = self.prev_trajectory

    def update(self, sh_env):

        with sh_env.lock():
            # TODO: extract only the things we actually need
            env = copy.deepcopy(sh_env)

        veh = env.vehicle_state

        if env.local_map is None:
            time.sleep(0.001)
            return self.trajectory

        with self.lock_shared():
            params = self.shared.params

            length_veh = env.vehicle_state.rear_axis_to_front + env.vehicle_state.rear_axis_to_rear

            params.dyn_prog.length_veh = length_veh
            params.dyn_prog.width_veh = env.vehicle_state.width
            params.dyn_prog.path_steps = int(
                    np.ceil(params.dyn_prog.s_max/params.dyn_prog.path_step_size))

            params.ref_line_steps = env.local_map.steps_ref
            params.ref_line_step_size = env.local_map.step_size_ref

            params.lat_sampling.k_abs_max = np.tan(veh.delta_max) / veh.wheel_base
            params.lat_sampling.rear_axis_to_rear = veh.rear_axis_to_rear
            params.lat_sampling.rear_axis_to_front = veh.rear_axis_to_front
            params.lat_sampling.width_veh = veh.width 
            params.lat_sampling.length_veh = length_veh

            sh_params = params.deepcopy()

        params = Params()
        otb.merge(params, sh_params)

        if self.last_time == env.t and not params.update_always:
            # only update if time changed
            time.sleep(0.001)
            return self.trajectory
        self.last_time = env.t

        dyn_objs = env.get_all_tracks()

        replan = self.update_initial_state(env, params)
        self.update_environment(env, params)

        #if not replan and not self.current_traj_valid():
        #    self.update_initial_state(env, params, force_reinit=True)
        #    self.update_environment(pred_dyn_objs, params.dyn_prog_env)
        #    replan = True

        if replan or params.update_always:
            self.prev_trajectory = self.trajectory
            self.trajectory = self.update_planner(veh, env.local_map, params)

        if params.write_debug_data:
            self.write_debug_data(env.t, params.dyn_prog, env.vehicle_state)

        return self.trajectory
