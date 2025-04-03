import time
import numpy as np
import objtoolbox as otb

from tplcpp import DynProgEnvironment, DynProgEnvParams

from tpl import util
from tpl.planning.utils import rampify_profile
from tpl.environment import prediction_module


class Params:

    def __init__(self):

        self.write_debug_data = True

        self.dead_time = 0.0

        self.a_lat_max = 2.5

        self.a_max_v_ref = 3.0
        self.a_min_v_ref = -3.0
        self.j_max_v_ref = 1.5
        self.j_min_v_ref = -1.5

        self.t_dist_on_map = 0.5
        self.t_dist_crossing = 3.0

        self.cpp = DynProgEnvParams()


class DpEnv:

    def __init__(self, shared, lock_shared):

        self.shared = shared 
        self.lock_shared = lock_shared

        self.ref_line = None
        self.ref_proj = None

        self.last_update_time = 0.0
        self.dt_start = None

        self.ref_line_shift = 0.0
        self.ref_line_step_size = 0.0

        self.cpp_env = DynProgEnvironment()

        self.runtime_environment = 0.0

        with self.lock_shared():
            self.shared.params.env = otb.bundle()
            self.shared.params.env = Params()
            self.shared.debug.env = otb.bundle()

    def update_params(self, env):

        with self.lock_shared():
            params = self.shared.params.env

            params.cpp.dilation = np.sqrt(2.0) * env.vehicle_state.width * 0.5

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

        # compute profile with limited acceleration and jerk

        idxs_zero = self.ref_line[:, 5] < 1.0

        self.ref_line[:, 5] = rampify_profile(
                None,
                None,
                self.ref_line[:, 5],
                params.a_min_v_ref, params.a_max_v_ref,
                params.j_min_v_ref, params.j_max_v_ref,
                1.0,
                env.local_map.step_size_ref)[:, 0]

        self.ref_line[idxs_zero, 5] = 0.0

        # semantic info (e.g. intersections)

        for ip in env.local_map.intersection_paths:
            if not ip.stop_proj.in_bounds:
                return
            conflict_zone_np = np.array([ip.stop_proj.end, ip.stop_proj.end + 10])
            self.ref_line[conflict_zone_np[0]:conflict_zone_np[1], 8] = 1.0

        # make sure to cover the entire range of the ref_line

        params.cpp.l_min = np.floor(np.min(-self.ref_line[:, 7]))
        params.cpp.l_max = np.ceil(np.max(self.ref_line[:, 6]))

    def update_environment(self, env, params):

        start = time.perf_counter()

        # write to environment and do actual update

        self.cpp_env.reinit_buffers(params.cpp)
        self.cpp_env.set_ref_line(self.ref_line, self.ref_line_step_size)

        maps = {m.uuid: m for m in env.get_relevant_maps()}

        for obj in env.predicted:
            for pred in obj.predictions:
                try:
                    m = maps[pred.uuid_assoc_map]
                except KeyError:
                    continue
                on_local_map = m.name == "local_map_behind"

                ts = np.array([0.0, *(params.cpp.dt_start + pred.states[:-1, 0])])
                ts += params.dead_time

                if on_local_map:
                    sweep_length = params.t_dist_on_map
                else:
                    sweep_length = params.t_dist_crossing
                    if env.vehicle_state.v > 20.0 or (obj.a is not None and obj.a > 1.0):
                        sweep_length = 4.0
                    elif env.vehicle_state.v > 15.0:
                        sweep_length = 3.0
                    elif env.vehicle_state.v > 10.0:
                        sweep_length = 2.0

                geom = util.gen_prediction_geometry(
                        pred.states,
                        obj.hull,
                        m.path[:, :2],
                        ts,
                        station_step_size=5.0,
                        expansion_rate=0.0,
                        sweep_length=sweep_length)

                geom[:, 2] -= params.dead_time

                self.cpp_env.insert_geometry(geom, obj.stationary)

        self.cpp_env.update()

        self.runtime_environment = (time.perf_counter() - start) * 1000.0

    def write_debug_data(self):

        with self.lock_shared():
            dbg = self.shared.debug.env

            dbg.runtime_environment = self.runtime_environment

            dbg.ref_line = self.ref_line

            dbg.occ_map_cart = self.cpp_env.get_occ_map_cartesian()

            dbg.occ_map = np.vstack(np.moveaxis(
                    self.cpp_env.get_occ_map(), 2, 0))
            dbg.dist_map_lon = np.vstack(np.moveaxis(
                    self.cpp_env.get_dist_map_lon(), 2, 0))
            dbg.dir_dist_map = self.cpp_env.get_dist_map_dir(0)

    def update(self, env):

        params = self.update_params(env)

        self.update_reference_line(env, params)
        self.update_environment(env, params)

        if params.write_debug_data:
            self.write_debug_data()

        self.last_update_time = env.t
