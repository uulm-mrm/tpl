import copy
import time
import numpy as np
import os.path as osp

import objtoolbox as otb

from tpl import util
from tplcpp import (
        PolySamplingPlanner as CppPlanner,
        PolySamplingParams,
        PolySamplingTrajPoint
    )

from tpl.planning import BasePlanner, Trajectory
from tpl.planning.utils import rampify_profile, curv_to_vel_profile


class Params:

    def __init__(self):

        self.a_min = -2.5
        self.a_max = 2.5

        self.j_min = -1.5
        self.j_max = 1.5
        
        self.max_lat_acc = 2.5

        self.path_sampling_step = 0.5
        self.path_length = 250

        self.poly_params = PolySamplingParams()

        self.dist_reset = 2.0


class PolySamplingPlanner(BasePlanner):

    def __init__(self, shared, lock_shared):
        """
        Find a path using polynomial sampling, according to the concept from
        "Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame"
        Werling et al., 2010.
        """

        self.shared = shared
        self.lock_shared = lock_shared

        self.runtime = 0.0

        self.trajectory = Trajectory()
        self.poly_traj = None

        self.last_time = 0.0
        self.last_update_time = 0.0

        self.init_state = PolySamplingTrajPoint()

        self.poly_sampling_planner = CppPlanner()

        with self.lock_shared():
            self.shared.params = Params()

    def update(self, sh_env):

        with sh_env.lock():
            # TODO: extract only the things we actually need
            env = copy.deepcopy(sh_env)

        # update params
        with self.lock_shared():
            params = self.shared.params.deepcopy()

        poly_params = PolySamplingParams()
        otb.merge(poly_params, params.poly_params)

        if self.last_time == env.t:
            # only update if time changed
            time.sleep(0.001)
            return self.trajectory

        self.last_time = env.t

        veh = env.vehicle_state

        cmap = env.local_map
        if cmap is None:
            return self.trajectory

        poly_params.rear_axis_to_rear = veh.rear_axis_to_rear
        poly_params.rear_axis_to_front = veh.rear_axis_to_front
        poly_params.width_ego = veh.width + 1.0
        
        dt_replan = env.t - self.last_update_time

        if dt_replan >= poly_params.dt or dt_replan < 0.0:

            start_time = time.perf_counter()

            ref_proj = util.project(cmap.path[:, :2], [veh.x, veh.y])

            path = util.resample_path(cmap.path,
                                      params.path_sampling_step,
                                      params.path_length,
                                      start_index=ref_proj.start,
                                      zero_vel_at_end=True)

            path[:, 5] = curv_to_vel_profile(path[:, 4], path[:, 5], params.max_lat_acc)
            path[:, 5] = rampify_profile(None, None,
                                         path[:, 5],
                                         params.a_min, params.a_max,
                                         params.j_min, params.j_max,
                                         1.0,
                                         params.path_sampling_step)[:, 0]

            dyn_objs = env.get_all_tracks()

            for do in dyn_objs:
                self.poly_sampling_planner.add_obstacle(
                        do.pos[0],
                        do.pos[1],
                        do.yaw,
                        do.v,
                        do.evade,
                        do.hull)

            path_traj = np.vstack([self.trajectory.x, self.trajectory.y]).T
            path_proj = util.project(path_traj, (veh.x, veh.y))

            do_reset = env.reset_required
            do_reset |= abs(path_proj.distance) > params.dist_reset
            do_reset |= self.poly_traj is None

            if do_reset:
                start_tp = PolySamplingTrajPoint()
                start_tp.s = 0.0
                start_tp.d = ref_proj.distance
                start_tp.s_d = veh.v
                start_tp.s_dd = veh.a
            else:
                start_idx = max(0, min(len(self.poly_traj.points)-1,
                                       int(dt_replan / poly_params.dt)))
                start_tp = self.poly_traj.points[start_idx]
                start_tp.s = 0.0
        
            self.poly_traj = self.poly_sampling_planner.update(start_tp, path, poly_params)
            traj_points = self.poly_traj.points

            self.trajectory = Trajectory()
            self.trajectory.time = env.t + np.array([tp.t for tp in traj_points])
            self.trajectory.x = [tp.x for tp in traj_points]
            self.trajectory.y = [tp.y for tp in traj_points]
            self.trajectory.s = np.insert(np.cumsum(
                    [tp.ds for tp in traj_points]), 0, 0.0, axis=0)[:-1]
            self.trajectory.velocity = [tp.s_d for tp in traj_points]
            self.trajectory.acceleration = [tp.s_dd for tp in traj_points]
            self.trajectory.orientation = [tp.yaw for tp in traj_points]
            self.trajectory.curvature = [tp.c for tp in traj_points]

            self.last_update_time = env.t

            self.runtime = duration = time.perf_counter() - start_time

        return self.trajectory
