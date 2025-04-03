import copy
import itertools
import numpy as np

from tpl import util
from tpl.environment import DynamicObject, map_module

from scipy.optimize import linear_sum_assignment


class Track:

    ID_COUNTER = 10000

    def __init__(self):

        Track.ID_COUNTER += 1
        self.id = Track.ID_COUNTER

        self.t = 0.0

        # x, y, v_x, v_y
        self.state = np.zeros((4,))

        self.covar = np.diag([0.1, 0.1, 0.1, 0.1])

        self.hull = np.zeros((0, 2))
        self.hull_radius = 0.0

        self.pos_prev = np.zeros((2,))
        self.hull_prev = np.zeros((0, 2))

        self.v_abs = 0.0
        self.a_abs = 0.0

        # stores direction of last significant movement
        self.heading = None

        self.object_class = ""

        self.existence = 0.15
        self.stationary = 0.0


class TrackingModule:

    def __init__(self):

        self.d_gating = 5.0
        self.d_gating_birth = 5.0

        self.maps = []

        self.tracks = []
        self.tracks_new = []

        self.v_min = 0.5

        self.covar_meas = np.diag([0.01, 0.1])
        self.covar_proc = np.diag([0.01, 0.01, 0.02, 0.02])

        self.last_update_time = -1.0
        self.newest_det_time = 0.0

    def filter_detections(self, env):

        # collect all detections
        all_dets = env.ir_pc_dets.copy()

        # filter out older detections
        all_dets = [d for d in all_dets if d.t > self.newest_det_time]
        if len(all_dets) > 0:
            self.newest_det_time = max(d.t for d in all_dets)

        if len(all_dets) == 0:
            return []

        # filter out everything not close to path_ref
        on_path_dets = []
        for d in all_dets:
            on_any_map = False
            d.on_local_map = False

            for m in self.maps:
                proj = util.project(m.path[:, :2], d.pos)

                assoc_tolerance = d.hull_radius
                if d.object_class == "pedestrian":
                    # also associate pedestrians close to road
                    assoc_tolerance += 2.0

                # very cheap intersection test
                left_bound = m.d_left[proj.index] + assoc_tolerance
                right_bound = -m.d_right[proj.index] - assoc_tolerance
                if not right_bound < proj.distance < left_bound:
                    continue

                on_any_map = True
                if m.name == "local_map_behind":
                    d.on_local_map = True

            if on_any_map:
                on_path_dets.append(d)

        # try to fuse remaining detections
        while True:
            did_merge = False
            pairs = list(itertools.combinations(on_path_dets, 2))
            for d, o in pairs:
                if d.object_class != o.object_class:
                    continue
                if util.intersect_polygons(d.hull, o.hull):
                    d.hull = util.convex_hull(np.vstack((d.hull, o.hull)))
                    d.pos = np.mean(d.hull, axis=0)
                    d.hull_radius = np.max(np.linalg.norm(
                            d.hull - d.pos[np.newaxis, :], axis=1))
                    try:
                        on_path_dets.remove(o)
                    except ValueError:
                        pass 
                    did_merge = True
            if not did_merge:
                # merge pairwise until no merges possible
                break

        return on_path_dets

    def association(self, detections):

        assocs = {}
        unused_dets = []

        all_tracks = self.tracks + self.tracks_new
        count_tracks = len(all_tracks)
        count_dets = len(detections)

        if count_tracks == 0:
            return assocs, detections

        # this contains for each row (detection) the association costs to all tracks
        # additionally a separate column is included for each detection
        # this column can be selected if no other match is possible
        # in this case the detection is marked as unused and converted to a tentative track
        mat_costs = np.zeros((count_dets, count_tracks + count_dets)) + 1e4

        for i, det in enumerate(detections):
            for j, tr in enumerate(all_tracks):
                if tr.t >= det.t:
                    mat_costs[i, j] = 1e10
                    continue
                if tr.object_class != det.object_class:
                    mat_costs[i, j] = 1e10
                    continue
                d = np.linalg.norm(det.pos - tr.state[:2])
                if d > self.d_gating:
                    mat_costs[i, j] = 1e10
                    continue
                mat_costs[i, j] = d

        _, assignment = linear_sum_assignment(mat_costs)
        for i, track_idx in enumerate(assignment):
            if track_idx < count_tracks:
                assocs[all_tracks[track_idx].id] = detections[i]
            else:
                unused_dets.append(detections[i])

        return assocs, unused_dets

    def predict_tracks(self, dt):

        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt

        for tr in self.tracks:
            tr.state[:2] += dt * tr.state[2:]
            tr.hull += dt * tr.state[np.newaxis, 2:]
            tr.covar = F @ tr.covar @ F.T + self.covar_proc

    def update_tracks(self, t, dt, veh, assocs):

        for tr in self.tracks:
            try:
                o = assocs[tr.id]
                dt_meas = o.t - tr.t
                tr.existence = min(1.0, tr.existence + dt_meas)
            except KeyError:
                tr.existence = max(0.0, tr.existence - dt)
                continue

            tr.t = o.t

            # shenanigans to correct velocity for partially visible object hulls

            hull_min_v = (np.min(o.hull, axis=0) - np.min(tr.hull_prev, axis=0)) / dt_meas
            hull_max_v = (np.max(o.hull, axis=0) - np.max(tr.hull_prev, axis=0)) / dt_meas

            if abs(hull_min_v[0]) < abs(hull_max_v[0]):
                v_box_x = hull_min_v[0]
            else:
                v_box_x = hull_max_v[0]

            if abs(hull_min_v[1]) < abs(hull_max_v[1]):
                v_box_y = hull_min_v[1]
            else:
                v_box_y = hull_max_v[1]

            tr.state[:2] = np.mean(o.hull, axis=0)
            tr.state[2] = tr.state[2] * 0.9 + v_box_x * 0.1
            tr.state[3] = tr.state[3] * 0.9 + v_box_y * 0.1

            # kalman update step
            H = np.eye(4)[:2, :]
            S = H @ tr.covar @ H.T + self.covar_meas
            K = tr.covar @ H.T @ np.linalg.inv(S)
            Z = np.eye(4) - K @ H

            #new_pos = tr.pos_prev + np.array([v_box_x, v_box_y]) * dt_meas
            #tr.state = Z @ tr.state + K @ new_pos

            tr.covar = Z @ tr.covar
            
            v_abs = np.linalg.norm(tr.state[2:])
            a_abs = tr.a_abs * 0.9 + (v_abs - tr.v_abs) / dt_meas * 0.1

            tr.v_abs = v_abs
            tr.a_abs = a_abs

            tr.hull_prev = copy.deepcopy(o.hull)
            tr.hull = copy.deepcopy(o.hull)
            tr.hull_radius = o.hull_radius
            tr.pos_prev = copy.deepcopy(tr.state[:2])

            if tr.v_abs > self.v_min:
                tr.heading = np.arctan2(tr.state[3], tr.state[2])

            if tr.v_abs < self.v_min:
                tr.stationary = min(1.0, tr.stationary + dt_meas)
            else:
                tr.stationary = max(0.0, tr.stationary - dt_meas)

    def init_tracks(self, dt, assocs):

        confirmed_tracks = []

        # match new tracks from previous frame
        for tr in self.tracks_new:
            try:
                o = assocs[tr.id]
                dt_meas = o.t - tr.t
                tr.existence = min(1.0, tr.existence + dt_meas)
            except KeyError:
                tr.existence = max(0.0, tr.existence - dt)
                continue

            if tr.object_class == "pedestrian":
                tr.state[2:] = 0.0
            else:
                tr.state[2:] = (o.pos - tr.state[:2]) / (o.t - tr.t)
            tr.t = o.t
            tr.state[:2] = o.pos
            tr.pos_prev = copy.deepcopy(o.pos)
            tr.hull_prev = copy.deepcopy(o.hull)
            tr.hull = copy.deepcopy(o.hull)
            tr.hull_radius = o.hull_radius

            confirmed_tracks.append(tr)

        self.tracks += confirmed_tracks
        self.tracks_new = [tr for tr in self.tracks_new if tr not in confirmed_tracks]

    def create_tracks(self, dets):

        # create new tracks from remaining detections
        for o in dets:
            tr = Track()
            tr.t = o.t
            tr.state[:2] = o.pos
            tr.pos_prev = copy.deepcopy(o.pos)
            tr.hull_prev = copy.deepcopy(o.hull)
            tr.hull = copy.deepcopy(o.hull)
            tr.hull_radius = o.hull_radius
            tr.object_class = o.object_class
            if o.on_local_map:
                tr.existence = 0.15
            else:
                # be very conservative with intersecting maps
                tr.existence = 1.0
            self.tracks_new.append(tr)

    def update(self, env):

        t = env.t
        veh = env.vehicle_state
        veh_pos = np.array([veh.x, veh.y])

        cmap = env.get_current_map()
        if cmap is None or env.local_map is None:
            return

        if self.last_update_time < 0:
            dt = 0.0
        else:
            dt = t - self.last_update_time
        self.last_update_time = t

        self.maps = env.get_relevant_maps()

        self.predict_tracks(dt)

        dets_remaining = self.filter_detections(env)
        assocs, dets_remaining = self.association(dets_remaining)

        self.update_tracks(env.t, dt, env.vehicle_state, assocs)
        self.init_tracks(dt, assocs)
        self.create_tracks(dets_remaining)

        # remove tracks which did not get an update for some time
        self.tracks_new = [tr for tr in self.tracks_new if tr.existence > 0]
        self.tracks = [tr for tr in self.tracks if tr.existence > 0]

        updated_tracks = []
        for tr in self.tracks:
            with_same_id = [t for t in env.tracks.internal if t.id == tr.id]
            if len(with_same_id) > 0:
                do = with_same_id[0]
            else:
                do = DynamicObject()
            do.id = tr.id
            do.t = tr.t
            do.object_class = tr.object_class
            do.pos = tr.state[:2]
            do.v = tr.v_abs
            do.a = tr.a_abs
            if tr.heading is None:
                do.yaw = np.arctan2(tr.state[3], tr.state[2])
            else:
                do.yaw = tr.heading
            do.covar = tr.covar
            do.hull = tr.hull
            do.hull_radius = tr.hull_radius
            do.stationary = tr.stationary == 1.0
            updated_tracks.append(copy.deepcopy(do))

        # write tracks to environment
        env.tracks.internal = updated_tracks
