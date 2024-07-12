import copy
import numpy as np

from tpl import util
from tpl.environment import DynamicObject, map_module


class Track:

    ID_COUNTER = 10000

    def __init__(self):

        Track.ID_COUNTER += 1
        self.id = Track.ID_COUNTER

        self.t = 0.0

        # x, y, v_x, v_y
        self.state = np.zeros((4,))

        self.covar = np.diag([1.0, 1.0, 1.0, 1.0])
        self.covar_meas = np.diag([0.1, 0.1])
        self.covar_proc = np.diag([0.1, 0.1, 0.5, 0.5])

        self.hull = np.zeros((0, 2))
        self.hull_radius = 0.0

        # length of velocity vector
        self.v_abs = 0.0
        # stores direction of last significant movement
        self.heading = None

        self.object_class = ""


class TrackingModule:

    def __init__(self):

        self.d_gating = 5.0
        self.d_gating_birth = 5.0

        self.track_kill_time = 0.5

        self.maps = []

        self.tracks = []
        self.tracks_new = []

        self.last_update_time = -1.0
        self.newest_det_time = 0.0

    def filter_detections(self, env):

        cmap = env.get_current_map()

        # collect all detections
        all_dets = []
        for src in env.ir_pc_dets.__slots__:
            all_dets += env.ir_pc_dets[src]

        # filter out older detections
        all_dets = [d for d in all_dets if d.t > self.newest_det_time]
        if len(all_dets) > 0:
            self.newest_det_time = max(d.t for d in all_dets)

        # filter out everything not close to path_ref
        on_path_dets = []
        for d in all_dets:
            for m in self.maps:
                proj = util.project(m.path[:, :2], d.pos)
                left_bound = m.d_left[proj.index] + d.hull_radius
                right_bound = -m.d_right[proj.index] - d.hull_radius
                if not right_bound < proj.distance < left_bound:
                    continue
                boundary_polygon = map_module.get_map_boundary_polygon(m)
                if util.intersect_polygons(boundary_polygon, d.hull):
                    on_path_dets.append(d)
                    break

        return on_path_dets

    def associate_track(self, det, tracks, gating_dist):

        best_tr = None
        best_dist = float("inf")

        for tr in tracks:
            if tr.object_class != det.object_class:
                continue
            d = np.linalg.norm(det.pos - tr.state[:2])
            if d > gating_dist:
                continue
            if d < best_dist:
                best_tr = tr
                best_dist = d

        return best_tr

    def predict_tracks(self, dt):

        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt

        for tr in self.tracks:
            tr.state[:2] += dt * tr.state[2:]
            tr.hull += dt * tr.state[np.newaxis, 2:]
            tr.covar = F @ tr.covar @ F.T + tr.covar_proc

    def update_tracks(self, dets):

        unused_dets = []

        # association and kalman update
        for o in dets:
            tr = self.associate_track(o, self.tracks, self.d_gating)
            if tr is None:
                unused_dets.append(o)
                continue

            # kalman update step
            H = np.eye(4)[:2, :]
            S = H @ tr.covar @ H.T + tr.covar_meas
            K = tr.covar @ H.T @ np.linalg.inv(S)
            Z = np.eye(4) - K @ H

            # use newest convex hulls
            if tr.t < o.t:
                tr.hull = copy.deepcopy(o.hull)
                tr.hull_radius = o.hull_radius

            tr.t = o.t
            tr.state = Z @ tr.state + K @ o.pos
            tr.covar = Z @ tr.covar
            tr.v_abs = np.linalg.norm(tr.state[2:])
            if tr.v_abs > 0.5:
                tr.heading = np.arctan2(tr.state[3], tr.state[2])

        return unused_dets

    def init_tracks(self, dets):

        unused_dets = []

        # match new tracks from previous frame
        for o in dets:
            tr = self.associate_track(o, self.tracks_new, self.d_gating_birth)
            if tr is None:
                unused_dets.append(o)
                continue

            tr = copy.deepcopy(tr)

            vel = (o.pos - tr.state[:2]) / (o.t - tr.t)
            tr.t = o.t
            tr.state[:2] = o.pos
            tr.state[2:] = vel
            tr.hull = copy.deepcopy(o.hull)
            tr.hull_radius = o.hull_radius

            self.tracks.append(tr)

        return unused_dets

    def create_tracks(self, dets):

        self.tracks_new = []

        # create new tracks from remaining detections
        for o in dets:
            tr = Track()
            tr.t = o.t
            tr.state[:2] = o.pos
            tr.hull = copy.deepcopy(o.hull)
            tr.object_class = o.object_class
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

        if len(dets_remaining) > 0:
            dets_remaining = self.update_tracks(dets_remaining)
            dets_remaining = self.init_tracks(dets_remaining)
            self.create_tracks(dets_remaining)

        # remove tracks which did not get an update for some time
        self.tracks = [tr for tr in self.tracks
                       if env.t - tr.t < self.track_kill_time]

        # write tracks to environment
        env.tracks.internal = []
        for tr in self.tracks:
            do = DynamicObject()
            do.id = tr.id
            do.t = tr.t
            do.object_class = tr.object_class
            do.pos = tr.state[:2]
            do.v = tr.v_abs
            if tr.heading is None:
                do.yaw = np.arctan2(tr.state[3], tr.state[2])
            else:
                do.yaw = tr.heading
            do.hull = tr.hull
            do.hull_radius = tr.hull_radius
            env.tracks.internal.append(copy.deepcopy(do))
