import numba
import numpy as np

from tpl import util
from tpl.environment import map_module, Prediction


@numba.njit(fastmath=True, cache=True)
def calc_pred_cv(x0, dt, horizon):

    l = int(horizon / dt) + 1
    x = np.zeros((l, x0.shape[0]))
    x[0, :] = x0

    for i in range(l-1):
        x[i+1, 0] = x[i, 0] + dt
        x[i+1, 1] = x[i, 1] + dt * x[i, 4] * np.cos(x[i, 3])
        x[i+1, 2] = x[i, 2] + dt * x[i, 4] * np.sin(x[i, 3])
        x[i+1, 3] = x[i, 3]
        x[i+1, 4] = x[i, 4]

    return x


@numba.njit(fastmath=True)
def calc_pred_cv_path(x0, d0, s0, path, dt, horizon):

    l = int(horizon / dt) + 1
    x = np.zeros((l, x0.shape[0]))
    x[0, :] = x0

    s = s0
    d = d0

    for i in range(l-1):
        s += dt * x[i, 4]
        pos = util.lerp(s, path[:, 3], path[:, :2])
        heading = util.lerp(s, path[:, 3], path[:, 2], angle=True)
        pos[0] += -np.sin(heading) * d
        pos[1] += np.cos(heading) * d
        x[i+1, 0] = x[i, 0] + dt
        x[i+1, 1:3] = pos
        x[i+1, 3] = heading
        x[i+1, 4] = x[i, 4]

    return x


class PredictionModule:

    def __init__(self):

        self.limit_assoc_vel = 3.6
        self.limit_assoc_angle = 0.8

        self.dt_pred = 0.5
        self.horizon_pred = 10.0

    def associate_maps_and_tracks(self, maps, tracks):

        associations = [[] for i in range(len(tracks))]
        map_counts = [0 for i in range(len(tracks))]

        for m in maps:
            boundary_polygon = map_module.get_map_boundary_polygon(m)
            for i, tr in enumerate(tracks):
                proj = util.project(m.path[:, :2], tr.pos)
                left_bound = m.d_left[proj.index] + tr.hull_radius
                right_bound = -m.d_right[proj.index] - tr.hull_radius
                if not right_bound < proj.distance < left_bound:
                    continue
                if not util.intersect_polygons(boundary_polygon, tr.hull):
                    continue
                map_counts[i] += 1
                if (tr.v > self.limit_assoc_vel 
                        and abs(np.cos(tr.yaw - proj.angle)) < self.limit_assoc_angle):
                    continue
                associations[i].append((proj, m))

        return associations, map_counts

    def clean_tracks(self, tracks, associations, map_counts):

        keep_tracks = []
        keep_associations = []
        for tr, assoc, mc in zip(tracks, associations, map_counts):
            if mc > 0:
                keep_tracks.append(tr)
                keep_associations.append(assoc)

        return keep_tracks, keep_associations

    def apply_predictions(self, tracks, associations):

        for tr, assocs in zip(tracks, associations):
            x0 = np.array([0.0, tr.pos[0], tr.pos[1], tr.yaw, tr.v])
            if len(assocs) == 0:
                states = calc_pred_cv(x0, self.dt_pred, self.horizon_pred)
                pred = Prediction()
                pred.states = states
                tr.predictions.append(pred)
            else:
                for (proj, m) in assocs:
                    states = calc_pred_cv_path(x0,
                                               proj.distance,
                                               proj.arc_len,
                                               m.path,
                                               self.dt_pred,
                                               self.horizon_pred)
                    pred = Prediction()
                    pred.states = states
                    pred.proj_assoc_map = proj
                    pred.uuid_assoc_map = m.uuid
                    tr.predictions.append(pred)

    def update(self, env):

        cmap = env.get_current_map()
        if cmap is None or env.local_map is None:
            return

        maps = env.get_relevant_maps()
        tracks = env.get_all_tracks()

        associations, maps_counts = self.associate_maps_and_tracks(maps, tracks)
        tracks, associations = self.clean_tracks(tracks, associations, maps_counts)

        self.apply_predictions(tracks, associations)

        env.predicted = tracks
