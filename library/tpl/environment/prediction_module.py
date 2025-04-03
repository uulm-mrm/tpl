import numba
import numpy as np

from tpl import util
from tpl.environment import map_module, Prediction

from scipy.interpolate import interp1d


@numba.njit(fastmath=True)
def lerp(x, xs, ys, angle=False, clip_alpha=False):
    """
    Simple linear interpolation function.
    Assumes xs values are scalar and equally spaced.
    Values in ys can be vector-valued.
    Out of bound values will be set to values at the boundary.
    """

    l = len(ys)

    if l == 0:
        return None
    elif l == 1:
        return ys[0]

    dx = xs[1] - xs[0]

    q = (x - xs[0]) / dx
    start = int(max(0, min(l-2, np.floor(q))))
    end = int(max(0, min(l-1, np.ceil(q))))
    alpha = q - start
    if clip_alpha:
        alpha = max(0.0, min(1.0, alpha))

    if angle:
        return ys[start] + util.short_angle_dist(ys[start], ys[end]) * alpha
    else:
        return ys[start] * (1.0 - alpha) + ys[end] * alpha


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


@numba.njit(fastmath=True, cache=True)
def calc_pred_cv_path(x0, d0, s0, path, dt, horizon, clip_pos_alpha=False):

    s = s0
    d = d0

    heading = lerp(s, path[:, 3], path[:, 2], angle=True, clip_alpha=True)
    cos_angle_dist = np.cos(x0[3] - heading)

    l = int(horizon / dt) + 1
    x = np.zeros((l, x0.shape[0]))
    x[0, :] = x0
    x[0, 3] = heading
    x[0, 4] *= cos_angle_dist

    for i in range(l-1):
        s += dt * x[i, 4] 
        pos = lerp(s, path[:, 3], path[:, :2], clip_alpha=clip_pos_alpha)
        heading = lerp(s, path[:, 3], path[:, 2], angle=True, clip_alpha=True)
        pos[0] += -np.sin(heading) * d
        pos[1] += np.cos(heading) * d
        x[i+1, 0] = x[i, 0] + dt
        x[i+1, 1:3] = pos
        x[i+1, 3] = heading
        x[i+1, 4] = x[i, 4] 

    return x


class PredictionModule:

    def __init__(self):

        self.limit_assoc_vel = 1.0
        self.limit_assoc_angle = 0.9

        self.pedestrian_assoc_tol = 5.0

        self.dt_pred = 1.0
        self.horizon_pred = 10.0
        
        self.pred_margin_acc = 0.2

    def associate_maps_and_tracks(self, maps, tracks):

        for i, tr in enumerate(tracks):

            pos_mean = np.mean(tr.hull, axis=0)
            for m in maps.values():

                proj = util.project(m.path[:, :2], pos_mean)
                if not proj.in_bounds:
                    continue

                assoc_tolerance = tr.hull_radius
                if tr.object_class == "pedestrian":
                    # also associate pedestrians close to road
                    assoc_tolerance += self.pedestrian_assoc_tol

                left_bound = m.d_left[proj.index] + assoc_tolerance
                right_bound = -m.d_right[proj.index] - assoc_tolerance
                if not right_bound < proj.distance < left_bound:
                    continue

                projs_hull = util.project(m.path[:, :2], tr.hull)
                projs_hull = [p for p in projs_hull if p.in_bounds]
                dists = np.array([p.distance for p in projs_hull])
                d_max = np.max(dists)
                d_min = np.min(dists)

                if d_max < -m.d_right[proj.index] or d_min > m.d_left[proj.index]:
                    continue

                pred = Prediction()
                pred.proj_assoc_map = proj
                pred.uuid_assoc_map = m.uuid
                pred.cos_angle_dist = np.cos(tr.yaw - proj.angle)
                tr.predictions.append(pred)

    def clean_tracks(self, env, maps, tracks):

        veh = env.vehicle_state
        keep_tracks = []
        for tr in tracks:

            if len(tr.predictions) == 0:
                continue

            reject = False
            for p in tr.predictions:
                m = maps[p.uuid_assoc_map]
                veh_proj = util.project(m.path[:, :2], [veh.x, veh.y])
                # only remove tracks if ego is actually on that map
                if (veh_proj.distance > m.d_left[veh_proj.index]
                    or veh_proj.distance < -m.d_right[veh_proj.index]):
                    continue
                dist = p.proj_assoc_map.arc_len - veh_proj.arc_len
                if dist < -3.0:
                    # remove tracks directly behind the vehicle
                    if abs(veh_proj.distance - p.proj_assoc_map.distance) < (veh.width + 0.25):
                        reject = True
                        break
                    # remove tracks behind the vehicle with higher temporal distance
                    temporal_dist = abs(dist) / max(0.001, abs(veh.v - tr.v))
                    if temporal_dist > 5.0:
                        reject = True
                        break

            if not reject:
                keep_tracks.append(tr)

        return keep_tracks

    def apply_predictions(self, maps, tracks):

        for tr in tracks:
            # determine start state
            pos_mean = np.mean(tr.hull, axis=0)
            if tr.object_class == "pedestrian":
                x0 = np.array([0.0, pos_mean[0], pos_mean[1], 0.0, 0.0])
            else:
                if tr.v < 0.5:
                    v_pred = 0.0
                else:
                    v_pred = tr.v 
                x0 = np.array([0.0, pos_mean[0], pos_mean[1], tr.yaw, v_pred])

            # do prediction

            for pred in tr.predictions:
                pred_along_map = (tr.object_class != "pedestrian"
                                  and tr.v > self.limit_assoc_vel
                                  and abs(pred.cos_angle_dist) > self.limit_assoc_angle)

                x0_p = x0.copy()
                
                if pred_along_map:
                    proj = pred.proj_assoc_map
                    m = maps[pred.uuid_assoc_map]
                    on_local_map = m.name == "local_map_behind"

                    pred.states = calc_pred_cv_path(
                            x0_p,
                            proj.distance,
                            proj.arc_len,
                            m.path,
                            self.dt_pred,
                            self.horizon_pred,
                            clip_pos_alpha=on_local_map)
                else:
                    x0_p[4] = 0.0
                    pred.states = calc_pred_cv(x0_p, self.dt_pred, self.horizon_pred)

            # remove additional prediction not along a path
            path_preds = [p for p in tr.predictions if p.states[0, 4] != 0.0]
            if len(path_preds) > 0:
                tr.predictions = path_preds

            # remove additional prediction in reverse direction
            non_reverse_preds = [p for p in tr.predictions if p.cos_angle_dist > 0.0]
            if len(non_reverse_preds) > 0:
                tr.predictions = non_reverse_preds

    def update(self, env):

        cmap = env.get_current_map()
        if cmap is None or env.local_map is None:
            return

        maps = {m.uuid: m for m in env.get_relevant_maps()}
        tracks = env.get_all_tracks()

        self.associate_maps_and_tracks(maps, tracks)
        tracks = self.clean_tracks(env, maps, tracks)
        self.apply_predictions(maps, tracks)

        env.predicted = tracks
