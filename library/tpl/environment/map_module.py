import os
import uuid
import traceback

import numpy as np 
import objtoolbox as otb

from tpl import util

from scipy.interpolate import UnivariateSpline, splrep, splev


class VelocityLimit:

    def __init__(self):

        self.__tag__ = "velocity_limit"
        self.uuid = uuid.uuid4().hex

        self.pos = np.array([0.0, 0.0])
        self.limit = 0.0
        self.length = 20.0
        self.min_distance = 1.0

        self.proj = util.Projection()
        self.active = True

    def __savestate__(self):

        s = self.__dict__.copy()
        del s["proj"]
        del s["active"]

        return s


class TrafficLight(VelocityLimit):

    UNKNOWN = -1
    RED = 0
    YELLOW = 1
    GREEN = 2

    def __init__(self):

        super().__init__()

        self.__tag__ = "traffic_light"
        self.uuid = uuid.uuid4().hex

        self.light_pos = np.array([0.0, 0.0])
        self.detection_radius = 1.0

        self.state = TrafficLight.UNKNOWN
        self.t_last_det = 0.0

    def __savestate__(self):

        s = super().__savestate__()
        s["light_pos"] = self.light_pos
        s["detection_radius"] = self.detection_radius

        return s


class CrossWalk(VelocityLimit):

    def __init__(self):

        super().__init__()

        self.__tag__ = "cross_walk"
        self.uuid = uuid.uuid4().hex

        self.corners = np.zeros((0,))
        self.limit = 10.0

    def __savestate__(self):

        s = super().__savestate__()
        s["corners"] = self.corners

        return s


class TurnIndPoint:

    OFF = 0
    RIGHT = 1
    LEFT = 1

    def __init__(self):

        self.__tag__ = "turn_ind_point"
        self.uuid = uuid.uuid4().hex

        self.pos = np.array([0.0, 0.0])
        self.dir = TurnIndPoint.OFF
        """off: 0, right: -1, left: 1"""

        self.activation_radius = 2.0


class Checkpoint(VelocityLimit):

    def __init__(self):

        super().__init__()

        self.__tag__ = "checkpoint"
        self.uuid = uuid.uuid4().hex

        self.length = 10.0
        self.active = False

        self.name = ""
        self.pos = np.array([0.0, 0.0])
        self.activation_radius = 5.0


class MapSwitchPoint:

    def __init__(self):

        self.__tag__ = "map_switch_point"
        self.uuid = uuid.uuid4().hex

        self.pos = np.array([0.0, 0.0])

        self.trigger_divisor = 1
        self.triggers = 0

        self.activation_radius = 5.0
        self.in_radius = False

        self.target_uuid = ""

    def __savestate__(self):

        s = self.__dict__.copy()
        del s["in_radius"]

        return s


class IntersectionPath:

    def __init__(self, pos=np.zeros((2,)), edit_path=False):

        self.__tag__ = "intersection_path"
        self.uuid = uuid.uuid4().hex

        self.pos = pos
        self.stop_pos = pos + np.array([2.0, 2.0])
        self.stop_proj = util.Projection()

        self.intersection_map_uuid = ""
        self.map_segment: Map = None
        self.map_segment_step_size = 2.0
        self.offset_path_begin = -20
        self.offset_path_end = 20

        self.t_min = float("inf")
        self.gap_acceptance = 5.0
        self.gap_hysteresis = 2.0

        self.stop = True
        self.v_max_approach = 3.0
        self.stop_always = False
        self.stop_for_any  = False
        self.d_min_stopped = 5.0
        self.stopped = False

    def __savestate__(self):

        d = self.__dict__.copy()
        del d["t_min"]
        del d["map_segment"]
        del d["stop"]
        del d["stopped"]

        return d


class Map:

    def __init__(self, name="unnamed_map"):

        self.__tag__ = "map"

        self.name = name
        self.uuid = uuid.uuid4().hex

        self.velocity_limits = []
        self.turn_ind_points = []
        self.map_switch_points = []
        self.intersection_paths = []

        # dims: x, y, d_left, d_right, speed_limit
        self.control_points = np.zeros((0, 5))
        self.spline_degree = 1
        self.smoothing = 0.0
        # step size for discretization
        self.step_size_discr = 0.5
        self.closed_path = False

        # these arrays are calculated with update_map_path(...):

        # discretized ref path, dims: x, y, orientation, s, curvature, speed_limit
        self.path = None
        # discretized points of right boundary, dims: x, y
        self.boundary_left = None
        # discretized points of left boundary, dims: x, y
        self.boundary_right = None
        # discretized absolute distance to left boundary, dims: d_left
        self.d_left = None
        # discretized absolute distance to right boundary, dims: d_right
        self.d_right = None

        # DEPRECATED
        self.route = None

    def __savestate__(self):

        d = self.__dict__.copy()
        del d["path"]
        del d["boundary_left"]
        del d["boundary_right"]
        del d["d_left"]
        del d["d_right"]
        del d["route"]

        return d

    def __str__(self):

        return self.name


class LocalMap(Map):

    def __init__(self):

        super().__init__("local_map")

        # how manys steps to sample the global map for
        self.steps_ref = 400
        # how large the sampling steps along the global map are
        self.step_size_ref = 0.5
        # the start index on the global map
        self.idx_start_ref = 0
        # the previous shift of the start index on the global map
        self.shift_idx_start_ref = 0


def copy_map_segment(dst_map, src_map, step_size, steps, start_idx):

    steps = min(len(src_map.path), steps)

    try:
        resample_info = util.resample(src_map.path[:, :2],
                                      step_size,
                                      steps,
                                      start_idx,
                                      closed=src_map.closed_path)
    except RuntimeError:
        return False

    dst_map.path = util.interp_resampled_path(
            src_map.path,
            resample_info,
            step_size,
            steps,
            zero_vel_at_end=not src_map.closed_path,
            closed=src_map.closed_path)

    alpha = resample_info[:, 2]
    alpha_inv = 1.0 - resample_info[:, 2]
    idx_prev = resample_info[:, 3].astype('int')
    idx_next = resample_info[:, 4].astype('int')

    dst_map.d_left = src_map.d_left[idx_prev] * alpha_inv + src_map.d_left[idx_next] * alpha
    dst_map.d_right = src_map.d_right[idx_prev] * alpha_inv + src_map.d_right[idx_next] * alpha

    cos_orth = np.cos(dst_map.path[:, 2] + np.pi/2)
    sin_orth = np.sin(dst_map.path[:, 2] + np.pi/2)

    dst_map.boundary_left = dst_map.path[:, :2].copy()
    dst_map.boundary_right = dst_map.path[:, :2].copy()
    dst_map.boundary_left[:, 0] += dst_map.d_left * cos_orth
    dst_map.boundary_left[:, 1] += dst_map.d_left * sin_orth
    dst_map.boundary_right[:, 0] -= dst_map.d_right * cos_orth
    dst_map.boundary_right[:, 1] -= dst_map.d_right * sin_orth

    return True


def reinit_map(cmap):

    if len(cmap.control_points) == 0:
        cmap.path = np.zeros((0, 6))
        cmap.boundary_left = np.zeros((0, 2))
        cmap.boundary_right = np.zeros((0, 2))
        cmap.d_left = np.zeros((0, 1))
        cmap.d_right = np.zeros((0, 1))
        return

    if len(cmap.control_points) == 1:
        cmap.path = np.array([[
            cmap.control_points[0, 0],
            cmap.control_points[0, 1],
            0.0,
            0.0,
            0.0,
            cmap.control_points[0, 4]]])
        cmap.boundary_left = cmap.control_points[:, :2].copy()
        cmap.boundary_left += np.array([[0.0, cmap.control_points[0, 2]]])
        cmap.boundary_right = cmap.control_points[:, :2].copy()
        cmap.boundary_right -= np.array([[0.0, cmap.control_points[0, 3]]])
        cmap.d_left = cmap.control_points[0, 2].reshape((1, 1))
        cmap.d_right = cmap.control_points[0, 3].reshape((1, 1))
        return

    # determine usable spline degree

    k = max(1, min(5, cmap.spline_degree))

    if len(cmap.control_points) < 5:
        k = min(1, k)
    elif len(cmap.control_points) < 7:
        k = min(3, k)

    # fit splines to generate path

    if cmap.closed_path:
        cps = np.vstack([cmap.control_points, cmap.control_points[0, :]])
    else:
        cps = cmap.control_points

    dists = np.linalg.norm(np.diff(cps[:, :2], axis=0), axis=1)
    dists_cum = np.array([0.0, *np.cumsum(dists)])

    smoothing = max(0.0, cmap.smoothing)

    xs = splrep(dists_cum, cps[:, 0], s=smoothing, k=k, per=cmap.closed_path)
    ys = splrep(dists_cum, cps[:, 1], s=smoothing, k=k, per=cmap.closed_path)
    vs = splrep(dists_cum, cps[:, 4], k=1, per=cmap.closed_path)

    step_size = max(0.1, min(5.0, cmap.step_size_discr))

    len_path_cp = dists_cum[-1]

    steps = np.arange(
            0.0,
            len_path_cp,
            step_size)

    path = np.zeros((len(steps), 6))
    path[:, 0] = splev(steps, xs)
    path[:, 1] = splev(steps, ys)
    path[:, 2] = np.arctan2(splev(steps, ys, der=1), splev(steps, xs, der=1))
    path[:, 3] = steps
    path[:, 5] = splev(steps, vs)

    # convert path into standard shape

    len_path = np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))
    resample_steps = round(len_path / step_size)

    # resample path in equidistant steps

    try:
        resample_info = util.resample(path[:, :2],
                                      step_size,
                                      resample_steps,
                                      cmap.closed_path)
    except RuntimeError as e:
        return

    cmap.path = util.interp_resampled_path(
            path,
            resample_info,
            step_size,
            resample_steps,
            False,
            cmap.closed_path)

    # compute boundaries

    ds_left = splrep(dists_cum, cps[:, 2], k=1, per=cmap.closed_path)
    ds_right = splrep(dists_cum, cps[:, 3], k=1, per=cmap.closed_path)

    ds_left_sampled = splev(steps, ds_left)
    ds_right_sampled = splev(steps, ds_right)

    alpha = resample_info[:, 2]
    alpha_inv = 1.0 - resample_info[:, 2]
    idx_prev = resample_info[:, 3].astype('int')
    idx_next = resample_info[:, 4].astype('int')

    cmap.d_left = ds_left_sampled[idx_prev] * alpha_inv + ds_left_sampled[idx_next] * alpha
    cmap.d_right = ds_right_sampled[idx_prev] * alpha_inv + ds_right_sampled[idx_next] * alpha

    cos_cmap_orth = np.cos(cmap.path[:, 2] + np.pi/2)
    sin_cmap_orth = np.sin(cmap.path[:, 2] + np.pi/2)

    cmap.boundary_left = cmap.path[:, :2].copy()
    cmap.boundary_right = cmap.path[:, :2].copy()
    cmap.boundary_left[:, 0] += cmap.d_left * cos_cmap_orth
    cmap.boundary_left[:, 1] += cmap.d_left * sin_cmap_orth
    cmap.boundary_right[:, 0] -= cmap.d_right * cos_cmap_orth
    cmap.boundary_right[:, 1] -= cmap.d_right * sin_cmap_orth


def reinit_map_items(cmap, map_store):

    maps = otb.get_obj_dict(map_store)

    for ip in cmap.intersection_paths:

        if ip.intersection_map_uuid not in maps:
            continue

        ip.offset_path_end = max(ip.offset_path_begin+1, ip.offset_path_end)
        ip.map_segment_step_size = max(0.1, ip.map_segment_step_size)

        src_map = maps[ip.intersection_map_uuid]

        proj = util.project(src_map.path[:, :2], ip.pos)

        f = src_map.step_size_discr / ip.map_segment_step_size
        steps = max(1, int(abs(ip.offset_path_end - ip.offset_path_begin) * f))
        start_idx = proj.index + ip.offset_path_begin

        ip.map_segment = Map()

        copy_map_segment(ip.map_segment,
                         src_map,
                         ip.map_segment_step_size,
                         steps,
                         start_idx)


def update_local_map(env):

    cmap = env.get_current_map()
    if cmap is None:
        env.local_map = None
        return

    veh = env.vehicle_state

    if env.local_map is None:
        proj_path_ref = None
        env.local_map = LocalMap()
        on_map = False
    else:
        proj_path_ref = util.project(env.local_map.path[:, :2], (veh.x, veh.y))
        d_r = -env.local_map.d_right[proj_path_ref.index]
        d_l = env.local_map.d_left[proj_path_ref.index]
        on_map = d_r <= proj_path_ref.distance <= d_l

    local_map = env.local_map
    local_map.velocity_limits = cmap.velocity_limits
    local_map.turn_ind_points = cmap.turn_ind_points
    local_map.map_switch_points = cmap.map_switch_points
    local_map.intersection_paths = cmap.intersection_paths

    # ensures that path_ref does not overlap anywhere
    local_map.shift_idx_start_ref = 0

    if not on_map:
        proj_route = util.project(cmap.path[:, :2], (veh.x, veh.y))
        local_map.idx_start_ref = proj_route.start
        env.reset_required = True
    elif proj_path_ref.start > 0:
        local_map.shift_idx_start_ref = proj_path_ref.start
        local_map.idx_start_ref += proj_path_ref.start
        if cmap.closed_path:
            local_map.idx_start_ref %= len(cmap.path)
        else:
            local_map.idx_start_ref = max(0, min(len(cmap.path), local_map.idx_start_ref))

    copy_success = copy_map_segment(local_map,
                                    cmap,
                                    local_map.step_size_ref,
                                    local_map.steps_ref,
                                    local_map.idx_start_ref)

    if not copy_success:
        return

    local_map.steps_ref = len(local_map.path)


def update_map_items(env):

    cmap = env.get_current_map()
    if cmap is None:
        return
    if env.local_map is None:
        return

    veh = env.vehicle_state
    veh_pos = np.array([veh.x, veh.y])

    # handle map switch points

    for msp in cmap.map_switch_points:
        if np.linalg.norm(msp.pos - veh_pos) < msp.activation_radius:
            if not msp.in_radius:
                msp.triggers += 1
                if msp.triggers % msp.trigger_divisor == 0:
                    env.selected_map = msp.target_uuid
                    env.reset_required = True
            msp.in_radius = True
        else:
            msp.in_radius = False

    # turn indicator points

    veh.turn_indicator = 0
    for tip in cmap.turn_ind_points:
        if np.linalg.norm(tip.pos - veh_pos) < tip.activation_radius:
            veh.turn_indicator = tip.dir.selected()

    # update velocity limits

    velocity_limits_path_ref = []

    for vl in cmap.velocity_limits:
        
        vl.proj = util.project(env.local_map.path[:, :2], vl.pos)
        on_ref_path = vl.proj.in_bounds and abs(vl.proj.distance) <= vl.min_distance

        if on_ref_path:
            velocity_limits_path_ref.append(vl)

        if vl.__tag__ == "velocity_limit":

            # automatically reactivate if not on path_ref
            if not on_ref_path:
                vl.active = True

        elif vl.__tag__ == "traffic_light":

            if abs(env.t - vl.t_last_det) > 1.0:
                vl.state = TrafficLight.UNKNOWN

            if not on_ref_path:
                continue

            min_proj = None
            min_det = None

            all_tds = []
            for src in list(env.tl_dets.__slots__):
                all_tds += getattr(env.tl_dets, src)

            # project points on traffic light detection
            # rays and find ray with smallest distance
            for det in all_tds:
                a = np.array([det.near_point[:2], det.far_point[:2]])
                p = util.project(a, vl.light_pos)
                if min_proj is None or abs(p.distance) < abs(min_proj.distance):
                    min_proj = p
                    min_det = det

            if min_det is None:
                continue

            if abs(min_proj.distance) > vl.detection_radius:
                continue

            vl.t_last_det = min_det.t
            vl.state = min_det.state

    # deactivate next velocity_limit on button press

    if veh.steering_wheel_button and len(velocity_limits_path_ref) > 0:
        vl_next = min(velocity_limits_path_ref, key=lambda vl: vl.proj.arc_len)
        vl_next.active = False

    # update intersection paths

    for ip in cmap.intersection_paths:

        ip.stop_proj = util.project(env.local_map.path[:, :2], ip.stop_pos)
        if not ip.stop_proj.in_bounds:
            ip.t_min = float("inf")
            ip.stop = False
            ip.stopped = False
            continue

        if np.linalg.norm(ip.stop_pos - veh_pos) < ip.d_min_stopped:
            if veh.v < 0.1:
                ip.stopped = True
        if ip.stop_always and not ip.stopped:
            ip.stop = True
            continue

        t_min = float("inf")
        for tr in env.predicted:
            for pred in tr.predictions:
                if ip.map_segment.uuid != pred.uuid_assoc_map:
                    continue
                dists = np.linalg.norm(pred.states[:, 1:3] - ip.pos[np.newaxis, :], axis=1)
                idx_closest = np.argmin(dists)
                t_closest = pred.states[idx_closest, 0]
                t_min = min(t_min, t_closest)

        if ip.stop_for_any and t_min < float("inf"):
            ip.stop = True
            continue

        if t_min - ip.t_min > ip.gap_hysteresis:
            ip.t_min = t_min
        elif t_min < ip.t_min:
            ip.t_min = t_min
        ip.stop = bool(ip.t_min < ip.gap_acceptance)


def get_map_boundary_polygon(cmap):

    return np.vstack([
        cmap.boundary_right,
        cmap.boundary_left[::-1],
        cmap.boundary_right[np.newaxis, 0]])


def load_map_store(file_path):

    file_path = os.path.join(util.PATH_MAPS, file_path)

    try:
        map_store = {}
        if not otb.load(map_store, file_path):
            return otb.bundle()
        for cmap in map_store.values():
            if len(cmap.control_points) == 0 and cmap.route is not None and len(cmap.route) > 0:
                # TODO: remove if all maps have been converted
                cmap.control_points = np.zeros((len(cmap.route), 5))
                cmap.control_points[:, 0] = cmap.route[:, 0]
                cmap.control_points[:, 1] = cmap.route[:, 1]
                cmap.control_points[:, 2] = 2.0
                cmap.control_points[:, 3] = 2.0
                cmap.control_points[:, 4] = cmap.route[:, 5]
            reinit_map(cmap)
        map_store = otb.bundle(map_store)
        for cmap in map_store.values():
            reinit_map_items(cmap, map_store)
    except Exception as e:
        traceback.print_exc()
        return otb.bundle()

    return map_store
