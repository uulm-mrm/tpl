import os
import uuid
import traceback

import numba
import numpy as np 
import objtoolbox as otb

from tpl import util


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

    RED = 0
    YELLOW = 1
    GREEN = 2
    NONE = 3

    def __init__(self):

        super().__init__()

        self.__tag__ = "traffic_light"
        self.uuid = uuid.uuid4().hex

        self.light_pos = np.array([0.0, 0.0])
        self.detection_radius = 1.0

        self.t = 0.0
        self.state = TrafficLight.NONE
        self.can_stop = False

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
        self.free_limit = 10.0

    def __savestate__(self):

        s = super().__savestate__()
        s["corners"] = self.corners

        return s


class TurnIndPoint:

    OFF = 0
    RIGHT = -1
    LEFT = 1
    HAZARD = 2

    def __init__(self):

        self.__tag__ = "turn_ind_point"
        self.uuid = uuid.uuid4().hex

        self.pos = np.array([0.0, 0.0])
        self.dir = TurnIndPoint.OFF

        self.activation_radius = 2.0


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

    def __init__(self, pos=np.zeros((2,))):

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

        self.stop = True

        self.d_decision = 30.0
        self.gap_acceptance = 5.0
        self.gap_rejection = 3.0

    def __savestate__(self):

        d = self.__dict__.copy()
        del d["map_segment"]
        del d["stop"]

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

        # dims: x, y, d_left, d_right, speed_limit, altitude
        self.control_points = np.zeros((0, 6))
        self.smoothing = 0.0
        # step size for discretization
        self.step_size_discr = 0.5
        self.closed_path = False

        # these arrays are calculated with reinit_map(...)

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
        # discretized altitude values
        self.altitude = None

        # DEPRECATED
        self.route = None

    def __savestate__(self):

        d = self.__dict__.copy()
        del d["path"]
        del d["boundary_left"]
        del d["boundary_right"]
        del d["d_left"]
        del d["d_right"]
        del d["altitude"]
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

        # by how much vel. constraints should be shifted
        self.shift_vel_lim = 0
        # if set the intersection path heuristic will be applied
        self.update_inters_paths = False
        # by how many indices the ref_line is shifted to follow the ego vehicle
        self.step_shift_idx = 2
        # at which vehicle position on the local map the vehicle should be
        self.position_vehicle = 0.0


def copy_map_segment(dst_map, src_map, step_size, steps, start_idx):

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

    # resample path into finer steps for smoothing

    if cmap.closed_path:
        cps = np.vstack([cmap.control_points, cmap.control_points[0, :]])
    else:
        cps = cmap.control_points

    step_size = max(0.1, min(5.0, cmap.step_size_discr))
    len_path = np.sum(np.linalg.norm(np.diff(cps[:, :2], axis=0), axis=1))
    resample_steps = int(len_path / step_size)

    try:
        resample_info = util.resample(cps[:, :2],
                                      step_size,
                                      resample_steps,
                                      closed=cmap.closed_path)
    except RuntimeError as e:
        return

    alpha = resample_info[:, 2]
    alpha_inv = 1.0 - resample_info[:, 2]
    idx_prev = resample_info[:, 3].astype('int')
    idx_next = resample_info[:, 4].astype('int')

    cmap.d_left = cps[idx_prev, 2] * alpha_inv + cps[idx_next, 2] * alpha
    cmap.d_right = cps[idx_prev, 3] * alpha_inv + cps[idx_next, 3] * alpha
    cmap.altitude = cps[idx_prev, 5] * alpha_inv + cps[idx_next, 5] * alpha

    diffs = np.diff(cps[:, :2], axis=0)
    angles = np.zeros((cps.shape[0],))
    angles[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
    if cmap.closed_path:
        angles[-1] = angles[0]
    else:
        angles[-1] = angles[-2]

    path = np.zeros((cps.shape[0], 6))
    path[:, :2] = cps[:, :2]
    path[:, 2] = angles
    path[:, 5] = cps[:, 4]

    path = util.interp_resampled_path(
            path,
            resample_info,
            step_size,
            resample_steps,
            False,
            cmap.closed_path)

    if cmap.smoothing > 1e-5:
        xys = util.smooth_path(
                resample_info[:, :2],
                step_size,
                0.0,
                10.0 * cmap.smoothing,
                1000.0 * cmap.smoothing,
                cmap.closed_path)

        len_path = np.sum(np.linalg.norm(np.diff(xys, axis=0), axis=1))
        resample_steps = int(len_path / step_size)

        try:
            resample_info = util.resample(
                    xys,
                    step_size,
                    resample_steps,
                    closed=cmap.closed_path)
        except RuntimeError as e:
            return

        alpha = resample_info[:, 2]
        alpha_inv = 1.0 - resample_info[:, 2]
        idx_prev = resample_info[:, 3].astype('int')
        idx_next = resample_info[:, 4].astype('int')

        cmap.d_left = cmap.d_left[idx_prev] * alpha_inv + cmap.d_left[idx_next] * alpha
        cmap.d_right = cmap.d_right[idx_prev] * alpha_inv + cmap.d_right[idx_next] * alpha
        cmap.altitude = cmap.altitude[idx_prev] * alpha_inv + cmap.altitude[idx_next] * alpha

        diffs = np.diff(xys[:, :2], axis=0)
        angles = np.zeros((xys.shape[0],))
        angles[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
        if cmap.closed_path:
            angles[-1] = angles[0]
        else:
            angles[-1] = angles[-2]

        prev_path = path

        path = np.zeros((xys.shape[0], 6))
        path[:, :2] = xys
        path[:, 2] = angles
        path[:, 5] = prev_path[:, 5]

        path = util.interp_resampled_path(
                path,
                resample_info,
                step_size,
                resample_steps,
                False,
                cmap.closed_path)

    cmap.path = path

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

        reinit_intersection_path(ip, cmap, maps)


def reinit_intersection_path(ip, cmap, maps):

    ip.map_segment_step_size = max(0.1, ip.map_segment_step_size)

    src_map = maps[ip.intersection_map_uuid]
    proj = util.project(src_map.path[:, :2], ip.pos)

    if src_map.closed_path:
        steps = (ip.offset_path_end - ip.offset_path_begin) % len(src_map.path)
    else:
        ip.offset_path_end = max(ip.offset_path_begin+1, ip.offset_path_end)
        steps = ip.offset_path_end - ip.offset_path_begin

    f = src_map.step_size_discr / ip.map_segment_step_size
    steps = max(1, int(abs(steps) * f))
    start_idx = proj.index + ip.offset_path_begin

    ip.map_segment = Map()
    ip.map_segment.name = src_map.name

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
        env.local_map_behind = Map()
        env.local_map_behind.name = "local_map_behind"
        on_map = False
    else:
        proj_path_ref = util.project(env.local_map.path[:, :2], (veh.x, veh.y))
        d_r = -env.local_map.d_right[proj_path_ref.index]
        d_l = env.local_map.d_left[proj_path_ref.index]
        on_map = (d_r <= proj_path_ref.distance <= d_l) & proj_path_ref.in_bounds

    local_map = env.local_map
    local_map.velocity_limits = cmap.velocity_limits
    local_map.turn_ind_points = cmap.turn_ind_points
    local_map.map_switch_points = cmap.map_switch_points
    local_map.intersection_paths = cmap.intersection_paths

    # ensures that path_ref does not overlap anywhere
    local_map.shift_idx_start_ref = 0
    local_map_vehicle_pos_steps = int(local_map.position_vehicle // local_map.step_size_ref)

    if not on_map:
        proj_route = util.project(cmap.path[:, :2], (veh.x, veh.y))
        local_map.idx_start_ref = proj_route.start
        env.reset_counter += 1
    elif abs(proj_path_ref.start - local_map_vehicle_pos_steps) > local_map.step_shift_idx:
        shift = ((proj_path_ref.start - local_map_vehicle_pos_steps) // local_map.step_shift_idx)
        local_map.shift_idx_start_ref = shift * local_map.step_shift_idx
        local_map.idx_start_ref += shift * local_map.step_shift_idx
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

    idx_start_ref_behind = local_map.idx_start_ref - local_map.steps_ref
    if cmap.closed_path:
        idx_start_ref_behind %= len(cmap.path)
    else:
        idx_start_ref_behind = max(0, min(len(cmap.path), idx_start_ref_behind))

    copy_success = copy_map_segment(env.local_map_behind,
                                    cmap,
                                    env.local_map.step_size_ref,
                                    env.local_map.steps_ref*2,
                                    idx_start_ref_behind)

    if not copy_success:
        return

    local_map.steps_ref = len(local_map.path)


@numba.njit(cache=True, fastmath=True)
def curv_to_vel_profile(k, lim_v, a_lat_max, k_eps=1e-6):

    out = np.zeros(len(lim_v))
    abs_k = np.abs(k)
    for i in range(len(abs_k)):
        k = abs_k[i]
        if abs(k) > k_eps:
            out[i] = min(lim_v[i], np.sqrt(a_lat_max / k))
        else:
            out[i] = lim_v[i]

    return out


@numba.njit(cache=True, fastmath=True)
def zero_after_first_zero(vel_profile):

    con_vp = np.zeros_like(vel_profile)
    for i in range(len(vel_profile)):
        if not vel_profile[i]:
            break
        con_vp[i] = 1.0

    return con_vp


def add_vel_constraint(lim_v, index, max_vel=0.0, length=10, shift=0):

    # let's use variable names, which look very similar

    i = int(index + shift)
    l = min(len(lim_v) - i, int(length))
    j = max(0, i + l)
    l = max(0, min(j, l))
    i = max(0, i)

    lim_v[i:j] = np.minimum(lim_v[i:j], np.ones((l,)) * max_vel)


def update_local_map_velocity(env):
    
    cmap = env.local_map
    if cmap is None:
        return

    v_lim = curv_to_vel_profile(cmap.path[:, 4],
                                cmap.path[:, 5],
                                env.vehicle_state.a_lat_max)

    for vl in cmap.velocity_limits:

        if not vl.active:
            continue
        if not vl.proj.in_bounds:
            continue
        if abs(vl.proj.distance) > vl.min_distance:
            continue

        add_vel_constraint(
                v_lim,
                vl.proj.start,
                vl.limit,
                vl.length,
                cmap.shift_vel_lim)

    cmap.path[:, 5] = v_lim


def update_local_map_inters_paths(env):

    cmap = env.local_map
    if cmap is None:
        return

    v_lim = cmap.path[:, 5]

    if not cmap.update_inters_paths:
        return

    for ip in env.local_map.intersection_paths:

        if not ip.stop_proj.in_bounds:
            continue
        if abs(ip.stop_proj.distance) > 1.0:
            continue
        if not ip.stop:
            continue

        add_vel_constraint(
            v_lim,
            ip.stop_proj.start,
            max_vel=0.0,
            length=10,
            shift=cmap.shift_vel_lim)

    cmap.path[:, 5] = v_lim


def update_map_items(env):

    cmap = env.local_map
    if cmap is None:
        return

    veh = env.vehicle_state
    veh_pos = np.array([veh.x, veh.y])
    proj_veh = util.project(env.local_map.path[:, :2], veh_pos)

    # handle map switch points

    for msp in cmap.map_switch_points:
        if np.linalg.norm(msp.pos - veh_pos) < msp.activation_radius:
            if not msp.in_radius:
                msp.triggers += 1
                if msp.triggers % msp.trigger_divisor == 0:
                    env.selected_map = msp.target_uuid
                    env.reset_counter += 1
            msp.in_radius = True
        else:
            msp.in_radius = False

    # turn indicator points

    veh.turn_indicator = 0
    for tip in cmap.turn_ind_points:
        if np.linalg.norm(tip.pos - veh_pos) < tip.activation_radius:
            veh.turn_indicator = tip.dir

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

        elif vl.__tag__ == "cross_walk":

            if not on_ref_path:
                continue

            stop = False
            only_stationary = True
            for tr in list(env.get_all_tracks()):
                if tr.object_class != "pedestrian":
                    continue
                if util.intersect_polygons(tr.hull, vl.corners):
                    stop = True
                only_stationary &= tr.stationary

            if stop:
                if only_stationary:
                    vl.limit = 3.0
                else:
                    vl.limit = 0.0
            else:
                vl.limit = vl.free_limit

        elif vl.__tag__ == "traffic_light":

            if abs(env.t - vl.t) > 3.0:
                vl.state = TrafficLight.RED
                vl.can_stop = True

            if not on_ref_path:
                continue

            all_tds = []
            for src in list(env.tl_dets.__slots__):
                all_tds += getattr(env.tl_dets, src)

            # associate detections with traffic light
            assoc_dets = []
            for det in all_tds:
                if det.confidence < 0.25:
                    # filter out low confidence detections
                    continue
                a = np.array([det.near_point, det.far_point])
                ray = a[1] - a[0]
                angle = np.arctan2(ray[1], ray[0])
                angle_dist = abs(np.degrees(util.short_angle_dist(vl.proj.angle, angle)))
                if angle_dist > 35.0:
                    # filter out detections with too high angle
                    continue
                p = util.project(a, vl.light_pos)
                if abs(p.distance) <= vl.detection_radius:
                    assoc_dets.append((det, p))

            if len(assoc_dets) > 0:

                # compute the detection state with weighted voting
                vote = np.zeros((4,))
                for det, p in assoc_dets:
                    w = (vl.detection_radius - abs(p.distance)) / vl.detection_radius
                    if det.state == TrafficLight.NONE:
                        w *= 0.1
                    vote[det.state] += w
                det_state = int(np.argmax(vote))

                # check if we can still stop
                if (vl.state in [TrafficLight.GREEN, TrafficLight.NONE]
                        and det_state not in [TrafficLight.GREEN, TrafficLight.NONE]):
                    d_to_tl = vl.proj.arc_len - proj_veh.arc_len
                    d_stop = veh.v**2 / (2 * 2.75)
                    vl.can_stop = d_to_tl >= d_stop

                vl.t = env.t
                vl.state = det_state

            vl.active = vl.state in [TrafficLight.RED, TrafficLight.YELLOW]
            vl.active &= vl.can_stop

    # deactivate next velocity_limit on button press

    if veh.steering_wheel_button and len(velocity_limits_path_ref) > 0:
        vl_next = min(velocity_limits_path_ref, key=lambda vl: vl.proj.arc_len)
        vl_next.active = False

    # update intersection paths

    for ip in cmap.intersection_paths:
        ip.stop_proj = util.project(env.local_map.path[:, :2], ip.stop_pos)
        if not ip.stop_proj.in_bounds:
            ip.stop = True
            continue

        dist_veh_to_stop_line = (np.linalg.norm(ip.stop_pos - veh_pos) 
                                 - veh.rear_axis_to_front)
        time_veh_to_stop_line = max(0.0, dist_veh_to_stop_line / max(2.0, veh.v))

        if dist_veh_to_stop_line > ip.d_decision:
            continue

        pos_critical = (ip.map_segment.path[-1, 3] * abs(ip.offset_path_begin)
                    / (ip.offset_path_end - ip.offset_path_begin))

        t_min = float("inf")
        for tr in env.predicted:
            if tr.stationary:
                continue
            for pred in tr.predictions:
                if ip.map_segment.uuid != pred.uuid_assoc_map:
                    continue
                if pred.cos_angle_dist < 0.0:
                    continue
                t_inters = ((5.0 + pos_critical - pred.proj_assoc_map.arc_len)
                            / max(5.0, tr.v * pred.cos_angle_dist))
                if t_inters < 0.0:
                    continue
                t_min = min(t_min, t_inters)

        if t_min - time_veh_to_stop_line > ip.gap_acceptance:
            ip.stop = False
        elif not ip.stop:
            if t_min < ip.gap_rejection:
                stop_acc = 6.0
                dist_break_to_stop = (veh.v**2) / (2 * stop_acc)
                if dist_veh_to_stop_line > dist_break_to_stop:
                    # can still stop, abort mission
                    ip.stop = True

    # synchronize with global map object

    mmap = env.get_current_map()
    mmap.velocity_limits = cmap.velocity_limits
    mmap.turn_ind_points = cmap.turn_ind_points
    mmap.map_switch_points = cmap.map_switch_points
    mmap.intersection_paths = cmap.intersection_paths


def get_map_boundary_polygon(cmap):

    return np.vstack([
        cmap.boundary_right,
        cmap.boundary_left[::-1],
        cmap.boundary_right[np.newaxis, 0]])


def load_map_store(file_path, mmap_arrays=False):

    file_path = os.path.join(util.PATH_MAPS, file_path)

    try:
        map_store = {}
        if not otb.load(map_store, file_path, mmap_arrays=mmap_arrays):
            return otb.bundle()
        for cmap in map_store.values():
            if len(cmap.control_points) == 0 and cmap.route is not None and len(cmap.route) > 0:
                # TODO: remove if all maps have been converted
                cmap.control_points = np.zeros((len(cmap.route), 6))
                cmap.control_points[:, 0] = cmap.route[:, 0]
                cmap.control_points[:, 1] = cmap.route[:, 1]
                cmap.control_points[:, 2] = 2.0
                cmap.control_points[:, 3] = 2.0
                cmap.control_points[:, 4] = cmap.route[:, 5]
            if cmap.control_points.shape[1] < 6:
                control_points = np.zeros((len(cmap.control_points), 6))
                control_points[:, :5] = cmap.control_points
                cmap.control_points = control_points
            reinit_map(cmap)
        map_store = otb.bundle(map_store)
        for cmap in map_store.values():
            reinit_map_items(cmap, map_store)
    except Exception as e:
        traceback.print_exc()
        return otb.bundle()

    return map_store
