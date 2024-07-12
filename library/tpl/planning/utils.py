import numba
import numpy as np

from tpl import util

from tpl.environment import TrafficLight


@numba.njit(cache=True, fastmath=True)
def rampify_profile(v0, a0, lim_v, a_min, a_max, j_min, j_max, v_min, step):
    """
    Integrates the spatial velocity dynamics backwards with
    minimum jerk and acceleration and forwards with maximum jerk
    and acceleration.

    This yields a velocity profile over space which is *driveable*,
    but neither foresighted nor comfortable. Therefore this profile
    is used only as reference (and limit) for a subsequent optimization.
    """

    lim_v = np.maximum(lim_v, v_min)

    horizon = len(lim_v)
    profile = np.zeros((horizon, 2))

    # backward pass

    current_v = lim_v[-1]
    current_a = 0.0
    for t in range(horizon-1, 0, -1):
        profile[t, 0] = current_v
        profile[t, 1] = current_a
        lim_a = max(a_min, (current_v - lim_v[t-1]) / step * current_v)
        if lim_a < 0.0:
            current_a = max(current_a + j_min / current_v * step, lim_a)
        else:
            current_a = 0.0
            current_v = lim_v[t]
        current_v += min(-current_a / current_v * step, lim_v[t-1] - current_v)

    # forward pass

    if v0 is None:
        profile[0, 0] = current_v
    else:
        current_v = max(v0, v_min)
        profile[0, 0] = max(v0, v_min)

    if a0 is None:
        current_a = -current_a
        profile[0, 1] = current_a
    else:
        current_a = a0
        profile[0, 1] = a0

    for t in range(0, horizon):
        if t < horizon-1:
            lim_a = min(a_max, (profile[t+1, 0] - current_v) / step * current_v)
        if lim_a > 0.0:
            current_a = min(current_a + j_max / current_v * step, lim_a)
        else:
            current_a = 0.0
            current_v = profile[t, 0]
        next_v = current_v + min(current_a / current_v * step, lim_v[t] - current_v)
        current_v = min(profile[t, 0], next_v)
        profile[t, 0] = current_v
        profile[t, 1] = current_a

    return profile


@numba.njit(cache=True, fastmath=True)
def curv_to_vel_profile(k, lim_v, max_lat_acc, k_eps=1e-6):

    abs_k = np.abs(k)
    for i in range(len(abs_k)):
        k = abs_k[i]
        if abs(k) > k_eps:
            lim_v[i] = min(lim_v[i], np.sqrt(max_lat_acc / k))

    return lim_v


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


def apply_velocity_limits(lim_v, cmap, safety_dist=1.0):

    for vl in cmap.velocity_limits:

        if not vl.active:
            continue
        if not vl.proj.in_bounds:
            continue
        if abs(vl.proj.distance) > vl.min_distance:
            continue

        if vl.__tag__ == "traffic_light":
            if not vl.proj.in_bounds:
                continue
            if (vl.state == TrafficLight.GREEN
                    or vl.state == TrafficLight.UNKNOWN):
                continue

        add_vel_constraint(
                lim_v,
                vl.proj.start,
                vl.limit,
                vl.length,
                -safety_dist)
