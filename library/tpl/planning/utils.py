import numba
import numpy as np


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
