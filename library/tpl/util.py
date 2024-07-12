import os
import re
import time

import numba
import numpy as np

import structstore as sts

import tpl.optim.optimizers as opts
from tplcpp import (
        point_in_polygon,
        intersect_polygons,
        convex_hull,
        project,
        Projection,
        resample
    )


TO_SNAKE_CASE = re.compile(r'(?<!^)(?=[A-Z])')


# easier access to important config directories
PATH_BASE = os.path.dirname(os.path.realpath(__file__))
PATH_DATA = os.path.abspath(os.path.join(PATH_BASE, "..", "..", "data"))
PATH_SCENARIOS = os.path.join(PATH_DATA, "scenarios")
PATH_MAPS = os.path.join(PATH_DATA, "maps")
PATH_PARAMS = os.path.join(PATH_DATA, "params")
PATH_GUI = os.path.join(PATH_DATA, "gui")


def to_snake_case(name):

    return TO_SNAKE_CASE.sub('_', name).lower()


def get_subclasses_recursive(cls):

    classes = []

    for c in cls.__subclasses__():
        cs = get_subclasses_recursive(c)
        classes += cs
        classes.append(c)

    return classes


def runtime(func):
    """
    Simple decorator to measure the runtime performance of functions.
    """

    def inner(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        inner.runtime = time.perf_counter() - start
        return res

    inner.runtime = 0.0

    return inner


@numba.njit(cache=True, fastmath=True)
def short_angle_dist(x, y):
    """
    Calculates the shortest angle distance between two angles a0 and a1.
    """

    x = normalize_angle(x)
    y = normalize_angle(y)

    a0 = y-x
    a1 = y-x+2*np.pi
    a2 = y-x-2*np.pi

    a = a0
    if np.abs(a1) < np.abs(a):
        a = a1
    if np.abs(a2) < np.abs(a):
        a = a2

    return a


@numba.njit(cache=True, fastmath=True)
def normalize_angle(a):

    a = a % (np.pi * 2)
    a = (a + np.pi * 2) % (np.pi * 2)
    if a > np.pi:
        a -= np.pi * 2;

    return a


@numba.njit(fastmath=True)
def lerp(x, xs, ys, angle=False):
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

    if angle:
        return ys[start] + short_angle_dist(ys[start], ys[end]) * alpha
    else:
        return ys[start] * (1.0 - alpha) + ys[end] * alpha


def load_route_from_csv(path, sampling_dist=0.5):

    route = np.genfromtxt(path, delimiter=",", )
    route = resample_path(route, 0.5, len(route))

    return build_route(route)


def resample_path(path,
                  step_size,
                  steps,
                  start_index=0,
                  zero_vel_at_end=False,
                  closed=False):
    """
    Resamples a path array in equidistant steps.
    Assumes x, y are the first two coordinates.
    """

    path = np.array(path)
    try:
        resample_info = resample(path[:, :2], step_size, steps, start_index, closed)
    except RuntimeError:
        return None
    return interp_resampled_path(
            path, resample_info, step_size, steps, zero_vel_at_end, closed)


@numba.njit(cache=True, fastmath=True)
def interp_resampled_path(path, rsi, step_size, steps, zero_vel_at_end, closed):

    rs = np.zeros((steps, 6))
    rs[:, :2] = rsi[:, :2]

    prevs = path[rsi[:, 3].astype('int')]
    nexts = path[rsi[:, 4].astype('int')]

    # recover orientation, distance, target velocity
    for i in range(len(rsi)):
        t = rsi[i, 2]
        if not closed and rsi[i, 4] == len(path)-1 and t > 1.0:
            # extrapolation
            rs[i, 2] = nexts[i, 2]
            rs[i, 3] = step_size * i
            rs[i, 5] = 0.0 if zero_vel_at_end else nexts[i, 5]
        else:
            # interpolation
            rs[i, 2] = prevs[i, 2] + t * short_angle_dist(prevs[i, 2], nexts[i, 2])
            rs[i, 3] = step_size * i
            rs[i, 5] = (1.0 - t) * prevs[i, 5] + t * nexts[i, 5]

    # recover curvature
    for i in range(1, len(rsi)):
        rs[i-1, 4] = 2 * np.sin(short_angle_dist(
                rs[i-1, 2], rs[i, 2]) / 2) / step_size
    if closed:
        gap = np.linalg.norm(rs[0, :2] - rs[-1, :2])
        rs[-1, 4] = 2 * np.sin(short_angle_dist(
            rs[-1, 2], rs[0, 2]) / 2) / gap
    else:
        rs[-1, 4] = rs[-2, 4]

    return rs


def smooth_route(path, step_size=0.5, smoothness=0.1):
    """
    Smoothes a given path by repeatedly solving an OCP.
    """

    path = np.array(path)

    RefLineSmoother = opts.genopt.build(opts.config_ref_line_smoother())

    opt = RefLineSmoother()
    opt.horizon = 20
    opt.step = step_size
    opt.params.w_pos = 1.0
    opt.params.s_start = 0.0
    opt.params.w_k = smoothness
    opt.params.w_dk = 1.0
    opt.params.ref_step = step_size
    opt.u_max[:] = 5.0
    opt.u_min[:] = -5.0

    rs_path = resample_path(path, step_size, opt.horizon, 0, zero_vel_at_end=True)
    opt.x[0] = rs_path[0, [0, 1, 2, 4]]
    opt.u[:-1] = np.diff(rs_path[:opt.horizon, 4]) / opt.step
    opt.u[-1] = opt.u[-2]

    arr = np.zeros_like(path)
    arr[:, 3] = path[:, 3]

    for i in range(0, len(path)):
        rs_path = resample_path(path, step_size, opt.horizon, i, zero_vel_at_end=True)
        opt.params.ref_x = rs_path[:, 0]
        opt.params.ref_y = rs_path[:, 1]
        opt.params.ref_s = rs_path[:, 3] - rs_path[0, 3]
        arr[i, :2] = opt.x[0, :2].copy()
        arr[i, 2] = normalize_angle(opt.x[0, 2])
        arr[i, 4] = opt.x[0, 3].copy()
        arr[i, 5] = path[i, 5].copy()
        opt.update()
        opt.shift(1)

    return arr


def build_route(route):
    """
    Beats the route array into the right shape (by force).
    """

    # calculate angles, distances and curvatures

    route_pos = route[:, :2]

    route_angles = []
    route_dists = [0.0]

    prev_p = route_pos[0]
    for p in route_pos[1:]:
        d = p - prev_p
        route_angles.append(np.arctan2(d[1], d[0]))
        route_dists.append(route_dists[-1] + np.linalg.norm(d))
        prev_p = p
    route_angles.append(route_angles[-1])

    route_curvs = []

    prev_p = route_pos[0]
    prev_a = route_angles[0]
    for i, p in enumerate(route_pos[1:]):
        d = np.linalg.norm(p - prev_p)
        a = route_angles[i+1]
        angle_diff = short_angle_dist(prev_a, a)
        route_curvs.append(angle_diff / d)
        prev_p = p
        prev_a = a
    route_curvs.append(route_curvs[-1])

    route_angles = np.reshape(np.array(route_angles), (-1, 1))
    route_dists = np.reshape(np.array(route_dists), (-1, 1))
    route_curvs = np.reshape(np.array(route_curvs), (-1, 1))

    # handle speed limits

    if route.shape[1] > 2:
        route_speed_limits = route[:, -1].reshape(-1, 1)
    else:
        route_speed_limits = (np.zeros((len(route), 1)) + 30.0) / 3.6

    route = np.hstack([
        route_pos,
        route_angles,
        route_dists,
        route_curvs,
        route_speed_limits])

    return route


def make_class_shared(cls):
    """
    This create a child class from cls, which gets all of its data
    from a given structstore. Methods from cls can still be called.
    """

    class SharedClass(cls):

        def __init__(self, sh_obj, init_args=(), init_kwargs={}):
            self.bind_sh_obj(sh_obj)
            with sh_obj.lock():
                super().__init__(*init_args, **init_kwargs)

        def bind_sh_obj(self, sh_obj):
            super().__setattr__("__sh", sh_obj)
            super().__setattr__("lock", sh_obj.lock)
            super().__setattr__("revalidate", sh_obj.lock)

        @staticmethod
        def fromparams(path, *args, init_args=(), init_kwargs={}, **kwargs):
            sh_obj = StructStoreRegistry.get(path, *args, **kwargs)
            return SharedClass(sh_obj, init_args=init_args, init_kwargs=init_kwargs)

        @staticmethod
        def fromshared(sh_obj):
            self = SharedClass.__new__(SharedClass)
            self.bind_sh_obj(sh_obj)
            return self

        @property
        def __dict__(self):
            raise AttributeError("__dict__ use is prohibited, use __slots__")

        def __getattr__(self, name):
            return getattr(self.__getattribute__("__sh"), name)

        def __setattr__(self, name, value):
            return setattr(self.__getattribute__("__sh"), name, value)

        def __deepcopy__(self, memo={}):
            n = SharedClass.__new__(SharedClass)
            c = self.__getattribute__("__sh").deepcopy()
            super(SharedClass, n).__setattr__("__sh", c)
            return n

        def wait_for_attr(self, attr_name, timeout=-1.0):
            start_time = time.time()
            while True:
                self.revalidate()
                if hasattr(self, attr_name):
                    return True
                if timeout >= 0.0:
                    if time.time() - start_time > timeout:
                        break
                time.sleep(0.001)
            return False

    SharedClass.__qualname__ = f"Shared{cls.__qualname__}"

    return SharedClass


class StructStoreRegistry:

    REGISTRY = {}

    @staticmethod
    def get(path, *args, **kwargs):

        try:
            store = StructStoreRegistry.REGISTRY[path]
            return store
        except KeyError:
            store = sts.StructStoreShared(path, *args, **kwargs)
            StructStoreRegistry.REGISTRY[path] = store

        return store
