import numba
import numpy as np

import objtoolbox as otb

from tpl import util
from tpl.util import runtime
from tpl.optim import optimizers as opts


@numba.njit(cache=True, fastmath=True)
def rampify_profile(step,
                    horizon,
                    evasion_sharpness,
                    proj_distance,
                    path,
                    gap,
                    lower,
                    upper):

    d_offset_fwd = np.zeros((len(path),))
    d_offset_bwd = np.zeros((len(path),))

    for pass_nr in range(0, 2):

        if pass_nr == 0:
            pd = d_offset_fwd
            d = lower[0]
            i_range = range(0, horizon)
        else:
            pd = d_offset_bwd
            d = lower[horizon-1]
            i_range = range(horizon-1, -1, -1)

        for i in i_range:
            if pass_nr == 0:
                slope_lim_range = range(i, horizon)
            else:
                slope_lim_range = range(i, -1, -1)

            d = max(lower[i], d)
            pd[i] = d

            # determine slope
            slope = -(evasion_sharpness / max(path[i, 5], 1e-8)**2)
            for k in slope_lim_range:
                slope = min(slope, (upper[k] - gap - d) / (max(1, abs(k-i)) * step))

            # adapt slope if very close to obstacle
            if pass_nr == 1:
                slope = min(slope, (proj_distance - d) / max(1, (i * step)))

            d += step * slope

    return np.maximum(d_offset_fwd, d_offset_bwd)


class CostFunctionParams:

    def __init__(self):

        self.w_d = 0.5
        self.w_v_d = 0.5
        self.w_a_d = 0.5
        self.w_k = 0.5


class Params:

    def __init__(self):

        self.horizon = 250
        self.step = 0.5
        self.ref_step = 0.5

        # limits for second derivatives of d
        self.min_d_dd = -2.5
        self.max_d_dd = 2.5

        self.max_lat_acc = 2.5

        self.lateral_min_gap = 2.0
        self.offset_center_line = 0.0

        self.evasion_lon_d_safe = 4
        # needs to account for vehicle width
        self.evasion_lat_d_safe = 2.0
        # for higher values we are getting closer to the obstacle
        self.evasion_sharpness = 20.0

        self.cost_func = CostFunctionParams()


class PathOptim:

    def __init__(self):

        self.opt = opts.lateral_profile()
        self.opt.lg_mult_limit = 0.0
        self.opt.barrier_weight[:] = 1000.0
        self.opt.lagrange_multiplier[:] = 0.0

        self.opt_path = np.zeros((1, 6))

        self.d_lower_constr = np.zeros((0, 1))
        self.d_upper_constr = np.zeros((0, 1))

        self.reset_required = False

    @runtime
    def update(self, env, params):

        local_map = env.local_map
        path_len = min(params.horizon, local_map.steps_ref)
        path = local_map.path[:path_len].copy()
        veh = env.vehicle_state
        proj_veh = util.project(path[:, :2], np.array([veh.x, veh.y]))

        opt = self.opt
        opt.horizon = path_len
        opt.step = params.step
        opt.params.ref_step = local_map.step_size_ref
        opt.u_min[:] = -params.max_d_dd
        opt.u_max[:] = params.max_d_dd

        # cost function params
        otb.merge(opt.params, params.cost_func)

        d_lower_constr = -local_map.d_right[:path_len] + veh.width / 2.0
        d_upper_constr = -local_map.d_left[:path_len] + veh.width / 2.0

        # (re-)initialization via warm starting

        si = local_map.shift_idx_start_ref 

        if env.reset_required or self.reset_required or not 0 <= si < path_len:
            # initialize from reference line
            opt.x[0, 0] = 0.0
            opt.x[0, 1] = 0.0
            opt.u[:] = 0.0
        else:
            # can reinitialize optimizer by shifting
            opt.shift(si)
            # keep values fixed, which have already been traversed by the vehicle
            length_veh = veh.rear_axis_to_rear + veh.rear_axis_to_front
            fix = int(np.ceil(length_veh / local_map.step_size_ref))
            # initizalize from previous profiles
            d_upper_constr[:fix] = -opt.params.d_upper_constr[si:si+fix]
            d_lower_constr[:fix] = opt.params.d_lower_constr[si:si+fix]

        self.reset_required = False

        # incorporate dynamic objects

        for obj in env.predicted:

            if obj.evade == "":
                continue

            pps = util.project(path[:, :2], obj.hull)

            p_min = min(pps, key=lambda p: p.arc_len)
            p_max = max(pps, key=lambda p: p.arc_len)

            v_scale = path[0, 5] / (path[0, 5] - obj.v)

            p_min.arc_len = (p_min.arc_len - params.evasion_lon_d_safe) * v_scale
            p_max.arc_len = (p_max.arc_len + params.evasion_lon_d_safe) * v_scale

            p_min.index = max(0, min(path.shape[0], int(p_min.arc_len / opt.step)))
            p_max.index = max(0, min(path.shape[0], int(p_max.arc_len / opt.step)))

            pps_in = [p for p in pps if p.in_bounds
                and -local_map.d_right[p.index] <= p.distance <= local_map.d_left[p.index]]

            if len(pps_in) == 0:
                continue

            if obj.evade == "right":
                d_mult = -1.0
                side = d_upper_constr
                other_side = d_lower_constr
            elif obj.evade == "left":
                d_mult = 1.0
                side = d_lower_constr
                other_side = d_upper_constr
            else:
                # TODO: decide if we want to evade anyway
                continue

            d_max = max(pps_in, key=lambda p: d_mult*p.distance).distance
            d_max += d_mult * params.evasion_lat_d_safe
            evade_dist = d_mult * d_max

            for i in range(p_min.index, p_max.index):
                side[i] = np.minimum(-other_side[i], np.maximum(side[i], evade_dist))

        self.d_lower_constr = d_lower_constr
        self.d_upper_constr = d_upper_constr

        # compute smoothed evasive profile

        d_lower_ref = rampify_profile(opt.step,
                                      opt.horizon,
                                      params.evasion_sharpness,
                                      proj_veh.distance,
                                      path,
                                      params.lateral_min_gap,
                                      d_lower_constr,
                                      -d_upper_constr)

        d_upper_ref = rampify_profile(opt.step,
                                      opt.horizon,
                                      params.evasion_sharpness,
                                      -proj_veh.distance,
                                      path,
                                      params.lateral_min_gap,
                                      d_upper_constr,
                                      -d_lower_constr)

        d_offset = np.maximum(params.offset_center_line, d_lower_ref)

        # copy data to optimizer and update

        opt.params.k_ref = path[:, 4]
        opt.params.d_lower_constr = d_lower_ref
        opt.params.d_upper_constr = -d_upper_ref
        opt.params.d_offset = d_offset

        opt.integrator_type = opt.EULER

        opt.update()

        # transform to cartesian coordinates and resample

        path[:, 0] += -np.sin(path[:, 2]) * opt.x[:-1, 0]
        path[:, 1] += np.cos(path[:, 2]) * opt.x[:-1, 0]
        path[:, 2] += np.arctan(opt.x[:-1, 1])

        self.opt_path = util.resample_path(path, opt.step, opt.horizon)
