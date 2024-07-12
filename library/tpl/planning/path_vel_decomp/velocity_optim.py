import numpy as np
import objtoolbox as otb
from scipy.interpolate import interp1d

from tpl import util
from tpl.util import runtime
from tpl.optim import optimizers as opts

from tpl.planning.utils import (
        rampify_profile,
        curv_to_vel_profile,
        zero_after_first_zero,
        add_vel_constraint,
        apply_velocity_limits
    )


class TimeConstr:

    def __init__(self, t=0.0, pos=None):

        self.pos = np.array([0.0, 0.0]) if pos is None else pos
        self.proj = util.Projection()
        self.t = t


class CostFunctionParams:

    def __init__(self):

        self.p_v = 0.1
        self.p_a = 1.0


class Params:

    def __init__(self):

        self.horizon = 250
        self.step = 0.5
        self.ref_step = 0.5

        self.a_min = -2.5
        self.a_max = 2.5
        self.j_min = -1.5
        self.j_max = 1.5
        self.max_a_total = 5.0
        self.max_lat_acc = 1.5
        self.max_traffic_light_stop_acc = 2.0

        self.d_lat_leader_safe = 1.0

        self.leader_reaction_exp = 1.0
        self.min_d_safe = 1.0
        self.min_v_profile = 1.0

        self.time_constr_alpha = 10.0
        self.time_constr_beta = 0.005

        self.cost_func = CostFunctionParams()


class VelocityOptim:

    def __init__(self):

        self.opt = opts.velocity_profile_space()
        self.opt.max_iterations = 20
        self.opt.lg_mult_limit = 0.1
        self.opt.barrier_weight[:] = 1000.0
        self.opt.lagrange_multiplier[:] = 0.0

        self.path_prev = None
        self.ss = np.zeros((1,))
        self.shifts = np.zeros((1,))

        self.v_lim = np.zeros((1,))
        self.v_ref = np.zeros((1, 2))
        self.v_opt = np.zeros((1,))

        self.stop_mask = np.zeros((1,))

        self.s_leader = 10**6
        self.v_leader = 0.0

    def update_shifts(self, path, params):

        self.ss = np.arange(0.0, params.horizon * params.step, params.step)

        if self.path_prev is not None:
            p = util.project(self.path_prev[:, :2], path[0, :2])
            self.shifts = self.ss + p.arc_len
        else:
            self.shifts = self.ss.copy()

        self.path_prev = path

    def shift_interp(self, arr, axis=0, interp_kind="linear"):
        
        return interp1d(self.ss,
                        arr,
                        kind=interp_kind,
                        axis=axis,
                        fill_value="extrapolate")(self.shifts)

    def update_leader(self, path, env, params):

        self.s_leader = 10.0**6
        self.v_leader = 0.0

        veh = env.vehicle_state
        d_lat_assoc = veh.width / 2.0 + params.d_lat_leader_safe
        veh_proj = util.project(path[:, :2], (veh.x, veh.y))

        for o in env.predicted:
            # fast, approximative check
            proj = util.project(path[:, :2], o.pos)
            if abs(proj.distance) - o.hull_radius >= d_lat_assoc:
                continue

            # slower, more precise check
            projs_hull = util.project(path[:, :2], o.hull)
            if np.any([not p.in_bounds for p in projs_hull]):
                continue
            dists = np.array([p.distance for p in projs_hull])
            if np.all(dists >= 0.0) or np.all(dists < 0.0):
                # only on one side of the ref line
                min_dist = np.min(np.abs(dists))
                if min_dist > d_lat_assoc:
                    continue

            d_lon_leader = np.min([p.arc_len for p in projs_hull]) - veh_proj.arc_len
            if d_lon_leader >= self.s_leader:
                continue
            self.s_leader = d_lon_leader
            self.v_leader = o.v * np.cos(proj.angle - o.yaw)

    @runtime
    def update(self, path, env, params):

        t = env.t
        veh = env.vehicle_state
        cmap = env.local_map

        params.horizon = min(len(path), params.horizon)

        opt = self.opt
        opt.horizon = params.horizon
        opt.step = params.step
        opt.params.ref_step = params.ref_step
        opt.params.max_a_total = params.max_a_total
        opt.u_max[:] = params.a_max
        opt.u_min[:] = params.a_min

        otb.merge(opt.params, params.cost_func)

        self.update_shifts(path, params)
        self.update_leader(path, env, params)

        opt.x[:-1] = self.shift_interp(opt.x[:-1, :])
        # normalizes time array start to 0 (required by time constraints)
        opt.x[:, 1] -= opt.x[0, 1]
        opt.u = self.shift_interp(opt.u, interp_kind="zero")
        opt.lagrange_multiplier = self.shift_interp(opt.lagrange_multiplier)

        # compute reference velocity from maximum allowed
        # velocity, lateral acceleration, and path curvature

        safety_dist = int((params.min_d_safe + veh.rear_axis_to_front) / opt.step)

        lim_v = curv_to_vel_profile(path[:, 4], path[:, 5], params.max_lat_acc)
        apply_velocity_limits(lim_v, cmap, safety_dist)

        # add leader vehicle to velocity profile

        ld_safety_dist = max(0.1, int(self.v_leader / opt.step)) + safety_dist

        add_vel_constraint(
            lim_v,
            (self.s_leader / params.step - ld_safety_dist),
            self.v_leader * min(1.0, self.s_leader / (ld_safety_dist * params.step)),
            length=20)

        # add intersection path constraints

        for ip in env.local_map.intersection_paths:
            if (not ip.stop_proj.in_bounds 
                    or abs(ip.stop_proj.distance) > 1.0
                    or ip.stop_proj.arc_len < veh.rear_axis_to_front):
                continue
            if ip.stop:
                add_vel_constraint(
                    lim_v,
                    ip.stop_proj.index - safety_dist,
                    max_vel=0.0,
                    length=10)
            if not ip.stopped:
                add_vel_constraint(
                    lim_v,
                    ip.stop_proj.index - safety_dist,
                    max_vel=ip.v_max_approach,
                    length=10,
                    shift=-10)

        # add velocity constraints from maneuver to lim_v

        for pos1, pos2, cons_v in env.man_vel_cons:
            proj1 = util.project(opt_path[:, :2], pos1)
            proj2 = util.project(opt_path[:, :2], pos2)

            add_vel_constraint(
                lim_v,
                proj1.index,
                cons_v,
                proj2.index - proj1.index,
                0)

        # compute reference profile for easier optimization

        if self.v_ref.shape[0] != opt.horizon:
            v_ref_new = np.zeros((opt.horizon, 2))
            v_ref_new[0] = self.v_ref[0]
            self.v_ref = v_ref_new

        if env.reset_required:
            self.v_ref[0, 0] = lim_v[0]
            self.v_ref[0, 1] = 0.0
        else:
            self.v_ref = self.shift_interp(self.v_ref)

        self.v_ref = rampify_profile(self.v_ref[0, 0], self.v_ref[0, 1],
                                     lim_v,
                                     params.a_min, params.a_max,
                                     params.j_min, params.j_max,
                                     params.min_v_profile,
                                     opt.step)

        # set optimizer velocity contraints

        if env.reset_required:
            opt.x[0, 0] = veh.v
            opt.x[0, 1] = veh.a

        opt.params.ref_v = self.v_ref[:, 0]
        opt.params.ref_k = path[:, 4]

        # handle time constraints

        opt.params.ref_t_max = np.ones((opt.horizon,)) * 10e10
        opt.params.ref_t_min = np.zeros((opt.horizon,))
        opt.params.ref_t_offset = np.ones((opt.horizon,))
        opt.params.ref_v_weight = np.ones((opt.horizon,))

        # current time at the vehicle position minus the 
        # time of the trajectory at the vehicle position
        # is the wall-time at the trajectory start

        ep = util.project(path[:, :2], np.array([veh.x, veh.y]))

        t_at_veh = ((1.0 - ep.alpha) * opt.x[ep.start, 1]
                    + ep.alpha * opt.x[ep.end, 1])

        time_at_traj_start = t - t_at_veh

        # consider time constraints from maneuvers
        man_min_time_cons = [
            TimeConstr(pos=pos, t=t_min)
            for pos, t_min, t_max in env.man_time_cons]
        man_max_time_cons = [
            TimeConstr(pos=pos, t=t_max)
            for pos, t_min, t_max in env.man_time_cons]

        for tc in man_min_time_cons:

            tc.proj = util.project(path[:, :2], tc.pos)
            idx = tc.proj.index

            if idx >= opt.horizon - 1 or t > tc.t:
                continue

            opt.params.ref_t_min[idx] = max(0.0, tc.t - time_at_traj_start)
            opt.params.ref_t_offset[idx] = (tc.t - time_at_traj_start) - opt.x[idx, 1]

            ss = np.arange(0, opt.horizon) * opt.step 
            rel_wp = tc.proj.arc_len - params.time_constr_alpha
            opt.params.ref_v_weight = np.minimum(
                    opt.params.ref_v_weight, 
                    ((ss - rel_wp) * params.time_constr_beta)**2)

        for tc in man_max_time_cons:

            tc.proj = util.project(path[:, :2], tc.pos)
            idx = tc.proj.index

            if idx >= opt.horizon - 1 or t > tc.t:
                continue

            opt.params.ref_t_max[idx] = max(0.0, tc.t - time_at_traj_start)

        # finally do the update step

        opt.update()

        # create stop mask

        self.stop_mask = (lim_v >= params.min_v_profile) \
                    * ((opt.params.ref_t_min - opt.x[:-1, 1] <= 0.0) 
                        | (opt.x[:-1, 0] > params.min_v_profile * 1.1))
        self.stop_mask = zero_after_first_zero(self.stop_mask.astype(float))

        self.v_lim = lim_v
        self.v_opt = opt.x[:-1, 0].copy() * self.stop_mask
