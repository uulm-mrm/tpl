"""
Contains ready-to-serve definitions for various optimizers.
"""

import sympy as sp
import numpy as np

from tpl.optim import genopt
from tpl.optim import symext as spx


def config_trajectory_tracking_mpc():

    t = sp.Symbol("t")
    T = sp.Symbol("T")
    dt = sp.Symbol("dt")

    # states

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    phi = sp.Symbol("phi")
    delta = sp.Symbol("delta")
    v = sp.Symbol("v")
    s_r = sp.Symbol("s_r")
    a = sp.Symbol("a")

    # actions

    j = sp.Symbol("j")
    delta_dot = sp.Symbol("delta_dot")

    # reference course arrays

    ref_x = spx.ArraySymbol("ref_x")
    ref_y = spx.ArraySymbol("ref_y")
    ref_phi = spx.ArraySymbol("ref_phi")
    ref_k = spx.ArraySymbol("ref_k")
    ref_v = spx.ArraySymbol("ref_v")
    ref_step = sp.Symbol("ref_step")

    l = sp.Symbol("l")
    v_ch = sp.Symbol("v_ch")
    max_delta = sp.Symbol("max_delta")
    max_acc = sp.Symbol("max_acc")
    min_acc = sp.Symbol("min_acc")
    a_offset = sp.Symbol("a_offset")

    # time constants

    # dynamics function

    r_x = spx.lerp(0.0, ref_step, s_r, ref_x)
    r_y = spx.lerp(0.0, ref_step, s_r, ref_y)
    r_phi = spx.lerp_angle(0.0, ref_step, s_r, ref_phi)
    r_k = spx.lerp(0.0, ref_step, s_r, ref_k)
    v_trg = spx.lerp(0.0, ref_step, spx.fixed(s_r), ref_v)

    phi_dot = v/(l*(1 + (v/v_ch)**2)) * sp.tan(delta)

    d_r = sp.cos(r_phi)*(y - r_y) - sp.sin(r_phi)*(x - r_x)

    s_dot = v * sp.cos(phi - r_phi) / (1 - d_r * r_k)

    dynamics = sp.Matrix([
            v * sp.cos(phi),
            v * sp.sin(phi),
            phi_dot,
            delta_dot,
            a + a_offset,
            s_dot,
            j
        ])

    # penalty coefficients

    pd = sp.Symbol("pd")
    pv = sp.Symbol("pv")
    pdelta = sp.Symbol("pdelta")
    pdelta_dot = sp.Symbol("pdelta_dot")
    p_phi_dot = sp.Symbol("p_phi_dot")
    min_p_phi_dot = sp.Symbol("min_p_phi_dot")
    min_pdelta_dot = sp.Symbol("min_pdelta_dot")
    p_phi = sp.Symbol("p_phi")
    p_phi_ref_dot_diff = sp.Symbol("p_phi_ref_dot_diff")
    pa = sp.Symbol("pa")
    pj = sp.Symbol("pj")

    # cost function definition

    costs = 0.0
    costs += (min_pdelta_dot + pdelta_dot * v**2) * delta_dot**2
    costs += (min_p_phi_dot + p_phi_dot * v**2) * phi_dot**2
    costs += pa * a**2
    costs += pj * j**2
    costs += pv * (v - v_trg)**2
    costs += pd * d_r**2
    costs += p_phi * (1.0 - sp.cos(phi - r_phi))
    costs += p_phi_ref_dot_diff * (phi_dot - s_dot * r_k)**2 * v**2

    # constraints

    delta_constr_max = delta - max_delta
    delta_constr_min = -max_delta - delta

    acc_constr_max = a - max_acc
    acc_constr_min = min_acc - a

    # build controller

    config = genopt.Config(
            [x, y, phi, delta, v, s_r, a],
            [j, delta_dot],
            [pd, pv, pdelta, min_pdelta_dot, pdelta_dot, min_p_phi_dot,
             p_phi_dot, p_phi, p_phi_ref_dot_diff, pa, pj,
             l, v_ch,
             ref_x, ref_y, ref_phi, ref_k, ref_v, ref_step,
             max_delta, max_acc, min_acc,
             a_offset],
            dynamics,
            costs,
            end_costs=0.0,
            constraints=[delta_constr_max, delta_constr_min, acc_constr_max, acc_constr_min],
            use_cache=True)

    return config


def config_trajectory_tracking_mpc_time():

    t = sp.Symbol("t")
    T = sp.Symbol("T")
    dt = sp.Symbol("dt")

    # states

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    phi = sp.Symbol("phi")
    delta = sp.Symbol("delta")
    v = sp.Symbol("v")
    a = sp.Symbol("a")

    # actions

    j = sp.Symbol("j")
    delta_dot = sp.Symbol("delta_dot")

    # reference traj arrays

    ref_x = spx.ArraySymbol("ref_x")
    ref_y = spx.ArraySymbol("ref_y")
    ref_phi = spx.ArraySymbol("ref_phi")
    ref_v = spx.ArraySymbol("ref_v")
    ref_dt = sp.Symbol("ref_dt")
    ref_t_offset = sp.Symbol("ref_t_offset")

    l = sp.Symbol("l")
    v_ch = sp.Symbol("v_ch")
    max_delta = sp.Symbol("max_delta")
    max_acc = sp.Symbol("max_acc")
    min_acc = sp.Symbol("min_acc")
    a_offset = sp.Symbol("a_offset")

    cog_pos = sp.Symbol("cog_pos")

    # time constants

    rt = ref_t_offset + dt * t

    # dynamics function

    r_x = spx.lerp(0.0, ref_dt, rt, ref_x)
    r_y = spx.lerp(0.0, ref_dt, rt, ref_y)
    r_phi = spx.lerp_angle(0.0, ref_dt, rt, ref_phi)
    v_trg = spx.lerp(0.0, ref_dt, rt, ref_v)

    beta = sp.atan(sp.tan(delta) * cog_pos)

    phi_dot = v * sp.tan(delta) * sp.cos(beta) / (l*(1 + (v/v_ch)**2))

    dynamics = sp.Matrix([
            v * sp.cos(phi + beta),
            v * sp.sin(phi + beta),
            phi_dot,
            delta_dot,
            a + a_offset,
            j
        ])

    # penalty coefficients

    pd = sp.Symbol("pd")
    pv = sp.Symbol("pv")
    pdelta = sp.Symbol("pdelta")
    pdelta_dot = sp.Symbol("pdelta_dot")
    p_phi_dot = sp.Symbol("p_phi_dot")
    min_p_phi_dot = sp.Symbol("min_p_phi_dot")
    min_pdelta_dot = sp.Symbol("min_pdelta_dot")
    p_phi = sp.Symbol("p_phi")
    pa = sp.Symbol("pa")
    pj = sp.Symbol("pj")

    # cost function definition

    costs = 0.0
    costs += (min_pdelta_dot + pdelta_dot * v**2) * delta_dot**2
    costs += (min_p_phi_dot + p_phi_dot * v**2) * phi_dot**2
    costs += pa * a**2
    costs += pj * j**2
    costs += pv * (v - v_trg)**2
    costs += pd * (x - r_x)**2 + pd * (y - r_y)**2
    costs += p_phi * (1.0 - sp.cos(phi - r_phi))

    # constraints

    delta_constr_max = delta - max_delta
    delta_constr_min = -max_delta - delta

    acc_constr_max = a - max_acc
    acc_constr_min = min_acc - a

    # build controller

    config = genopt.Config(
            [x, y, phi, delta, v, a],
            [j, delta_dot],
            [pd, pv, pdelta, min_pdelta_dot, pdelta_dot, min_p_phi_dot,
             p_phi_dot, p_phi, pa, pj,
             l, v_ch, cog_pos,
             ref_x, ref_y, ref_phi, ref_v, ref_dt, ref_t_offset,
             max_delta, max_acc, min_acc,
             a_offset],
            dynamics,
            costs,
            end_costs=0.0,
            constraints=[delta_constr_max, delta_constr_min, acc_constr_max, acc_constr_min],
            use_cache=True)

    return config


def config_lateral_profile():
    """
    This optimizer is used for lateral planning.
    """

    idx = sp.Symbol("t")
    ds = sp.Symbol("dt")

    d = sp.Symbol("d")
    v_d = sp.Symbol("v_d")
    a_d = sp.Symbol("a_d")

    k_ref = spx.ArraySymbol("k_ref")
    d_offset = spx.ArraySymbol("d_offset")
    d_lower_constr = spx.ArraySymbol("d_lower_constr")
    d_upper_constr = spx.ArraySymbol("d_upper_constr")

    ref_step = sp.Symbol("ref_step")

    s = idx * ds
    k_r = spx.lerp(0.0, ref_step, s, k_ref)
    d_o = spx.lerp(0.0, ref_step, s, d_offset)
    d_lower = spx.lerp(0.0, ref_step, s, d_lower_constr)
    d_upper = spx.lerp(0.0, ref_step, s, d_upper_constr)

    cstr_lower = d_lower - d
    cstr_upper = d - d_upper

    dynamics = sp.Matrix([v_d, a_d])

    w_d = sp.Symbol("w_d")
    w_v_d = sp.Symbol("w_v_d")
    w_a_d = sp.Symbol("w_a_d")
    w_k = sp.Symbol("w_k")
    
    # calculate cartesian curvature from frenet coordinates
    k = (a_d / (v_d**2 + 1) + k_r) * sp.cos(sp.atan(v_d)) / (1 - d * k_r)

    costs = w_d * (d - d_o)**2 + w_v_d * v_d**2 + w_a_d * a_d**2 + w_k * k**2
    end_costs = w_d * (d - d_o)**2 + w_v_d * v_d**2

    config = genopt.Config(
            [d, v_d],
            [a_d],
            [k_ref, d_offset, d_lower_constr, d_upper_constr, ref_step, w_d, w_v_d, w_a_d, w_k],
            dynamics,
            costs,
            end_costs=end_costs,
            constraints=[cstr_lower, cstr_upper],
            use_cache=True)

    return config


def config_velocity_profile_time():

    # integration step index
    t = sp.Symbol("t")

    # states
    s = sp.Symbol("s")
    v = sp.Symbol("v")

    # actions
    a = sp.Symbol("a")

    # weights
    w_v = sp.Symbol("w_v")
    w_a = sp.Symbol("w_a")

    # max velocity
    ref_step = sp.Symbol("ref_step")
    ref_v = spx.ArraySymbol("ref_v")
    v_max = spx.lerp(0.0, ref_step, s, ref_v)

    # spatio-temporal 
    ref_s_max = spx.ArraySymbol("ref_s_max")
    ref_s_min = spx.ArraySymbol("ref_s_min")
    s_max = spx.get_array_value(ref_s_max, t)
    s_min = spx.get_array_value(ref_s_min, t)

    # constraints
    constr_v_min = 0.0 - v
    constr_v_max = v - v_max
    constr_s_max = sp.Piecewise((s - s_max, s_max > 0), (0.0, True))
    constr_s_min = sp.Piecewise((s_min - s, s_min > 0), (0.0, True))

    dynamics = sp.Matrix([v, a])

    costs = sp.Matrix([
                w_v * (1000 - v) # constant gradient
                + w_a * a**2
            ])

    end_costs = 0.0

    config = genopt.Config(
            [s, v],
            [a],
            [w_v, w_a, ref_v, ref_step, ref_s_max, ref_s_min],
            dynamics,
            costs,
            end_costs=end_costs,
            constraints=[constr_v_min, constr_v_max, constr_s_max, constr_s_min],
            use_cache=True)

    return config


def config_velocity_profile_space():

    t = sp.Symbol("t")
    dt = sp.Symbol("dt")

    st = sp.Symbol("st")
    v = sp.Symbol("v")

    a = sp.Symbol("a")

    ref_step = sp.Symbol("ref_step")

    ref_t_offset = spx.ArraySymbol("ref_t_offset")
    t_offset = spx.box_interp(ref_step, t * dt, ref_t_offset)

    dynamics = sp.Matrix([
        sp.Piecewise((a / v, v > 1.0 + 1e-3), (a, True)),
        sp.Piecewise((1.0 / v, v > 1.0 + 1e-3), (t_offset, True)),
    ])

    p_v = sp.Symbol("p_v")
    p_a = sp.Symbol("p_a")

    max_a_total = sp.Symbol("max_a_total")

    ref_v = spx.ArraySymbol("ref_v")
    ref_k = spx.ArraySymbol("ref_k")
    ref_t_max = spx.ArraySymbol("ref_t_max")
    ref_t_min = spx.ArraySymbol("ref_t_min")

    v_trg = spx.lerp(0.0, ref_step, t * dt, ref_v)
    kk = spx.box_interp(ref_step, t * dt, ref_k)

    constr_v_min = 1.0 - v
    constr_v_max = v - v_trg

    # offset time

    ost = st + t_offset

    # time constraints

    t_min = spx.lerp(0.0, ref_step, t * dt, ref_t_min)
    constr_t_min = (t_min - st) * sp.Piecewise((v - 1.0, t_min > 0.0), (1.0, True))

    t_max = spx.lerp(0.0, ref_step, t * dt, ref_t_max)
    constr_t_max = ost - t_max
    
    a_lat = v**2 * kk
    constr_a_max = (a**2 + a_lat**2) - max_a_total**2

    ref_v_weight = spx.ArraySymbol("ref_v_weight")
    v_weight = spx.lerp(0.0, ref_step, t * dt, ref_v_weight)

    costs = sp.Matrix([p_v * (v_trg - v)**2 * v_weight + p_a * a**2])
    end_costs = 0.0

    config = genopt.Config(
            [v, st],
            [a],
            [p_v, p_a, 
             max_a_total,
             ref_v, ref_k, ref_step,
             ref_t_max, ref_t_min, ref_t_offset, 
             ref_v_weight],
            dynamics,
            costs,
            end_costs=end_costs,
            constraints=[
                constr_a_max, 
                constr_v_min, 
                constr_v_max, 
                constr_t_max, 
                constr_t_min],
            use_cache=True)

    return config


def config_ref_line_smoother_k():
    """
    This optimizer is used for reference line approximation.
    """

    # states

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    phi = sp.Symbol("phi")

    t = sp.Symbol("t")
    dt = sp.Symbol("dt")

    # dynamics function

    k = sp.Symbol("k")

    dynamics = sp.Matrix([
                sp.cos(phi),
                sp.sin(phi),
                k,
            ])

    # penalty parameters for position and curvature

    w_pos = sp.Symbol("w_pos")
    w_k = sp.Symbol("w_k")

    # the reference line parameter arrays

    ref_x = spx.ArraySymbol("ref_x")
    ref_y = spx.ArraySymbol("ref_y")

    ref_step = sp.Symbol("ref_step")

    # costs

    s = t * dt
    x_ref = spx.lerp(0.0, ref_step, s, ref_x)
    y_ref = spx.lerp(0.0, ref_step, s, ref_y)

    costs = (w_pos * (x - x_ref)**2
             + w_pos * (y - y_ref)**2
             + w_k * k**2)

    end_costs = 0.0

    # optimizer compilation

    config = genopt.Config(
            [x, y, phi],
            [k],
            [w_pos, w_k, ref_x, ref_y, ref_step],
            dynamics,
            costs,
            end_costs=end_costs,
            use_cache=True)

    return config


def config_ref_line_smoother_dk():
    """
    This optimizer is used for reference line approximation.
    """

    # states

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    phi = sp.Symbol("phi")
    k = sp.Symbol("k")

    t = sp.Symbol("t")
    dt = sp.Symbol("dt")

    # dynamics function

    dk = sp.Symbol("dk")

    dynamics = sp.Matrix([
                sp.cos(phi),
                sp.sin(phi),
                k,
                dk
            ])

    # penalty parameters for position and curvature

    w_pos = sp.Symbol("w_pos")
    w_k = sp.Symbol("w_k")
    w_dk = sp.Symbol("w_dk")
    s_start = sp.Symbol("s_start")

    # the reference line parameter arrays

    ref_x = spx.ArraySymbol("ref_x")
    ref_y = spx.ArraySymbol("ref_y")

    ref_step = sp.Symbol("ref_step")

    # costs

    s = s_start + t * dt
    x_ref = spx.lerp(0.0, ref_step, s, ref_x)
    y_ref = spx.lerp(0.0, ref_step, s, ref_y)

    costs = (w_pos * (x - x_ref)**2
             + w_pos * (y - y_ref)**2
             + w_k * k**2
             + w_dk * dk**2)

    end_costs = 0.0

    # optimizer compilation

    config = genopt.Config(
            [x, y, phi, k],
            [dk],
            [w_pos, w_k, w_dk, s_start, ref_x, ref_y, ref_step],
            dynamics,
            costs,
            end_costs=end_costs,
            use_cache=True)

    return config


def build_optimizers(force_rebuild=False):

    config_functions = [
            config_trajectory_tracking_mpc,
            config_trajectory_tracking_mpc_time,
            config_lateral_profile,
            config_velocity_profile_space,
            config_ref_line_smoother_k,
            config_ref_line_smoother_dk,
        ]

    configs = [f() for f in config_functions]
    names = ["_".join(f.__name__.split("_")[1:]) for f in config_functions]

    if not force_rebuild:
        configs = [c for c, n in zip(configs, names) if n not in globals()]

    if len(configs) == 0:
        return

    opt_classes = genopt.build_parallel(configs)

    globals().update({k: v for k, v in zip(names, opt_classes)})
