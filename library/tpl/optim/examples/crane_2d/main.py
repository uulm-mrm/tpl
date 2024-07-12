import time

import numpy as np
import sympy as sp
import imviz as viz

import os.path as osp

from tpl.optim import genopt
from tpl.optim import symext as spx


def build_opt():
    """
    Dynamics according to: "Model Predictive Control of an Overhead Crane
    Using Constraint Substitution" by Käpernik and Graichen (ACC 2013)
    """

    # states

    # horizontal position
    r = sp.Symbol("r")
    # horizontal velocity
    v = sp.Symbol("v")
    # length of the rope
    l = sp.Symbol("l")
    # rope length velocity
    l_dot = sp.Symbol("l_dot")
    # rope angle
    phi = sp.Symbol("phi")
    # rope angle velocity
    phi_dot = sp.Symbol("phi_dot")

    # actions

    # acceleration of cart
    a_c = sp.Symbol("a_c")
    # acceleration of rope length
    a_r = sp.Symbol("a_r")

    # params

    # target position
    r_trg = sp.Symbol("r_trg")
    # rope target length
    l_trg = sp.Symbol("l_trg")

    w_e_r = sp.Symbol("w_e_r")
    w_v = sp.Symbol("w_v")
    w_e_l = sp.Symbol("w_e_l")
    w_l_dot = sp.Symbol("w_l_dot")
    w_phi = sp.Symbol("w_phi")
    w_phi_dot = sp.Symbol("w_phi_dot")
    w_a_c = sp.Symbol("w_a_c")
    w_a_r = sp.Symbol("w_a_r")

    g = 9.81

    dynamics = sp.Matrix([
            v,
            a_c,
            l_dot,
            a_r,
            phi_dot,
	        -((g * sp.sin(phi) + sp.cos(phi)*a_c + 2 * l_dot * phi_dot) / l)
        ])

    costs = (w_e_r * (r_trg - r)**2
          + w_v * v**2
          + w_e_l * (l_trg - l)**2
          + w_l_dot * l_dot**2
          + w_phi * phi**2
          + w_phi_dot * phi_dot**2
          + w_a_c * a_c**2
          + w_a_r * a_r**2)

    # build controller

    config = genopt.Config([r, v, l, l_dot, phi, phi_dot],
                           [a_c, a_r],
                           [w_e_r, w_v, w_e_l, w_l_dot, w_phi, w_phi_dot, w_a_c, w_a_r, r_trg, l_trg],
                           dynamics,
                           costs)

    Optimizer = genopt.build(config)

    return Optimizer


def main():

    print("Building optimizer ...")
    Optimizer = build_opt()
    opt = Optimizer()

    # step size = 0.1 seconds
    opt.step = 0.1

    # optimize for 40 steps = 4.0 seconds
    opt.horizon = 40

    # control constraints
    opt.u_min[:, 0] = -0.5
    opt.u_min[:, 1] = -0.5
    opt.u_max[:, 0] = 0.5
    opt.u_max[:, 1] = 0.5

    # parameters
    opt.params.w_e_r = 1.0
    opt.params.w_v = 1.0
    opt.params.w_e_l = 0.1
    opt.params.w_l_dot = 0.01
    opt.params.w_phi = 0.001
    opt.params.w_phi_dot = 10.0
    opt.params.w_a_c = 0.1
    opt.params.w_a_r = 0.1
    opt.params.r_trg = 0.0
    opt.params.l_trg = 2.0

    # options EULER, HEUN, RK4
    opt.integrator_type = opt.HEUN

    sim_running = False

    # setup gui config
    ini_path = osp.join(osp.dirname(__file__), "gui.ini")
    viz.set_ini_path(ini_path)
    viz.load_ini(ini_path)

    # initialize non-zero rope length to avoid singularity
    opt.x[0, 2] = 1.0

    while viz.wait():

        opt.update()

        r = opt.x[:, 0]
        l = opt.x[:, 2]
        phi = opt.x[:, 4]
        x_end = r + l * np.sin(phi)
        y_end = -l * np.cos(phi)

        if viz.begin_figure("Crane", flags=viz.PlotFlags.EQUAL):
            colors = np.zeros((len(x_end), 4))
            colors[:, 0] = 1.0
            colors[:, 1] = 1.0
            colors[:, 2] = 0.0
            colors[:, 3] = np.linspace(1.0, 0.0, len(x_end))**2
            viz.plot(x_end, y_end, color=colors, fmt="-o")
            viz.plot([opt.x[0, 0]], [0.0], label="crane", fmt="-o")
            viz.plot([opt.x[0, 0], x_end[0]], [0.0, y_end[0]], label="crane", fmt="-")
            opt.params.r_trg, _ = viz.drag_point("target_position",
                                                 (opt.params.r_trg, 0.0),
                                                 color="white",
                                                 flags=viz.PlotDragToolFlags.DELAYED)
            opt.params.l_trg = -viz.drag_hline("target_rope_length",
                                               -opt.params.l_trg,
                                               color="white",
                                               flags=viz.PlotDragToolFlags.DELAYED)
            viz.plot_dummy("rope_end_trajectory", legend_color="yellow")
            viz.plot_dummy("target", legend_color="white")
        viz.end_figure()

        if viz.begin_window("Settings"):
            if viz.button(("stop" if sim_running else "start") + " simulation"):
                sim_running = not sim_running
            viz.separator()
            viz.autogui(opt.params)
            viz.separator()
            error_r = abs(opt.params.r_trg - opt.x[0, 0])
            error_l = abs(opt.params.l_trg - opt.x[0, 2])
            viz.text(f"runtime in ms: {opt.runtime:.4}")
            viz.text(f"error r (position) in m: {error_r:.4}")
            viz.text(f"error l (rope length) in m: {error_l:.4}")
        viz.end_window()

        if viz.begin_window("Internals"):
            viz.autogui(opt)
        viz.end_window()
        
        if sim_running:
            opt.shift(1)


if __name__ == "__main__":
    main()
