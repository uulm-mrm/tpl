import sympy as sp
import os.path as osp

from tpl.optim import genopt
from tpl.optim import symext as spx

import imviz as viz


def build_opt():

    # states

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    heading = sp.Symbol("heading")

    # actions

    v = sp.Symbol("v")
    steer = sp.Symbol("steer")

    # params

    x_trg = sp.Symbol("x_trg")
    y_trg = sp.Symbol("y_trg")

    w_e_pos = sp.Symbol("w_e_pos")
    w_e_v = sp.Symbol("w_e_v")
    w_steer = sp.Symbol("w_steer")

    dynamics = sp.Matrix([
        v * sp.cos(heading),
        v * sp.sin(heading),
        v * steer
    ])

    costs = (w_e_pos * (x - x_trg)**2 
          + w_e_pos * (y - y_trg)**2
          + w_e_v * v**2
          + w_steer * steer**2)

    end_costs = (w_e_pos * (x - x_trg)**2 
              + w_e_pos * (y - y_trg)**2)

    # build optimizer

    config = genopt.Config([x, y, heading],
                           [v, steer],
                           [x_trg, y_trg, w_e_pos, w_e_v, w_steer],
                           dynamics,
                           costs,
                           end_costs,
                           use_cache=True)

    Optimizer = genopt.build(config)

    return Optimizer


def main():

    print("Building optimizer ...")
    Optimizer = build_opt()
    opt = Optimizer()

    # step size = 0.05 seconds
    opt.step = 0.05

    # optimize for 40 steps = 2.0 seconds
    opt.horizon = 40

    # control constraints
    opt.u_min[:, 0] = -2.0
    opt.u_min[:, 1] = -1.0
    opt.u_max[:, 0] = 2.0
    opt.u_max[:, 1] = 1.0

    # parameters
    opt.params.x_trg = 10.0
    opt.params.y_trg = 10.0
    opt.params.w_e_pos = 1.0
    opt.params.w_e_v = 0.5
    opt.params.w_steer = 0.1

    # options EULER, HEUN, RK4
    opt.integrator_type = opt.EULER

    sim_running = False

    # setup gui config
    ini_path = osp.join(osp.dirname(__file__), "gui.ini")
    viz.set_ini_path(ini_path)
    viz.load_ini(ini_path)

    while viz.wait():

        opt.update()

        if viz.begin_figure("Vehicle Position", flags=viz.PlotFlags.EQUAL):
            viz.plot(opt.x[:, 0], opt.x[:, 1], label="trajectory", fmt="-o")
            opt.params.x_trg, opt.params.y_trg = viz.drag_point(
                    "target",
                    (opt.params.x_trg, opt.params.y_trg))
            viz.plot_dummy("target", legend_color="white")
        viz.end_figure()

        if viz.begin_window("Settings"):
            if viz.button(("stop" if sim_running else "start") + " simulation"):
                sim_running = not sim_running
            viz.separator()
            viz.autogui(opt.params)
            viz.separator()
            error_x = abs(opt.params.x_trg - opt.x[0, 0])
            error_y = abs(opt.params.y_trg - opt.x[0, 1])
            viz.text(f"runtime in ms: {opt.runtime:.4}")
            viz.text(f"error x in m: {error_x:.4}")
            viz.text(f"error y in m: {error_y:.4}")
        viz.end_window()

        if viz.begin_window("Internals"):
            viz.autogui(opt)
        viz.end_window()
        
        if sim_running:
            opt.shift(1)

if __name__ == "__main__":
    main()
