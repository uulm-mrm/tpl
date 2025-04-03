import time
import numpy as np
import scipy.linalg

import imviz as viz

from casadi import SX, vertcat, sin, cos
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver


def export_robot_model() -> AcadosModel:

    model_name = "unicycle"

    # set up states & controls
    x = SX.sym("x")
    y = SX.sym("y")
    heading = SX.sym("heading")

    x = vertcat(x, y, heading)

    v = SX.sym("v")
    steer = SX.sym("steer")
    u = vertcat(v, steer)

    # xdot
    x_dot = SX.sym("x_dot")
    y_dot = SX.sym("y_dot")
    heading_dot = SX.sym("heading_dot")

    xdot = vertcat(x_dot, y_dot, heading_dot)

    # dynamics
    f_expl = vertcat(v * cos(heading), v * sin(heading), v * steer)

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.x = x
    #model.xdot = xdot
    model.u = u
    model.name = model_name

    T_horizon = 2.0  # Define the prediction horizon
    N_horizon = 40  # Define the number of discretization steps

    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost
    Q_mat = np.diag([1.0, 1.0, 0.0])
    R_mat = np.diag([1.0, 1.0])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    x0 = np.array([0.0, 0.0, 0.0])  # Inital state

    # set constraints
    ocp.constraints.lbu = np.array([-2.0, -1.0])
    ocp.constraints.ubu = np.array([2.0, 1.0])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.ext_fun_compile_flags = "-O3 -march=native"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM" #"FULL_CONDENSING_QPOASES" 
    ocp.solver_options.hpipm_mode = "BALANCE"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 10
    ocp.solver_options.sim_method_num_stages = 1
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.qp_solver_iter_max = 10
    ocp.solver_options.tf = T_horizon
    ocp.solver_options.regularize_method = "PROJECT"
    ocp.solver_options.qp_solver_cond_N = int(ocp.dims.N / 2)

    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", np.zeros((nx,)))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    sim_running = True

    yref = np.array([10.0, 10.0, 0, 0, 0])
    yref_N = np.array([10.0, 10.0, 0])

    while viz.wait():

        for j in range(N_horizon):
            acados_ocp_solver.set(j, "yref", yref)
        acados_ocp_solver.set(N_horizon, "yref", yref_N)

        acados_ocp_solver.set(0, "lbx", x0)
        acados_ocp_solver.set(0, "ubx", x0)

        acados_ocp_solver.solve()
        print(acados_ocp_solver.get_stats("time_tot") * 1000.0)

        sim_u = acados_ocp_solver.get(0, "u")

        x = np.zeros((N_horizon, 3))
        for i in range(N_horizon):
            x[i] = acados_ocp_solver.get(i, "x")

        if viz.begin_figure("Vehicle Position", flags=viz.PlotFlags.EQUAL):
            viz.plot(x[:, 0], x[:, 1], label="trajectory", fmt="-o")
            yref[0], yref[1] = viz.drag_point("target", (yref[0], yref[1]))
            yref_N[:2] = yref[:2]
            viz.plot_dummy("target", legend_color="white")
        viz.end_figure()

        if sim_running:
            acados_integrator.set("T", 0.05)
            x0 = acados_integrator.simulate(x0, sim_u)


def main():
    model = export_robot_model()


if __name__ == "__main__":
    main()
