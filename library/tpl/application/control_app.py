import time
import numpy as np
import os.path as osp
import objtoolbox as otb
import structstore as sts

import tpl.optim.optimizers as opts

from scipy.interpolate import interp1d

from tpl.control import BaseController
from tpl.planning import Trajectory
from tpl.environment import VehicleState
from tpl.util import PATH_PARAMS, get_subclasses_recursive, to_snake_case, project, short_angle_dist, StructStoreRegistry


class ControlInput:

    def __init__(self):

        self.t = 0.0
        self.vehicle = VehicleState()
        self.trajectory = Trajectory()


class ControlStats:

    def __init__(self):

        self.runtime = 0.0
        self.err_d_lat = 0.0
        self.err_vel = 0.0
        self.err_heading = 0.0


class ControlApp:

    SHM_CONTROLLER_INPUT_SIZE = 10**6
    SHM_CONTROLLERS_SIZE = 10**6

    def __init__(self,
                 app_id="",
                 control_params_path=None):

        opts.build_optimizers()

        self.app_id = app_id

        self.sh_input = StructStoreRegistry.get(
                f"/{self.app_id}tpl_control_input" ,
                ControlApp.SHM_CONTROLLER_INPUT_SIZE,
                reinit=True)

        with self.sh_input.lock():
            ci = ControlInput()
            self.sh_input.t = ci.t
            self.sh_input.vehicle = ci.vehicle
            self.sh_input.trajectory = ci.trajectory

        self.controllers = {}
        controller_classes = [BaseController] + get_subclasses_recursive(BaseController)
        controller_names = [to_snake_case(cls.__name__) for cls in controller_classes]

        self.sh_controllers = StructStoreRegistry.get(
                f"/{self.app_id}tpl_control" ,
                ControlApp.SHM_CONTROLLERS_SIZE,
                reinit=True)

        with self.sh_controllers.lock():
            self.sh_controllers.__storage__ = "default"
            self.sh_controllers.__renderer__ = "tpl.gui.state_and_params.render_controllers"
            self.sh_controllers.runtime = 0.0
            self.sh_controllers.stats = ControlStats()
            self.sh_controllers.controls = (0.0, 0.0)
            self.sh_controllers.active_controller = controller_names[0]
            self.sh_controllers.active_controller_state = otb.bundle()
            self.sh_controllers.controller_names = controller_names
            for cls in controller_classes:
                name = to_snake_case(cls.__name__)
                setattr(self.sh_controllers, name, otb.bundle())
                state = getattr(self.sh_controllers, name)
                self.controllers[name] = cls(state, self.sh_controllers.lock)
            load_control_params(self.sh_controllers, control_params_path)

    def calc_stats(self, con_input, runtime):

        veh = con_input.vehicle
        veh_pos = np.array([veh.x, veh.y])

        traj = con_input.trajectory
        traj_path = np.vstack([traj.x, traj.y]).T

        ivel = interp1d(traj.s, traj.velocity, fill_value="extrapolate")

        proj = project(traj_path, veh_pos)

        stats = ControlStats()
        stats.runtime = runtime
        stats.err_d_lat = proj.distance
        stats.err_heading = short_angle_dist(proj.angle, veh.phi)
        stats.err_vel = ivel(proj.arc_len) - veh.v

        return stats

    def update(self):

        runtime_start = time.perf_counter()

        with self.sh_input.lock():
            con_input = self.sh_input.deepcopy()

        with self.sh_controllers.lock():
            active_controller = self.sh_controllers.active_controller
        try:
            controller = self.controllers[active_controller]
        except KeyError:
            controller = self.controllers["base_controller"]

        controls, control_traj = controller.update(con_input)

        runtime = time.perf_counter() - runtime_start

        stats = self.calc_stats(con_input, runtime)

        with self.sh_controllers.lock():
            self.sh_controllers.runtime = runtime
            self.sh_controllers.stats = stats
            self.sh_controllers.controls = controls
            self.sh_controllers.control_trajectory = control_traj


def load_control_params(sh_controllers, path=None):

    if path is None:
        path = sh_controllers.__storage__

    abs_path = osp.join(PATH_PARAMS, "control", path)
    if otb.load(sh_controllers, abs_path):
        sh_controllers.__storage__ = path


def save_control_params(sh_controllers):

    params = otb.bundle()
    params.active_controller = sh_controllers.active_controller

    for cn in sh_controllers.controller_names:
        controller_state = getattr(sh_controllers, cn)
        if hasattr(controller_state, "params"):
            cs = otb.bundle()
            cs.params = getattr(controller_state, "params")
            params[cn] = cs

    abs_path = osp.join(PATH_PARAMS, "control", sh_controllers.__storage__)
    otb.save(params, abs_path)


def main(*args, **kwargs):

    try:
        app = ControlApp(*args, **kwargs)
        while True:
            app.update()
    except KeyboardInterrupt:
        exit(0)
