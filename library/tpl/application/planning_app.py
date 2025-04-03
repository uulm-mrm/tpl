import time
import os.path as osp
import minireload as mr
import objtoolbox as otb
import structstore as sts

import tpl.optim.optimizers as opts

from tpl.planning import BasePlanner, Trajectory
from tpl.environment import SharedEnvironmentState
from tpl.util import PATH_PARAMS, get_subclasses_recursive, to_snake_case, StructStoreRegistry


class PlanningApp:

    SHM_PLANNERS_SIZE = 10**8

    def __init__(self,
                 app_id="",
                 planning_params_path=None):

        opts.build_optimizers()

        self.app_id = app_id
        self.last_time = 0.0

        self.planners = {}
        planner_classes = [BasePlanner] + get_subclasses_recursive(BasePlanner)
        planner_names = [to_snake_case(cls.__name__) for cls in planner_classes]

        self.env = SharedEnvironmentState.fromshared(
                StructStoreRegistry.get(f"/{self.app_id}tpl_env"))
        self.env.wait_for_attr("t")

        self.sh_planners = StructStoreRegistry.get(
                f"/{self.app_id}tpl_planning" ,
                PlanningApp.SHM_PLANNERS_SIZE,
                reinit=True)

        with self.sh_planners.lock():
            self.sh_planners.__storage__ = "default"
            self.sh_planners.__renderer__ = "tpl.gui.state_and_params.render_planners"
            self.sh_planners.runtime = 0.0
            self.sh_planners.trajectory = Trajectory()
            self.sh_planners.active_planner = planner_names[0]
            self.sh_planners.planner_names = planner_names
            for cls in planner_classes:
                name = to_snake_case(cls.__name__)
                setattr(self.sh_planners, name, otb.bundle())
                state = getattr(self.sh_planners, name)
                self.planners[name] = cls(state, self.sh_planners.lock)
            load_planning_params(self.sh_planners, planning_params_path)

        self.last_active_planner = ""

    def update(self):

        # update trajectory

        runtime_start = time.perf_counter()

        do_update = True
        with self.env.lock():
            if self.last_time == self.env.t:
                # only update if time changed
                time.sleep(0.001)
                do_update = False
            self.last_time = self.env.t

        do_update = True

        with self.sh_planners.lock():
            active_planner = self.sh_planners.active_planner
            do_update = do_update or not hasattr(self.sh_planners, "has_new_traj")

        if self.last_active_planner != active_planner:
            with self.env.lock():
                self.env.reset()
        self.last_active_planner = active_planner

        try:
            planner = self.planners[active_planner]
        except KeyError:
            planner = None

        if planner is not None and do_update:
            trajectory = planner.update(self.env)

        if hasattr(planner, "runtime"):
            runtime = planner.runtime
        else:
            runtime = time.perf_counter() - runtime_start

        with self.sh_planners.lock():
            self.sh_planners.runtime = runtime
            if not do_update:
                return
            self.sh_planners.trajectory = trajectory
            self.sh_planners.has_new_traj = True


def load_planning_params(sh_planners, path=None):

    if path is None:
        path = sh_planners.__storage__

    abs_path = osp.join(PATH_PARAMS, "planning", path)
    if otb.load(sh_planners, abs_path):
        sh_planners.__storage__ = path


def save_planning_params(sh_planners):

    params = otb.bundle()
    params.active_planner = sh_planners.active_planner

    for pn in sh_planners.planner_names:
        planner_state = getattr(sh_planners, pn)
        if hasattr(planner_state, "params"):
            ps = otb.bundle()
            ps.params = getattr(planner_state, "params").deepcopy()
            params[pn] = ps

    abs_path = osp.join(PATH_PARAMS, "planning", sh_planners.__storage__)
    otb.save(params, abs_path)


def main(*args, **kwargs):

    app = PlanningApp(*args, **kwargs)
    update_func = mr.WrappingReloader(app.update)

    try:
        while True:
            update_func()
    except KeyboardInterrupt:
        exit(0)
