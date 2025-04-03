import time
import numpy as np
import imviz as viz

from contextlib import contextmanager

import imdash.utils as utils
from imdash.views.view_2d import View2DComponent

from tpl.simulation import SimCar, SimTrafficLight, SimTimeConstraint


class SimulationComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Simulation"

    def __init__(self):

        super().__init__()

        self.label = "simulation"

        self.sim_source = utils.DataSource(path="{/structstores/tpl_sim}")

    @contextmanager
    def locked_sim(self):

        src = self.sim_source.get_used_source()
        if src is None:
            yield None
            return

        try:
            store = src.store
            with store.lock():
                yield store.sim
        except AttributeError:
            yield None
            return

    def render_sim_state(self, sim):

        flags = viz.PlotDragToolFlags.DELAYED
        if self.no_fit:
            flags |= viz.PlotDragToolFlags.NO_FIT

        del_idx = None

        for i, sc in enumerate(sim.cars):

            viz.plot_rect(
                    (sc.x, sc.y),
                    (sc.length, sc.width),
                    color="red",
                    rotation=sc.yaw,
                    flags=flags)

            sc.x, sc.y = viz.drag_point(
                    sc.uuid,
                    (sc.x, sc.y),
                    color="red",
                    flags=flags)

            if utils.begin_context_drag_item(sc.uuid, sc.x, sc.y):
                if viz.begin_menu("Edit"):
                    viz.autogui(sc)
                    viz.end_menu()
                if viz.menu_item("Delete"):
                    del_idx = i
                viz.end_popup()

        if del_idx is not None:
            sim.cars.pop(del_idx)

        del_idx = None

        for i, stl in enumerate(sim.traffic_lights):

            if stl.state == SimTrafficLight.RED:
                color = np.array([1.0, 0.0, 0.0])
            elif stl.state == SimTrafficLight.YELLOW:
                color = np.array([1.0, 1.0, 0.0])
            elif stl.state == SimTrafficLight.GREEN:
                color = np.array([0.0, 1.0, 0.0])
            else:
                color = np.array([0.5, 0.5, 0.5])

            color *= 0.4 + abs(np.sin(time.time() * 2.0)) * 0.6

            stl.x, stl.y = viz.drag_point(
                    stl.uuid,
                    (stl.x, stl.y),
                    color=color,
                    flags=flags)

            viz.plot_circle((stl.x, stl.y),
                            1.0,
                            "###{stl.uuid}",
                            color=color,
                            flags=flags)

            if utils.begin_context_drag_item(stl.uuid, stl.x, stl.y):
                if viz.begin_menu("Edit"):
                    viz.autogui(stl)
                    viz.end_menu()
                if viz.menu_item("Delete"):
                    del_idx = i
                viz.end_popup()

        if del_idx is not None:
            sim.traffic_lights.pop(del_idx)

        del_idx = None

        for i, tc in enumerate(sim.time_constraints):

            tc.x, tc.y = viz.drag_point(
                    tc.uuid,
                    (tc.x, tc.y),
                    color=(1.0, 0.0, 1.0),
                    flags=flags)

            viz.plot_annotation(tc.x, tc.y,
                    f"t_min: {tc.t_min:.2f}, t_max: {tc.t_max:.2f}",
                    (1.0, 0.0, 1.0), (10, 10))

            if utils.begin_context_drag_item(tc.uuid, tc.x, tc.y):
                if viz.begin_menu("Edit"):
                    viz.autogui(tc)
                    viz.end_menu()
                if viz.menu_item("Delete"):
                    del_idx = i
                viz.end_popup()

        if del_idx is not None:
            sim.time_constraints.pop(del_idx)

        sim.ego.x, sim.ego.y = viz.drag_point("ego_pos",
                                              (sim.ego.x, sim.ego.y),
                                              color=1.0,
                                              flags=flags)

    def render_context_menu(self, sim):

        disabled = self.sim_source.get_used_source() is None

        if viz.begin_plot_popup():

            viz.begin_disabled(disabled)
            pp = viz.get_plot_popup_point()

            if viz.begin_menu("Simulation"):
                if viz.menu_item("Running", selected=sim.settings.running, shortcut="Space"):
                    sim.settings.running = not sim.settings.running
                if viz.begin_menu("Create"):
                    if viz.menu_item("Sim car"):
                        sc = SimCar()
                        sc.x = pp[0]
                        sc.y = pp[1]
                        sim.cars.append(sc)
                    if viz.menu_item("Sim traffic light"):
                        stl = SimTrafficLight()
                        stl.x = pp[0]
                        stl.y = pp[1]
                        sim.traffic_lights.append(stl)
                    if viz.menu_item("Sim time constraint"):
                        tc = SimTimeConstraint()
                        tc.x = pp[0]
                        tc.y = pp[1]
                        sim.time_constraints.append(tc)
                    viz.end_menu()
                viz.end_menu()

            viz.separator()
            viz.end_disabled()
        viz.end_plot_popup()

    def update_hotkeys(self, sim):

        if not viz.is_plot_hovered():
            return

        for ke in viz.get_key_events():
            if ke.action != viz.RELEASE and ke.key == viz.KEY_RIGHT:
                with self.locked_sim() as sim:
                    sim.settings.single_step_requested = True
            if ke.action == viz.PRESS and ke.key == viz.KEY_SPACE:
                with self.locked_sim() as sim:
                    sim.settings.running = not sim.settings.running
            if ke.action == viz.PRESS and ke.key == viz.KEY_R:
                with self.locked_sim() as sim:
                    sim.settings.running = False
                    sim.settings.reload_requested = True

    def render(self, idx, view):

        viz.push_mod_any()

        with self.locked_sim() as sim:
            if sim is None:
                raise RuntimeError("cannot connect to simulation")
            self.render_context_menu(sim)
            self.render_sim_state(sim)
            self.update_hotkeys(sim)

        viz.plot_dummy(f"{self.label}###{idx}", "white")

        if viz.pop_mod_any():
            pass
