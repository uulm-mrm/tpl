import time
import copy
import uuid
import imviz as viz
import numpy as np

from imdash.utils import begin_context_drag_item
from tpl.environment.map_module import TrafficLight


class MapObjectsRenderer:

    def __init__(self, no_fit=True, show_all=False):

        self.no_fit = no_fit
        self.show_all = show_all

        self.map_store = None

        self.update_flags()

    def update_flags(self):

        self.line_flags = viz.PlotLineFlags.NONE
        if self.no_fit:
            self.line_flags |= viz.PlotItemFlags.NO_FIT

        self.drag_tool_flags = viz.PlotDragToolFlags.DELAYED
        if self.no_fit:
            self.drag_tool_flags |= viz.PlotDragToolFlags.NO_FIT

    def render_map_store(self, store, selected_map=""):

        self.map_store = store

        for k, m in self.map_store.items():
            if k == selected_map:
                self.render(m)
            elif self.show_all:
                self.render(m)

    def render(self, obj):
        
        render_func_name = "render_" + obj.__class__.__name__

        if hasattr(self, render_func_name):
            getattr(self, render_func_name)(obj)
        elif hasattr(obj, "__tag__"):
            getattr(self, "render_" + obj.__tag__)(obj)
        else:
            print("MapObjectsRenderer: cannot render object!")
            print("     Object:", obj)

    def menu(self, obj):

        menu_func_name = "menu_" + obj.__class__.__name__

        try:
            if hasattr(self, menu_func_name):
                getattr(self, menu_func_name)(obj)
            elif hasattr(obj, "__tag__"):
                getattr(self, "menu_" + obj.__tag__)(obj)
        except AttributeError:
            viz.autogui(obj)

    def map_selector(self, label, selected_map):

        uuids = ["none"] + list(self.map_store.keys())
        names = [""] + [m.name for m in self.map_store.values()]
        try:
            active_idx = uuids.index(selected_map)
        except ValueError:
            active_idx = 0
        active_idx = viz.combo(label, names, active_idx)
        sel = uuids[active_idx]

        return sel

    def base_point(self, obj, col=(0.5, 0.0, 0.0), radius=6):

        obj_name = str(obj.uuid)
        res = np.array(viz.drag_point(
                obj_name,
                obj.pos,
                color=col,
                radius=radius,
                flags=self.drag_tool_flags))
        obj.pos = res

    def render_map(self, obj):

        self.update_flags()

        self.render(obj.velocity_limits)
        self.render(obj.turn_ind_points)
        self.render(obj.map_switch_points)
        self.render(obj.intersection_paths)

    def render_StructStoreList(self, obj):

        self.render_list(obj)

    def render_list(self, obj):

        obj_deleted = None
        obj_duplicated = None

        for i, o in enumerate(obj):
            self.render(o)
            obj_name = str(o.uuid)
            if begin_context_drag_item(obj_name, o.pos[0], o.pos[1]):
                if viz.begin_menu("Edit"):
                    self.menu(o)
                    viz.end_menu()
                if hasattr(o, "active"):
                    if not o.active:
                        if viz.menu_item("Activate"):
                            o.active = True
                    else:
                        if viz.menu_item("Deactivate"):
                            o.active = False
                if viz.menu_item("Duplicate"):
                    obj_duplicated = i
                if viz.menu_item("Delete"):
                    obj_deleted = i
                viz.end_popup()

        if obj_deleted is not None:
            obj.pop(obj_deleted)
            viz.set_mod(True)

        if obj_duplicated is not None:
            new_obj = copy.deepcopy(obj[obj_duplicated])
            new_obj.uuid = uuid.uuid4().hex
            obj.insert(obj_duplicated, new_obj)
            viz.set_mod(True)

    def render_traffic_light(self, obj):

        if obj.state == TrafficLight.RED:
            light_col = (1.0, 0.0, 0.0)
        elif obj.state == TrafficLight.YELLOW:
            light_col = (1.0, 1.0, 0.0)
        elif obj.state == TrafficLight.GREEN:
            light_col = (0.0, 1.0, 0.0)
        else:
            light_col = (0.2, 0.2, 0.2)

        viz.plot([obj.pos[0], obj.light_pos[0]],
                 [obj.pos[1], obj.light_pos[1]],
                 "-",
                 color=(1.0, 1.0, 1.0),
                 flags=self.line_flags)

        obj.light_pos[:] = viz.drag_point(f"{obj.uuid}_light",
                                          obj.light_pos,
                                          color=light_col,
                                          radius=10,
                                          flags=self.drag_tool_flags)

        viz.plot_circle(obj.light_pos,
                        obj.detection_radius,
                        color=light_col,
                        flags=self.line_flags)

        self.base_point(obj, col=(1.0, 1.0, 1.0))

    def render_cross_walk(self, obj):

        xs = [*obj.corners[:, 0], obj.corners[0, 0]]
        ys = [*obj.corners[:, 1], obj.corners[0, 1]]
        
        col = (1.0, 1.0, 1.0)

        viz.plot(xs, ys, color=col)
        viz.plot([obj.pos[0], np.mean(obj.corners[:, 0])],
                 [obj.pos[1], np.mean(obj.corners[:, 1])],
                 "-o",
                 color=col,
                 flags=self.line_flags)

        for i, c in enumerate(obj.corners):
            c[0], c[1] = viz.drag_point(f"{obj.uuid}_{i}",
                    c, color=(1.0, 1.0, 1.0), radius=2, flags=self.drag_tool_flags)

        self.base_point(obj, col=col)

    def render_velocity_limit(self, obj):

        if obj.active:
            col = (1.0, 0.0, 0.0)
        else:
            col = (0.5, 0.0, 0.0)

        self.base_point(obj, col=col)

    def render_turn_ind_point(self, obj):

        viz.plot_circle(obj.pos,
                        obj.activation_radius,
                        color=(1.0, 0.5, 0.0),
                        line_weight=1.0,
                        flags=self.line_flags)

        self.base_point(obj, col=(1.0, 0.5, 0.0))

    def render_map_switch_point(self, obj):

        viz.plot_circle(obj.pos,
                        obj.activation_radius,
                        color=(1.0, 0.0, 0.5),
                        line_weight=1.0,
                        flags=self.line_flags)

        self.base_point(obj, col=(1.0, 0.0, 0.5))

    def menu_map_switch_point(self, obj):

        obj.pos = viz.autogui(obj.pos, "pos")
        obj.trigger_divisor = viz.autogui(obj.trigger_divisor, "trigger_divisor")
        obj.triggers = viz.autogui(obj.triggers, "triggers")
        obj.activation_radius = viz.autogui(obj.activation_radius, "activation_radius")
        obj.in_radius = viz.autogui(obj.triggers, "in_radius")
        obj.target_uuid = self.map_selector("target_map", obj.target_uuid)

    def render_checkpoint(self, obj):

        viz.plot_circle(obj.pos,
                        obj.activation_radius,
                        color=(1.0, 1.0, 0.0),
                        line_weight=1.0,
                        flags=self.line_flags)

        self.base_point(obj, col=(1.0, 1.0, 0.0))

    def render_dynamic_object(self, obj):

        self.base_point(obj, col=(1.0, 1.0, 0.0))

    def render_time_constr(self, obj):

        self.base_point(obj, col=(1.0, 0.0, 0.5))

    def render_intersection_path(self, obj):

        if obj.stop:
            col_stop = (1.0, 0.0, 0.0)
        else:
            col_stop = (0.5, 0.5, 0.5)

        viz.plot([obj.pos[0], obj.stop_pos[0]],
                 [obj.pos[1], obj.stop_pos[1]],
                 fmt="-",
                 color=col_stop)

        self.base_point(obj, col=(1.0, 1.0, 1.0))

        obj.stop_pos = np.array(viz.drag_point(
                f"{obj.uuid}_stop",
                obj.stop_pos,
                radius=6.0,
                color=col_stop))

        if obj.map_segment is not None:
            viz.plot(obj.map_segment.path[:, 0],
                     obj.map_segment.path[:, 1],
                     fmt="-",
                     line_weight=5.0,
                     color="white")

    def menu_intersection_path(self, obj):

        obj.pos = viz.autogui(obj.pos, "pos")
        obj.stop_pos = viz.autogui(obj.stop_pos, "stop_pos")
        obj.offset_path_begin = viz.autogui(obj.offset_path_begin, "offset_path_begin")
        obj.offset_path_end = viz.autogui(obj.offset_path_end, "offset_path_end")
        obj.intersection_map_uuid = self.map_selector("intersection_map", obj.intersection_map_uuid)
        obj.map_segment_step_size = viz.autogui(obj.map_segment_step_size, "map_segment_step_size")
        obj.v_max_approach = viz.autogui(obj.v_max_approach, "v_max_approach")
        obj.stop_always = viz.autogui(obj.stop_always, "stop_always")
        obj.stop_for_any = viz.autogui(obj.stop_for_any, "stop_for_any")
        obj.gap_acceptance = viz.autogui(obj.gap_acceptance, "gap_acceptance")
        obj.gap_hysteresis = viz.autogui(obj.gap_hysteresis, "gap_hysteresis")
