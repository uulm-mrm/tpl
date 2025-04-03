import os
import copy
import time
import uuid
import traceback

import numpy as np
import imviz as viz
import objtoolbox as otb

import imdash.utils as utils

import tpl.util as tpl_utils
from tpl.environment import (
        Map,
        VelocityLimit,
        TrafficLight,
        CrossWalk,
        TurnIndPoint,
        MapSwitchPoint,
        IntersectionPath
    )

import tpl.environment.map_module as map_module

from tpl.gui.renderers import MapObjectsRenderer
from tpl.gui.renderers import MapPathsRenderer
from imdash.views.view_2d import View2D


class MapEditor(View2D):

    DISPLAY_NAME = "Tpl/Map editor"

    def __init__(self):

        super().__init__()

        self.title = "Map Editor"

        self.map_store_path = None
        self.maps = {}
        self.selected_map = None

        self.show_velocity = True
        self.show_d_left = True
        self.show_d_right = True
        self.show_altitude = True
        self.show_curvature = True
        self.show_all_paths = True

        self.menu_file_path = tpl_utils.PATH_MAPS
        self.menu_file_func = None
        self.menu_file_path_needed = False

        self.path_editor_reset()
        self.path_editor_focused = False

        self.show_as_kph = False
        self.vel_set_value = 0.0

        self.set_value = 0.0

        self.recording = False
        self.recorded_len = 0.0
        self.recorded_path = []
        self.rec_x_source = utils.DataSource()
        self.rec_y_source = utils.DataSource()
        self.rec_z_source = utils.DataSource()
        self.rec_sources_err = ""

        self.save_req = False
        self.save_req_time = 0.0

        self.proj_point_idx = None

    def __savestate__(self):

        d = self.__dict__.copy()

        if d["map_store_path"] is not None:
            d["map_store_path"] = os.path.relpath(
                    d["map_store_path"],
                    os.getcwd())

        if d["menu_file_path"] is not None:
            d["menu_file_path"] = os.path.relpath(
                    d["menu_file_path"],
                    os.getcwd())

        del d["maps"]
        del d["menu_file_func"]
        del d["menu_file_path_needed"]
        del d["selection"]
        del d["recording"]
        del d["recorded_len"]
        del d["recorded_path"]
        del d["save_req"]
        del d["save_req_time"]
        del d["proj_point_idx"]

        return d

    def __loadstate__(self, s):

        if "map_store_path" in s and s["map_store_path"] is not None:
            s["map_store_path"] = os.path.abspath(os.path.normpath(
                    os.path.join(os.getcwd(), s["map_store_path"])))

        if "menu_file_path" in s and s["menu_file_path"] is not None:
            s["menu_file_path"] = os.path.abspath(os.path.normpath(
                    os.path.join(os.getcwd(), s["menu_file_path"])))

        self.__dict__.update(s)
        self.open_store(self.map_store_path)

    def path_editor_reset(self):

        self.selection = np.zeros((0, 2))

    def open_store(self, path=None):

        self.map_store_path = path

        if self.map_store_path is not None:
            self.maps = map_module.load_map_store(
                    self.map_store_path, mmap_arrays=True)

    def cmap(self):

        try:
            return self.maps[self.selected_map]
        except KeyError:
            return None

    def get_path_length(self, cmap):

        vs = cmap.path[:-1, :2] 
        ds = np.linalg.norm(vs, axis=1)
        return float(np.sum(ds))

    def import_map_from_csv(self, file_path):

        route = tpl_utils.load_route_from_csv(file_path)

        new_map = Map()
        new_map.name = os.path.basename(file_path).split(".")[0]
        new_map.control_points = np.zeros((len(route), 5))
        new_map.control_points[:, 0] = route[:, 0]
        new_map.control_points[:, 1] = route[:, 1]
        new_map.control_points[:, 2] = 2.0
        new_map.control_points[:, 3] = 2.0
        new_map.control_points[:, 4] = route[:, 5]
        map_module.reinit_map(new_map)

        self.maps[new_map.uuid] = new_map

    def export_map_as_csv(self, path):

        cmap = self.cmap()
        if cmap is None:
            return

        np.savetxt(path,
               cmap.path[:, [0,1,4]],
               delimiter=",",
               header="utm_east,utm_north,curvature",
               comments="")

    def export_vel_lim_as_csv(self, path):

        cmap = self.cmap()
        if cmap is None:
            return

        vl_points = []
        for vl in cmap.velocity_limits:
            vl.proj = tpl_utils.project(np.array(cmap.path[:, :2]), vl.pos)
            vl_points.append(vl.proj.point)
        vl_points = np.array(vl_points)

        np.savetxt(path,
                   vl_points,
                   delimiter=",",
                   header="utm_east,utm_north",
                   comments="")

    def export_map_vel_lim_as_csv(self, path):

        cmap = self.cmap()
        if cmap is None:
            return

        folder_path = os.path.join(path, cmap.name)

        os.makedirs(folder_path, exist_ok=True)

        self.export_map_as_csv(os.path.join(folder_path, "map.csv"))
        self.export_vel_lim_as_csv(os.path.join(folder_path, "stop_points.csv"))

    def defer_file_func(self, func):

        self.menu_file_func = func
        self.menu_file_path_needed = True

    def handle_menu_file_func(self):

        if self.menu_file_path_needed:
            viz.open_popup("Select path")

        self.menu_file_path = viz.file_dialog_popup(
                "Select path", self.menu_file_path)
        if viz.mod():
            self.menu_file_func(self.menu_file_path)

        self.menu_file_path_needed = False

    def render_create_menu(self, pp):

        cmap = self.cmap()
        has_cmap = cmap is not None

        if viz.begin_menu("Create"):
            viz.separator()
            if viz.menu_item("Map point", shortcut="A", enabled=has_cmap):
                self.insert_point()
            if viz.menu_item("Velocity limit", enabled=has_cmap):
                vl = VelocityLimit()
                vl.pos = pp
                cmap.velocity_limits.append(vl)
            if viz.menu_item("Traffic light", enabled=has_cmap):
                tl = TrafficLight()
                tl.pos = pp
                tl.light_pos = tl.pos + (5.0, 5.0)
                cmap.velocity_limits.append(tl)
            if viz.menu_item("Crosswalk", enabled=has_cmap):
                vl = CrossWalk()
                vl.pos = pp
                vl.corners = np.array([
                        [-5.0, -5.0],
                        [5.0, -5.0],
                        [5.0, 5.0],
                        [-5.0, 5.0]
                    ]) + vl.pos
                cmap.velocity_limits.append(vl)
            if viz.menu_item("Turn indicator point", enabled=has_cmap):
                vl = TurnIndPoint()
                vl.pos = pp
                cmap.turn_ind_points.append(vl)
            if viz.menu_item("Map switch point", enabled=has_cmap):
                vl = MapSwitchPoint()
                vl.pos = pp
                cmap.map_switch_points.append(vl)
            if viz.menu_item("Intersection path", enabled=has_cmap):
                ip = IntersectionPath(pp)
                cmap.intersection_paths.append(ip)
            viz.end_menu()

    def render_menu_bar(self):

        if viz.begin_menu_bar():
            if viz.begin_menu("File"):
                if viz.menu_item("Open store"):
                    self.defer_file_func(self.open_store)
                if viz.menu_item("Save store"):
                    viz.set_mod(True)
                viz.separator()
                if viz.menu_item("New empty map"):
                    m = Map()
                    map_module.reinit_map(m)
                    map_module.reinit_map_items(m, self.maps)
                    self.maps[m.uuid] = m
                if viz.menu_item("New map from csv"):
                    self.defer_file_func(lambda p: print(p))
                if viz.begin_menu("Import"):
                    if viz.menu_item("Map from csv"):
                        self.defer_file_func(self.import_map_from_csv)
                    viz.end_menu()
                if viz.begin_menu("Export"):
                    if viz.menu_item("Map as csv"):
                        self.defer_file_func(self.export_map_as_csv)
                    if viz.menu_item("Velocity limits csv"):
                        self.defer_file_func(self.export_vel_lim_as_csv)
                    if viz.menu_item("Map and vel. limits as csv"):
                        self.defer_file_func(self.export_map_vel_lim_as_csv)
                    viz.end_menu()
                viz.separator()
                if viz.menu_item("Delete view"):
                    self.destroyed = True
                viz.end_menu()
            if viz.begin_menu("Edit"):
                if viz.menu_item("Reverse map direction"):
                    self.reverse_map_direction(self.cmap())
                viz.end_menu()
            if viz.begin_menu("Show"):
                if viz.menu_item("Velocity", selected=self.show_velocity):
                    self.show_velocity = not self.show_velocity
                if viz.menu_item("Right border", selected=self.show_d_right):
                    self.show_d_right = not self.show_d_right
                if viz.menu_item("Left border", selected=self.show_d_left):
                    self.show_d_left = not self.show_d_left
                if viz.menu_item("Curvature", selected=self.show_curvature):
                    self.show_curvature = not self.show_curvature
                if viz.menu_item("All paths", selected=self.show_all_paths):
                    self.show_all_paths = not self.show_all_paths
                viz.end_menu()
            viz.end_menu_bar()

    def render_context_menu(self):

        if viz.begin_plot_popup():
            pp = viz.get_plot_popup_point()
            self.render_create_menu(pp)
        viz.end_plot_popup()

    def reverse_map_direction(self, cmap):

        cmap.control_points = cmap.control_points[::-1]
        map_module.reinit_map(cmap)
        viz.set_mod(True)

    def record_map_data(self):

        try:
            x = self.rec_x_source()
            y = self.rec_y_source()
            z = self.rec_z_source()
            if self.recording:
                if self.recorded_path == []:
                    self.recorded_path.append([x, y, z])
                else:
                    l = np.linalg.norm(np.array(self.recorded_path[-1])[:2]
                                       - np.array([x, y]))
                    if l > 0.5:
                        self.recorded_path.append([x, y, z])
                        self.recorded_len += l
            self.rec_sources_err = ""
        except Exception as e:
            self.rec_sources_err = traceback.format_exc()

    def render_recording_window(self, sources):

        cmap = self.cmap()

        w, _ = viz.get_content_region_avail()

        if viz.begin_window("Map Editor - Recording"):

            viz.autogui(self.rec_x_source, "x", sources=sources)
            viz.autogui(self.rec_y_source, "y", sources=sources)
            viz.autogui(self.rec_z_source, "z", sources=sources)

            if not self.recording:
                if self.rec_sources_err == "":
                    if viz.button("record"):
                        self.recording = True
                else:
                    viz.text("Error", color="red")
                    if viz.is_item_hovered():
                        viz.begin_tooltip()
                        viz.text(self.rec_sources_err)
                        viz.end_tooltip()
            else:
                if viz.button("save to new###rec_save_new"):
                    rec_path_np = np.array(self.recorded_path)
                    control_points = np.zeros((len(self.recorded_path), 6))
                    control_points[:, :2] = rec_path_np[:, :2]
                    control_points[:, 2] = 2.0
                    control_points[:, 3] = 2.0
                    control_points[:, 4] = 0.0
                    control_points[:, 5] = rec_path_np[:, 2]

                    m = Map()
                    m.control_points = control_points
                    self.maps[m.uuid] = m
                    self.selected_map = m.uuid
                    map_module.reinit_map(m)

                    self.recorded_path = []
                    self.recorded_len = 0.0
                    self.recording = False
                    self.path_editor_reset()

                    viz.set_mod(True)

                viz.same_line()

                cmap = self.cmap()
                if cmap is not None:
                    if viz.button("save to current###rec_save_current"):
                        rec_path_np = np.array(self.recorded_path)
                        control_points = np.zeros((len(self.recorded_path), 6))
                        control_points[:, :2] = rec_path_np[:, :2]
                        control_points[:, 2] = 2.0
                        control_points[:, 3] = 2.0
                        control_points[:, 4] = 0.0
                        control_points[:, 5] = rec_path_np[:, 2]
                        cmap.control_points = control_points
                        map_module.reinit_map(cmap)

                        self.recorded_path = []
                        self.recorded_len = 0.0
                        self.recording = False
                        self.path_editor_reset()

                viz.same_line()

                if viz.button("cancel###rec_cancel"):
                    self.recorded_path = []
                    self.recorded_len = 0.0
                    self.recording = False

        viz.end_window()

    def setup_plot(self):

        viz.setup_axes("utm_east [m]", "utm_north [m]")

        if viz.plot_selection_ended():
            viz.hard_cancel_plot_selection()

    def render_path_editor(self):

        x_min, y_min, x_max, y_max = viz.get_plot_limits()
        subsample_step = min(x_max - x_min, y_max - y_min) / 30
        subsample_step = max(1, min(100, int(subsample_step)))

        map_paths_renderer = MapPathsRenderer(self.show_all_paths)
        map_paths_renderer.name = "maps"
        map_paths_renderer.render_map_store(
                self.maps,
                self.selected_map)

        cmap = self.cmap()

        if cmap is None:
            return

        if cmap.control_points.shape[0] == 0:
            cmap.control_points = np.zeros((0, 6))
        
        indices = np.array(range(len(cmap.control_points))).reshape((-1, 1))
        points = np.concatenate([indices, cmap.control_points], axis=1)

        viz.plot(points[:, 1], points[:, 2],
                 label=f"{cmap.name}",
                 color=(0.2, 0.6, 0.8),
                 fmt="o")

        sel = viz.get_plot_selection()

        if len(points) > 0: 

            pos_plot_mouse = viz.get_plot_mouse_pos()
            proj = tpl_utils.project(points[:, 1:3], pos_plot_mouse)
            focused = viz.is_window_focused()

            if (viz.get_mouse_drag_delta() == 0.0).all():
                self.proj_point_idx = max(0, min(len(points) - 1, proj.index))

            if np.sum(sel) != 0.0:
                self.selection = points[
                        (points[:, 1] >= sel[0])
                        & (points[:, 2] >= sel[1])
                        & (points[:, 1] <= sel[2])
                        & (points[:, 2] <= sel[3])]
            elif len(points) > 0 and len(self.selection) < 2 and focused:
                self.selection = points[np.newaxis, self.proj_point_idx]
            else:
                if not focused and self.path_editor_focused:
                    self.path_editor_reset()
            self.path_editor_hovered = focused

            # positional drag points

            diff = np.array([0.0, 0.0])
            for i, p in enumerate(self.selection):
                new_pos = viz.drag_point(
                        f"{i}_sel_point",
                        p[1:3],
                        color=(1.0, 1.0, 0.0),
                        flags=viz.PlotDragToolFlags.DELAYED)
                if viz.mod():
                    diff = new_pos - p[1:3]

            idx = self.selection[:, 0].astype("int")

            if (diff != 0.0).any():
                self.selection[:, 1:3] += diff
                cmap.control_points[np.array(idx), 0:2] = self.selection[:, 1:3]

        # keyboard bindings

        if viz.is_window_focused():
            for ke in viz.get_key_events():

                if ke.action == viz.RELEASE:
                    continue

                if ke.key == viz.KEY_A:
                    self.insert_point(use_mouse_pos=True)

                if len(self.selection) == 0:
                    continue

                mod = False
                idx = self.selection[:, 0].astype("int")

                step = 1.0
                if ke.mod == viz.MOD_SHIFT:
                    step = 10
                if ke.mod == viz.MOD_ALT:
                    step = 0.1

                if ke.key == viz.KEY_LEFT:
                    self.selection[:, 1] -= step
                    mod = True
                if ke.key == viz.KEY_RIGHT:
                    self.selection[:, 1] += step
                    mod = True
                if ke.key == viz.KEY_UP:
                    self.selection[:, 2] += step
                    mod = True
                if ke.key == viz.KEY_DOWN:
                    self.selection[:, 2] -= step
                    mod = True

                if mod:
                    cmap.control_points[idx] = self.selection[:, 1:]
                    viz.set_mod(True)

                if ke.key == viz.KEY_BACKSPACE or ke.key == viz.KEY_DELETE:
                    cmap.control_points = np.delete(cmap.control_points, idx.reshape(-1), axis=0)
                    self.selection = np.zeros((0, cmap.control_points.shape[1]))
                    viz.set_mod(True)

        if viz.mod_any():
            map_module.reinit_map(cmap)
            map_module.reinit_map_items(cmap, self.maps)

    def render_map_objects(self):

        cmap = self.cmap()
        if cmap is None:
            return

        viz.push_mod_any()

        renderer = MapObjectsRenderer()
        renderer.render_map_store(self.maps, self.selected_map)

        if viz.pop_mod_any():
            map_module.reinit_map_items(cmap, self.maps)

    def render_recorded_path(self):

        if self.recording and len(self.recorded_path) > 0:
            path = np.array(self.recorded_path)
            l = round(self.recorded_len, 2)
            viz.plot(path[:, 0], path[:, 1],
                     color=(1.0, 0.5, 0.0),
                     line_weight=2.0,
                     label=f"recorded [{l:.2f}m]###recorded")

    def render_velocity_toolbar(self):

        cmap = self.cmap()

        set_to = False

        if viz.button("set to"):
            set_to = True
        viz.same_line()

        self.vel_set_value = viz.drag("value", self.vel_set_value)

        if self.show_as_kph:
            cf = 3.6
        else:
            cf = 1.0
        
        if len(self.selection) > 0 and (set_to or viz.mod()):
            idx = list(self.selection[:, 0].astype("int"))
            cmap.control_points[idx, 4] = self.vel_set_value / cf
        elif set_to:
            cmap.control_points[:, 4] = self.vel_set_value / cf

    def render_velocity_editor(self):

        cmap = self.cmap()
        if cmap is None:
            return

        if viz.plot_selection_ended():
            viz.hard_cancel_plot_selection()

        viz.setup_axes("s [m]", "velocity [km/h]"
                       if self.show_as_kph else "velocity [m/s]")

        if self.show_as_kph:
            cf = 3.6
        else:
            cf = 1.0

        indices = np.array(range(len(cmap.control_points))).reshape((-1, 1))
        points = np.concatenate([indices, cmap.control_points], axis=1)

        dists = np.linalg.norm(np.diff(cmap.control_points[:, :2], axis=0), axis=1)
        dists_cum = np.array([0.0, *np.cumsum(dists)])

        points[:, 5] *= cf

        viz.plot(dists_cum,
                 points[:, 5],
                 color=(0.2, 0.6, 0.8),
                 fmt="-o",
                 label=f"velocity")
        sel = viz.get_plot_selection()

        if np.sum(sel) != 0.0:
            self.selection = points[
                    (dists_cum >= sel[0])
                    & (points[:, 5] >= sel[1])
                    & (dists_cum <= sel[2])
                    & (points[:, 5] <= sel[3])]

        if len(self.selection) > 0:

            idx = self.selection[:, 0].astype("int")

            self.selection[:, 1:] = cmap.control_points[idx]
            self.selection[:, 5] *= cf

            diff = 0.0
            for i, p in enumerate(self.selection):
                new_pos = viz.drag_point(
                        f"{i}_sel_point", (dists_cum[int(p[0])], p[5]), color=(1.0, 1.0, 0.0))
                if viz.mod():
                    diff = new_pos[1] - p[5]

            if diff != 0.0:
                self.selection[:, 5] += diff
                cmap.control_points[idx, 4] = self.selection[:, 5] / cf

            if viz.is_window_focused():
                for ke in viz.get_key_events():
                    if ke.action == viz.RELEASE:
                        continue
                    if ke.key == viz.KEY_DOWN:
                        self.selection[:, 5] -= 1.0
                        cmap.control_points[idx, 4] = self.selection[:, 5] / cf
                        viz.set_mod(True)
                    if ke.key == viz.KEY_UP:
                        self.selection[:, 4] += 1.0
                        cmap.control_points[idx, 4] = self.selection[:, 5] / cf
                        viz.set_mod(True)

                    if ke.key == viz.KEY_BACKSPACE or ke.key == viz.KEY_DELETE:
                        cmap.control_points = np.delete(cmap.control_points, idx.reshape(-1), axis=0)
                        self.selection = np.zeros((0, cmap.control_points.shape[1]))
                        viz.set_mod(True)

    def render_val_toolbar(self, idx_val):

        viz.push_mod_any()

        cmap = self.cmap()

        set_to = False

        if viz.button("set to"):
            set_to = True
        viz.same_line()

        self.set_value = viz.drag("value", self.set_value)

        if len(self.selection) > 0 and (set_to or viz.mod()):
            idx = list(self.selection[:, 0].astype("int"))
            cmap.control_points[idx, idx_val] = self.set_value
            viz.set_mod(True)
        elif set_to:
            cmap.control_points[:, idx_val] = self.set_value
            viz.set_mod(True)

        if viz.pop_mod_any():
            map_module.reinit_map(cmap)
            map_module.reinit_map_items(cmap, self.maps)

    def render_val_editor(self, label, idx_val):

        cmap = self.cmap()
        if cmap is None:
            return

        if viz.plot_selection_ended():
            viz.hard_cancel_plot_selection()

        viz.setup_axes("s [m]", label)

        if len(cmap.control_points) > 0:

            indices = np.array(range(len(cmap.control_points))).reshape((-1, 1))
            points = np.concatenate([indices, cmap.control_points], axis=1)

            dists = np.linalg.norm(np.diff(cmap.control_points[:, :2], axis=0), axis=1)
            dists_cum = np.array([0.0, *np.cumsum(dists)])

            viz.plot(dists_cum,
                     points[:, idx_val+1],
                     color=(0.2, 0.6, 0.8),
                     fmt="-o",
                     label=label)
            sel = viz.get_plot_selection()

            if np.sum(sel) != 0.0:
                self.selection = points[
                        (dists_cum >= sel[0])
                        & (points[:, idx_val+1] >= sel[1])
                        & (dists_cum <= sel[2])
                        & (points[:, idx_val+1] <= sel[3])]

        if len(self.selection) > 0:

            viz.push_mod_any()

            idx = self.selection[:, 0].astype("int")

            self.selection[:, 1:] = cmap.control_points[idx]

            diff = 0.0
            for i, p in enumerate(self.selection):
                new_pos = viz.drag_point(
                        f"{i}_sel_point", (dists_cum[int(p[0])], p[idx_val+1]), color=(1.0, 1.0, 0.0))
                if viz.mod():
                    diff = new_pos[1] - p[idx_val+1]

            if diff != 0.0:
                self.selection[:, idx_val+1] += diff
                cmap.control_points[idx, idx_val] = self.selection[:, idx_val+1]

            if viz.is_window_focused():
                for ke in viz.get_key_events():
                    if ke.action == viz.RELEASE:
                        continue
                    if ke.key == viz.KEY_DOWN:
                        self.selection[:, idx_val+1] -= 1.0
                        cmap.control_points[idx, idx_val] = self.selection[:, idx_val+1]
                        viz.set_mod(True)
                    if ke.key == viz.KEY_UP:
                        self.selection[:, idx_val+1] += 1.0
                        cmap.control_points[idx, idx_val] = self.selection[:, idx_val+1]
                        viz.set_mod(True)

                    if ke.key == viz.KEY_BACKSPACE or ke.key == viz.KEY_DELETE:
                        cmap.control_points = np.delete(cmap.control_points, idx.reshape(-1), axis=0)
                        self.selection = np.zeros((0, cmap.control_points.shape[1]))
                        viz.set_mod(True)

            if viz.pop_mod_any():
                map_module.reinit_map(cmap)
                map_module.reinit_map_items(cmap, self.maps)

    def render_velocity_context_menu(self):

        if viz.begin_plot_popup():
            if viz.menu_item("Show as km/h", selected=self.show_as_kph):
                self.show_as_kph = not self.show_as_kph
        viz.end_plot_popup()

    def insert_point(self, use_mouse_pos=False):

        cmap = self.cmap()
        if cmap is None:
            return

        if use_mouse_pos:
            pp = viz.get_plot_mouse_pos()
        else:
            pp = viz.get_plot_popup_point()

        proj = tpl_utils.project(cmap.control_points[:, :2], pp)
        if len(cmap.control_points) > 0:
            new_point = cmap.control_points[np.newaxis, proj.start].copy()
        else:
            new_point = np.zeros((1, cmap.control_points.shape[1]))
            new_point[0, 2] = 2.0
            new_point[0, 3] = 2.0
            new_point[0, 4] = 30.0 / 3.6
        new_point[0, :2] = pp
        i = proj.end
        if not proj.in_bounds:
            if proj.start == 0:
                i = 0
            else:
                i = len(cmap.control_points)
        cmap.control_points = np.insert(cmap.control_points, i, new_point, axis=0)

        self.path_editor_reset()

        viz.set_mod(True)

    def render_map_info(self):

        cmap = self.cmap()
        if cmap is None:
            return

        viz.setup_axis(viz.Axis.X1, "s [m]")
        viz.setup_axis(viz.Axis.Y1, "orientation [rad]")
        viz.setup_axis(viz.Axis.Y2, "curvature [1/m]")

        viz.set_axis(viz.Axis.Y1)
        viz.plot(cmap.path[:, 3],
                 cmap.path[:, 2],
                 fmt="-",
                 label="orientation")

        viz.set_axis(viz.Axis.Y2)
        viz.plot(cmap.path[:, 3],
                 cmap.path[:, 4],
                 fmt="-",
                 label="curvature")

    def render_maps_window(self):

        remove_map = None
        duplicate_map = None
        
        cmap = self.cmap()

        if viz.begin_window("Map Editor - Maps"):
            for k, m in self.maps.items():
                flags = viz.TreeNodeFlags.NONE
                if cmap is not None and cmap.uuid == m.uuid:
                    flags |= viz.TreeNodeFlags.SELECTED
                viz.push_id(m.uuid)
                tree_open = viz.tree_node(f"{m.name}###{m.uuid}", flags=flags)
                if viz.begin_popup_context_item():
                    if viz.menu_item("Select"):
                        self.selected_map = k
                        self.path_editor_reset()
                        viz.set_mod(True)
                    if viz.menu_item("Duplicate"):
                        duplicate_map = k
                    if viz.menu_item("Remove"):
                        remove_map = k
                    viz.end_popup()
                if tree_open:
                    viz.push_id(m.uuid)
                    viz.push_mod_any()
                    viz.autogui(m)
                    if viz.pop_mod_any():
                        map_module.reinit_map(m)
                        map_module.reinit_map_items(m, self.maps)
                    viz.pop_id()
                    m.uuid = k
                    viz.tree_pop()
                viz.pop_id()
        viz.end_window()

        if duplicate_map is not None:
            keys_list = list(self.maps.keys())
            i = keys_list.index(duplicate_map)

            new_map = copy.deepcopy(self.maps[duplicate_map])
            new_map.uuid = uuid.uuid4().hex
            self.maps[new_map.uuid] = new_map
            keys_list.insert(i, new_map.uuid)

            self.maps = {k: self.maps[k] for k in keys_list}
            duplicate_map = None
            viz.set_mod(True)

        if remove_map is not None:
            del self.maps[remove_map]
            remove_map = None
            viz.set_mod(True)

    def render(self, sources):

        self.record_map_data()

        if self.maps == None:
            self.maps = {}
            self.open_store()

        if not self.show:
            return

        viz.push_mod_any()

        window_open = viz.begin_window(f"{self.title}###{self.uuid}", menu_bar=True)
        self.show = viz.get_window_open()
        if window_open:
            self.render_menu_bar()
            if viz.begin_figure(f"{self.title}###{self.uuid}",
                                flags=viz.PlotFlags.EQUAL 
                                      | viz.PlotFlags.NO_TITLE
                                      | viz.PlotFlags.NONE):
                self.setup_plot()
                self.render_components(sources)
                self.render_path_editor()
                self.render_context_menu()
                self.render_map_objects()
                self.render_recorded_path()
                self.render_maps_window()
                self.render_recording_window(sources)
            viz.end_figure()
        viz.end_window()

        if self.show_velocity:
            window_open = viz.begin_window(
                    f"{self.title} - Velocity###{self.uuid}velocity")
            self.show_velocity = viz.get_window_open()
            if window_open:
                self.render_velocity_toolbar()
                if viz.begin_figure(f"{self.title} - Velocity###{self.uuid}velocity",
                                    flags=viz.PlotFlags.NO_TITLE
                                          | viz.PlotFlags.NONE):
                    self.render_velocity_editor()
                    self.render_velocity_context_menu()
                viz.end_figure()
            viz.end_window()

        if self.show_d_left:
            window_open = viz.begin_window(
                    f"{self.title} - Left Border###{self.uuid}d_left")
            self.show_d_left = viz.get_window_open()
            if window_open:
                self.render_val_toolbar(2)
                if viz.begin_figure(f"{self.title} - Left Border###{self.uuid}d_left",
                                    flags=viz.PlotFlags.NO_TITLE
                                          | viz.PlotFlags.NONE):
                    self.render_val_editor("d_left [m]", 2)
                viz.end_figure()
            viz.end_window()

        if self.show_d_right:
            window_open = viz.begin_window(
                    f"{self.title} - Right Border###{self.uuid}d_right")
            self.show_d_right = viz.get_window_open()
            if window_open:
                self.render_val_toolbar(3)
                if viz.begin_figure(f"{self.title} - Right Border###{self.uuid}d_right",
                                    flags=viz.PlotFlags.NO_TITLE
                                          | viz.PlotFlags.NONE):
                    self.render_val_editor("d_right [m]", 3)
                viz.end_figure()
            viz.end_window()

        if self.show_altitude:
            window_open = viz.begin_window(
                    f"{self.title} - Altitude###{self.uuid}altitude")
            self.show_altitude = viz.get_window_open()
            if window_open:
                self.render_val_toolbar(5)
                if viz.begin_figure(f"{self.title} - Altitude###{self.uuid}altitude",
                                    flags=viz.PlotFlags.NO_TITLE
                                          | viz.PlotFlags.NONE):
                    self.render_val_editor("altitude [m]", 5)
                viz.end_figure()
            viz.end_window()

        if self.show_curvature:
            window_open = viz.begin_window(
                    f"{self.title} - Curvature###{self.uuid}curvature")
            self.show_curvature = viz.get_window_open()
            if window_open:
                if viz.begin_figure(f"{self.title} - Curvature###{self.uuid}curvature",
                                    flags=viz.PlotFlags.NO_TITLE
                                          | viz.PlotFlags.NONE):
                    self.render_map_info()
                viz.end_figure()
            viz.end_window()

        self.handle_menu_file_func()

        if viz.pop_mod_any():
            self.save_req = True
            self.save_req_time = time.time()

        if self.save_req and time.time() - self.save_req_time > 0.5:
            # throttled saving
            if self.map_store_path is not None:
                otb.save(self.maps, self.map_store_path, mmap_arrays=True)
            self.save_req = False
