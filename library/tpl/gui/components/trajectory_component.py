import copy
import numpy as np
import imviz as viz

from imdash.utils import DataSource, ColorEdit
from imdash.views.view_2d import View2DComponent

from tpl.gui.renderers import DynamicObjectsRenderer
from tpl.planning import Trajectory
from tpl.util import lerp

import matplotlib.cm as colormaps
from matplotlib.colors import LinearSegmentedColormap
colormaps.blrd = LinearSegmentedColormap.from_list('blrd', ['blue', 'red'])
colormaps.blgrrd = LinearSegmentedColormap.from_list('blgrrd', ['blue', '#333333', 'red'])


class TrajectoryComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Trajectory"

    def __init__(self):

        super().__init__()

        self.label = "trajectory"

        self.traj = DataSource(path="{/structstores/tpl_planning/trajectory}")

        self.show_line = True
        self.line_weight = 1.0

        self.show_markers = True
        self.marker_size = 4.0
        self.format_markers = "o"
        self.dt_markers = 1.0

        self.color_base = ColorEdit([1.0, 0.0, 0.0, 1.0])
        self.color_source = viz.Selection(["base", "time", "velocity", "acceleration"])
        self.color_map = viz.Selection(["jet", "turbo", "gnuplot", "blrd", "blgrrd", "bwr", "coolwarm"])
        self.use_color_max_min = False
        self.color_min_val = 0.0
        self.color_max_val = 1.0

    def render(self, idx, view):

        traj: Trajectory = self.traj()

        label = f"{self.label}###{idx}"

        ts_markers = np.arange(traj.time[0], traj.time[-1], self.dt_markers)

        markers_xs = lerp(ts_markers, traj.time, traj.x)
        markers_ys = lerp(ts_markers, traj.time, traj.y)

        col_source = self.color_source.selected()

        if col_source == "base":
            c = self.color_base()
            c_markers = self.color_base()
        else:
            color_vals = np.array(getattr(traj, col_source))
            if self.use_color_max_min:
                color_vals -= self.color_min_val
                color_vals /= self.color_max_val - self.color_min_val
            else:
                color_vals -= np.min(color_vals)
                color_vals /= np.max(color_vals)
            markers_col_vals = lerp(ts_markers, traj.time, color_vals)
            cm = getattr(colormaps, self.color_map.selected())
            c = cm(color_vals)
            c_markers = cm(markers_col_vals)

        viz.plot(traj.x,
                 traj.y,
                 "-",
                 label=label,
                 color=c,
                 line_weight=self.line_weight)

        viz.plot(markers_xs,
                 markers_ys,
                 self.format_markers,
                 label=label,
                 color=c_markers,
                 marker_size=self.marker_size)
