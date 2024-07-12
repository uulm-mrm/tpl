import numpy as np

import imviz as viz


class MapPathsRenderer:

    def __init__(self, show_all=False, no_fit=False, div_subsample=30):

        self.no_fit = no_fit
        self.div_subsample = div_subsample
        self.show_all = show_all

        self.name = None

        self.format = "-"
        self.line_weight = 1.0
        self.marker_size = 3.0

        self.color_default = np.array([0.5, 0.5, 0.5])
        self.color_selected = np.array([0.2, 0.6, 0.8])

        self.line_flags = viz.PlotLineFlags.NONE
        if self.no_fit:
            self.line_flags |= viz.PlotItemFlags.NO_FIT

    def render_map_store(self, maps, selected_map=""):

        x_min, y_min, x_max, y_max = viz.get_plot_limits()
        subsample_step = min(x_max - x_min, y_max - y_min) / self.div_subsample
        subsample_step = max(1, min(100, int(subsample_step)))

        for k, m in maps.items():
            if k == selected_map:
                self.render_map(m, self.color_selected, self.line_flags, subsample_step)
            elif self.show_all:
                flags = viz.PlotItemFlags.NO_FIT
                self.render_map(m, self.color_default, flags, subsample_step)

    def render_map(self, m, color, flags, subsample_step=1):

        if self.name is None:
            label = f"{m.name}"
        else:
            label = self.name

        kwargs = dict(
             fmt="-",
             label=label,
             color=color,
             line_weight=self.line_weight,
             marker_size=self.marker_size,
             flags=flags)

        if len(m.path) == 0:
            return

        idx_last = int((len(m.path)-1) / subsample_step) * subsample_step
        if m.closed_path:
            idxs_end = [idx_last, 0]
        else:
            idxs_end = [idx_last, -1]

        viz.plot(m.path[::subsample_step, 0],
                 m.path[::subsample_step, 1],
                 **kwargs)
        viz.plot(m.path[idxs_end, 0],
                 m.path[idxs_end, 1],
                 **kwargs)

        kwargs["color"] = color * 0.5

        viz.plot(m.boundary_left[::subsample_step, 0],
                 m.boundary_left[::subsample_step, 1],
                 **kwargs)
        viz.plot(m.boundary_left[idxs_end, 0],
                 m.boundary_left[idxs_end, 1],
                 **kwargs)

        viz.plot(m.boundary_right[::subsample_step, 0],
                 m.boundary_right[::subsample_step, 1],
                 **kwargs)
        viz.plot(m.boundary_right[idxs_end, 0],
                 m.boundary_right[idxs_end, 1],
                 **kwargs)
