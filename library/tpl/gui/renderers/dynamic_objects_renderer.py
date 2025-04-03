import numpy as np
import imviz as viz

from imdash.utils import ColorEdit


class DynamicObjectsRenderer:

    def __init__(self):

        self.color = ColorEdit(default=np.array((1.0, 0.0, 0.0, 1.0)))
        self.line_weight = 3.0

        self.show_predictions = False
        self.show_vel_line = False

    def render(self, all_dyn_objs, idx, comp):

        col = self.color()

        label = f"{comp.label}###{idx}"
        viz.plot_dummy(label=label, legend_color=col)

        plot_line_flags = viz.PlotLineFlags.NONE 
        if comp.no_fit:
            plot_line_flags |= viz.PlotItemFlags.NO_FIT

        for o in all_dyn_objs:

            viz.plot(o.hull[:, 0],
                     o.hull[:, 1],
                     color=col,
                     label=label,
                     line_weight=self.line_weight,
                     flags=plot_line_flags)
            viz.plot(o.hull[[-1,0], 0],
                     o.hull[[-1,0], 1],
                     color=col,
                     label=label,
                     line_weight=self.line_weight,
                     flags=plot_line_flags)

            if self.show_vel_line and o.yaw is not None and o.v is not None:
                vv = o.v * np.array([np.cos(o.yaw), np.sin(o.yaw)])
                viz.plot([o.pos[0], o.pos[0] + vv[0]],
                         [o.pos[1], o.pos[1] + vv[1]],
                         color=col,
                         label=label,
                         line_weight=self.line_weight,
                         flags=plot_line_flags)

            if self.show_predictions:
                for pred in o.predictions:
                    viz.plot(pred.states[:, 1],
                             pred.states[:, 2],
                             color=col,
                             label=label,
                             fmt="-o",
                             line_weight=self.line_weight,
                             flags=plot_line_flags)
