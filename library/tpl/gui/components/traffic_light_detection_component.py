import copy
import imviz as viz

from imdash.utils import DataSource
from imdash.views.view_2d import View2DComponent

from tpl.environment.map_module import TrafficLight


class TrafficLightDetectionComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Traffic light detection"

    def __init__(self):

        super().__init__()

        self.label = "traffic_light_detection"

        self.tpl_env = DataSource(path="{/structstores/tpl_env}")

        self.line_weight = 3.0

        self.history = []
        self.size_hist_max = 200
        self.t_prev = 0.0

    def __savestate__(self):

        d = self.__dict__.copy()
        del d["history"]
        del d["t_prev"]

        return d

    def render(self, idx, view):

        viz.plot_dummy(label=f"{self.label}###{idx}", legend_color=1.0)

        plot_line_flags = viz.PlotLineFlags.NONE 
        if self.no_fit:
            plot_line_flags |= viz.PlotItemFlags.NO_FIT

        src = self.tpl_env.get_used_source()

        store = src.store

        with store.lock():
            env = copy.deepcopy(store)
            tl_dets = env.tl_dets

        if self.t_prev != env.t:
            all_tds = []
            for src in list(tl_dets.__dict__):
                all_tds += getattr(tl_dets, src)

            self.history += all_tds
            self.history = self.history[-self.size_hist_max:]
            self.t_prev = env.t

        for td in self.history:

            if td.state == TrafficLight.RED:
                col = (1.0, 0.0, 0.0)
            elif td.state == TrafficLight.YELLOW:
                col = (1.0, 1.0, 0.0)
            elif td.state == TrafficLight.GREEN:
                col = (0.0, 1.0, 0.0)
            else:
                col = (0.3, 0.3, 0.3)

            viz.plot([td.near_point[0], td.far_point[0]],
                     [td.near_point[1], td.far_point[1]],
                     color=col,
                     line_weight=self.line_weight,
                         flags=plot_line_flags)
