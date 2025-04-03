import numpy as np
import imviz as viz

from imdash.utils import DataSource, ColorEdit
from imdash.components.view_2d import History2DComp
from imdash.views.view_2d import View2DComponent

from tpl.gui.renderers import DynamicObjectsRenderer


class DynamicObjectsHistoryComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Dynamic objects history"

    def __init__(self):

        super().__init__()

        self.label = "dynamic_objects_history"

        self.history = History2DComp()
        self.history.x_source.path = "{/structstores/tpl_env/t}"
        self.history.y_source.path = "{/structstores/tpl_env/predicted}"
        self.history.plot_history = self.plot_history
        self.renderer = DynamicObjectsRenderer()

    def plot_history(self, idx):

        xs = np.array([x for x, y in self.history.history])
        ys = [y for x, y in self.history.history]

        xs -= np.min(xs)
        if np.max(xs) > 0:
            xs /= np.max(xs)
        else:
            xs = np.ones_like(xs)

        for t, dyn_objs in zip(xs, ys):

            if type(dyn_objs) == list:
                all_dyn_objs = dyn_objs
            else:
                all_dyn_objs = []
                for l in dyn_objs.__dict__.values():
                    all_dyn_objs += l

            self.renderer.color.alt_val[3] = t
            self.renderer.render(all_dyn_objs, idx, self)

    def render(self, idx, view):

        self.history.render(idx, view)
