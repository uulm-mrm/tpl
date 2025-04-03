import copy
import numpy as np
import imviz as viz

from imdash.utils import DataSource, ColorEdit
from imdash.views.view_2d import View2DComponent

from tpl.gui.renderers import DynamicObjectsRenderer


class DynamicObjectsComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Dynamic objects"

    def __init__(self):

        super().__init__()

        self.label = "dynamic_objects"

        self.dyn_objs = DataSource(path="{/structstores/tpl_env/predicted}")

        self.renderer = DynamicObjectsRenderer()

    def render(self, idx, view):

        dyn_objs = self.dyn_objs()
        if type(dyn_objs) == list:
            all_dyn_objs = dyn_objs
        else:
            all_dyn_objs = []
            for l in dyn_objs.__dict__.values():
                all_dyn_objs += l

        self.renderer.render(all_dyn_objs, idx, self)
