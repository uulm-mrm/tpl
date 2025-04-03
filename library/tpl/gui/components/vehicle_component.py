import numpy as np
import imviz as viz

from imdash.utils import DataSource, ColorEdit
from imdash.views.view_2d import View2DComponent

from tpl.gui.renderers import VehicleRenderer


class VehicleComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Vehicle"

    def __init__(self):

        super().__init__()

        self.label = "vehicle"

        self.vehicle = DataSource(path="{/structstores/tpl_env/vehicle_state}")
        self.renderer = VehicleRenderer()

    def render(self, idx, view):

        veh = self.vehicle()
        self.renderer.render(veh, idx, self)
