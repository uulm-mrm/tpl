import numpy as np
import imviz as viz

from imdash.views.view_2d import View2DComponent
from imdash.utils import DataSource, ColorEdit


class CarlaBirdseyeComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Carla Birdseye Component"

    def __init__(self):

        super().__init__()

        self.title = "carla_birdseye"

        self.cam_info_source = DataSource(default=None, use_expr=True)

    def render(self, idx, view):

        c = self.cam_info_source()

        flags = viz.PlotImageFlags.NONE
        if self.no_fit:
            flags |= viz.PlotItemFlags.NO_FIT

        item_id = self.label + f"###{self.uuid}"

        view_size = np.tan(np.radians(c.fov / 2.0)) * c.pos[2] * 2.0

        x = c.pos[0] - view_size / 2.0
        y = c.pos[1] - view_size / 2.0

        scale_x = view_size / c.img.shape[1]
        scale_y = view_size / c.img.shape[0]

        viz.plot_dummy(item_id)
        viz.plot_image(
            item_id,
            np.swapaxes(c.img, 0, 1),
            x,
            y,
            c.img.shape[1] * scale_x,
            c.img.shape[0] * scale_y,
            uv0=(1.0, 0.0),
            uv1=(0.0, 1.0),
            flags=flags)
