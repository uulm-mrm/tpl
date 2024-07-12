import numpy as np

from imdash.utils import DataSource, ColorEdit
from imdash.views.view_2d import View2DComponent

from tpl.gui.renderers import MapPathsRenderer


class MapPathsComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Map paths"

    def __init__(self):

        super().__init__()

        self.label = "map_paths"

        self.tpl_env = DataSource()

        self.format = "-"
        self.line_weight = 1.0
        self.marker_size = 3.0

        self.show_all = False

        self.color_default = ColorEdit(default=np.array([0.2, 0.2, 0.2]))
        self.color_selected = ColorEdit(default=np.array([0.5, 0.5, 0.5]))

    def render(self, idx, view):

        src = self.tpl_env.get_used_source()
        env = src.store

        renderer = MapPathsRenderer(self.show_all, self.no_fit)
        renderer.name = f"{self.label}###{idx}"
        renderer.format = self.format
        renderer.line_weight = self.line_weight
        renderer.marker_size = self.marker_size
        renderer.color_default = self.color_default()
        renderer.color_selected = self.color_selected()

        with env.lock():
            maps_dict = {s: getattr(env.maps, s) for s in env.maps.__slots__}
            renderer.render_map_store(maps_dict, env.selected_map)
