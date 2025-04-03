import numpy as np
import imviz as viz

from imdash.utils import DataSource
from imdash.views.view_2d import View2DComponent


class FollowCamComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Follow camera"

    def __init__(self):

        super().__init__()

        self.label = "follow_cam"

        self.tpl_env = DataSource(path="{/structstores/tpl_env}")

        self.cam_pos = np.array([0.0, 0.0])
        self.cam_scale = np.array([100.0, 100.0])

        self.smoothing = 0.99
        self.lookahead_secs = 5.0
        self.lookahead_min = 100.0
        self.scale_min = 50.0

        self.inters_path_start = 0.5
        self.inters_path_end = 0.8

    def __savestate__(self):

        d = self.__dict__.copy()

        del d["cam_pos"]
        del d["cam_scale"]

        return d

    def render(self, idx, view):

        env = self.tpl_env.get_used_source().store
        with env.lock():
            veh = env.vehicle_state
            ref_pos = np.array([veh.x, veh.y])
            if env.local_map is not None:
                lookahead_dist = max(self.lookahead_min, veh.v * self.lookahead_secs)
                lookahead_idx = min(len(env.local_map.path) - 1,
                        int(lookahead_dist / env.local_map.step_size_ref))
                ref_ps = np.vstack([ref_pos, env.local_map.path[:lookahead_idx, :2]])
                for ip in env.local_map.intersection_paths:
                    if ip.stop_proj.in_bounds and abs(ip.stop_proj.distance) < 1.0:
                        l = len(ip.map_segment.path)
                        ref_ps = np.vstack([ref_ps, ip.map_segment.path[
                            int(l*self.inters_path_start):int(l*self.inters_path_end), :2]])
                ref_ps_min = np.min(ref_ps, axis=0)
                ref_ps_max = np.max(ref_ps, axis=0)
                ref_ps_size = ref_ps_max - ref_ps_min
                ref_ps_size_max = max(self.scale_min, np.max(ref_ps_size))
                ref_pos = ref_ps_min + ref_ps_size / 2.0

        cam_scale = np.array([ref_ps_size_max, ref_ps_size_max]) * 1.1

        if np.linalg.norm(ref_pos - self.cam_pos) > 100.0:
            self.cam_pos = ref_pos
            self.cam_scale = cam_scale
        else:
            self.cam_pos = (self.cam_pos * self.smoothing 
                            + ref_pos * (1.0 - self.smoothing))

        viz.setup_axes_limits(self.cam_pos[0] - self.cam_scale[0]/2.0,
                              self.cam_pos[0] + self.cam_scale[0]/2.0,
                              self.cam_pos[1] - self.cam_scale[1]/2.0,
                              self.cam_pos[1] + self.cam_scale[1]/2.0,
                              viz.PlotCond.ALWAYS)

        ws = viz.get_plot_size()
        cam_scale[1] *= ws[1] / ws[0]

        self.cam_scale = (self.cam_scale * self.smoothing 
                          + cam_scale * (1.0 - self.smoothing))
