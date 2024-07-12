import copy
from typing import List, Tuple

import numpy as np
import objtoolbox as otb

from tpl import util
from tpl.environment import VehicleState


class EnvironmentState:

    def __init__(self):

        self.reset()

    def reset(self):

        self.t = 0.0

        self.reset_required = True
        """
        Indicates that something fundamental has changed in the environment.
        Planning algorithms can use this hint to reinitialize.
        """

        self.vehicle_state = VehicleState()
        """
        State of the ego vehicle.
        """

        self.map_store_path = ""
        self.maps = otb.bundle()
        self.selected_map = ""
        self.local_map = None
        """
        Members for map data management.
        """

        self.tl_dets = otb.bundle()
        """
        Traffic light detections.
        """

        self.ir_pc_dets = otb.bundle()
        """
        Dynamic objects from early fusion.
        """

        self.tracks = otb.bundle()
        """
        Tracked dynamic objects
        """

        self.predicted = []
        """
        Filtered dynamic objects with attached predictions
        """

        self.man_time_cons: List[Tuple[np.ndarray, float, float]] = []
        """
        Min/max time constraints from maneuver module (earliest/latest crossing).
        List of position and min/max absolute timestamps.
        """

        self.man_vel_cons: List[Tuple[np.ndarray, np.ndarray, float]] = []
        """
        Velocity constraints from maneuver module.
        List of start/end position and max velocity.
        """

    def set_selected_map(self, name_or_uuid):

        for k, m in otb.get_obj_dict(self.maps).items():
            if k == name_or_uuid or v.name == name_or_uuid:
                self.selected_map = k
                self.reset_required = True
                return True

        return False

    def get_current_map(self):

        return getattr(self.maps, self.selected_map, None)

    def get_relevant_maps(self):
        """
        Returns the local map and all intersection path maps.
        """

        cmap = self.get_current_map()
        if cmap is None:
            return []

        maps = [self.local_map]
        for ip in cmap.intersection_paths:
            p = util.project(self.local_map.path[:, :2], ip.stop_pos)
            if p.in_bounds:
                maps.append(ip.map_segment)

        return maps

    def get_all_tracks(self):

        dyn_objs = []
        for v in otb.get_obj_dict(self.tracks).values():
            dyn_objs += copy.deepcopy(v)

        return dyn_objs


SharedEnvironmentState = util.make_class_shared(EnvironmentState)
