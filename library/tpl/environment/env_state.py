import copy
from typing import List, Tuple

import numpy as np
import objtoolbox as otb

from tpl import util
from tpl.environment import VehicleState


class EnvironmentState:

    def __init__(self):

        self.full_reset()

    def reset(self):

        self.reset_counter += 1
        self.local_map = None

    def full_reset(self):

        self.t = 0.0

        self.reset_counter = 1
        """
        Increment if something fundamental has changed in the environment.
        Planning algorithms can use this counter to reinitialize.
        """

        self.vehicle_state = VehicleState()
        """
        State of the ego vehicle.
        """

        self.map_store_path = ""
        self.maps = otb.bundle()
        self.selected_map = ""
        self.local_map = None
        self.local_map_behind = None
        """
        Members for map data management.
        """

        self.tl_dets = otb.bundle()
        """
        Traffic light detections.
        """

        self.ir_pc_dets = []
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

        self.cpms = []
        """
        Dynamic objects from infrastructure.
        """

        self.overtake_obj = []
        """
        Overtake objects from maneuver handling.
        """

        self.man_cam_ids = []
        """
        Object CAM IDs from maneuver handling.
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

        for k, v in otb.get_obj_dict(self.maps).items():
            if k == name_or_uuid or v.name == name_or_uuid:
                self.selected_map = k
                self.reset()
                return True

        return False

    def auto_select_map(self):

        min_map = None
        min_proj = None

        for k, v in otb.get_obj_dict(self.maps).items():
            proj = util.project(v.path[:, :2], (self.vehicle_state.x, self.vehicle_state.y))
            if not proj.in_bounds:
                continue
            if np.degrees(np.abs(util.short_angle_dist(proj.angle, self.vehicle_state.phi))) > 30.0:
                continue
            if min_proj is None or abs(proj.distance) < abs(min_proj.distance):
                min_map = v
                min_proj = proj

        if min_map is not None:
            self.selected_map = min_map.uuid

    def get_current_map(self):

        return getattr(self.maps, self.selected_map, None)

    def get_relevant_maps(self):
        """
        Returns the local map and all intersection path maps.
        """

        if self.local_map is None:
            return []

        maps = [self.local_map_behind]
        for ip in self.local_map.intersection_paths:
            p = util.project(self.local_map.path[:, :2], ip.stop_pos)
            if p.in_bounds:
                maps.append(ip.map_segment)

        return maps

    def get_all_tracks(self):

        dyn_objs = []
        for v in otb.get_obj_dict(self.tracks).values():
            dyn_objs += copy.deepcopy(v)
        dyn_objs += copy.deepcopy(self.cpms)

        return dyn_objs

    def get_cpms(self):

        dyn_objs = []
        for v in otb.get_obj_dict(self.tracks).values():
            dyn_objs += copy.deepcopy(v)
        dyn_objs += copy.deepcopy(self.cpms)

        return dyn_objs


SharedEnvironmentState = util.make_class_shared(EnvironmentState)
