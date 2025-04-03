import os
import sys
import uuid
import importlib

import numpy as np
import objtoolbox as otb

from tpl import util


class SimIdmParams:

    def __init__(self):

        self.time_headway = 1.5
        self.gap_min = 2.0
        self.a = 1.5
        self.b = 3.0
        self.delta = 4.0


class SimCar:

    def __init__(self):

        self.uuid = uuid.uuid4().hex

        self.map_uuid = ""

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.width = 2.0
        self.length = 4.0

        self.proj = None
        self.reverse = False

        self.v = 10.0
        self.target_v = 10.0
        self.target_change_dt = -1.0
        self.target_last_change_t = 0.0
        self.target_v_low = 5.0
        self.target_v_high = 20.0

        self.use_route_velocity = False
        self.react_to_curvature = False

        self.use_idm = False
        self.params_idm = SimIdmParams()

        self.noise_x = 0.0
        self.noise_y = 0.0
        self.noise_yaw = 0.0
        self.noise_v = 0.0
        self.noise_hull = 0.0

        self.evade = ""

    @staticmethod
    def get_convex_hull(self):

        l = self.length
        w = self.width

        hull_points = np.array([
                [l/2, w/2, 0.0, 1.0],
                [l/2, -w/2, 0.0, 1.0],
                [-l/2, -w/2, 0.0, 1.0],
                [-l/2, w/2, 0.0, 1.0]
            ])

        model_mat = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0.0, self.x],
            [np.sin(self.yaw), np.cos(self.yaw), 0.0, self.y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        return ((model_mat @ hull_points.T).T)[:, :2]


class SimEgo:

    def __init__(self):

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.v = 0.0
        self.a = 0.0
        self.min_v = 0.0
        self.max_v = 100.0

        self.steer_angle = 0.0
        self.max_steer_angle = np.radians(40.96)

        self.acc_dead_time = 0.0
        self.steer_dead_time = 0.0

        self.control_acc = 0.0
        self.control_steer = 0.0

        self.width = 1.89
        self.length = 5.1

        self.wheel_base = 3.165
        self.cog_axes_ratio = 0.5

        self.rear_to_rear_axis = 2.665 - (self.wheel_base * self.cog_axes_ratio)
        self.front_to_front_axis = 2.23 - (self.wheel_base * (1.0 - self.cog_axes_ratio))

        self.track_width = 1.6

        self.v_ch = 32.0

    @staticmethod
    def get_convex_hull(self):

        bounds = np.array([
            (-self.rear_to_rear_axis, -self.width/2, 1),
            (self.wheel_base + self.front_to_front_axis, -self.width/2, 1),
            (self.wheel_base + self.front_to_front_axis, self.width/2, 1),
            (-self.rear_to_rear_axis, self.width/2, 1),
            (-self.rear_to_rear_axis, -self.width/2, 1)
        ])

        trans = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), self.x],
            [np.sin(self.yaw), np.cos(self.yaw), self.y],
            [0.0, 0.0, 1.0]
        ])

        return (bounds @ trans.T)[:, :2]


class SimTrafficLight:

    UNKNOWN = -1
    RED = 0
    YELLOW = 1
    GREEN = 2

    def __init__(self):

        self.uuid = uuid.uuid4().hex

        self.x = 0.0
        self.y = 0.0

        self.state = SimTrafficLight.UNKNOWN


class SimTimeConstraint:

    def __init__(self):

        self.uuid = uuid.uuid4().hex

        self.x = 0.0
        self.y = 0.0

        self.t_min = 0.0
        self.t_max = 10.0**10


class SimSettings:

    def __init__(self):

        self.running = False
        self.reload_requested = True
        self.single_step_requested = False
        self.reload_if_finished = False

        self.use_real_time = True
        self.fixed_time_step = 0.01
        self.int_step = 0.005

        self.update_logic = True
        self.update_vehicle_state = True
        self.update_dynamic_objects = True

        self.set_env_time = True
        self.set_env_vehicle_state = True
        self.set_env_dynamic_objects = True
        self.set_env_dynamic_objects_dt = 0.01
        self.set_env_traffic_lights = True
        self.set_env_time_constraints = True


class SimRuleViolation:

    COLLISION = 0
    OFF_ROAD = 1
    WRONG_WAY = 2
    SPEED_LIMIT = 3

    def __init__(self, t, kind, msg=""):

        self.t = t
        self.kind = kind
        self.msg = msg


class SimRuleChecker:

    def __init__(self):

        self.enable = False

        self.off_road_dist_limit = 1.0
        self.v_max_tol = 1.0

        self.violations = []


class SimState:

    def __init__(self):

        self.t = 0.0
        self.finished = False

        self.map_store_path = ""
        self.selected_map = ""
        self.available_maps = []

        self.init_env_params = ""
        self.init_planning_params = ""
        self.init_control_params = ""

        self.ego = SimEgo()
        self.cars = []
        self.traffic_lights = []
        self.time_constraints = []

        self.rule_checker = SimRuleChecker()
        self.settings = SimSettings()

        self.manager = None

    def __savestate__(self):

        s = self.__dict__.copy()
        del s["available_maps"]
        del s["manager"]

        return s


manager_template = """class SimulationManager:

    def __init__(self, sim):
        pass

    def update(self, sim):
        pass
"""


class SimulationModuleFinder(importlib.abc.MetaPathFinder):
    """
    This is necessary so that autoreload works correctly. It provides an
    interface for the reload(...) function to retreive the module spec.
    """

    PATH_MAP = {}

    def find_spec(self, fullname, path, target=None):

        if not fullname in self.PATH_MAP:
            return None

        return importlib.util.spec_from_file_location(
                fullname,
                self.PATH_MAP[fullname])

    def find_module(self, fullname, path):
        return None 


sys.meta_path.append(SimulationModuleFinder())


def state_from_shstate(sim):

    save_state = SimState()
    otb.merge(save_state, sim)

    # convert to SimCar to retain correct type
    for i, c in enumerate(save_state.cars):
        sc = SimCar()
        otb.merge(sc, c)
        sc.proj = None
        save_state.cars[i] = sc

    # convert to SimTrafficLight to retain correct type
    for i, tl in enumerate(save_state.traffic_lights):
        stl = SimTrafficLight()
        otb.merge(stl, tl)
        save_state.traffic_lights[i] = stl

    # convert to SimTimeConstraint to retain correct type
    for i, tc in enumerate(save_state.time_constraints):
        stc = SimTimeConstraint()
        otb.merge(stc, tc)
        save_state.time_constraints[i] = stc

    return save_state


def save_sim_state(sim, path):

    path = os.path.join(util.PATH_SCENARIOS, path)

    save_state = state_from_shstate(sim)
    otb.save(save_state, path)

    manager = os.path.join(path, "manager.py")

    if not os.path.exists(manager):
        manager.write_text(manager_template)


def load_sim_state(path):

    path = os.path.join(util.PATH_SCENARIOS, path)

    sim = SimState()
    if not otb.load(sim, path):
        return None

    manager = os.path.join(path, "manager.py")

    if os.path.exists(manager):
        module_path = str(manager)
        module_name = "_".join(module_path.strip(".py").split("/")[1:])

        SimulationModuleFinder.PATH_MAP[module_name] = module_path

        manager_module = importlib.import_module(module_name)
        sim.manager = manager_module.SimulationManager(sim)

    return sim
