import time
import copy
import numpy as np
import os.path as osp

from tpl import util
from tpl.application.environment_app import load_env_params
from tpl.application.planning_app import load_planning_params
from tpl.application.control_app import load_control_params
from tpl.simulation import (
        SimState,
        SimCar,
        SimEgo,
        SimRuleViolation,
        load_sim_state
    )
from tpl.environment import (
        DynamicObject,
        TrafficLightDetection,
        load_map_store
    )

import objtoolbox as otb
import structstore as sts


class SimCore:

    def __init__(self, app_id="", scenario_path=None):

        self.accumulator = 0.0
        self.last_update_time = -1.0
        self.set_env_dynamic_objects_dt = 0.0

        self.acc_buffer = []
        self.steering_angle_buffer = []

        self.manager = None

        if app_id != "":
            app_id += "_"
        self.app_id = app_id

        self.sh_state = util.StructStoreRegistry.get(f"/{self.app_id}tpl_sim", 10**7, reinit=True)
        with self.sh_state.lock():
            if scenario_path is None:
                scenario_path = "default"
            self.sh_state.__storage__ = scenario_path
            self.sh_state.__renderer__ = "tpl.gui.state_and_params.render_simulation"
            self.sh_state.sim = SimState()

    def reload_scenario(self, scenario_path, env, planning, control):

        if scenario_path is None:
            return

        sim = load_sim_state(scenario_path)

        if sim is None:
            print(f"Loading scenario {scenario_path} failed!")
            return

        self.accumulator = 0.0
        self.last_update_time = -1.0
        self.set_env_dynamic_objects_dt = 0.0

        sim.t = 0.0
        sim.rule_checker.violations = []
        sim.settings.reload_requested = False
        self.manager = sim.manager

        with env.lock():
            env.full_reset()
            load_env_params(env, sim.init_env_params)
            env.selected_map = sim.selected_map
            env.map_store_path = sim.map_store_path 
            env.maps = load_map_store(sim.map_store_path)
            env.reset_counter = 1
            sim.available_maps = [(k, v.name) for k,v in otb.get_obj_dict(env.maps).items()]
        with planning.lock():
            load_planning_params(planning, sim.init_planning_params)
        with control.lock():
            load_control_params(control, sim.init_control_params)

        self.sh_state.sim = sim
        self.sh_state.__storage__ = scenario_path

        self.acc_buffer = []
        self.steering_angle_buffer = []

    def update_ego(self, ego, t, dt):

        # simulate acceleration/steering deadtime

        if dt > 0.0:
            self.acc_buffer.append((t, ego.control_acc))
            self.steering_angle_buffer.append((t, ego.control_steer))

            while len(self.acc_buffer) > ego.acc_dead_time // dt + 1:
                self.acc_buffer.pop(0)
            while len(self.steering_angle_buffer) > ego.steer_dead_time // dt + 1:
                self.steering_angle_buffer.pop(0)

        if ego.acc_dead_time == 0.0 and len(self.acc_buffer) > 0:
            ego.a = self.acc_buffer[-1][1]
        else:
            for at, ac in self.acc_buffer:
                if t - at <= ego.acc_dead_time:
                    ego.a = ac
                    break

        if ego.steer_dead_time == 0.0 and len(self.steering_angle_buffer) > 0:
            ego.steer_angle = self.steering_angle_buffer[-1][1]
        else:
            for st, sc in self.steering_angle_buffer:
                if t - st <= ego.steer_dead_time:
                    ego.steer_angle = sc
                    break

        # update ego using kinematic bicycle model with characteristic velocity

        ego.x += dt * ego.v * np.cos(ego.yaw)
        ego.y += dt * ego.v * np.sin(ego.yaw)

        ego.yaw += dt * ego.v / (ego.wheel_base 
                * (1 + (ego.v/ego.v_ch)**2)) * np.tan(ego.steer_angle)

        ego.yaw = util.normalize_angle(ego.yaw)

        ego.v += dt * ego.a
        ego.v = min(ego.max_v, max(ego.min_v, ego.v))

        ego.steer_angle = min(ego.max_steer_angle, 
                max(-ego.max_steer_angle, ego.steer_angle))

    def update_cars(self, t, ego, cars, maps, dt):

        for sc in cars:
            try:
                cmap = getattr(maps, sc.map_uuid)
            except AttributeError:
                continue

            pos = np.array([sc.x, sc.y])

            if (sc.proj is None
                    or np.linalg.norm(pos - sc.proj.point) > 0.0
                    or abs(np.cos(sc.yaw - sc.proj.angle)) > 0.5):
                sc.proj = util.project(cmap.path[:, :2],
                                       np.array([sc.x, sc.y]),
                                       cmap.closed_path)
                if sc.reverse:
                    sc.proj.angle = util.normalize_angle(sc.proj.angle + np.pi)
                sc.x = sc.proj.point[0]
                sc.y = sc.proj.point[1]
                sc.yaw = sc.proj.angle

            if sc.use_idm:
                steps = min(200, max(50, int(abs(sc.v) * 4 / cmap.step_size_discr)))
                path_leader = util.path_segment(
                        cmap.path,
                        -steps if sc.reverse else steps,
                        sc.proj.end if sc.reverse else sc.proj.start,
                        cmap.closed_path)

                pos_sc = util.project(path_leader[:, :2], (sc.x, sc.y)).arc_len
                pos_sc += sc.length

                ego_x = ego.x + np.cos(ego.yaw) * ego.wheel_base * 0.5
                ego_y = ego.y + np.sin(ego.yaw) * ego.wheel_base * 0.5
                pos_other = [[ego_x, ego_y]] + [(o.x, o.y) for o in cars if o != sc]
                projs = util.project(path_leader[:, :2], pos_other)
                projs = [p for p in projs if p.in_bounds]
                projs = [p for p in projs if abs(p.distance) < sc.width]
                if len(projs) > 0:
                    proj_min = min(projs, key=lambda p: p.arc_len)
                    s_net = proj_min.arc_len - pos_sc
                    v_lead = ego.v * np.cos(proj_min.angle - ego.yaw)
                else:
                    s_net = 10.0**6
                    v_lead = 0.0

                ip = sc.params_idm
                v_delta = sc.v - v_lead
                s_star = (ip.gap_min
                          + sc.v * ip.time_headway
                          + (sc.v * v_delta) / (2.0 * np.sqrt(ip.a * ip.b)))
                acc = ip.a * (1.0 
                           - (sc.v / max(0.001, sc.target_v))**ip.delta
                           - (s_star / s_net)**2)
                acc = max(-ip.b, min(ip.a, acc))
                sc.v += dt * acc
                if sc.v < 0.001 and acc <= 0.0:
                    sc.v = 0.0
            else:
                sc.v = sc.target_v

            # udpate and reproject

            sc.x += dt * sc.v * np.cos(sc.yaw)
            sc.y += dt * sc.v * np.sin(sc.yaw)

            proj = util.project(cmap.path[:, :2], np.array([sc.x, sc.y]))
            sc.proj = proj
            if sc.reverse:
                sc.proj.angle = util.normalize_angle(sc.proj.angle + np.pi)

            sc.yaw = sc.proj.angle

            # velocity updates

            if sc.target_change_dt > 0.0:
                if t - sc.target_last_change_t > sc.target_change_dt:
                    sc.target_v = np.random.uniform(sc.target_v_low, sc.target_v_high)
                    sc.target_last_change_t = t

            if sc.use_route_velocity:
                sc.target_v = path[min(proj.index, len(path)-1), 5]

            if sc.react_to_curvature:
                curv = abs(path[proj.start, 4])
                if curv > 10e-6:
                    sc.v = min(np.sqrt(sc.max_lat_acc / curv), sc.v)

    def measure_ego(self, veh, t, ego):

        veh.t = t
        veh.x = ego.x
        veh.y = ego.y
        veh.phi = ego.yaw
        veh.v = ego.v
        veh.a = ego.a
        veh.delta = ego.steer_angle
        veh.lat_acc = ego.v**2 * np.tan(veh.delta) / ego.wheel_base

        veh.wheel_base = ego.wheel_base
        veh.track_width = ego.track_width
        veh.rear_axis_to_rear = ego.rear_to_rear_axis
        veh.rear_axis_to_front = ego.wheel_base + ego.front_to_front_axis
        veh.cog_axes_ratio = ego.cog_axes_ratio

        veh.dead_time_steer = ego.steer_dead_time
        veh.dead_time_acc = ego.acc_dead_time

        veh.steering_wheel_button = False
        veh.imu_state = 3

    def measure_cars(self, env, cars):

        env.tracks.simulation = []
        env.ir_pc_dets = []

        for sc in cars:
            noise_x = np.random.normal(0.0, sc.noise_x)
            noise_y = np.random.normal(0.0, sc.noise_y)
            noise_yaw = np.random.normal(0.0, sc.noise_yaw)
            noise_v = np.random.normal(0.0, sc.noise_v)

            sc = copy.deepcopy(sc)
            sc.x += noise_x
            sc.y += noise_y
            sc.yaw += noise_yaw
            sc.v += noise_v

            obj = DynamicObject()
            obj.id = sc.uuid
            obj.t = env.t
            obj.pos = np.array([sc.x, sc.y])
            obj.v = sc.v
            obj.yaw = sc.yaw
            obj.evade = sc.evade
            obj.hull = SimCar.get_convex_hull(sc)
            obj.hull += np.random.normal(0.0, sc.noise_hull, obj.hull.shape)
            obj.hull_radius = np.max(np.linalg.norm(
                    obj.hull - obj.pos[np.newaxis, :], axis=1))

            obj = copy.deepcopy(obj)
            obj.yaw = None
            obj.v = None
            env.ir_pc_dets.append(obj)

    def measure_traffic_lights(self, env, traffic_lights):

        env.tl_dets.simulation = []

        veh = env.vehicle_state
        near_point = np.array([veh.x, veh.y])

        for tl in traffic_lights:
            tl_det = TrafficLightDetection()
            tl_det.t = env.t
            tl_det.near_point = near_point
            tl_det.far_point = np.array([tl.x, tl.y])
            tl_det.state = tl.state
            tl_det.confidence = 1.0

            env.tl_dets.simulation.append(tl_det)

    def measure_time_constraints(self, env, time_cons):

        env.man_time_cons = []
        for tc in time_cons:
            env.man_time_cons.append(
                    (np.array([tc.x, tc.y]), tc.t_min, tc.t_max))

    def measure_env(self, env, sim):

        settings = sim.settings
        veh = env.vehicle_state

        if settings.set_env_time:
            env.t = sim.t
        if settings.set_env_vehicle_state:
            self.measure_ego(veh, env.t, sim.ego)
        if settings.set_env_dynamic_objects:
            dt_update = settings.set_env_dynamic_objects_dt
            if sim.t - self.set_env_dynamic_objects_dt > dt_update:
                self.measure_cars(env, sim.cars)
                self.set_env_dynamic_objects_dt = sim.t
        if settings.set_env_traffic_lights:
            self.measure_traffic_lights(env, sim.traffic_lights)
        if settings.set_env_time_constraints:
            self.measure_time_constraints(env, sim.time_constraints)

    def update_rule_checks(self, sim, cmap):

        rc = sim.rule_checker

        if not rc.enable:
            return

        # collisions

        ego_hull = SimEgo.get_convex_hull(sim.ego)
        for sc in sim.cars:
            collision = util.intersect_polygons(SimCar.get_convex_hull(sc), ego_hull)
            if collision:
                v = SimRuleViolation(
                        sim.t,
                        SimRuleViolation.COLLISION,
                        f"ego collides with {sc.uuid}")
                rc.violations.append(v)

        if cmap is not None:

            # off road 

            proj = util.project(cmap.path[:, :2],
                                np.array([sim.ego.x, sim.ego.y]))
            d_right = -cmap.d_right[proj.index]
            d_left = cmap.d_left[proj.index]
            if proj.distance < 0.0:
                if proj.distance + d_right > 0.0:
                    v = SimRuleViolation(
                            sim.t,
                            SimRuleViolation.OFF_ROAD,
                            f"d_ego: {proj.distance} < d_right: {d_right}")
                    rc.violations.append(v)
            if proj.distance > 0.0:
                if proj.distance - d_left > 0.0:
                    v = SimRuleViolation(
                            sim.t,
                            SimRuleViolation.OFF_ROAD,
                            f"d_ego: {proj.distance} > d_left: {d_left}")
                    rc.violations.append(v)

            # wrong way / heading diff

            angle_diff = np.cos(proj.angle - sim.ego.yaw)
            if angle_diff < 0.0:
                v = SimRuleViolation(
                        sim.t,
                        SimRuleViolation.WRONG_WAY,
                        f"yaw_ego: {sim.ego.yaw} not aligned with yaw_path: {proj.angle}")
                rc.violations.append(v)

            # maximum route velocity violation

            v_max = cmap.path[proj.index, 5]
            v_max_violation = float(max(0.0, sim.ego.v - (v_max + rc.v_max_tol)))
            if v_max_violation > 0.0:
                v = SimRuleViolation(
                        sim.t,
                        SimRuleViolation.SPEED_LIMIT,
                        f"v_ego: {sim.ego.v} > v_max: {v_max}")
                rc.violations.append(v)

    def get_next_sim_state(self, env, planning, control):

        with self.sh_state.lock():
            sh_sim = self.sh_state.sim
            # reload scenario if required
            if sh_sim.finished and sh_sim.settings.reload_if_finished:
                sh_sim.settings.reload_requested = True
            if sh_sim.settings.reload_requested:
                self.reload_scenario(self.sh_state.__storage__,
                                     env,
                                     planning,
                                     control)
            # copy for updates
            sim = copy.deepcopy(self.sh_state.sim)

            # reset single step request flag
            sh_sim.settings.single_step_requested = False

        settings = sim.settings

        # execute simulation manager for custom behavior
        if self.manager is not None:
            self.manager.update(sim)

        # calculate accumulator for updates

        if settings.update_logic:
            if not settings.running:
                if settings.single_step_requested:
                    dt = settings.fixed_time_step
                    self.accumulator = settings.fixed_time_step
                else:
                    dt = 0.0
                    self.accumulator = 0.0
            else:
                if self.last_update_time > 0:
                    if settings.use_real_time:
                        self.accumulator += time.time() - self.last_update_time
                    else:
                        self.accumulator += settings.fixed_time_step
                dt = settings.int_step
                # limits accumulator, so it does not build up during long hangs
                self.accumulator = min(dt * 100, self.accumulator)
        else:
            dt = 0.0
            self.accumulator = 0.0

        self.last_update_time = time.time()

        with env.lock():
            cmap = env.get_current_map()

            # update simulation physics
            if settings.update_logic:
                while self.accumulator >= dt:
                    if settings.update_vehicle_state:
                        self.update_ego(sim.ego, sim.t, dt)
                    if settings.update_dynamic_objects:
                        self.update_cars(sim.t, sim.ego, sim.cars, env.maps, dt)
                    self.accumulator -= dt
                    self.accumulator = round(self.accumulator, 5)
                    sim.t += dt
                    sim.t = round(sim.t, 5)
                    if not settings.running:
                        # do one update with dt = 0.0, then exit
                        break

            self.update_rule_checks(sim, cmap)
            self.measure_env(env, sim)

        return sim

    def write_sim_state(self, sim):

        with self.sh_state.lock():

            new_settings = copy.deepcopy(self.sh_state.sim.settings)

            if (sim.settings.running
                    or sim.settings.single_step_requested):
                self.sh_state.sim = sim

            self.sh_state.sim.available_maps = sim.available_maps
            self.sh_state.sim.settings = new_settings
