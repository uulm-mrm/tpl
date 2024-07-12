import os
import copy
import uuid
import random
import unittest
import numpy as np
import os.path as osp

import objtoolbox as otb

from parameterized import parameterized_class

from tpl.simulation import SimStandalone, SimState, state_from_shstate


@parameterized_class([
    {
        "scenario_path": "acc_2024/cv_3o",
    },
    {
        "scenario_path": "acc_2024/ot_2o",
    },
    {
        "scenario_path": "acc_2024/rb_3o",
    },
    {
        "scenario_path": "acc_2024/cv_3o",
        "active_planner": "poly_sampling_planner"
    },
    {
        "scenario_path": "acc_2024/ot_2o",
        "active_planner": "poly_sampling_planner"
    },
    {
        "scenario_path": "acc_2024/rb_3o",
        "active_planner": "poly_sampling_planner"
    }
])
class TestSimScenarios(unittest.TestCase):

    scenario_path = ""
    active_planner = "path_vel_decomp_planner"

    def setUp(self):

        self.base_path = osp.normpath(osp.realpath(
                osp.dirname(__file__) + "/../../data/"))

        self.sim_states = []
        self.runtimes = []
        self.res_path = ""

    def tearDown(self):

        try:
            del self.standalone
        except AttributeError:
            pass

        # build and save log

        log = otb.bundle()
        log.sim_states = self.sim_states
        log.runtimes = self.runtimes
        log.runtime_mean = np.mean(self.runtimes)
        log.runtime_stddev = np.std(self.runtimes)
        log.runtime_max = np.max(self.runtimes)

        log_path = osp.join(osp.dirname(osp.realpath(__file__)), 
                "test_sim_results",
                self.res_path)
        otb.save(log, log_path)

        # build and save state csv files

        csv_ego_states = []
        for s in self.sim_states:
            veh = s.ego
            csv_ego_states.append([s.t, veh.x, veh.y, veh.yaw, veh.steer_angle, veh.v, veh.a])
        csv_ego_states = np.array(csv_ego_states)

        csv_path = osp.join(log_path, "ego_state.csv")
        np.savetxt(csv_path,
                   csv_ego_states,
                   delimiter=",",
                   header="t,x,y,phi,delta,v,a",
                   comments="")

    def create_standalone(self):

        # create random app id to reduce interference with other instances
        #app_id = uuid.uuid4().hex
        app_id = ""

        standalone = SimStandalone(
                app_id=app_id,
                scenario_path=self.scenario_path)

        with standalone.core.sh_state.lock():
            sim = standalone.core.sh_state.sim
            sim.settings.running = True
            sim.settings.use_real_time = False

        return standalone

    def check_rules(self, sim):

        rc = sim.rule_checker

        self.assertEqual(list(rc.collisions), [], "scenario was not collision free")
        self.assertEqual(rc.off_road, False, "vehicle went off road")
        self.assertEqual(rc.wrong_way, False, "vehicle went the wrong way")
        self.assertEqual(rc.v_max_violation, 0.0, "violated route velocity limit")

    def test_scenario(self):

        self.standalone = self.create_standalone()
        self.res_path = os.path.join(self.active_planner, self.scenario_path)

        with self.standalone.planning_app.sh_planners.lock():
            self.standalone.planning_app.sh_planners.active_planner = self.active_planner

        finished = False

        # run simulation

        while not finished:
            self.standalone.update()
            with self.standalone.core.sh_state.lock():
                sim = copy.deepcopy(self.standalone.core.sh_state.sim)

            self.sim_states.append(state_from_shstate(sim))

            with self.standalone.planning_app.sh_planners.lock():
                self.runtimes.append(self.standalone.planning_app.sh_planners.runtime)

            self.check_rules(sim)
            finished = sim.finished
