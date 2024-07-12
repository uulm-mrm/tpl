import copy

from tpl import util
from tpl.application import EnvironmentApp, PlanningApp, ControlApp
from tpl.simulation import SimCore


class SimStandalone:

    def __init__(self,
                 app_id="",
                 scenario_path=None,
                 env_params=None,
                 planning_params=None,
                 control_params=None):

        self.core = SimCore(app_id, scenario_path)

        with self.core.sh_state.lock():
            scenario_planning_params = self.core.sh_state.sim.init_planning_params
            scenario_control_params = self.core.sh_state.sim.init_control_params

        if planning_params is None and scenario_planning_params != "":
            planning_params = scenario_planning_params
        if control_params is None and scenario_control_params != "":
            control_params = scenario_control_params

        self.env_app = EnvironmentApp(app_id, env_params)
        self.planning_app = PlanningApp(app_id, planning_params)
        self.control_app = ControlApp(app_id, control_params)

        self.core.reload_scenario(
                scenario_path,
                self.env_app.env,
                self.planning_app.sh_planners,
                self.control_app.sh_controllers)

    def update(self):

        with self.control_app.sh_controllers.lock():
            controls = copy.deepcopy(self.control_app.sh_controllers.controls)

        with self.core.sh_state.lock():
            sim = self.core.sh_state.sim
            sim.ego.control_acc = controls[0]
            sim.ego.control_steer = controls[1]

        with self.env_app.env.lock():
            sim = self.core.get_next_sim_state(
                    self.env_app.env,
                    self.planning_app.sh_planners,
                    self.control_app.sh_controllers)
            self.env_app.update(sim.t)
            veh = copy.deepcopy(self.env_app.env.vehicle_state)

        self.planning_app.update()

        with self.planning_app.sh_planners.lock():
            traj = copy.deepcopy(self.planning_app.sh_planners.trajectory)

        with self.control_app.sh_input.lock():
            self.control_app.sh_input.t = sim.t
            self.control_app.sh_input.vehicle = veh
            self.control_app.sh_input.trajectory = traj

        self.control_app.update()

        self.core.write_sim_state(sim)
