import unittest
from tpl.simulation import SimStandalone


class TestNoCrashOnInit(unittest.TestCase):

    def test_no_config(self):
        """
        Checks in simulation, if all planners and controllers, can start with
        no external config and can update with each other.
        """

        try:
            sim = SimStandalone()
            for p in sim.planning_app.planners.keys():
                with sim.planning_app.sh_planners.lock():
                    sim.planning_app.sh_planners.active_planner = p
                for c in sim.control_app.controllers.keys():
                    with sim.control_app.sh_controllers.lock():
                        sim.control_app.sh_controllers.active_controller = c
                    sim.update()
        except Exception as e:
            self.fail(e)
