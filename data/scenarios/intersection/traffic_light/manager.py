import numpy as np
from tpl.simulation.state import SimTrafficLight


class SimulationManager:

    def __init__(self, sim):

        sim.ego.x = np.random.uniform(30, 50)
        sim.ego.v = np.random.uniform(5, 10)

        for c in sim.cars:
            c.y += np.random.uniform(-20, 20)
            c.v_target = np.random.uniform(3, 15)

        self.red_phase = np.random.uniform(1.0, 10.0)
        self.yellow_phase = self.red_phase + np.random.uniform(1.0, 3.0)

    def update(self, sim):

        if sim.t < self.red_phase:
            sim.traffic_lights[0].state = SimTrafficLight.RED
        elif sim.t < self.yellow_phase:
            sim.traffic_lights[0].state = SimTrafficLight.YELLOW
        else:
            sim.traffic_lights[0].state = SimTrafficLight.GREEN

        if sim.ego.x > 130.0:
            sim.finished = True
