import numpy as np
from tpl.simulation.state import SimTrafficLight


class SimulationManager:

    def __init__(self, sim):

        sim.ego.x = np.random.uniform(50, 80)
        sim.ego.v = np.random.uniform(5, 10)

        for c in sim.cars:
            c.y += np.random.uniform(-20, 20)
            c.v_target = np.random.uniform(3, 15)

        sim.traffic_lights[0].state = SimTrafficLight.RED

    def update(self, sim):

        if sim.t > 15:
            sim.finished = True
