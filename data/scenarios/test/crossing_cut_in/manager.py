import time
import numpy as np


class SimulationManager:

    def __init__(self, sim):

        sim.ego.x = np.random.uniform(-100, -30)
        sim.ego.v = np.random.uniform(5, 15)

        self.cut_in_time = np.random.uniform(2.0, 5.0)

        for c in sim.cars:
            c.v = 0.0

    def update(self, sim):
        
        if sim.t > self.cut_in_time:
            for c in sim.cars:
                c.target_v = 5.0

        sim.finished = sim.ego.x > 20.0

