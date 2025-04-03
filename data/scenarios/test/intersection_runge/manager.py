import numpy as np


class SimulationManager:

    def __init__(self, sim):

        sim.ego.v = np.random.uniform(5, 10)

        for c in sim.cars:
            c.y += np.random.uniform(-20, 20)
            c.v_target = np.random.uniform(3, 15)

    def update(self, sim):
        
        if sim.t > 20.0:
            sim.finished = True
