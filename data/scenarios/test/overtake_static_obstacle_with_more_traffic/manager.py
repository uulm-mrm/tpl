import numpy as np


class SimulationManager:

    def __init__(self, sim):

        sim.ego.x = np.random.uniform(-100, -30)
        sim.ego.v = np.random.uniform(5, 15)

    def update(self, sim):

        if sim.ego.x > 100:
            sim.finished = True
