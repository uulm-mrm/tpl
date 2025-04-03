import numpy as np


class SimulationManager:

    def __init__(self, sim):

        sim.ego.x += np.random.uniform(-10, 10)

    def update(self, sim):
        
        if sim.ego.x > 50:
            sim.finished = True
