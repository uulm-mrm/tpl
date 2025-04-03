import numpy as np

class SimulationManager:

    def __init__(self, sim):
        pass

    def update(self, sim):
        
        sim.finished = sim.ego.x > 50
