import numpy as np
from tpl.simulation.state import SimTrafficLight


class SimulationManager:

    def __init__(self, sim):
        pass

    def update(self, sim):
        
        if sim.t > 20.0:
            sim.finished = True
