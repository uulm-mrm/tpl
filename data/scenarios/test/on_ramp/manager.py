import numpy as np
from tpl.simulation.state import SimTrafficLight


class SimulationManager:

    def __init__(self, sim):

        for c in sim.cars:
            c.x += np.random.uniform(-30, 30)

    def update(self, sim):

        if sim.ego.x > -100.0:
            sim.finished = True
