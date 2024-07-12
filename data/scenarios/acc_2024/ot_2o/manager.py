class SimulationManager:

    def __init__(self, sim):

        sim.ego.v = 10.0

    def update(self, sim):

        if sim.t > 15.0:
            sim.finished = True
