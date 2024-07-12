class SimulationManager:

    def __init__(self, sim):
        pass

    def update(self, sim):

        if sim.t > 26.0:
            sim.finished = True
