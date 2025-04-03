class SimulationManager:

    def __init__(self, sim):
        pass

    def update(self, sim):

        sim.finished = sim.t > 20.0
