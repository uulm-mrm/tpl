class SimulationManager:

    def __init__(self, sim):
        pass

    def update(self, sim):

        tl = sim.traffic_lights[0]
        tl.state = 2

        if sim.t > 14.0:
            tl.state = 0
