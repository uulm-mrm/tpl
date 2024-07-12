from tpl.planning import Trajectory


class BasePlanner:

    def __init__(self, state, params):
        pass

    def update(self, env):
        return Trajectory()
