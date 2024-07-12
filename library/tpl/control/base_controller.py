from tpl.planning import Trajectory

class BaseController:

    def __init__(self, shared, lock_shared):

        self.shared = shared
        self.lock_shared = lock_shared

    def reinit_state(self):
        pass

    def reinit_params(self):
        pass

    def update(self, con_input):
        return (0.0, 0.0), Trajectory()
