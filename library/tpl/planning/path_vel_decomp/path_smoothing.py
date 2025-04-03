import numpy as np

from tpl.util import runtime
from tpl.optim import optimizers as opts


class Params:

    def __init__(self):

        self.horizon = 250
        self.step = 0.5
        self.ref_step = 0.5

        self.k_min = -1.0
        self.k_max = 1.0

        self.w_pos = 1.0
        self.w_k = 0.1


class PathSmoothing:

    def __init__(self):

        self.opt = opts.ref_line_smoother_k()
        self.opt.lg_mult_limit = 0.1
        self.opt.barrier_weight[:] = 1000.0
        self.opt.lagrange_multiplier[:] = 0.0

        self.opt_path = np.zeros((1, 6))

        self.reset_counter = 0
        self.reset_required = False

    @runtime
    def update(self, env, params):

        local_map = env.local_map
        path_len = min(params.horizon, local_map.steps_ref)
        path = local_map.path[:path_len]

        opt = self.opt
        opt.horizon = path_len
        opt.step = params.step
        opt.u_min[:] = params.k_min
        opt.u_max[:] = params.k_max
        opt.integrator_type = opt.EULER

        opt.params.w_pos = params.w_pos
        opt.params.w_k = params.w_k
        opt.params.ref_x = path[:, 0]
        opt.params.ref_y = path[:, 1]
        opt.params.ref_step = local_map.step_size_ref

        # (re-)initialization via warm starting

        index_shift = local_map.shift_idx_start_ref

        self.reset_required |= self.reset_counter != env.reset_counter
        self.reset_counter = env.reset_counter

        if self.reset_required or not 0 <= index_shift < path_len:
            # initialize from reference line
            opt.x[0, :] = path[0, :3]
            opt.u[:] = path[:path_len, 4]
        else:
            # can reinitialize optimizer by shifting
            opt.shift(index_shift)

        self.reset_required = False

        # copy data to optimizer and update

        opt.update()

        # build optimized path structure

        self.opt_path = np.zeros((path_len, 6))
        self.opt_path[:, :3] = opt.x[:-1, :3]
        self.opt_path[:, 3] = local_map.path[:path_len, 3]
        self.opt_path[:, 4] = opt.u
        self.opt_path[:, 5] = local_map.path[:path_len, 5]
