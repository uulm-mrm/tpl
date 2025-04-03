# initialize cuda
from tplcpp import cuda_get_device_count, cuda_set_device
cuda_dev_count = cuda_get_device_count()
if cuda_dev_count > 0:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_dev_count-1}"
    cuda_set_device(cuda_dev_count-1)

from tpl.planning.trajectory import Trajectory
from tpl.planning.base_planner import BasePlanner

from tpl.planning.path_vel_decomp.path_smoothing import PathSmoothing
from tpl.planning.path_vel_decomp.path_optim import PathOptim
from tpl.planning.path_vel_decomp.velocity_optim import VelocityOptim
from tpl.planning.path_vel_decomp.path_vel_decomp_planner import PathVelDecompPlanner

from tpl.planning.dyn_prog.dp_lat_lon_planner import DpLatLonPlanner
from tpl.planning.dyn_prog.poly_lat_dp_lon_planner import PolyLatDpLonPlanner
#from tpl.planning.dyn_prog.lattice_planner import LatticePlanner

from tpl.planning.idm_sampling.idm_sampling_planner import IdmSamplingPlanner
