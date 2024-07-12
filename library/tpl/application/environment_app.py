import time
import os.path as osp
import objtoolbox as otb

from tpl import util
from tpl.environment import (
        SharedEnvironmentState,
        TrackingModule,
        PredictionModule,
        map_module
    )


class EnvironmentApp:

    SHM_ENV_SIZE = 10**7

    def __init__(self,
                 app_id="",
                 env_params_path=None):

        self.app_id = app_id
        self.last_time = 0.0

        self.env = SharedEnvironmentState.fromparams(
                f"/{self.app_id}tpl_env",
                EnvironmentApp.SHM_ENV_SIZE,
                reinit=True)
        with self.env.lock():
            self.env.__storage__ = "default"
            self.env.__renderer__ = "tpl.gui.state_and_params.render_environment"
            load_env_params(self.env, env_params_path)

        self.tracking_module = TrackingModule()
        self.prediction_module = PredictionModule()

    def update(self, t):

        with self.env.lock():
            map_module.update_local_map(self.env)

        with self.env.lock():
            if self.last_time == t:
                # only update if time changed
                time.sleep(0.001)
                return
            elif t < self.last_time:
                # reinit if time jumps backwards
                self.tracking_module = TrackingModule()
                self.prediction_module = PredictionModule()
                self.last_time = 0.0
            self.last_time = t

        with self.env.lock():
            self.env.t = t
            self.tracking_module.update(self.env)
            self.prediction_module.update(self.env)
            map_module.update_map_items(self.env)


def load_env_params(sh_env, path=None):

    if path is None:
        path = sh_env.__storage__
    abs_path = osp.join(util.PATH_PARAMS, "env", path)

    if not otb.load(sh_env, abs_path):
        return
    
    sh_env.__storage__ = path

    map_store = map_module.load_map_store(sh_env.map_store_path)
    if map_store is None:
        sh_env.map_store_path = ""
        sh_env.maps = otb.bundle()
    else:
        sh_env.maps = map_store

    return True


def save_env_params(sh_env):

    params = otb.bundle()
    params.map_store_path = sh_env.map_store_path
    params.selected_map = sh_env.selected_map

    abs_path = osp.join(PATH_PARAMS, "env", sh_env.__storage__)
    otb.save(params, abs_path)
