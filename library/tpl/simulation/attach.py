import time
from tpl.simulation import SimCore

import structstore as sts


class SimAttach:

    def __init__(self, sh_env_path, scenario_path=None):

        self.core = SimCore(app_id="", scenario_path=scenario_path)

        self.sh_env_path = sh_env_path
        self.sh_env = sts.StructStoreShared(sh_env_path)

    def validate_env(self):

        env_ok = False
        while not env_ok:
            self.sh_env.revalidate()
            with self.sh_env.lock():
                env_ok = hasattr(self.sh_env, "env")
            if not env_ok:
                time.sleep(1.0)
                print(f"Waiting for valid environment at {self.sh_env_path} ...")

    def update(self):

        self.validate_env()

        sim = self.core.get_next_sim_state(self.sh_env)
        self.core.write_sim_state(sim)
