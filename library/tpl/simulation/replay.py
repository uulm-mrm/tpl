import time
import copy

import objtoolbox as otb
import structstore as sts


class SimReplayState:

    def __init__(self):

        self.running = True
        self.sleep_time = 0.01
        self.step = 0


class SimReplay:

    def __init__(self, app_id="", recording_path=None):

        self.recording = otb.bundle()
        if not otb.load(self.recording, recording_path):
            print(f"Loading recording from {recording_path} failed")

        if app_id != "":
            app_id += "_"
        self.app_id = app_id

        self.sh_replay = sts.StructStoreShared(f"/{self.app_id}tpl_sim_replay",
                                               10**5,
                                               reinit=True)
        with self.sh_replay.lock():
            self.sh_replay.state = SimReplayState()

        self.sh_state = sts.StructStoreShared(f"/{self.app_id}tpl_sim")

        sim_initialized = False

        while not sim_initialized:
            self.sh_state.revalidate()
            with self.sh_state.lock():
                sim_initialized = hasattr(self.sh_state, "sim")
                print(self.sh_state)
            print("Waiting for tplsim ...")
            time.sleep(1.0)

        print("Found to tplsim.")

        with self.sh_state.lock():
            self.sh_state.sim = self.recording.sim_states[0]
            self.sh_state.sim.settings.running = False

    def update(self):

        self.sh_state.revalidate()

        with self.sh_replay.lock():
            if self.sh_replay.state.running:
                self.sh_replay.state.step += 1 
                self.sh_replay.state.step = max(0, min(
                    len(self.recording.sim_states) - 1,
                    self.sh_replay.state.step)
                )
            state = copy.deepcopy(self.sh_replay.state)

        with self.sh_state.lock():
            self.sh_state.sim = self.recording.sim_states[state.step]
            self.sh_state.sim.settings.running = False

        time.sleep(max(0.0, state.sleep_time))
