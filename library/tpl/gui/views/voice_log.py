import time
import imviz as viz

from imdash.views.view_2d import View2D
from imdash.utils import speak as util_speak, DataSource


class VoiceLog(View2D):

    DISPLAY_NAME = "Tpl/Voice log"

    def __init__(self):

        super().__init__()

        self.title = "Voice Log"

        self.initialized = False
        self.last_time_speak = 0

        self.active = True

        self.env_log_active = True
        self.env_source = DataSource()
        self.env_error = ""
        
        self.last_reset_counter = None
        self.last_imu_state = None
        self.last_automated = None

        self.planning_log_active = True
        self.planning_source = DataSource()
        self.planning_error = ""

        self.last_reinit_dp_lat_lon_msg = None

    def speak(self, msg):

        now = time.time()
        if now - self.last_time_speak < 3.0:
            return
        util_speak(msg)
        self.last_time_speak = now

    def __savestate__(self):

        d = self.__dict__.copy()

        del d["initialized"]

        del d["last_reset_counter"]
        del d["last_imu_state"]
        del d["last_automated"]

        del d["last_reinit_dp_lat_lon_msg"]

        return d

    def update_speech(self):

        try:
            env = self.env_source.get_used_source().store

            with env.lock():
                imu_state = env.vehicle_state.imu_state
                automated = env.vehicle_state.automated
                reset_counter = env.reset_counter

            if self.last_reset_counter is not None and self.last_reset_counter != reset_counter:
                self.speak("Warning: Environment reset")
            self.last_reset_counter = reset_counter

            if self.last_imu_state is not None and self.last_imu_state != imu_state:
                if imu_state == 0:
                    self.speak("Error: No GPS available")
                elif imu_state == 1:
                    self.speak("Warning: No R.T.K. available")
                elif imu_state == 2:
                    self.speak("Warning: R.T.K. floating")
                elif imu_state == 3:
                    self.speak("Info: R.T.K. locked")
            self.last_imu_state = imu_state

            if self.last_automated is not None and self.last_automated != automated:
                if automated:
                    self.speak("Autonomous mode engaged")
                else:
                    self.speak("Autonomous mode disengaged")
            self.last_automated = automated

            self.env_error = ""
        except Exception as e:
            self.env_error = "Environment not found"

        try:
            planning = self.planning_source.get_used_source().store

            with planning.lock():
                reinit_dp_lat_lon_msg = planning.dp_lat_lon_planner.debug.reinit_msg

            if (self.last_reinit_dp_lat_lon_msg is not None
                and self.last_reinit_dp_lat_lon_msg != reinit_dp_lat_lon_msg):
                self.speak(reinit_dp_lat_lon_msg.split("#")[0])
            self.last_reinit_dp_lat_lon_msg = reinit_dp_lat_lon_msg

            self.planning_error = ""
        except Exception as e:
            self.planning_error = "Planning not found"

    def render(self, sources):

        if not self.initialized and self.active:
            self.speak("Voice log activated")
            self.initialized = True

        if self.active:
            self.update_speech()

        if not self.show:
            return

        if viz.begin_window(self.title):
            self.active = viz.checkbox("voice_log_active", self.active)
            if viz.mod() and self.active:
                self.speak("Voice log activated")
            viz.separator()
            self.env_log_active = viz.checkbox("env_log_active", self.env_log_active)
            viz.autogui(self.env_source, "env_source", sources=sources)
            if self.env_error != "":
                viz.text(self.env_error, color="red")
            viz.separator()
            self.planning_log_active = viz.checkbox("planning_log_active", self.planning_log_active)
            viz.autogui(self.planning_source, "planning_source", sources=sources)
            if self.planning_error != "":
                viz.text(self.planning_error, color="red")
        viz.end_window()
