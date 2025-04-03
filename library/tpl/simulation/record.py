import os
import copy
import os.path as osp
import objtoolbox as otb
import multiprocessing as mp

from tpl.simulation import state_from_shstate


class ImdashRecorder():

    def __init__(self, imdash_config, task_queue, done_queue):

        from imdash.main import GLOBAL_CONF_DIR
        from tpl.gui.main import GuiMain

        if osp.isabs(imdash_config):
            self.imdash_conf_path = imdash_config
        else:
            self.imdash_conf_path = os.path.join(
                GLOBAL_CONF_DIR, "config_store", imdash_config)

        self.gui_main = GuiMain()
        self.gui_main.imm.open_config(self.imdash_conf_path)

        self.task_queue = task_queue
        self.done_queue = done_queue

    def update(self):

        self.gui_main.imm.update()


def export_imdash_plot(output_path, name, postfix):

    export_path = os.path.join(os.path.dirname(output_path), "implots")
    os.makedirs(export_path, exist_ok=True)

    import imviz as viz
    pb_cls = viz.export.PlotBuffer
    pb: viz.export.PlotBuffer = None
    for k, pb in pb_cls.plots.items():
        if pb.label.split("#")[0] == name:
            break

    if pb is None:
        print(f"Warning: could not find plot '{name}' to be exported")
        return

    pb.capture = True

    fn = viz.export.clean_label(pb.label).lower().replace(" ", "_")
    if postfix != "":
        fn += "_" + postfix
    fn += "." + pb.export.filetype
    pb.export.path = os.path.join(export_path, fn)


def start_imdash_recorder(output_path, imdash_config, task_queue, done_queue):

    # force offscreen rendering
    os.environ["DISPLAY"] = "no-display"

    import imviz as viz
    from imdash.utils import FfmpegRecorder

    # keep plots for later
    viz.export.PlotBuffer.persistent = True

    rec = ImdashRecorder(imdash_config, task_queue, done_queue)
    frame_recorder = None

    while True:
        cmd, *args = task_queue.get()

        if cmd == "finish":
            break

        rec.update()

        if cmd == "frame":
            if frame_recorder is None:
                menu_bar_width = int(viz.get_global_font_size() * 2.4)
                frame_size = viz.get_main_window_size()
                frame_size[1] -= menu_bar_width
                frame_recorder = FfmpegRecorder(output_path, *frame_size)
            frame = viz.get_pixels(0, menu_bar_width, *frame_size)
            frame_recorder.record(frame)
            done_queue.put(True)

        if cmd == "export_plots":
            export_imdash_plot(output_path, *args)
            rec.update()

    frame_recorder.finish()


class SimRecorder:

    def __init__(self,
                 standalone,
                 path, 
                 imdash_video_config=None,
                 dt_state_log=0.05):

        self.sta = standalone
        self.path = path
        self.imdash_video_config = imdash_video_config
        self.dt_state_log = dt_state_log

        self.video_path = osp.join(self.path, f"video.mp4")
        os.makedirs(self.path, exist_ok=True)

        self.t_last_video_frame = -1.0
        self.t_last_state_log = -1.0

        if self.imdash_video_config is not None:

            mm = mp.Manager()

            self.task_queue = mm.Queue()
            self.done_queue = mm.Queue()

            ctx = mp.get_context("spawn")
            self.imdash_proc = ctx.Process(
                    target=start_imdash_recorder,
                    args=(self.video_path,
                          self.imdash_video_config,
                          self.task_queue,
                          self.done_queue),
                    daemon=True)
            self.imdash_proc.start()

        self.log = otb.bundle()
        self.log.state_sim = []
        self.log.runtime_planner = []
        self.log.active_planner = []
        self.log.runtime_controller = []
        self.log.active_controller = []

    def push_task(self, cmd, *args):

        self.task_queue.put((cmd, *args))

    def capture(self):

        with self.sta.core.sh_state.lock():
            sim = copy.deepcopy(self.sta.core.sh_state.sim)

        if round(sim.t - self.t_last_state_log, 5) >= self.dt_state_log:
            self.log.state_sim.append(state_from_shstate(sim))

            with self.sta.planning_app.sh_planners.lock():
                self.log.runtime_planner.append(
                        self.sta.planning_app.sh_planners.runtime)
                self.log.active_planner.append(
                        self.sta.planning_app.sh_planners.active_planner)

            with self.sta.control_app.sh_controllers.lock():
                self.log.runtime_controller.append(
                        self.sta.control_app.sh_controllers.runtime)
                self.log.active_controller.append(
                        self.sta.control_app.sh_controllers.active_controller)

            self.t_last_state_log = sim.t

        if (self.imdash_video_config is not None 
                and sim.t - self.t_last_video_frame > 1.0/24.0):
            # do some dummy updates before first frame
            if self.t_last_video_frame < 0.0:
                for i in range(3):
                    self.push_task("dummy")
            self.push_task("frame")
            self.done_queue.get()
            self.t_last_video_frame = sim.t

    def export_plot(self, plot_name, postfix=""):

        if self.imdash_video_config is not None:
            self.push_task("export_plots", plot_name, postfix)

    def finish(self):

        otb.save(self.log, self.path)

        if self.imdash_video_config is not None:
            self.push_task("finish")
            self.imdash_proc.join()
