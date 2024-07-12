import os
import glob
import argparse

import imviz as viz

from tpl.gui.views import *
from tpl.gui.components import *

from imdash.main import Main as ImdashMain


class GuiMain:

    def __init__(self):

        self.imm = ImdashMain()

        prev_config_path = self.imm.global_config.last_config_path

        # importing base configs
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
        base_path = os.path.normpath(os.path.join(base_path, "data", "gui"))
        for gui_config in glob.glob(os.path.join(base_path, "*")):
            self.imm.import_config(gui_config, overwrite=False)

        # open previous config
        self.imm.open_config(prev_config_path)

    def update(self):
        self.imm.update()


def main():

    parser = argparse.ArgumentParser(prog="tplgui",
                                     description="tpl graphical user interface")

    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    if args.dev:
        viz.dev.launch(GuiMain, "update")
    else:
        gm = GuiMain()
        while True:
            gm.update()
