import os
import io
import glob
import shutil
import zipfile
import threading

import numpy as np
import imviz as viz

from PIL import Image
from urllib.request import urlopen
from concurrent.futures import Future, ThreadPoolExecutor

from imdash.utils import DataSource, ColorEdit
from imdash.views.view_2d import View2DComponent


class Tile:

    def __init__(self):

        self.x = 0
        self.y = 0
        self.scale_x = 0
        self.scale_y = 0

        self.img_path = None
        self.img = None


class TileCacheManager:

    def __init__(self):

        self.plot_limits = None

        self.download_thread = None
        self.download_error = ""

        self.cache_dir = os.path.expanduser("~/.cache/tplgui_map_tiles")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.load_thread_pool = ThreadPoolExecutor(1)

        self.cache = {}
        self.load_from_cache()

    def compress_cache(self):

        for img_path in glob.glob(self.cache_dir + "/**/*.tif", recursive=True):
            Image.open(img_path).save(os.path.splitext(img_path)[0] + ".jpg", 'jpeg')
            os.remove(img_path)

    def load_from_cache(self):

        cache = {}

        for img_path in glob.glob(self.cache_dir + "/**/*.jpg", recursive=True):
            tfw_path = os.path.splitext(img_path)[0] + ".tfw"

            with open(tfw_path, "r") as fd:
                x_width_pix = float(fd.readline())
                y_width_pix = float(fd.readline())
                x_height_pix = float(fd.readline())
                y_height_pix = float(fd.readline())
                x = float(fd.readline())
                y = float(fd.readline())

            tile = Tile()
            tile.x = x - x_width_pix/2
            tile.y = y - y_height_pix/2
            tile.scale_x = abs(x_width_pix)
            tile.scale_y = abs(y_height_pix)
            tile.img_path = img_path
            tile.img = self.load_thread_pool.submit(
                    lambda p: np.asarray(Image.open(p)), img_path)

            tile_id = f"{int(tile.x/1000)}_{int(tile.y/1000)}"
            cache[tile_id] = tile

        self.cache = cache

    def get_current_tile_coords(self):

        w = self.plot_limits[2] - self.plot_limits[0]
        h = self.plot_limits[3] - self.plot_limits[1]
        x = self.plot_limits[0] + w/2
        y = self.plot_limits[1] + h/2
        x = int(x / 1000)
        y = int(y / 1000)
        if x % 2 == 0:
            x -= 1
        if y % 2 > 0:
            y -= 1

        return x, y

    def download_tiles(self):

        try:
            if self.plot_limits == None:
                return

            x, y = self.get_current_tile_coords()

            # TODO: add other tile sources

            url = "https://opengeodata.lgl-bw.de/data/dop20/"
            url += f"dop20rgb_32_{x}_{y}_2_bw.zip"

            http_response = urlopen(url)
            zf = zipfile.ZipFile(io.BytesIO(http_response.read()))
            zf.extractall(path=self.cache_dir)
        except Exception as e:
            self.download_error = str(e)

        self.compress_cache()

    def delete_tiles(self):

        x, y = self.get_current_tile_coords()
        for tile_dir in glob.glob(self.cache_dir + f"/*{x}_{y}*"):
            shutil.rmtree(tile_dir)

        self.load_from_cache()

    def start_download_thread(self):

        self.download_error = ""
        self.download_thread = threading.Thread(target=self.download_tiles)
        self.download_thread.start()

    def update(self):

        if (self.download_thread is not None and 
                not self.download_thread.is_alive()):
            self.load_from_cache()
            self.download_thread = None

        self.plot_limits = viz.get_plot_limits()


class MapTilesComponent(View2DComponent):

    DISPLAY_NAME = "Tpl/Map tiles component"
    CACHE = TileCacheManager()

    def __init__(self):

        super().__init__()

        self.label = "map_tiles"
        self.hide = False

    def __autogui__(self, *args, **kwargs):

        self.label = viz.autogui(self.label, "label")
        self.hide = viz.autogui(self.hide, "hide")

        c = MapTilesComponent.CACHE

        if c.download_thread == None:
            if viz.button("Download tiles"):
                c.start_download_thread()
            if viz.is_item_hovered():
                viz.begin_tooltip()
                viz.text("Download the tiles at the current center of the map")
                viz.end_tooltip()
            if viz.button("Delete tiles"):
                c.delete_tiles()
            if viz.is_item_hovered():
                viz.begin_tooltip()
                viz.text("Delete the tiles at the current center of the map")
                viz.end_tooltip()
        else:
            viz.text("Downloading, this may take a while ...")

        if c.download_error != "":
            viz.text("Download error: " + c.download_error, color="red")

    def render(self, idx, view):
        
        c = MapTilesComponent.CACHE
        c.update()

        if not self.hide:
            for tile_id, tile in c.cache.items():
                if type(tile.img) == Future:
                    if tile.img.done():
                        tile.img = tile.img.result()
                    else:
                        continue
                viz.plot_image(f"###{idx}_{tile_id}",
                               tile.img,
                               tile.x,
                               tile.y - tile.img.shape[0] * tile.scale_x,
                               tile.img.shape[1] * tile.scale_y,
                               tile.img.shape[0] * tile.scale_x,
                               skip_upload=True)

        viz.plot_dummy(f"{self.label}###{idx}", legend_color="white")
