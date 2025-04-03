import os
import copy
import time
import numba
import argparse

import numpy as np
import moderngl as mgl
import cv2 as cv
import structstore as sts

from tpl.environment import load_map_store
from tpl import util

import pyrr
import pywavefront


class SimModel:

    CACHE = {}

    def __init__(self, ctx, prog, path=None):

        if path is not None:
            if path not in SimModel.CACHE:

                self.car_model = pywavefront.Wavefront(
                        os.path.join(util.PATH_DATA, "models", path))

                vtxs = []
                cols = []
                for name, mtl in self.car_model.materials.items():
                    vtxs += mtl.vertices
                    cols += [mtl.diffuse[:3] for i in range(len(mtl.vertices) // 3)]
                vtxs = np.array(vtxs).reshape((-1, 3))
                cols = np.array(cols).reshape((-1, 3))

                vtx_data = np.zeros((len(vtxs), 2, 3))
                vtx_data[:, 0, :] = vtxs
                vtx_data[:, 1, :] = cols

                vbo = ctx.buffer(vtx_data.astype("f4").tobytes())
                vao = ctx.vertex_array(prog, vbo, "in_vert", "in_color")

                SimModel.CACHE[path] = (vbo, vao)

            self.vbo, self.vao = SimModel.CACHE[path] 
        else:
            self.vbo = None
            self.vao = None

        self.pos = np.array([0.0, 0.0, 0.0])
        self.rot = np.array([0.0, 0.0, 0.0])
        self.scale = np.array([1.0, 1.0, 1.0])

    def mat_model(self):

        mat_trans = pyrr.matrix44.create_from_translation(self.pos)
        mat_rot = (pyrr.matrix44.create_from_x_rotation(self.rot[1])
                   @ pyrr.matrix44.create_from_y_rotation(self.rot[2])
                   @ pyrr.matrix44.create_from_z_rotation(self.rot[0]))

        return (mat_trans.T @ mat_rot.T).T


class SimCamera:

    def __init__(self,
                 ctx,
                 pos=(0.0, 0.0, 0.0),
                 rot=(0.0, 0.0, 0.0),
                 width_img=512,
                 height_img=512,
                 fovy=100.0):

        self.pos = np.array(pos)
        # roll, pitch, yaw
        self.rot = np.array(rot)

        self.width_img = width_img
        self.height_img = height_img

        self.fovy = fovy
        self.z_near = 0.1
        self.z_far = 1000.0

        self._tex_color = ctx.texture((1, 1), 3)
        self._tex_depth = ctx.depth_texture((1, 1))
        self._fbo = None

        self.resize(ctx)
        
    def resize(self, ctx):

        do_resize = False
        do_resize |= self._tex_color.width != self.width_img
        do_resize |= self._tex_color.height != self.height_img

        if not do_resize:
            return

        self._tex_color = ctx.texture(
                (self.width_img, self.height_img), 3)
        self._tex_depth = ctx.depth_texture(
                (self.width_img, self.height_img))

        self._fbo = ctx.framebuffer(
            color_attachments=[self._tex_color],
            depth_attachment=self._tex_depth)

    def mat_view(self):

        mat_trans = pyrr.matrix44.create_from_translation(self.pos)
        mat_rot = (pyrr.matrix44.create_from_x_rotation(self.rot[1])
                   @ pyrr.matrix44.create_from_y_rotation(self.rot[2])
                   @ pyrr.matrix44.create_from_z_rotation(self.rot[0]))

        return (mat_trans.T @ mat_rot.T).T

    def mat_proj(self):

        return pyrr.matrix44.create_perspective_projection(
                self.fovy,
                self.width_img/self.height_img,
                self.z_near,
                self.z_far)

    def get_color_img(self):

        return np.frombuffer(
                cam._tex_color.read(),
                dtype=np.uint8).reshape(
                    (cam.height_img, cam.width_img, 3))

    def get_depth_img(self, metric=False):

        img_depth = np.frombuffer(
                cam._fbo.depth_attachment.read(),
                dtype=np.float32).reshape(
                    (cam.height_img, cam.width_img))

        if metric:
            # convert to metric depth
            img_depth_norm = 2.0 * cam.z_near * cam.z_far / (
                    cam.z_far + cam.z_near - img_depth_norm 
                    * (cam.z_far - cam.z_near));
            img_depth_norm = np.maximum(0.0, 100.0 - img_depth_norm)
            img_depth_norm /= np.max(100.0)
        else:
            img_depth_norm = 1.0 - (2.0 * img_depth - 1.0)
            img_depth_norm *= 255.0

        return img_depth


class FpsCamera(SimCamera):

    def __init__(self, ctx, **kwargs):

        super().__init__(ctx, **kwargs)

        self.locked = False
        self.last_mouse_pos = np.array(viz.get_mouse_pos())
        self.last_window_hovered = False

    def update(self):

        yaw = self.rot[2]

        speed = 0.1
        speed_angle = 0.005

        if viz.get_key(viz.KEY_LEFT_CONTROL).action != viz.RELEASE:
            speed *= 20.0

        c = speed * -np.cos(yaw)
        s = speed * np.sin(yaw)

        if viz.get_key(viz.KEY_W).action != viz.RELEASE:
            self.pos[0] += s
            self.pos[2] += c
        if viz.get_key(viz.KEY_A).action != viz.RELEASE:
            self.pos[0] += c
            self.pos[2] += -s
        if viz.get_key(viz.KEY_S).action != viz.RELEASE:
            self.pos[0] -= s
            self.pos[2] -= c
        if viz.get_key(viz.KEY_D).action != viz.RELEASE:
            self.pos[0] += -c
            self.pos[2] += s
        if viz.get_key(viz.KEY_LEFT_SHIFT).action != viz.RELEASE:
            self.pos[1] -= speed
        if viz.get_key(viz.KEY_SPACE).action != viz.RELEASE:
            self.pos[1] += speed

        if viz.is_window_hovered() and viz.is_mouse_clicked(0):
            self.locked = not self.locked

        if not self.last_window_hovered and viz.is_window_hovered():
            self.last_mouse_pos = np.array(viz.get_mouse_pos())
        self.last_window_hovered = viz.is_window_hovered()

        if not self.last_window_hovered:
            self.locked = False
        
        mouse_pos = np.array(viz.get_mouse_pos())
        mouse_delta = mouse_pos - self.last_mouse_pos
        self.last_mouse_pos = mouse_pos

        if self.locked:
            self.rot[1] += np.arcsin(min(1.0, max(-1.0, mouse_delta[1] * speed_angle)))
            self.rot[1] = min(1.56, max(-1.56, self.rot[1]))
            self.rot[2] += np.arcsin(min(1.0, max(-1.0, mouse_delta[0] * speed_angle)))


class Renderer:

    def __init__(self, interactive=False):

        super().__init__()

        self.interactive = interactive
        if self.interactive:
            import imviz as viz
            globals()["viz"] = viz

        self.ctx = mgl.create_context(standalone=not self.interactive)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 mat_vp;
                uniform mat4 mat_model;

                in vec3 in_vert;
                in vec3 in_color;
        
                out vec3 v_color;
        
                void main() {
                    v_color = in_color;
                    gl_Position = mat_vp * mat_model * vec4(in_vert, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
        
                in vec3 v_color;
                out vec3 f_color;
        
                void main() {
                    f_color = v_color;
                }
            """)

        self.sh_sim = sts.StructStoreShared("/tpl_sim")
        sim_initialized = False
        while not sim_initialized:
            self.sh_sim.revalidate()
            with self.sh_sim.lock():
                sim_initialized = hasattr(self.sh_sim, "sim")
            print("Waiting for tplsim ...")
            time.sleep(1.0)

        self.maps = load_map_store("ulm")
        for m in self.maps.values():
            if m.name == "albert_einstein_allee":
                break
        
        self.ctx.enable(mgl.DEPTH_TEST)

        self.origin_utm = m.path[0, :2]
        self.mo = self.gen_map_geom(m)

    def gen_map_geom(self, cmap, steps_lat=10):

        ps = cmap.path[:, :2] - self.origin_utm
        os = cmap.path[:, 2] 
        ls = cmap.d_left.copy()
        rs = cmap.d_right.copy()

        if cmap.closed_path:
            ps = np.array([*ps, ps[0]])
            os = np.array([*os, os[0]])
            ls = np.array([*ls, ls[0]])
            rs = np.array([*rs, rs[0]])

        n = np.vstack([-np.sin(os), np.cos(os)]).T
        vertices = []

        col_road = np.array([0.5, 0.5, 0.5])
        col_line = np.array([0.1, 0.1, 0.1])

        for i in range(1, len(ps)):
            p_p = ps[i-1]
            p_n = ps[i]
            step_ls_p = (ls[i-1] + rs[i-1]) / (steps_lat - 1)
            step_ls_n = (ls[i] + rs[i]) / (steps_lat - 1)
            for j in range(1, steps_lat):
                c0 = p_p + n[i-1] * ((j-1) * step_ls_p - rs[i-1])
                c1 = p_p + n[i-1] * (j * step_ls_p - rs[i-1])
                c2 = p_n + n[i] * ((j-1) * step_ls_n - rs[i])
                c3 = p_n + n[i] * (j * step_ls_n - rs[i])
                c0 = np.array([c0[0], 0.0, -c0[1], *col_road])
                c1 = np.array([c1[0], 0.0, -c1[1], *col_road])
                c2 = np.array([c2[0], 0.0, -c2[1], *col_road])
                c3 = np.array([c3[0], 0.0, -c3[1], *col_road])
                vertices += [c0, c3, c2]
                vertices += [c0, c1, c3]

        stripe_width = 0.2
        stripe_period = 8
        stripe_length = 3

        for i in range(1, len(ps)):
            if i % stripe_period < stripe_length:
                continue
            p_p = ps[i-1]
            p_n = ps[i]
            c0 = p_p + n[i-1] * -rs[i-1]
            c1 = p_p + n[i-1] * (stripe_width - rs[i-1])
            c2 = p_n + n[i] * -rs[i]
            c3 = p_n + n[i] * (stripe_width - rs[i])
            c0 = np.array([c0[0], 0.001, -c0[1], *col_line])
            c1 = np.array([c1[0], 0.001, -c1[1], *col_line])
            c2 = np.array([c2[0], 0.001, -c2[1], *col_line])
            c3 = np.array([c3[0], 0.001, -c3[1], *col_line])
            vertices += [c0, c3, c2]
            vertices += [c0, c1, c3]

        for i in range(1, len(ps)):
            if i % stripe_period < stripe_length:
                continue
            p_p = ps[i-1]
            p_n = ps[i]
            c0 = p_p + n[i-1] * (ls[i-1] - stripe_width)
            c1 = p_p + n[i-1] * ls[i-1]
            c2 = p_n + n[i] * (ls[i] - stripe_width)
            c3 = p_n + n[i] * ls[i]
            c0 = np.array([c0[0], 0.001, -c0[1], *col_line])
            c1 = np.array([c1[0], 0.001, -c1[1], *col_line])
            c2 = np.array([c2[0], 0.001, -c2[1], *col_line])
            c3 = np.array([c3[0], 0.001, -c3[1], *col_line])
            vertices += [c0, c3, c2]
            vertices += [c0, c1, c3]

        vertices = np.array(vertices)

        mo = SimModel(self.ctx, self.prog)
        mo.vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        mo.vao = self.ctx.vertex_array(self.prog, mo.vbo, "in_vert", "in_color")

        return mo

    def loop(self):

        if self.interactive:
            self.interactive_loop()
        else:
            # TODO: implement
            pass

    def interactive_loop(self):

        cam = FpsCamera(self.ctx, width_img=1024, height_img=750)
        cam.pos[0] = 0.0
        cam.pos[1] = 2.0
        cam.pos[2] = 0.0

        while viz.wait():
            self.render_camera(cam)
            if viz.begin_window("Simulation"):
                viz.image_texture(cam._tex_color.glo, 
                                  (cam.width_img, cam.height_img))
                w, h = viz.get_window_size()
                cam.width_img = int(w)
                cam.height_img = int(h)
                cam.resize(self.ctx)
                cam.update()
            viz.end_window()

    def render_camera(self, cam):

        self.sh_sim.revalidate()

        # flip up and down to avoid flipping rendered textures later
        mat_flip = np.eye(4)
        mat_flip[1, 1] *= -1
        mat_vp = (cam.mat_proj().T @ mat_flip @ np.linalg.inv(cam.mat_view().T)).T

        self.prog["mat_vp"].write(mat_vp.astype(np.float32).copy())

        cam._fbo.use()
        cam._fbo.clear(1.0, 1.0, 1.0, 1.0)

        with self.sh_sim.lock():
            sim_state = copy.deepcopy(self.sh_sim.sim)

        models = []

        for c in sim_state.cars:
            m = SimModel(self.ctx, self.prog, "car.obj")
            m.pos = np.array([c.x, 0.0, -c.y])
            m.rot = np.array([0.0, 0.0, -c.yaw - np.pi*0.5])
            models.append(m)

        for m in models:
            m.pos[0] -= self.origin_utm[0]
            m.pos[2] -= -self.origin_utm[1]

        models.append(self.mo)

        for o in models:
            self.prog["mat_model"].write(
                    o.mat_model().astype(np.float32).copy())
            o.vao.render(mgl.TRIANGLES)


def main():

    parser = argparse.ArgumentParser(
                    prog='tplsim_renderer',
                    description='Renders the simulation environment')

    parser.add_argument('-i', '--interactive', action='store_true') 
    args = parser.parse_args()

    renderer = Renderer(interactive=args.interactive)
    renderer.loop()


if __name__ == "__main__":
    main()
