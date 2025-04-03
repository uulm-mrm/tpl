import glob
import pathlib
import imviz as viz
import os.path as osp
import objtoolbox as otb

from tpl import util
from tpl.environment import load_map_store
from tpl.application.environment_app import load_env_params, save_env_params
from tpl.application.planning_app import load_planning_params, save_planning_params
from tpl.application.control_app import load_control_params, save_control_params
from tpl.simulation import save_sim_state, load_sim_state


def sub_path_selector(name, base_path, path):

    paths = list(pathlib.Path(base_path).glob("**"))
    paths = [p for p in paths if (p / "state.json").exists()]
    paths = [str(p.relative_to(base_path)) for p in paths]
    paths = [p for p in paths if p != "."]
    paths = sorted(paths)

    try:
        active_idx = paths.index(path)
    except ValueError:
        active_idx = 0
    active_idx = viz.combo(f"{name}###{name}", paths, active_idx)

    return paths[active_idx]


def render_environment(env, name, ctx, **kwargs):

    viz.push_mod_any()

    env.__storage__ = sub_path_selector(
            "params",
            osp.join(util.PATH_PARAMS, "env"),
            env.__storage__)
    if viz.mod():
        load_env_params(env)

    viz.separator()

    if viz.button("Open"):
        ctx.request_file_path("open_map_store")
    viz.same_line()
    if viz.button("Reload"):
        env.maps = load_map_store(env.map_store_path)
    if fp := ctx.get_file_path("open_map_store"):
        env.map_store_path = fp
        env.maps = load_map_store(env.map_store_path)

    viz.same_line()
    viz.text(f'map_store_path: "{osp.basename(env.map_store_path)}"')

    all_maps = otb.get_obj_dict(env.maps)
    uuids = ["none"] + list(all_maps.keys())
    names = [""] + [v.name for v in otb.get_obj_dict(env.maps).values()]

    try:
        active_idx = uuids.index(env.selected_map)
    except ValueError:
        active_idx = 0

    active_idx = viz.combo("selected_map", names, active_idx)
    if viz.mod():
        env.reset_counter += 1
        env.selected_map = uuids[active_idx]

    viz.separator()

    if viz.button(f"Reset [{env.reset_counter}]###reset_button"):
        env.reset_counter += 1

    ctx.render(env.maps, "maps")
    ctx.render(env.local_map, "local_map")
    ctx.render(env.vehicle_state, "vehicle_state")
    ctx.render(env.tl_dets, "tl_dets")
    ctx.render(env.ir_pc_dets, "ir_pc_dets")
    ctx.render(env.tracks, "tracks")
    ctx.render(env.predicted, "predicted")

    if viz.pop_mod_any() and env.__storage__ != "":
        save_env_params(env)


def render_planners(s, name, ctx, **kwargs):

    viz.push_mod_any()

    s.__storage__ = sub_path_selector(
            "params",
            osp.join(util.PATH_PARAMS, "planning"),
            s.__storage__)
    if viz.mod():
        load_planning_params(s)

    viz.separator()

    ctx.render(s.runtime, "runtime")
    ctx.render(s.trajectory, "trajectory")

    plan_names = sorted(list(s.planner_names))
    active_idx = plan_names.index(s.active_planner)
    active_idx = viz.combo("active_planner", plan_names, active_idx)
    if viz.mod():
        s.active_planner = plan_names[active_idx]

    viz.separator()

    for p in plan_names:
        ctx.render(getattr(s, p), p)

    if viz.pop_mod_any():
        save_planning_params(s)

    return s


def render_controllers(s, name, ctx, **kwargs):

    viz.push_mod_any()

    s.__storage__ = sub_path_selector(
            "params",
            osp.join(util.PATH_PARAMS, "control"),
            s.__storage__)
    if viz.mod():
        load_control_params(s)

    viz.separator()

    ctx.render(s.runtime, "runtime")
    ctx.render(s.stats, "stats")
    ctx.render(s.controls[0], "acceleration")
    ctx.render(s.controls[1], "steering_angle")

    con_names = sorted(list(s.controller_names))
    ctx.render(s.active_controller_state, "active_controller_state")

    try:
        active_idx = con_names.index(s.active_controller)
    except ValueError:
        active_idx = 0

    active_idx = viz.combo("active_controller", con_names, active_idx)
    if viz.mod():
        s.active_controller = con_names[active_idx]

    viz.separator()

    for c in con_names:
        ctx.render(getattr(s, c), c)

    if viz.pop_mod_any():
        save_control_params(s)

    return s


def render_simulation(sh_sim, name, ctx, **kwargs):

    viz.push_mod_any()

    if viz.button("Save"):
        save_sim_state(sh_sim.sim, sh_sim.__storage__)

    sh_sim.__storage__ = sub_path_selector(
            "scenario",
            util.PATH_SCENARIOS,
            sh_sim.__storage__)
    if viz.mod():
        sh_sim.sim.settings.reload_requested = True

    viz.separator()

    sim = sh_sim.sim

    sim.map_store_path = sub_path_selector(
            "map_store_path",
            util.PATH_MAPS,
            sim.map_store_path)
    if viz.mod():
        sim.map_store_reload_requested = True

    def map_selector(selected_map):

        uuids = ["none"] + [m[0] for m in sim.available_maps]
        names = [""] + [m[1] for m in sim.available_maps]
        try:
            active_idx = uuids.index(selected_map)
        except ValueError:
            active_idx = 0
        active_idx = viz.combo("selected_map", names, active_idx)
        sel = uuids[active_idx]

        return sel

    new_sel = map_selector(sim.selected_map)
    if viz.mod():
        sim.settings.reset_required = True
        sim.selected_map = new_sel

    sim.init_env_params = sub_path_selector(
            "init_env_params",
            osp.join(util.PATH_PARAMS, "env"),
            sim.init_env_params)

    sim.init_planning_params = sub_path_selector(
            "init_planning_params",
            osp.join(util.PATH_PARAMS, "planning"),
            sim.init_planning_params)

    sim.init_control_params = sub_path_selector(
            "init_control_params",
            osp.join(util.PATH_PARAMS, "control"),
            sim.init_control_params)

    viz.separator()

    sim_dict = otb.get_obj_dict(sim)
    del sim_dict["map_store_path"]
    del sim_dict["selected_map"]
    del sim_dict["available_maps"]
    del sim_dict["init_env_params"]
    del sim_dict["init_planning_params"]
    del sim_dict["init_control_params"]
    del sim_dict["cars"]

    ctx.render(sim_dict)

    if viz.tree_node("cars"):
        for i, c in enumerate(sim.cars):
            viz.push_id(c.uuid)
            if viz.tree_node(f'car_{i}'):
                car_dict = otb.get_obj_dict(c)
                del car_dict["map_uuid"]
                del car_dict["uuid"]
                ns = map_selector(c.map_uuid)
                if viz.mod():
                    c.map_uuid = ns
                for k, v in car_dict.items():
                    setattr(c, k, ctx.render(v, k))
                viz.tree_pop()
            viz.pop_id()
        viz.tree_pop()

    if viz.pop_mod_any() and sh_sim.__storage__ != "" and not sh_sim.sim.settings.reload_requested:
        save_sim_state(sh_sim.sim, sh_sim.__storage__)
