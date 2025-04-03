import numpy as np
import imviz as viz

from imdash.utils import ColorEdit


class VehicleRenderer:

    def __init__(self):

        self.color = ColorEdit(default=np.array((0.0, 0.7, 1.0, 1.0)))
        self.line_weight = 1.0
        self.show_imu_state = False
        self.show_origin = True
        self.show_wheels = True
        self.show_dir_triangle = False

    def render(self, veh, idx, comp):

        line_flags = viz.PlotLineFlags.NONE
        if comp.no_fit:
            line_flags |= viz.PlotLineFlags.NO_FIT

        if self.show_dir_triangle:
            bounds = np.array([
                (-veh.rear_axis_to_rear, -veh.width/2, 1),
                (veh.rear_axis_to_front * 0.5, -veh.width/2, 1),
                (veh.rear_axis_to_front, 0.0, 1),
                (veh.rear_axis_to_front * 0.5, veh.width/2, 1),
                (veh.rear_axis_to_front * 0.5, -veh.width/2, 1),
                (veh.rear_axis_to_front, -veh.width/2, 1),
                (veh.rear_axis_to_front, veh.width/2, 1),
                (-veh.rear_axis_to_rear, veh.width/2, 1),
                (-veh.rear_axis_to_rear, -veh.width/2, 1)
            ])
        else:
            bounds = np.array([
                (-veh.rear_axis_to_rear, -veh.width/2, 1),
                (veh.rear_axis_to_front, -veh.width/2, 1),
                (veh.rear_axis_to_front, veh.width/2, 1),
                (-veh.rear_axis_to_rear, veh.width/2, 1),
                (-veh.rear_axis_to_rear, -veh.width/2, 1)
            ])

        wheel_bounds = np.array([
            (-0.4, -0.1, 1),
            (0.4, -0.1, 1),
            (0.4, 0.1, 1),
            (-0.4, 0.1, 1),
            (-0.4, -0.1, 1)
        ])

        trans = np.array([
            [np.cos(veh.phi), -np.sin(veh.phi), veh.x],
            [np.sin(veh.phi), np.cos(veh.phi), veh.y],
            [0.0, 0.0, 1.0]
        ])

        rl_wheel_trans = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, veh.track_width/2.0],
            [0.0, 0.0, 1.0]
        ])
        rr_wheel_trans = rl_wheel_trans.copy()
        rr_wheel_trans[1, 2] -= veh.track_width
        fl_wheel_trans = np.array([
            [np.cos(veh.delta), -np.sin(veh.delta), veh.wheel_base],
            [np.sin(veh.delta), np.cos(veh.delta), veh.track_width/2.0],
            [0.0, 0.0, 1.0]
        ])
        fr_wheel_trans = fl_wheel_trans.copy()
        fr_wheel_trans[1, 2] -= veh.track_width

        fl_wheel_trans = trans @ fl_wheel_trans
        fr_wheel_trans = trans @ fr_wheel_trans
        rl_wheel_trans = trans @ rl_wheel_trans
        rr_wheel_trans = trans @ rr_wheel_trans

        rl_wheel_bounds = wheel_bounds @ rl_wheel_trans.T
        rr_wheel_bounds = wheel_bounds @ rr_wheel_trans.T
        fl_wheel_bounds = wheel_bounds @ fl_wheel_trans.T
        fr_wheel_bounds = wheel_bounds @ fr_wheel_trans.T

        bounds = bounds @ trans.T

        veh_color = self.color()

        if self.show_imu_state:
            if veh.imu_state == 0:
                viz.plot_annotation(veh.x, veh.y + 3, "NO GPS")
                veh_color = (1.0, 0.0, 0.0, 1.0)
            elif veh.imu_state == 1:
                viz.plot_annotation(veh.x, veh.y + 3, "GPS")
                veh_color = (1.0, 0.3, 0.0, 1.0)
            elif veh.imu_state == 2:
                viz.plot_annotation(veh.x, veh.y + 3, "RTK FLOAT")
                veh_color = (1.0, 0.7, 0.0, 1.0)
        
        label = f"{comp.label}###{idx}"

        if self.show_wheels:
            viz.plot(rr_wheel_bounds[:, 0], rr_wheel_bounds[:, 1],
                     label=label,
                     line_weight=self.line_weight,
                     color=veh_color,
                     flags=line_flags)
            viz.plot(rl_wheel_bounds[:, 0], rl_wheel_bounds[:, 1],
                     label=label,
                     line_weight=self.line_weight,
                     color=veh_color,
                     flags=line_flags)
            viz.plot(fr_wheel_bounds[:, 0], fr_wheel_bounds[:, 1],
                     label=label,
                     line_weight=self.line_weight,
                     color=veh_color,
                     flags=line_flags)
            viz.plot(fl_wheel_bounds[:, 0], fl_wheel_bounds[:, 1],
                     label=label,
                     line_weight=self.line_weight,
                     color=veh_color,
                     flags=line_flags)

        if self.show_origin:
            viz.plot([veh.x], [veh.y],
                     fmt="o",
                     label=label,
                     color=veh_color,
                     flags=line_flags)

        viz.plot(bounds[:, 0], bounds[:, 1],
                 label=label,
                 line_weight=self.line_weight,
                 color=veh_color,
                 flags=line_flags)
