import numpy as np
import pyvista as pv

from .._plot_resp_base import PlotResponseBase
from .plot_utils import PLOT_ARGS, _plot_all_mesh, _plot_lines

slider_widget_args = {
    "pointa": (0.01, 0.925),
    "pointb": (0.45, 0.925),
    "title": "Step",
    "title_opacity": 1,
    # title_color="black",
    "fmt": "%.0f",
    "title_height": 0.03,
    "slider_width": 0.03,
    "tube_width": 0.008,
}


class PlotResponsePyvistaBase(PlotResponseBase):
    def __init__(self, odb_tag, lazy_load=True):
        super().__init__(odb_tag, lazy_load=lazy_load)

        self.slider_widget_args = slider_widget_args
        self.pargs = PLOT_ARGS

        pv.set_plot_theme(PLOT_ARGS.theme)

    def _plot_outline(self, plotter: pv.Plotter):
        plotter.show_bounds(
            grid=False,
            location="outer",
            bounds=self.bounds,
            show_zaxis=self.show_zaxis,
        )

    def _plot_bc(self, plotter: pv.Plotter, step: int, defo_scale: float, bc_scale: float):
        bc_grid = None
        fixed_node_data = self._get_bc_da(step)
        if len(fixed_node_data) > 0:
            fix_tags = fixed_node_data["nodeTags"].values
            fixed_data = fixed_node_data.to_numpy()
            fixed_dofs = fixed_data[:, -6:].astype(int)
            if defo_scale == 0.0:
                node_deform_coords_da = self._get_node_da(step)
            else:
                node_deform_coords_da = self._get_defo_coord_da(step, defo_scale)
            coords_fix = node_deform_coords_da.sel({"nodeTags": fix_tags}).to_numpy()
            s = (self.min_bound_size + self.max_bound_size) / 75 * bc_scale
            bc_grid = _plot_bc(
                plotter,
                fixed_dofs,
                coords_fix,
                s,
                color=self.pargs.color_bc,
                show_zaxis=self.show_zaxis,
            )
        return bc_grid

    def _plot_bc_update(self, bc_grid, step: int, defo_scale: float, bc_scale: float):
        if defo_scale == 0.0:
            return bc_grid
        node_deform_coords_da = self._get_defo_coord_da(step, defo_scale)
        fixed_node_data = self._get_bc_da(step)
        fix_tags = fixed_node_data["nodeTags"].values
        fixed_data = fixed_node_data.to_numpy()
        fixed_dofs = fixed_data[:, -6:].astype(int)
        fixed_node_deform_coords = node_deform_coords_da.sel({"nodeTags": fix_tags}).to_numpy()
        s = (self.max_bound_size + self.min_bound_size) / 75 * bc_scale
        bc_points, _ = _get_bc_points_cells(
            fixed_node_deform_coords,
            fixed_dofs,
            s,
            show_zaxis=self.show_zaxis,
        )
        bc_grid.points = bc_points
        return bc_grid

    def _plot_mp_constraint(self, plotter: pv.Plotter, step: int, defo_scale):
        mp_grid = None
        mp_constraint_data = self._get_mp_constraint_da(step)
        if len(mp_constraint_data) > 0:
            if defo_scale == 0.0:
                node_deform_coords = self._get_node_da(step).to_numpy()
            else:
                node_deform_coords = np.array(self._get_defo_coord_da(step, defo_scale))
            cells = mp_constraint_data.to_numpy()[:, :3].astype(int)
            mp_grid = _plot_mp_constraint(
                plotter,
                node_deform_coords,
                cells,
                None,
                None,
                self.pargs.line_width / 2,
                self.pargs.color_constraint,
                show_dofs=False,
            )
        return mp_grid

    def _plot_mp_constraint_update(self, mp_grid, step: int, defo_scale: float):
        if defo_scale == 0.0:
            return mp_grid
        node_deform_coords = np.array(self._get_defo_coord_da(step, defo_scale))
        mp_grid.points = node_deform_coords
        return mp_grid

    def _plot_all_mesh(self, plotter, color="gray", step=0):
        if self.ModelUpdate or step == 0:
            pos = self._get_node_da(step).to_numpy()
            line_cells, _ = self._get_line_cells(self._get_line_da(step, enforce=True))
            _, unstru_cell_types, unstru_cells = self._get_unstru_cells(self._get_unstru_da(step))

            _plot_all_mesh(
                plotter,
                pos,
                line_cells,
                unstru_cells,
                unstru_cell_types,
                color=color,
                render_lines_as_tubes=False,
            )

    def _update_plotter(self, plotter: pv.Plotter, cpos):
        if isinstance(cpos, str):
            cpos = cpos.lower()
            viewer = {
                "xy": plotter.view_xy,
                "yx": plotter.view_yx,
                "xz": plotter.view_xz,
                "zx": plotter.view_zx,
                "yz": plotter.view_yz,
                "zy": plotter.view_zy,
                "iso": plotter.view_isometric,
            }
            if not self.show_zaxis and cpos not in ["xy", "yx"]:
                cpos = "xy"
                plotter.enable_2d_style()
                plotter.enable_parallel_projection()
            viewer[cpos]()

            if cpos == "iso":  # rotate camera
                plotter.camera.Azimuth(180)
        else:
            plotter.camera_position = cpos
            if not self.show_zaxis:
                plotter.view_xy()
                plotter.enable_2d_style()
                plotter.enable_parallel_projection()

        plotter.add_axes()
        return plotter


def _plot_bc(plotter, fixed_dofs, fixed_coords, s, color, show_zaxis):
    bc_plot = None
    if len(fixed_coords) > 0:
        points, cells = _get_bc_points_cells(fixed_coords, fixed_dofs, s, show_zaxis=show_zaxis)
        bc_plot = _plot_lines(
            plotter,
            points,
            cells,
            color=color,
            render_lines_as_tubes=False,
            width=1,
        )
    else:
        print("Warning:: Model has no fixed nodes!", stacklevel=2)
    return bc_plot


def _plot_mp_constraint(
    plotter,
    points,
    cells,
    dofs,
    midcoords,
    lw,
    color,
    show_dofs=False,
):
    pplot = _plot_lines(plotter, points, cells, width=lw, color=color, label="MP Constraint")
    dofs = ["".join(map(str, row)) for row in dofs]
    if show_dofs and len(cells) > 0:
        plotter.add_point_labels(
            midcoords,
            dofs,
            text_color=color,
            font_size=12,
            bold=True,
            show_points=False,
            always_visible=True,
            shape_opacity=0,
        )
    return pplot


def _get_bc_points_cells(fixed_coords, fixed_dofs, s, show_zaxis):
    if show_zaxis:
        points, cells = _get_bc_points_cells_3d(fixed_coords, fixed_dofs, s)
    else:
        points, cells = _get_bc_points_cells_2d(fixed_coords, fixed_dofs, s)
    return points, cells


def _get_bc_points_cells_2d(fixed_coords, fixed_dofs, s):
    points, cells = [], []
    fixed_dofs = ["".join(map(str, row)) for row in fixed_dofs]
    for coord, dof in zip(fixed_coords, fixed_dofs):
        x, y, z = coord
        idx = len(points)
        if dof[2] == "1":
            y -= s / 2
            points.extend([
                [x - s / 2, y - s / 2, z],
                [x + s / 2, y - s / 2, z],
                [x + s / 2, y + s / 2, z],
                [x - s / 2, y + s / 2, z],
            ])
            cells.extend([2, idx, idx + 1, 2, idx + 1, idx + 2, 2, idx + 2, idx + 3, 2, idx + 3, idx])
        elif dof[0] == "1" and dof[1] == "1":
            points.extend([
                [x - s * 0.5, y - s, z],
                [x + s * 0.5, y - s, z],
                [x, y, z],
            ])
            cells.extend([2, idx, idx + 1, 2, idx + 1, idx + 2, 2, idx + 2, idx])
        else:
            angles = np.linspace(0, 2 * np.pi, 21)
            coords = np.zeros((len(angles), 3))
            coords[:, 0] = 0.5 * s * np.cos(angles)
            coords[:, 1] = 0.5 * s * np.sin(angles)
            coords[:, 2] = z
            cell_i = []
            for i in range(len(angles) - 1):
                cell_i.extend([2, idx + i, idx + i + 1])
            cell_i.extend([2, idx + len(angles) - 1, idx])
            cells.extend(cell_i)
            if dof[0] == "1":
                coords[:, 0] += x - s / 2
                coords[:, 1] += y
            elif dof[1] == "1":
                coords[:, 0] += x
                coords[:, 1] += y - s / 2
            points.extend(coords)
    return points, cells


def _get_bc_points_cells_3d(fixed_coords, fixed_dofs, s):
    points, cells = [], []
    fixed_dofs = ["".join(map(str, row)) for row in fixed_dofs]
    for coord, dof in zip(fixed_coords, fixed_dofs):
        x, y, z = coord
        if dof[0] == "1":
            idx = len(points)
            points.extend([
                [x, y - s / 2, z - s / 2],
                [x, y + s / 2, z - s / 2],
                [x, y + s / 2, z + s / 2],
                [x, y - s / 2, z + s / 2],
            ])
            cells.extend([2, idx, idx + 1, 2, idx + 1, idx + 2, 2, idx + 2, idx + 3, 2, idx + 3, idx])
        if dof[1] == "1":
            idx = len(points)
            points.extend([
                [x - s / 2, y, z - s / 2],
                [x + s / 2, y, z - s / 2],
                [x + s / 2, y, z + s / 2],
                [x - s / 2, y, z + s / 2],
            ])
            cells.extend([2, idx, idx + 1, 2, idx + 1, idx + 2, 2, idx + 2, idx + 3, 2, idx + 3, idx])
        if dof[2] == "1":
            idx = len(points)
            points.extend([
                [x - s / 2, y - s / 2, z],
                [x + s / 2, y - s / 2, z],
                [x + s / 2, y + s / 2, z],
                [x - s / 2, y + s / 2, z],
            ])
            cells.extend([2, idx, idx + 1, 2, idx + 1, idx + 2, 2, idx + 2, idx + 3, 2, idx + 3, idx])
    return points, cells
