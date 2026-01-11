from functools import partial
from typing import Optional, Union

import numpy as np
import pyvista as pv

from .._plot_nodal_resp_base import PlotNodalResponseBase
from .plot_resp_base import PlotResponsePyvistaBase
from .plot_utils import PLOT_ARGS, _plot_all_mesh_cmap, _plot_lines, _plot_lines_cmap, _update_point_label_actor


class PlotNodalResponse(PlotNodalResponseBase, PlotResponsePyvistaBase):
    def __init__(self, odb_tag, lazy_load=True):
        super().__init__(odb_tag, lazy_load=lazy_load)

    def _make_title(self, step, time):
        max_norm, min_norm = np.nanmax(self.resps_norm[step]), np.nanmin(self.resps_norm[step])
        title = "Nodal Responses"
        if self.resp_type == "disp":
            resp_type = "Displacement"
        elif self.resp_type == "vel":
            resp_type = "Velocity"
        elif self.resp_type == "accel":
            resp_type = "Acceleration"
        else:
            resp_type = f"{self.resp_type.capitalize()}"
        dof = ",".join(self.component) if isinstance(self.component, (list, tuple)) else self.component
        size_symbol = ("norm.min", "norm.max") if isinstance(self.component, (list, tuple)) else ("min", "max")
        info = {
            "title": title,
            "resp_type": resp_type,
            "dof": dof,
            "min": min_norm,
            "max": max_norm,
            "step": step,
            "time": time,
        }
        lines = [
            f"{info['step']} (step)",
            f"{info['time']:.3f} (time)",
            f"{info['min']:.3E} ({size_symbol[0]})",
            f"{info['max']:.3E} ({size_symbol[1]})",
            "",
            f"{info['title']}",
            f"{info['resp_type']}",
            f"{info['dof']} (DOF)",
        ]
        if self.unit_symbol:
            lines.append(f"{self.unit_symbol} (unit)")

        max_len = max(len(line) for line in lines)
        padded_lines = [line.rjust(max_len) for line in lines]
        text = "\n".join(padded_lines)
        return text + "\n"

    def get_dataset(self, step, defo_scale=1.0):
        cmin, cmax, step = self._get_resp_clim_peak(idx=step)
        if self.resps_norm is not None:
            scalars = self._get_step_norm(step)
        else:
            node_resp = np.array(self._get_resp_da(step, self.resp_type, self.component))
            scalars = node_resp if node_resp.ndim == 1 else np.linalg.norm(node_resp, axis=-1)
        # ----------------------------------
        pos = np.array(self._get_defo_coord_da(step, defo_scale))
        line_cells, _ = self._get_line_cells(self._get_line_da(step))
        _, unstru_cell_types, unstru_cells = self._get_unstru_cells(self._get_unstru_da(step))

        cells, celltypes = [], []
        for cell in line_cells:
            cells.extend(cell)
            celltypes.append(pv.CellType.LINE)
        for cell_type, cell in zip(unstru_cell_types, unstru_cells):
            cells.extend(cell)
            celltypes.append(cell_type)

        if len(cells) > 0:
            unstru_grid = pv.UnstructuredGrid(cells, celltypes, pos)
            unstru_grid[self.resp_type] = scalars
        else:
            unstru_grid = pv.UnstructuredGrid()  # Empty UnstructuredGrid if no cells are present
        return unstru_grid

    def _create_mesh(
        self,
        plotter,
        value,
        alpha=1.0,
        clim=None,
        style="surface",
        show_outline=False,
        show_origin=False,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = False,
        cpos="iso",
        show_max_min: bool = True,
    ):
        step = round(value)
        line_cells, _ = self._get_line_cells(self._get_line_da(step))
        _, unstru_cell_types, unstru_cells = self._get_unstru_cells(self._get_unstru_da(step))
        t_ = self.time[step]
        node_no_deform_coords = np.array(self._get_node_da(step))
        node_defo_coords, scalars = self._get_mesh_data(step, alpha)
        # ----------------------------------------------------------------------------------------------
        # plotter.clear_actors()  # ! clear
        self.clear_plotter(plotter)
        point_grid, line_grid, solid_grid = _plot_all_mesh_cmap(
            plotter,
            node_defo_coords,
            line_cells,
            unstru_cells,
            unstru_cell_types,
            scalars=scalars,
            cmap=self.pargs.cmap,
            clim=clim,
            lw=self.pargs.line_width,
            show_edges=self.pargs.show_mesh_edges,
            edge_color=self.pargs.mesh_edge_color,
            edge_width=self.pargs.mesh_edge_width,
            opacity=self.pargs.mesh_opacity,
            style=style,
            show_scalar_bar=False,
            point_size=self.pargs.point_size,
            render_lines_as_tubes=self.pargs.render_lines_as_tubes,
            render_points_as_spheres=self.pargs.render_lines_as_tubes,
            show_origin=show_origin,
            pos_origin=node_no_deform_coords,
        )
        title = self._make_title(step, t_)
        scalar_bar = plotter.add_scalar_bar(title=title, **self.pargs.scalar_bar_kargs)
        if scalar_bar:
            # scalar_bar.SetTitle(title)
            title_prop = scalar_bar.GetTitleTextProperty()
            # title_prop.SetJustificationToRight()
            title_prop.BoldOn()

        if self.interp_beam_disp_on:
            points_origin_interp, points_defo_interp, cells_interp, scalars_interp = self.get_interp_beam_data(
                step, alpha
            )
            line_interp_grid = _plot_lines_cmap(
                plotter=plotter,
                pos=points_defo_interp,
                cells=cells_interp,
                scalars=scalars_interp,
                cmap=self.pargs.cmap,
                width=self.pargs.line_width,
                render_lines_as_tubes=self.pargs.render_lines_as_tubes,
                clim=None,
                show_scalar_bar=False,
            )
            if show_origin:
                _plot_lines(
                    plotter=plotter,
                    pos=points_origin_interp,
                    cells=cells_interp,
                    color="gray",
                    width=self.pargs.mesh_edge_width,
                    render_lines_as_tubes=self.pargs.render_lines_as_tubes,
                )
        else:
            line_interp_grid = None

        # Max Min Labels
        if show_max_min:
            idxs = [np.argmax(scalars), np.argmin(scalars)]
            labels = [f"Max: {scalars[idxs[0]]: .3e}", f"Min: {scalars[idxs[1]]: .3e}"]
            max_min_label_grid = plotter.add_point_labels(
                node_defo_coords[idxs],
                labels,
                show_points=False,
                point_size=self.pargs.point_size + 10,
                font_size=self.pargs.font_size + 5,
                shape_color="#c0fb2d",
                always_visible=True,
            )
        else:
            max_min_label_grid = None

        if show_outline:
            self._plot_outline(plotter)
        bc_grid, mp_grid = None, None
        if show_bc:
            bc_grid = self._plot_bc(plotter, step, defo_scale=alpha, bc_scale=bc_scale)
        if show_mp_constraint:
            mp_grid = self._plot_mp_constraint(plotter, step, defo_scale=alpha)
        self._update_plotter(plotter, cpos=cpos)
        return point_grid, line_grid, solid_grid, scalar_bar, bc_grid, mp_grid, max_min_label_grid, line_interp_grid

    def _update_mesh(
        self,
        value,
        point_grid=None,
        line_grid=None,
        solid_grid=None,
        scalar_bar=None,
        bc_grid=None,
        mp_grid=None,
        max_min_label_grid=None,
        line_interp_grid=None,
        alpha=1.0,
        bc_scale: float = 1.0,
        plotter=None,
    ):
        step = round(value)
        t_ = self.time[step]
        node_defo_coords, scalars = self._get_mesh_data(step, alpha)
        if point_grid:
            point_grid["scalars"] = scalars
            point_grid.points = node_defo_coords
        if line_grid:
            line_grid["scalars"] = scalars
            line_grid.points = node_defo_coords
        if solid_grid:
            solid_grid["scalars"] = scalars
            solid_grid.points = node_defo_coords
        # plotter.update_scalar_bar_range(clim=[np.min(scalars), np.max(scalars)])
        if scalar_bar:
            title = self._make_title(step, t_)
            # cbar.SetTitle(title)
            scalar_bar.SetTitle(title)
        if mp_grid:
            self._plot_mp_constraint_update(mp_grid, step, defo_scale=alpha)
        if bc_grid:
            self._plot_bc_update(bc_grid, step, defo_scale=alpha, bc_scale=bc_scale)
        if max_min_label_grid:
            idxs = [np.argmax(scalars), np.argmin(scalars)]
            labels = [f"Max: {scalars[idxs[0]]: .3e}", f"Min: {scalars[idxs[1]]: .3e}"]
            text_property = pv.TextProperty(
                bold=True,
                font_size=self.pargs.font_size + 10,
            )
            _update_point_label_actor(
                max_min_label_grid,
                node_defo_coords[idxs],
                labels,
                text_property=text_property,
                renderer=plotter.renderer,
                shape_opacity=100.0,
                always_visible=True,
            )
        if line_interp_grid:
            _, points_defo, _, scalars = self.get_interp_beam_data(step, alpha)
            line_interp_grid["scalars"] = scalars
            line_interp_grid.points = points_defo

    def plot_slide(
        self,
        plotter,
        alpha=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        style="surface",
        show_outline=False,
        show_origin=False,
        cpos="iso",
        show_max_min: bool = True,
        **kargs,
    ):
        cmin, cmax, _ = self._get_resp_clim_peak()
        clim = (cmin, cmax)
        alpha_ = alpha if show_defo else 0.0
        if self.ModelUpdate:
            func = partial(
                self._create_mesh,
                plotter,
                alpha=alpha_,
                clim=clim,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                style=style,
                show_outline=show_outline,
                show_origin=show_origin,
                cpos=cpos,
                show_max_min=show_max_min,
            )
        else:
            point_grid, line_grid, solid_grid, cbar, bc_grid, mp_grid, max_min_label_grid, line_interp_grid = (
                self._create_mesh(
                    plotter,
                    self.num_steps - 1,
                    alpha=alpha_,
                    clim=clim,
                    show_bc=show_bc,
                    bc_scale=bc_scale,
                    show_mp_constraint=show_mp_constraint,
                    style=style,
                    show_outline=show_outline,
                    show_origin=show_origin,
                    cpos=cpos,
                    show_max_min=show_max_min,
                )
            )
            func = partial(
                self._update_mesh,
                point_grid=point_grid,
                line_grid=line_grid,
                solid_grid=solid_grid,
                scalar_bar=cbar,
                bc_grid=bc_grid,
                mp_grid=mp_grid,
                alpha=alpha_,
                bc_scale=bc_scale,
                max_min_label_grid=max_min_label_grid,
                line_interp_grid=line_interp_grid,
                plotter=plotter,
                **kargs,
            )
        plotter.add_slider_widget(func, [0, self.num_steps - 1], value=self.num_steps - 1, **self.slider_widget_args)

    def plot_peak_step(
        self,
        plotter,
        step="absMax",
        alpha=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        style="surface",
        show_outline=False,
        show_origin=False,
        cpos="iso",
        show_max_min: bool = True,
    ):
        cmin, cmax, step = self._get_resp_clim_peak(idx=step)
        clim = (cmin, cmax)
        alpha_ = alpha if show_defo else 0.0
        self._create_mesh(
            plotter=plotter,
            value=step,
            alpha=alpha_,
            clim=clim,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_outline=show_outline,
            show_origin=show_origin,
            cpos=cpos,
            show_max_min=show_max_min,
        )

    def plot_anim(
        self,
        plotter,
        alpha=1.0,
        show_defo=True,
        framerate=None,
        savefig: str = "NodalRespAnimation.gif",
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        style="surface",
        show_outline=False,
        show_origin=False,
        cpos="iso",
        show_max_min: bool = True,
    ):
        if framerate is None:
            framerate = np.ceil(self.num_steps / 10)
        if savefig.endswith(".gif"):
            plotter.open_gif(savefig, fps=framerate)
        else:
            plotter.open_movie(savefig, framerate=framerate)
        cmin, cmax, max_step = self._get_resp_clim_peak()
        clim = (cmin, cmax)
        alpha_ = alpha if show_defo else 0.0
        # plotter.write_frame()  # write initial data
        if self.ModelUpdate:
            for step in range(self.num_steps):
                self._create_mesh(
                    plotter=plotter,
                    value=step,
                    alpha=alpha_,
                    clim=clim,
                    show_bc=show_bc,
                    bc_scale=bc_scale,
                    show_mp_constraint=show_mp_constraint,
                    style=style,
                    show_outline=show_outline,
                    show_origin=show_origin,
                    cpos=cpos,
                    show_max_min=show_max_min,
                )
                plotter.write_frame()
        else:
            point_grid, line_grid, solid_grid, scalar_bar, bc_grid, mp_grid, max_min_label_grid, line_interp_grid = (
                self._create_mesh(
                    plotter,
                    self.num_steps - 1,
                    alpha=alpha_,
                    show_bc=show_bc,
                    bc_scale=bc_scale,
                    show_mp_constraint=show_mp_constraint,
                    style=style,
                    show_outline=show_outline,
                    show_origin=show_origin,
                    cpos=cpos,
                    show_max_min=show_max_min,
                )
            )
            plotter.write_frame()
            for step in range(self.num_steps):
                self._update_mesh(
                    value=step,
                    point_grid=point_grid,
                    line_grid=line_grid,
                    solid_grid=solid_grid,
                    scalar_bar=scalar_bar,
                    bc_grid=bc_grid,
                    mp_grid=mp_grid,
                    alpha=alpha_,
                    bc_scale=bc_scale,
                    max_min_label_grid=max_min_label_grid,
                    line_interp_grid=line_interp_grid,
                    plotter=plotter,
                )
                plotter.write_frame()


def plot_nodal_responses(
    odb_tag: Union[int, str] = 1,
    slides: bool = False,
    step: Union[int, str] = "absMax",
    show_defo: bool = True,
    interpolate_beam_disp: bool = False,
    defo_scale: Union[float, int, bool] = 1.0,
    resp_type: str = "disp",
    resp_dof: Union[list, tuple, str] = ("UX", "UY", "UZ"),
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_max_min: bool = True,
    cpos: str = "iso",
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_undeformed: bool = False,
    style: str = "surface",
    show_outline: bool = False,
    lazy_load: bool = False,
) -> pv.Plotter:
    """Visualizing Node Responses.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    slides: bool, default: False
        Display the response for each step in the form of a slideshow.
        Otherwise, show the step with the following ``step`` parameter.
    step: Union[int, str], default: "absMax"
        If slides = False, this parameter will be used as the step to plot.
        If str, Optional: [absMax, absMin, Max, Min].
        If int, this step will be demonstrated (counting from 0).
    defo_scale: Union[float, int, bool] = 1.0
        Scales the size of the deformation presentation.
        If set to False, the deformed shape will not be scaled (original deformation).
        If set to True or "auto", the deformed shape will be scaled by the default scale (i.e., 1/20 of the maximum model dimensions).
        If set to a float or int, it will scale the deformed shape by that factor.
    show_defo: bool, default: True
        Whether to display the deformed shape.
    interpolate_beam_disp: bool, default: False, added since version 1.0.25.
        Whether to interpolate beam displacements.
        Shape functions will be used to interpolate the displacements of beam elements for a smoother visualization.
        If you have a large number of beam elements, enabling this option may slow down the plotting process, and it is recommended to disable it.
        If True, You need to ensure that the data has been saved in ``CreateODB`` with ``interpolate_beam_disp=True`` for this option to take effect.
    resp_type: str, default: disp
        Type of response to be visualized.
        Optional: "disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure".
    resp_dof: str, default: ("UX", "UY", "UZ")
        Component to be visualized.
        Optional: "UX", "UY", "UZ", "RX", "RY", "RZ".
        You can also pass on a list or tuple to display multiple dimensions, for example, ["UX", "UY"],
        ["UX", "UY", "UZ"], ["RX", "RY", "RZ"], ["RX", "RY"], ["RY", "RZ"], ["RX", "RZ"], and so on.

        .. Note::
            If the nodes include fluid pressure dof,
            such as those used for ...UP elements, the pore pressure should be extracted using ``resp_type="vel"``,
            and ``resp_dof="RZ"``.

    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
    unit_factor: float, default: None
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    show_max_min: bool, default: True
        Whether to show the maximum and minimum labels on the plot.
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_undeformed: bool, default: False
        Whether to show the undeformed shape of the model.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    lazy_load: bool, default: False, added since version 1.0.25.
        Whether to lazily load the response data.
        If True, the response data will be loaded on demand when needed for plotting.
        This can save memory when dealing with large datasets.
        If False, all response data will be loaded into memory at once.
        If you encounter memory issues, consider setting this parameter to True, elsewise, set it to False for plotting in safety.

    Returns
    -------
    Plotting object of PyVista to display vtk meshes or numpy arrays.
    See `pyvista.Plotter <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter>`_.

    You can use
    `Plotter.show <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.show#pyvista.Plotter.show>`_.
    to display the plotting window.

    You can also use
    `Plotter.export_html <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.export_html#pyvista.Plotter.export_html>`_.
    to export this plotter as an interactive scene to an HTML file.
    """
    plotter = pv.Plotter(
        notebook=PLOT_ARGS.notebook,
        line_smoothing=PLOT_ARGS.line_smoothing,
        polygon_smoothing=PLOT_ARGS.polygon_smoothing,
        off_screen=PLOT_ARGS.off_screen,
    )
    plotbase = PlotNodalResponse(odb_tag=odb_tag, lazy_load=lazy_load)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.set_comp_resp_type(resp_type=resp_type, component=resp_dof)
    plotbase.set_interp_beam_on(interpolate_beam_disp)
    if slides:
        plotbase.plot_slide(
            plotter,
            alpha=defo_scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_outline=show_outline,
            show_origin=show_undeformed,
            cpos=cpos,
            show_max_min=show_max_min,
        )
    else:
        plotbase.plot_peak_step(
            plotter,
            step=step,
            alpha=defo_scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_outline=show_outline,
            show_origin=show_undeformed,
            cpos=cpos,
            show_max_min=show_max_min,
        )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)
    return plotbase._update_plotter(plotter, cpos)


def plot_nodal_responses_animation(
    odb_tag: Union[int, str] = 1,
    framerate: Optional[int] = None,
    savefig: str = "NodalRespAnimation.gif",
    off_screen: bool = True,
    defo_scale: Union[float, int, bool] = 1.0,
    show_defo: bool = True,
    interpolate_beam_disp: bool = False,
    resp_type: str = "disp",
    resp_dof: Union[list, tuple, str] = ("UX", "UY", "UZ"),
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_bc: bool = False,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_max_min: bool = True,
    cpos: str = "iso",
    show_undeformed: bool = False,
    style: str = "surface",
    show_outline: bool = False,
    lazy_load: bool = False,
) -> pv.Plotter:
    """Visualize node response animation.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    framerate: int, default: 5
        Framerate for the display, i.e., the number of frames per second.
    savefig: str, default: NodalRespAnimation.gif
        Path to save the animation. The suffix can be ``.gif`` or ``.mp4``.
    off_screen: bool, default: True
        Whether to display the plotting window.
        If True, the plotting window will not be displayed.
    defo_scale: Union[float, int, bool] = 1.0
        Scales the size of the deformation presentation.
        If set to False, the deformed shape will not be scaled (original deformation).
        If set to True or "auto", the deformed shape will be scaled by the default scale (i.e., 1/20 of the maximum model dimensions).
        If set to a float or int, it will scale the deformed shape by that factor.
    interpolate_beam_disp: bool, default: False, added since version 1.0.25.
        Whether to interpolate beam displacements.
        Shape functions will be used to interpolate the displacements of beam elements for a smoother visualization.
        If you have a large number of beam elements, enabling this option may slow down the plotting process, and it is recommended to disable it.
        If True, You need to ensure that the data has been saved in ``CreateODB`` with ``interpolate_beam_disp=True`` for this option to take effect.
    show_defo: bool, default: True
        Whether to display the deformed shape.
    resp_type: str, default: disp
        Type of response to be visualized.
        Optional: "disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure".
    resp_dof: str, default: ("UX", "UY", "UZ")
        Component to be visualized.
        Optional: "UX", "UY", "UZ", "RX", "RY", "RZ".
        You can also pass on a list or tuple to display multiple dimensions, for example, ["UX", "UY"],
        ["UX", "UY", "UZ"], ["RX", "RY", "RZ"], ["RX", "RY"], ["RY", "RZ"], ["RX", "RZ"], and so on.
    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
    unit_factor: float, default: None
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_max_min: bool, default: True
        Whether to show the maximum and minimum labels on the plot.
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    show_undeformed: bool, default: False
        Whether to show the undeformed shape of the model.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    lazy_load: bool, default: False, added since version 1.0.25.
        Whether to lazily load the response data.
        If True, the response data will be loaded on demand when needed for plotting.
        This can save memory when dealing with large datasets.
        If False, all response data will be loaded into memory at once.
        If you encounter memory issues, consider setting this parameter to True, elsewise, set it to False for plotting in safety.

    Returns
    -------
    Plotting object of PyVista to display vtk meshes or numpy arrays.
    See `pyvista.Plotter <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter>`_.

    You can use
    `Plotter.show <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.show#pyvista.Plotter.show>`_.
    to display the plotting window.

    You can also use
    `Plotter.export_html <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.export_html#pyvista.Plotter.export_html>`_.
    to export this plotter as an interactive scene to an HTML file.
    """
    plotter = pv.Plotter(
        notebook=PLOT_ARGS.notebook,
        line_smoothing=PLOT_ARGS.line_smoothing,
        polygon_smoothing=PLOT_ARGS.polygon_smoothing,
        off_screen=off_screen,
    )
    plotbase = PlotNodalResponse(odb_tag=odb_tag, lazy_load=lazy_load)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.set_comp_resp_type(resp_type=resp_type, component=resp_dof)
    plotbase.set_interp_beam_on(interpolate_beam_disp)
    plotbase.plot_anim(
        plotter,
        alpha=defo_scale,
        show_defo=show_defo,
        framerate=framerate,
        savefig=savefig,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
        style=style,
        show_outline=show_outline,
        show_origin=show_undeformed,
        cpos=cpos,
        show_max_min=show_max_min,
    )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)

    print(f"Animation has been saved to {savefig}!")

    return plotbase._update_plotter(plotter, cpos)


def get_nodal_responses_dataset(
    odb_tag: Union[int, str] = 1,
    step: Union[int, str] = "absMax",
    resp_type: str = "disp",
    resp_dof: Union[list, tuple, str] = ("UX", "UY", "UZ"),
    defo_scale: Union[float, int, bool] = 1.0,
    lazy_load: bool = False,
) -> pv.UnstructuredGrid:
    """Get nodal responses dataset.
    Scalars are stored in the ``resp_type`` field of the dataset.

    Added in version 1.0.18+.

    Data Model in PyVista can be found at `PyVista Data Model <https://docs.pyvista.org/user-guide/data_model>`_.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    step: Union[int, str], default: "absMax"
        If slides = False, this parameter will be used as the step to plot.
        If str, Optional: [absMax, absMin, Max, Min].
        If int, this step will be demonstrated (counting from 0).
    resp_type: str, default: disp
        Type of response to be visualized.
        Optional: "disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure".
    resp_dof: str, default: ("UX", "UY", "UZ")
        Component to be visualized.
        Optional: "UX", "UY", "UZ", "RX", "RY", "RZ".
        You can also pass on a list or tuple to display multiple dimensions, for example, ["UX", "UY"],
        ["UX", "UY", "UZ"], ["RX", "RY", "RZ"], ["RX", "RY"], ["RY", "RZ"], ["RX", "RZ"], and so on.

        .. Note::
            If the nodes include fluid pressure dof,
            such as those used for ...UP elements, the pore pressure should be extracted using ``resp_type="vel"``,
            and ``resp_dof="RZ"``.
    defo_scale: Union[float, int, bool] = 1.0
        Scales the size of the deformation presentation.
        If set to False, the deformed shape will not be scaled (original deformation).
        If set to True or "auto", the deformed shape will be scaled by the default scale (i.e., 1/20 of the maximum model dimensions).
        If set to a float or int, it will scale the deformed shape by that factor.
    lazy_load: bool, default: False, added since version 1.0.25.
        Whether to lazily load the response data.
        If True, the response data will be loaded on demand when needed for plotting.
        This can save memory when dealing with large datasets.
        If False, all response data will be loaded into memory at once.
        If you encounter memory issues, consider setting this parameter to True, elsewise, set it to False for plotting in safety.

    Returns
    -------
    unstru_grid: `pyvista.UnstructuredGrid <https://docs.pyvista.org/api/core/_autosummary/pyvista.unstructuredgrid#pyvista.UnstructuredGrid>`_.
        Unstructured grid with unstructured cells and response scalars.
    """
    plotbase = PlotNodalResponse(odb_tag=odb_tag, lazy_load=lazy_load)
    plotbase.set_comp_resp_type(resp_type=resp_type, component=resp_dof)
    unstru_grid = plotbase.get_dataset(step, defo_scale=defo_scale)
    return unstru_grid
