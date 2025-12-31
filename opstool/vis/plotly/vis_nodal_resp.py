from typing import Optional, Union

import numpy as np
import plotly.graph_objs as go

from .._plot_nodal_resp_base import PlotNodalResponseBase
from .plot_resp_base import PlotResponsePlotlyBase
from .plot_utils import _plot_lines_cmap, _plot_points_cmap, _plot_unstru_cmap


class PlotNodalResponse(PlotNodalResponseBase, PlotResponsePlotlyBase):
    def __init__(self, odb_tag: Union[int, str], lazy_load=True):
        super().__init__(odb_tag, lazy_load=lazy_load)
        self.FIGURE = go.Figure()

        self.resps_norm = None
        self.comp_type = None

        title = f"<b>{self.PKG_NAME} :: Nodal Responses 3D Viewer</b><br><br>"
        self.title = {"text": title, "font": {"size": self.pargs.title_font_size}}

    def _make_title(self, step, add_title=False):
        max_norm, min_norm = np.nanmax(self._get_step_norm(step)), np.nanmin(self._get_step_norm(step))
        title = "Nodal Responses"
        if self.resp_type == "disp":
            resp_type = "Displacement"
        elif self.resp_type == "vel":
            resp_type = "Velocity"
        elif self.resp_type == "accel":
            resp_type = "Acceleration"
        else:
            resp_type = f"{self.resp_type.capitalize()}"
        comp = ",".join(self.component) if isinstance(self.component, (list, tuple)) else self.component
        size_symbol = ("norm.min", "norm.max") if isinstance(self.component, (list, tuple)) else ("min", "max")

        t_ = self.time[step]
        title = f"<b>{self._set_txt_props(resp_type)} *</b><br>"
        title += f"<b>(DOF) {self._set_txt_props(comp)} *</b><br>"
        if self.unit_symbol:
            unit_txt = self._set_txt_props(self.unit_symbol)
            title += f"<b>(unit) {unit_txt}</b><br>"
        max_norm = self._set_txt_props(f"{max_norm:.3E}")
        min_norm = self._set_txt_props(f"{min_norm:.3E}")
        title += f"<b>{size_symbol[1]}:</b> {max_norm}<br><b>{size_symbol[0]}:</b> {min_norm}"
        step_txt = self._set_txt_props(f"{step}")
        title += f"<br><b>step:</b> {step_txt}; "
        t_txt = self._set_txt_props(f"{t_:.3f}")
        title += f"<b>time</b>: {t_txt}<br> <br>"
        if add_title:
            title = self.title["text"] + title
        txt = {
            "font": {"size": self.pargs.font_size},
            "text": title,
        }
        return txt

    def _create_mesh(
        self,
        plotter,
        value,
        alpha=1.0,
        clim=None,
        style="surface",
        coloraxis=None,
        show_origin=False,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_hover: bool = True,
        show_max_min: bool = False,
    ):
        step = round(value)
        line_cells, _ = self._get_line_cells(self._get_line_da(step))
        _, unstru_cell_types, unstru_cells = self._get_unstru_cells(self._get_unstru_da(step))
        node_defo_coords = np.array(self._get_defo_coord_da(step, alpha))
        node_resp = np.array(self._get_resp_da(step, self.component))
        if self.resps_norm is not None:
            scalars = self._get_step_norm(step)
        else:
            scalars = node_resp if node_resp.ndim == 1 else np.linalg.norm(node_resp, axis=1)

        if show_bc:
            self._plot_bc(plotter=plotter, step=step, defo_scale=alpha, bc_scale=bc_scale)
        if show_mp_constraint:
            self._plot_mp_constraint(plotter, step=step, defo_scale=alpha)
        if show_origin:
            self._plot_all_mesh(plotter=plotter, step=step)

        if len(unstru_cells) > 0:
            (
                face_points,
                face_line_points,
                face_mid_points,
                veci,
                vecj,
                veck,
                face_scalars,
                face_line_scalars,
            ) = self._get_plotly_unstru_data(node_defo_coords, unstru_cell_types, unstru_cells, scalars)
            _plot_unstru_cmap(
                plotter,
                face_points,
                veci=veci,
                vecj=vecj,
                veck=veck,
                scalars=face_scalars,
                clim=clim,
                coloraxis=coloraxis,
                style=style,
                line_width=self.pargs.line_width,
                opacity=self.pargs.mesh_opacity,
                show_edges=self.pargs.show_mesh_edges,
                edge_color=self.pargs.mesh_edge_color,
                edge_width=self.pargs.mesh_edge_width,
                edge_points=face_line_points,
                edge_scalars=face_line_scalars,
            )
        if len(line_cells) > 0:
            line_points, line_mid_points, line_scalars = self._get_plotly_line_data(
                node_defo_coords, line_cells, scalars
            )
            _plot_lines_cmap(
                plotter,
                line_points,
                scalars=line_scalars,
                coloraxis=coloraxis,
                clim=clim,
                width=self.pargs.line_width,
            )
        if self.interp_beam_disp_on:
            points_origin_interp, points_defo_interp, cells_interp, scalars_interp = self.get_interp_beam_data(
                step, alpha
            )
            line_points_interp, _, line_scalars_interp = self._get_plotly_line_data(
                points_defo_interp, cells_interp, scalars_interp
            )
            _plot_lines_cmap(
                plotter,
                line_points_interp,
                scalars=line_scalars_interp,
                coloraxis=coloraxis,
                width=self.pargs.line_width,
            )
        _plot_points_cmap(
            plotter,
            node_defo_coords,
            scalars=scalars,
            clim=clim,
            coloraxis=coloraxis,
            name=self.resp_type,
            size=self.pargs.point_size,
            show_hover=show_hover,
        )

        if show_max_min:
            idxs = [np.argmin(scalars), np.argmax(scalars)]
            ps = node_defo_coords[idxs]
            labels = ["Min", "Max"]
            txt_plot = go.Scatter3d(
                x=ps[:, 0],
                y=ps[:, 1],
                z=ps[:, 2],
                text=labels,
                mode="markers+text",
                marker={"size": self.pargs.point_size + 2, "color": "#b790d4", "symbol": "x"},
                textfont={"color": "#5170d7", "size": self.pargs.font_size, "weight": "bold"},
                name="Max/Min Values",
            )
            plotter.append(txt_plot)

    def plot_slide(
        self,
        alpha=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        style="surface",
        show_origin=False,
        show_max_min: bool = False,
    ):
        cmin, cmax, _ = self._get_resp_clim_peak()
        clim = (cmin, cmax)
        alpha_ = alpha if show_defo else 0.0
        ndatas = []
        ndata_cum = 0
        for i in range(self.num_steps):
            plotter = []
            self._create_mesh(
                plotter,
                i,
                alpha=alpha_,
                clim=clim,
                coloraxis=f"coloraxis{i + 1}",
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                style=style,
                show_origin=show_origin,
                show_max_min=show_max_min,
            )
            self.FIGURE.add_traces(plotter)
            ndatas.append(len(self.FIGURE.data) - ndata_cum)
            ndata_cum = len(self.FIGURE.data)
        self._update_slider_layout(ndatas=ndatas, clim=clim)

    def plot_peak_step(
        self,
        step="absMax",
        alpha=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        style="surface",
        show_origin=False,
        show_max_min: bool = False,
    ):
        cmin, cmax, step = self._get_resp_clim_peak(idx=step)
        clim = (cmin, cmax)
        alpha_ = alpha if show_defo else 0.0
        plotter = []
        self._create_mesh(
            plotter=plotter,
            value=step,
            alpha=alpha_,
            clim=clim,
            coloraxis="coloraxis",
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_origin=show_origin,
            show_max_min=show_max_min,
        )
        self.FIGURE.add_traces(plotter)
        txt = self._make_title(step)
        self.FIGURE.update_layout(
            coloraxis={
                "colorscale": self.pargs.cmap,
                "cmin": cmin,
                "cmax": cmax,
                "colorbar": {"tickfont": {"size": self.pargs.font_size - 2}, "title": txt},
            },
            title=self.title,
        )

    def plot_anim(
        self,
        alpha=1.0,
        show_defo=True,
        framerate: Optional[int] = None,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        style="surface",
        show_origin=False,
        show_max_min: bool = False,
    ):
        if framerate is None:
            framerate = np.ceil(self.num_steps / 10)
        cmin, cmax, _ = self._get_resp_clim_peak()
        clim = (cmin, cmax)
        alpha_ = alpha if show_defo else 0.0
        nb_frames = self.num_steps
        duration = 1000 / framerate
        # -----------------------------------------------------------------------------
        # start plot
        frames = []
        for i in range(nb_frames):
            plotter = []
            self._create_mesh(
                plotter=plotter,
                value=i,
                alpha=alpha_,
                clim=clim,
                coloraxis="coloraxis",
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                style=style,
                show_origin=show_origin,
                show_hover=False,
                show_max_min=show_max_min,
            )
            frames.append(go.Frame(data=plotter, name="step:" + str(i)))
        # Add data to be displayed before animation starts
        plotter0 = []
        self._create_mesh(
            plotter0,
            0,
            alpha=alpha_,
            coloraxis="coloraxis",
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_origin=show_origin,
            show_hover=False,
            show_max_min=show_max_min,
        )
        self.FIGURE = go.Figure(frames=frames, data=plotter0)

        self._update_antimate_layout(duration=duration, cbar_title=self.resp_type.capitalize())


def plot_nodal_responses(
    odb_tag: Union[int, str] = 1,
    slides: bool = False,
    step: Union[int, str] = "absMax",
    defo_scale: Union[float, int, bool] = 1.0,
    show_defo: bool = True,
    interpolate_beam_disp: bool = False,
    resp_type: str = "disp",
    resp_dof: Union[list, tuple, str] = ("UX", "UY", "UZ"),
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_undeformed: bool = False,
    style: str = "surface",
    show_outline: bool = False,
    show_max_min: bool = False,
    lazy_load: bool = False,
) -> go.Figure:
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
    defo_scale: float, default: 1.0
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
        This feature is added since v1.0.15.
    unit_factor: float, default: None
        This feature is added since v1.0.15.
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_undeformed: bool, default: False
        Whether to show the undeformed shape of the model.
        Set to False can improve the performance of the visualization.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    show_max_min: bool, default: False
        Whether to show the maximum and minimum response value annotations in the plot.
    lazy_load: bool, default: False, added since version 1.0.25.
        Whether to lazily load the response data.
        If True, the response data will be loaded on demand when needed for plotting.
        This can save memory when dealing with large datasets.
        If False, all response data will be loaded into memory at once.
        If you encounter memory issues, consider setting this parameter to True, elsewise, set it to False for plotting in safety.

    Returns
    -------
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
    """
    plotbase = PlotNodalResponse(odb_tag, lazy_load=lazy_load)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.set_comp_resp_type(resp_type=resp_type, component=resp_dof)
    plotbase.set_interp_beam_on(interpolate_beam_disp)
    if slides:
        plotbase.plot_slide(
            alpha=defo_scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_origin=show_undeformed,
            show_max_min=show_max_min,
        )
    else:
        plotbase.plot_peak_step(
            step=step,
            alpha=defo_scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_origin=show_undeformed,
            show_max_min=show_max_min,
        )
    return plotbase.update_fig(show_outline=show_outline)


def plot_nodal_responses_animation(
    odb_tag: Union[int, str] = 1,
    framerate: Optional[int] = None,
    defo_scale: Union[float, int, bool] = 1.0,
    show_defo: bool = True,
    interpolate_beam_disp: bool = False,
    resp_type: str = "disp",
    resp_dof: Union[list, tuple, str] = ("UX", "UY", "UZ"),
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_undeformed: bool = False,
    style: str = "surface",
    show_outline: bool = False,
    show_max_min: bool = False,
    lazy_load: bool = False,
) -> go.Figure:
    """Visualize node response animation.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    framerate: int, default: 5
        Framerate for the display, i.e., the number of frames per second.
        For example, if an earthquake analysis has 1000 steps and you want to complete the demonstration in ten seconds, you should set ``framerate = 1000/10 = 100``.
    defo_scale: float, default: 1.0
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
    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
        This feature is added since v1.0.15.
    unit_factor: float, default: None
        This feature is added since v1.0.15.
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_undeformed: bool, default: False
        Whether to show the undeformed shape of the model.
        Set to False can improve the performance of the visualization.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface' or style='wireframe'
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    show_max_min: bool, default: False
        Whether to show the maximum and minimum response value annotations in the plot.
    lazy_load: bool, default: False, added since version 1.0.25.
        Whether to lazily load the response data.
        If True, the response data will be loaded on demand when needed for plotting.
        This can save memory when dealing with large datasets.
        If False, all response data will be loaded into memory at once.
        If you encounter memory issues, consider setting this parameter to True, elsewise, set it to False for plotting in safety.

    Returns
    -------
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
    """
    plotbase = PlotNodalResponse(odb_tag, lazy_load=lazy_load)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.set_comp_resp_type(resp_type=resp_type, component=resp_dof)
    plotbase.set_interp_beam_on(interpolate_beam_disp)
    plotbase.plot_anim(
        alpha=defo_scale,
        show_defo=show_defo,
        framerate=framerate,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
        style=style,
        show_origin=show_undeformed,
        show_max_min=show_max_min,
    )
    return plotbase.update_fig(show_outline=show_outline)
