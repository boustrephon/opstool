from typing import Optional, Union

import numpy as np
import plotly.graph_objs as go

from .._plot_unstru_resp_base import PlotUnstruResponseBase
from .plot_resp_base import PlotResponsePlotlyBase
from .plot_utils import _plot_points_cmap, _plot_unstru_cmap


class PlotUnstruResponse(PlotUnstruResponseBase, PlotResponsePlotlyBase):
    def __init__(self, odb_tag, lazy_load=True):
        super().__init__(odb_tag, lazy_load=lazy_load)
        self.ele_type = "Shell"
        self.fiber_point = None  # fiber point for shell fiber response
        self.title = None

    def _set_title(self):
        title = f"<b>{self.PKG_NAME}:: {self.ele_type.capitalize()} Responses 3D Viewer</b><br><br>"
        self.title = {"text": title, "font": {"size": self.pargs.title_font_size}}

    def _create_mesh(
        self,
        plotter,
        value,
        ele_tags=None,
        plot_all_mesh=True,
        clim=None,
        coloraxis="coloraxis",
        style="surface",
        show_values=False,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_max_min: bool = False,
    ):
        step = round(value)
        tags, pos, cells, cell_types = self._make_unstru_info(ele_tags, step)
        pos = np.array(self._get_defo_coord_da(step, defo_scale))
        resps = self.resp_step[step].to_numpy()
        scalars = resps
        #  ---------------------------------
        if plot_all_mesh:
            self._plot_all_mesh(plotter=plotter, step=step)
        if show_bc:
            self._plot_bc(plotter=plotter, step=step, defo_scale=defo_scale, bc_scale=bc_scale)
        if show_mp_constraint:
            self._plot_mp_constraint(plotter, step=step, defo_scale=defo_scale)
        # ---------------------------------------------------------------
        scalars_by_element = len(scalars) != len(pos)
        (
            face_points,
            face_line_points,
            face_mid_points,
            veci,
            vecj,
            veck,
            face_scalars,
            face_line_scalars,
        ) = self._get_plotly_unstru_data(pos, cell_types, cells, scalars, scalars_by_element=scalars_by_element)
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
        _plot_points_cmap(
            plotter,
            face_points,
            scalars=face_scalars,
            clim=clim,
            coloraxis=coloraxis,
            name=self.resp_type,
            size=self.pargs.point_size,
            show_hover=show_values,
        )
        if self.title is None:
            self._set_title()

        if show_max_min:
            idxs = [np.argmin(face_scalars), np.argmax(face_scalars)]
            ps = face_points[idxs]
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

    def _make_title(self, step, add_title=False):
        resp = self.resp_step[step].to_numpy()
        maxv, minv = np.nanmax(resp), np.nanmin(resp)
        t_ = self.time[step]

        if self.resp_type.lower() in ["stressmeasures", "stressmeasuresatnodes"]:
            resp_type = "Stress Measures"
        elif self.resp_type.lower() in ["strainmeasures", "strainmeasuresatnodes"]:
            resp_type = "Strain Measures"
        elif self.resp_type.lower() in ["sectionforces", "sectionforcesatnodes"]:
            resp_type = "Section Forces"
        elif self.resp_type.lower() in ["sectiondeformations", "sectiondeformationsatnodes"]:
            resp_type = "Section Deformations"
        elif self.resp_type.lower() in ["stresses", "stressesatnodes"]:
            resp_type = "Stresses"
        elif self.resp_type.lower() in ["strains", "strainsatnodes"]:
            resp_type = "Strains"
        else:
            resp_type = self.resp_type.capitalize()

        title = f"<b>{self._set_txt_props(resp_type)} *</b><br>"
        comp = self.component if isinstance(self.component, str) else " ".join(self.component)
        title += f"<b>(DOF) {self._set_txt_props(comp)}</b><br>"
        if self.unit_symbol:
            unit_txt = self._set_txt_props(self.unit_symbol)
            title += f"<b>(unit) {unit_txt}</b><br>"
        if self.fiber_point and "Sec" not in resp_type and self.ele_type.lower() == "shell":
            fiber_point = self._set_txt_props(self.fiber_point)
            title += f"<b>* (Fiber) {fiber_point}</b><br>"
        maxv = self._set_txt_props(f"{maxv:.3E}")
        minv = self._set_txt_props(f"{minv:.3E}")
        title += f"<b>Max.:</b> {maxv}<br><b>Min.:</b> {minv}"
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

    def plot_slide(
        self,
        ele_tags=None,
        style="surface",
        show_values=False,
        plot_all_mesh=False,
        show_defo=True,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_max_min: bool = False,
    ):
        _, clim = self._get_resp_peak()
        alpha_ = defo_scale if show_defo else 0.0
        ndatas = []
        ndata_cum = 0
        for i in range(self.num_steps):
            plotter = []
            self._create_mesh(
                plotter,
                i,
                ele_tags=ele_tags,
                clim=clim,
                coloraxis=f"coloraxis{i + 1}",
                show_values=show_values,
                plot_all_mesh=plot_all_mesh,
                style=style,
                defo_scale=alpha_,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                show_max_min=show_max_min,
            )
            self.FIGURE.add_traces(plotter)
            ndatas.append(len(self.FIGURE.data) - ndata_cum)
            ndata_cum = len(self.FIGURE.data)

        self._update_slider_layout(ndatas=ndatas, clim=clim)

    def plot_peak_step(
        self,
        ele_tags=None,
        step="absMax",
        style="surface",
        show_values=False,
        plot_all_mesh=False,
        show_defo=True,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_max_min: bool = False,
    ):
        max_step, clim = self._get_resp_peak(idx=step)
        alpha_ = defo_scale if show_defo else 0.0
        plotter = []
        self._create_mesh(
            plotter=plotter,
            value=max_step,
            ele_tags=ele_tags,
            clim=clim,
            coloraxis="coloraxis",
            show_values=show_values,
            plot_all_mesh=plot_all_mesh,
            style=style,
            defo_scale=alpha_,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_max_min=show_max_min,
        )
        self.FIGURE.add_traces(plotter)
        txt = self._make_title(max_step)
        self.FIGURE.update_layout(
            coloraxis={
                "colorscale": self.pargs.cmap,
                "cmin": clim[0],
                "cmax": clim[1],
                "colorbar": {"tickfont": {"size": self.pargs.font_size - 2}, "title": txt},
            },
        )

    def plot_anim(
        self,
        ele_tags=None,
        framerate: Optional[int] = None,
        style="surface",
        show_values=False,
        plot_all_mesh=False,
        show_defo=True,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_max_min: bool = False,
    ):
        if framerate is None:
            framerate = np.ceil(self.num_steps / 11)
        nb_frames = self.num_steps
        duration = 1000 / framerate
        # ---------------------------------------------
        _, clim = self._get_resp_peak()
        alpha_ = defo_scale if show_defo else 0.0
        # -----------------------------------------------------------------------------
        # start plot
        frames = []
        for i in range(nb_frames):
            plotter = []
            self._create_mesh(
                plotter=plotter,
                value=i,
                ele_tags=ele_tags,
                clim=clim,
                coloraxis="coloraxis",
                show_values=show_values,
                plot_all_mesh=plot_all_mesh,
                style=style,
                defo_scale=alpha_,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                show_max_min=show_max_min,
            )
            frames.append(go.Frame(data=plotter, name="step:" + str(i)))
        # Add data to be displayed before animation starts
        plotter0 = []
        self._create_mesh(
            plotter0,
            0,
            ele_tags=ele_tags,
            clim=clim,
            coloraxis="coloraxis",
            show_values=show_values,
            style=style,
            plot_all_mesh=plot_all_mesh,
            defo_scale=alpha_,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_max_min=show_max_min,
        )
        self.FIGURE = go.Figure(frames=frames, data=plotter0)
        # self.title = self._make_txt(0, add_title=True)

        self._update_antimate_layout(duration=duration, cbar_title=self.component)


def plot_unstruct_responses(
    odb_tag: Union[int, str] = 1,
    ele_type: str = "Shell",
    ele_tags: Optional[Union[int, list]] = None,
    slides: bool = False,
    step: Union[int, str] = "absMax",
    resp_type: str = "sectionForces",
    resp_dof: str = "MXX",
    shell_fiber_loc: Optional[Union[str, int]] = "top",
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    style: str = "surface",
    show_outline: bool = False,
    show_values: bool = False,
    show_defo: bool = False,
    defo_scale: Union[float, int, bool] = 1.0,
    show_bc: bool = False,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_model: bool = False,
    show_max_min: bool = False,
    lazy_load: bool = False,
) -> go.Figure:
    """Visualizing unstructured element (Shell, Plane, Brick) Response.

    .. Note::
        The responses at all Gaussian points are averaged.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of elements to be visualized.
        If None, all elements are selected.
    slides: bool, default: False
        Display the response for each step in the form of a slideshow.
        Otherwise, show the step with the following ``step`` parameter.
    step: Union[int, str], default: "absMax"
        If slides = False, this parameter will be used as the step to plot.
        If str, Optional: [absMax, absMin, Max, Min].
        If int, this step will be demonstrated (counting from 0).
    ele_type: str, default: "Shell"
        Element type, optional, one of ["Shell", "Plane", "Solid"].
    resp_type: str, default: None
        Response type, which dependents on the element type `ele_type`.

        #. For ``Shell`` elements, one of ["sectionForces", "sectionDeformations", "sectionForcesAtNodes", "sectionDeformationsAtNodes", "Stresses", "Strains", "StressesAtNodes", "StrainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element (per unit length).
            If None, defaults to "sectionForces".
        #. For ``Plane`` elements, one of ["stresses", "strains", "stressesAtNodes", "strainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element.
            If None, defaults to "stresses".
        #. For ``Brick`` or ``Solid`` elements, one of ["stresses", "strains", "stressesAtNodes", "strainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element.
            If None, defaults to "stresses".

    resp_dof: str, default: None
        Dof to be visualized, which dependents on the element type `ele_type`.

        .. note::
            The `resp_dof` here is consistent with stress-strain (force-deformation),
            and whether it is stress or strain depends on the parameter `resp_type`.

        .. important::
            The `resp_dof` parameter depends on the argument `nd_material_type` in :py:class:`opstool.post.CreateODB`.
            It can only be one of the returned responses.

        #. For ``Shell`` elements

            * If resp_type is the section responses, one of ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"].
            * If resp_type is the stress or strain, one of ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"].
            * If None, defaults to "MXX".

        #. For ``Plane`` elements, one of ["sigma11", "sigma22", "sigma12", "sigma33", "p1", "p2", "p3", "sigma_vm", "p_mean", "q_triaxial", "q_cs", "q_oct", "tau_max"].

            * "sigma11, sigma22, sigma12": Normal stress and shear stress (strain) in the x-y plane.
            * "sigma33": Out-of-plane normal stress (strain).
            * "p1, p2, p3": Principal stresses.
            * "sigma_vm": Von Mises stress.
            * "p_mean": Hydrostatic stress.
            * "q_triaxial": Deviatoric stress in triaxial test: q_tri = p1 - p3.
            * "q_cs": Deviatoric stress in critical state soil mechanics, q_cs = √(3J₂), where J2 = 1/6 * [ (p1-p2)^2 + (p2-p3)^2 + (p3-p1)^2 ].
            * "q_oct": Deviatoric stress in octahedral shear stress, τ_oct = √(2/3) * √(J2).
            * "tau_max": Maximum shear stress, 0.5*(p1-p3).
            * If None, defaults to "sigma11".

        #. For ``Brick`` or ``Solid`` elements, one of ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13", "p1", "p2", "p3", "sigma_vm", "p_mean", "q_triaxial", "q_cs", "q_oct", "tau_max"]

            * "sigma11, sigma22, sigma33": Normal stress (strain) along x, y, z.
            * "sigma12, sigma23, sigma13": Shear stress (strain).
            * "p1, p2, p3": Principal stresses.
            * "sigma_vm": Von Mises stress.
            * "p_mean": Hydrostatic stress.
            * "q_triaxial": Deviatoric stress in triaxial test: q_tri = p1 - p3.
            * "q_cs": Deviatoric stress in critical state soil mechanics, q_cs = √(3J₂), where J2 = 1/6 * [ (p1-p2)^2 + (p2-p3)^2 + (p3-p1)^2 ].
            * "q_oct": Deviatoric stress in octahedral shear stress, τ_oct = √(2/3) * √(J2).
            * "tau_max": Maximum shear stress, 0.5*(p1-p3).
            * If None, defaults to "sigma11".

    shell_fiber_loc: Optional[Union[str, int]], default: "top", added in v1.0.16
        The location of the fiber point for shell elements.
        If str, one of ["top", "bottom", "middle"].
        If int, the index of the fiber point to be visualized, from 1 (bottom) to N (top).
        The fiber point is the fiber layer in the shell section.
        Note that this parameter is only valid for stresses and strains in shell elements.

    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
        This feature is added since v1.0.15.
    unit_factor: float, default: None
        This feature is added since v1.0.15.
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface' or style='wireframe'
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    show_values: bool, default: False
        Whether to display the response value by hover.
        Set to False can improve the performance of the visualization.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    show_defo: bool, default: False
        Whether to display the deformed shape.
    defo_scale: float, default: 1.0
        Scales the size of the deformation presentation when show_defo is True.
        If set to False, the deformed shape will not be scaled (original deformation).
        If set to True or "auto", the deformed shape will be scaled by the default scale (i.e., 1/20 of the maximum model dimensions).
        If set to a float or int, it will scale the deformed shape by that factor.
    show_bc: bool, default: False
        Whether to display boundary supports.
        Set to False can improve the performance of the visualization.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
        Set to False can improve the performance of the visualization.
    show_model: bool, default: False
        Whether to plot the all model or not.
        Set to False can improve the performance of the visualization.
    show_max_min: bool, default: False
        Whether to show the maximum and minimum response values in the plot.
    lazy_load: bool, default: False
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
    plotbase = PlotUnstruResponse(odb_tag, lazy_load=lazy_load)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(
        ele_tags=ele_tags, ele_type=ele_type, resp_type=resp_type, component=resp_dof, fiber_point=shell_fiber_loc
    )
    if slides:
        plotbase.plot_slide(
            ele_tags=ele_tags,
            style=style,
            show_values=show_values,
            show_defo=show_defo,
            defo_scale=defo_scale,
            plot_all_mesh=show_model,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_max_min=show_max_min,
        )
    else:
        plotbase.plot_peak_step(
            ele_tags=ele_tags,
            step=step,
            style=style,
            show_values=show_values,
            show_defo=show_defo,
            defo_scale=defo_scale,
            plot_all_mesh=show_model,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_max_min=show_max_min,
        )
    return plotbase.update_fig(show_outline)


def plot_unstruct_responses_animation(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    framerate: Optional[int] = None,
    ele_type: str = "Shell",
    resp_type: Optional[str] = None,
    resp_dof: Optional[str] = None,
    shell_fiber_loc: Optional[Union[str, int]] = "top",
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    style: str = "surface",
    show_outline: bool = False,
    show_values: bool = False,
    show_defo: bool = True,
    defo_scale: Union[float, int, bool] = 1.0,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_model: bool = True,
    show_max_min: bool = False,
    lazy_load: bool = False,
) -> go.Figure:
    """Unstructured element (Shell, Plane, Brick) response animation.

    .. Note::
        The responses at all Gaussian points are averaged.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of truss elements to be visualized. If None, all truss elements are selected.
    framerate: int, default: None
        Framerate for the display, i.e., the number of frames per second.
    ele_type: str, default: "Shell"
        Element type, optional, one of ["Shell", "Plane", "Solid"].
    resp_type: str, default: None
        Response type, which dependents on the element type `ele_type`.

        #. For ``Shell`` elements, one of ["sectionForces", "sectionDeformations", "sectionForcesAtNodes", "sectionDeformationsAtNodes", "Stresses", "Strains", "StressesAtNodes", "StrainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element (per unit length).
            If None, defaults to "sectionForces".
        #. For ``Plane`` elements, one of ["stresses", "strains", "stressesAtNodes", "strainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element.
            If None, defaults to "stresses".
        #. For ``Brick`` or ``Solid`` elements, one of ["stresses", "strains", "stressesAtNodes", "strainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element.
            If None, defaults to "stresses".

    resp_dof: str, default: None
        Dof to be visualized, which dependents on the element type `ele_type`.

        .. note::
            The `resp_dof` here is consistent with stress-strain (force-deformation),
            and whether it is stress or strain depends on the parameter `resp_type`.

        .. important::
            The `resp_dof` parameter depends on the argument `nd_material_type` in :py:class:`opstool.post.CreateODB`.
            It can only be one of the returned responses.

        #. For ``Shell`` elements

            * If resp_type is the section responses, one of ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"].
            * If resp_type is the stress or strain, one of ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"].
            * If None, defaults to "MXX".

        #. For ``Plane`` elements, one of ["sigma11", "sigma22", "sigma12", "sigma33", "p1", "p2", "p3", "sigma_vm", "p_mean", "q_triaxial", "q_cs", "q_oct", "tau_max"].

            * "sigma11, sigma22, sigma12": Normal stress and shear stress (strain) in the x-y plane.
            * "sigma33": Out-of-plane normal stress (strain).
            * "p1, p2, p3": Principal stresses.
            * "sigma_vm": Von Mises stress.
            * "p_mean": Hydrostatic stress.
            * "q_triaxial": Deviatoric stress in triaxial test: q_tri = p1 - p3.
            * "q_cs": Deviatoric stress in critical state soil mechanics, q_cs = √(3J₂), where J2 = 1/6 * [ (p1-p2)^2 + (p2-p3)^2 + (p3-p1)^2 ].
            * "q_oct": Deviatoric stress in octahedral shear stress, τ_oct = √(2/3) * √(J2).
            * "tau_max": Maximum shear stress, 0.5*(p1-p3).
            * If None, defaults to "sigma11".

        #. For ``Brick`` or ``Solid`` elements, one of ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13", "p1", "p2", "p3", "sigma_vm", "p_mean", "q_triaxial", "q_cs", "q_oct", "tau_max"]

            * "sigma11, sigma22, sigma33": Normal stress (strain) along x, y, z.
            * "sigma12, sigma23, sigma13": Shear stress (strain).
            * "p1, p2, p3": Principal stresses.
            * "sigma_vm": Von Mises stress.
            * "p_mean": Hydrostatic stress.
            * "q_triaxial": Deviatoric stress in triaxial test: q_tri = p1 - p3.
            * "q_cs": Deviatoric stress in critical state soil mechanics, q_cs = √(3J₂), where J2 = 1/6 * [ (p1-p2)^2 + (p2-p3)^2 + (p3-p1)^2 ].
            * "q_oct": Deviatoric stress in octahedral shear stress, τ_oct = √(2/3) * √(J2).
            * "tau_max": Maximum shear stress, 0.5*(p1-p3).
            * If None, defaults to "sigma11".

    shell_fiber_loc: Optional[Union[str, int]], default: "top", added in v1.0.16
        The location of the fiber point for shell elements.
        If str, one of ["top", "bottom", "middle"].
        If int, the index of the fiber point to be visualized, from 1 (bottom) to N (top).
        The fiber point is the fiber layer in the shell section.
        Note that this parameter is only valid for stresses and strains in shell elements.

    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
        This feature is added since v1.0.15.
    unit_factor: float, default: None
        This feature is added since v1.0.15.
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface' or style='wireframe'
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    show_values: bool, default: False
        Whether to display the response value by hover.
        Set to False can improve the performance of the visualization.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    show_defo: bool, default: False
        Whether to display the deformed shape.
    defo_scale: float, default: 1.0
        Scales the size of the deformation presentation when show_defo is True.
        If set to False, the deformed shape will not be scaled (original deformation).
        If set to True or "auto", the deformed shape will be scaled by the default scale (i.e., 1/20 of the maximum model dimensions).
        If set to a float or int, it will scale the deformed shape by that factor.
    show_bc: bool, default: False
        Whether to display boundary supports.
        Set to False can improve the performance of the visualization.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
        Set to False can improve the performance of the visualization.
    show_model: bool, default: False
        Whether to plot the all model or not.
        Set to False can improve the performance of the visualization.
    show_max_min: bool, default: False
        Whether to show the maximum and minimum response values in the plot.
    lazy_load: bool, default: False
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
    plotbase = PlotUnstruResponse(odb_tag, lazy_load=lazy_load)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(
        ele_tags=ele_tags, ele_type=ele_type, resp_type=resp_type, component=resp_dof, fiber_point=shell_fiber_loc
    )
    plotbase.plot_anim(
        ele_tags=ele_tags,
        framerate=framerate,
        style=style,
        show_values=show_values,
        show_defo=show_defo,
        defo_scale=defo_scale,
        plot_all_mesh=show_model,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
        show_max_min=show_max_min,
    )
    return plotbase.update_fig(show_outline)
