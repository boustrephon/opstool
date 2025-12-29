from functools import partial
from typing import Optional, Union

import numpy as np
import pyvista as pv

from .._plot_unstru_resp_base import PlotUnstruResponseBase
from .plot_resp_base import PlotResponsePyvistaBase
from .plot_utils import PLOT_ARGS, _plot_unstru_cmap, _update_point_label_actor


class PlotUnstruResponse(PlotUnstruResponseBase, PlotResponsePyvistaBase):
    def __init__(self, odb_tag: Union[int, str], lazy_load: bool = False):
        super().__init__(odb_tag, lazy_load=lazy_load)

    def _make_title(self, scalars, step, time):
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
        info = {
            "title": self.ele_type.capitalize(),
            "resp_type": resp_type,
            "dof": self.component.capitalize(),
            "min": np.nanmin(scalars),
            "max": np.nanmax(scalars),
            "step": step,
            "time": time,
        }
        lines = [
            f"{info['step']} (step)",
            f"{info['time']:.3f} (time)",
            f"{info['min']:.3E} (min)",
            f"{info['max']:.3E} (max)",
            "",
            f"{info['title']} Responses",
            f"{info['resp_type']}",
            f"{info['dof']} (DOF)",
        ]
        if self.fiber_point and "Sec" not in resp_type and self.ele_type.lower() == "shell":
            info["fiber_point"] = self.fiber_point
            lines.append(f"{info['fiber_point']} (Fiber)")
        if self.unit_symbol:
            info["unit"] = self.unit_symbol
            lines.append(f"{info['unit']} (unit)")

        max_len = max(len(line) for line in lines)
        padded_lines = [line.rjust(max_len) for line in lines]
        text = "\n".join(padded_lines)
        return text + "\n"

    def _get_mesh_data(self, step, ele_tags, defo_scale):
        pos_defo = np.array(self._get_defo_coord_da(step, defo_scale))
        tags, pos, cells, cell_types = self._make_unstru_info(ele_tags, step)
        scalars = self.resp_step[step].to_numpy()
        return pos_defo, cells, cell_types, scalars

    def get_dataset(self, step, ele_tags=None, defo_scale=1.0):
        """Get the dataset for the specified step and element tags."""
        step, clim = self._get_resp_peak(idx=step)
        pos, cells, cell_types, scalars = self._get_mesh_data(step, ele_tags, defo_scale=defo_scale)
        grid = pv.UnstructuredGrid(cells, cell_types, pos)
        grid[self.resp_type] = scalars
        return grid

    def _create_mesh(
        self,
        plotter,
        value,
        ele_tags=None,
        plot_all_mesh=True,
        clim=None,
        style="surface",
        cpos="iso",
        defo_scale=1.0,
        show_outline=False,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = False,
        show_max_min: bool = False,
    ):
        step = round(value)
        node_no_deform_coords = np.array(self._get_node_da(step))
        pos, cells, cell_types, scalars = self._get_mesh_data(step, ele_tags, defo_scale)
        #  ---------------------------------
        plotter.clear_actors()  # !!!!!!
        if plot_all_mesh:
            self._plot_all_mesh(plotter, color="gray", step=step)
        resp_plot = _plot_unstru_cmap(
            plotter,
            pos=pos,
            cells=cells,
            cell_types=cell_types,
            scalars=scalars,
            cmap=self.pargs.cmap,
            clim=clim,
            show_scalar_bar=False,
            show_edges=self.pargs.show_mesh_edges,
            edge_color=self.pargs.mesh_edge_color,
            edge_width=self.pargs.mesh_edge_width,
            opacity=self.pargs.mesh_opacity,
            style=style,
            pos_origin=node_no_deform_coords,
        )

        title = self._make_title(scalars, step, self.time[step])
        scalar_bar = plotter.add_scalar_bar(title=title, **self.pargs.scalar_bar_kargs)
        if scalar_bar:
            # scalar_bar.SetTitle(title)
            title_prop = scalar_bar.GetTitleTextProperty()
            # title_prop.SetJustificationToRight()
            title_prop.BoldOn()

        # Max Min Labels
        if show_max_min:
            idxs = [np.argmax(scalars), np.argmin(scalars)]
            labels = [f"Max: {scalars[idxs[0]]: .3e}", f"Min: {scalars[idxs[1]]: .3e}"]
            if len(scalars) == len(pos):
                ps = pos[idxs]
            elif len(scalars) == len(cells):
                cell_max = cells[idxs[0], 1:]
                cell_min = cells[idxs[1], 1:]
                ps_max = np.mean(pos[cell_max, :], axis=0)
                ps_min = np.mean(pos[cell_min, :], axis=0)
                ps = np.vstack([ps_max, ps_min])
            else:
                ps = None
            if ps is not None:
                max_min_label_grid = plotter.add_point_labels(
                    ps,
                    labels,
                    point_size=self.pargs.point_size + 10,
                    font_size=self.pargs.font_size + 5,
                    shape_color="#c0fb2d",
                    always_visible=True,
                )
            else:
                max_min_label_grid = None
        else:
            max_min_label_grid = None

        if show_outline:
            self._plot_outline(plotter)
        bc_grid, mp_grid = None, None
        if show_bc:
            bc_grid = self._plot_bc(plotter, step, defo_scale=defo_scale, bc_scale=bc_scale)
        if show_mp_constraint:
            mp_grid = self._plot_mp_constraint(plotter, step, defo_scale=defo_scale)

        self._update_plotter(plotter, cpos)
        return resp_plot, scalar_bar, bc_grid, mp_grid, max_min_label_grid

    def _update_mesh(
        self,
        step,
        ele_tags,
        resp_plot,
        scalar_bar,
        bc_grid=None,
        mp_grid=None,
        max_min_label_grid=None,
        defo_scale=1.0,
        bc_scale: float = 1.0,
        plotter=None,
    ):
        step = round(step)
        pos, cells, cell_types, scalars = self._get_mesh_data(step, ele_tags, defo_scale)

        if resp_plot:
            resp_plot.points = pos
            resp_plot["scalars"] = scalars

        if scalar_bar:
            title = self._make_title(scalars, step, self.time[step])
            scalar_bar.SetTitle(title)

        if mp_grid:
            self._plot_mp_constraint_update(mp_grid, step, defo_scale=defo_scale)
        if bc_grid:
            self._plot_bc_update(bc_grid, step, defo_scale=defo_scale, bc_scale=bc_scale)

        if max_min_label_grid:
            idxs = [np.argmax(scalars), np.argmin(scalars)]
            labels = [f"Max: {scalars[idxs[0]]: .3e}", f"Min: {scalars[idxs[1]]: .3e}"]
            text_property = pv.TextProperty(
                bold=True,
                font_size=self.pargs.font_size + 10,
            )
            _update_point_label_actor(
                max_min_label_grid,
                pos[idxs],
                labels,
                text_property=text_property,
                renderer=plotter.renderer,
                shape_opacity=100.0,
                always_visible=True,
            )

    def plot_slide(
        self,
        plotter,
        ele_tags=None,
        style="surface",
        plot_model=True,
        cpos="iso",
        show_defo=True,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_outline=False,
        show_max_min: bool = True,
    ):
        _, clim = self._get_resp_peak()
        alpha_ = defo_scale if show_defo else 0.0
        if self.ModelUpdate:
            func = partial(
                self._create_mesh,
                plotter,
                ele_tags=ele_tags,
                clim=clim,
                plot_all_mesh=plot_model,
                style=style,
                cpos=cpos,
                defo_scale=alpha_,
                show_outline=show_outline,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                show_max_min=show_max_min,
            )
        else:
            resp_plot, scalar_bar, bc_grid, mp_grid, max_min_label_grid = self._create_mesh(
                plotter,
                self.num_steps - 1,
                ele_tags=ele_tags,
                clim=clim,
                plot_all_mesh=plot_model,
                style=style,
                cpos=cpos,
                defo_scale=alpha_,
                show_outline=show_outline,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                show_max_min=show_max_min,
            )
            func = partial(
                self._update_mesh,
                ele_tags=ele_tags,
                resp_plot=resp_plot,
                scalar_bar=scalar_bar,
                bc_grid=bc_grid,
                mp_grid=mp_grid,
                bc_scale=bc_scale,
                defo_scale=alpha_,
                max_min_label_grid=max_min_label_grid,
                plotter=plotter,
            )
        plotter.add_slider_widget(func, [0, self.num_steps - 1], value=self.num_steps - 1, **self.slider_widget_args)

    def plot_peak_step(
        self,
        plotter,
        step="absMax",
        ele_tags=None,
        style="surface",
        plot_model=True,
        cpos="iso",
        defo_scale=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_outline=False,
        show_max_min: bool = True,
    ):
        step, clim = self._get_resp_peak(idx=step)
        alpha_ = defo_scale if show_defo else 0.0
        self._create_mesh(
            plotter=plotter,
            value=step,
            ele_tags=ele_tags,
            clim=clim,
            plot_all_mesh=plot_model,
            style=style,
            cpos=cpos,
            defo_scale=alpha_,
            show_outline=show_outline,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_max_min=show_max_min,
        )

    def plot_anim(
        self,
        plotter,
        ele_tags=None,
        framerate: Optional[int] = None,
        savefig: str = "ShellRespAnimation.gif",
        style="surface",
        plot_model=True,
        cpos="iso",
        defo_scale=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_outline=False,
        show_max_min: bool = True,
    ):
        if framerate is None:
            framerate = np.ceil(self.num_steps / 11)
        if savefig.endswith(".gif"):
            plotter.open_gif(savefig, fps=framerate)
        else:
            plotter.open_movie(savefig, framerate=framerate)
        _, clim = self._get_resp_peak()
        alpha_ = defo_scale if show_defo else 0.0
        # plotter.write_frame()  # write initial data

        if self.ModelUpdate:
            for step in range(self.num_steps):
                self._create_mesh(
                    plotter,
                    step,
                    ele_tags=ele_tags,
                    clim=clim,
                    plot_all_mesh=plot_model,
                    style=style,
                    cpos=cpos,
                    defo_scale=alpha_,
                    show_outline=show_outline,
                    show_bc=show_bc,
                    bc_scale=bc_scale,
                    show_mp_constraint=show_mp_constraint,
                    show_max_min=show_max_min,
                )
                plotter.write_frame()
        else:
            resp_plot, scalar_bar, bc_grid, mp_grid, max_min_label_grid = self._create_mesh(
                plotter,
                0,
                ele_tags=ele_tags,
                clim=clim,
                plot_all_mesh=plot_model,
                style=style,
                cpos=cpos,
                defo_scale=alpha_,
                show_outline=show_outline,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                show_max_min=show_max_min,
            )
            plotter.write_frame()
            for step in range(1, self.num_steps):
                self._update_mesh(
                    step=step,
                    ele_tags=ele_tags,
                    resp_plot=resp_plot,
                    scalar_bar=scalar_bar,
                    bc_grid=bc_grid,
                    mp_grid=mp_grid,
                    bc_scale=bc_scale,
                    defo_scale=alpha_,
                    max_min_label_grid=max_min_label_grid,
                    plotter=plotter,
                )
                plotter.write_frame()


def plot_unstruct_responses(
    odb_tag: Union[int, str] = 1,
    ele_type: str = "Shell",
    ele_tags: Optional[Union[int, list]] = None,
    slides: bool = False,
    step: Union[int, str] = "absMax",
    resp_type: str = "sectionForces",
    resp_dof: str = "MXX",
    shell_fiber_loc: Optional[Union[str, int]] = "top",
    style: str = "surface",
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_defo: bool = True,
    defo_scale: Union[float, int, bool] = 1.0,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_outline: bool = False,
    cpos: str = "iso",
    show_model: bool = True,
    show_max_min: bool = True,
    lazy_load: bool = False,
) -> pv.Plotter:
    """Visualizing unstructured element (Shell, Plane, Brick) Response.

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
    unit_factor: float, default: None
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    style: str, default: surface
        Visualization the mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    show_defo: bool, default: True
        Whether to display the deformed shape.
    defo_scale: float, default: 1.0
        Scales the size of the deformation presentation when show_defo is True.
        If set to False, the deformed shape will not be scaled (original deformation).
        If set to True or "auto", the deformed shape will be scaled by the default scale (i.e., 1/20 of the maximum model dimensions).
        If set to a float or int, it will scale the deformed shape by that factor.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    show_model: bool, default: True
        Whether to plot the all model or not.
    show_max_min: bool, default: True
        Whether to show the maximum and minimum response value labels.
    lazy_load: bool, default: False
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
    plotbase = PlotUnstruResponse(odb_tag, lazy_load=lazy_load)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(
        ele_tags=ele_tags, ele_type=ele_type, resp_type=resp_type, component=resp_dof, fiber_point=shell_fiber_loc
    )
    if slides:
        plotbase.plot_slide(
            plotter,
            ele_tags=ele_tags,
            style=style,
            cpos=cpos,
            plot_model=show_model,
            defo_scale=defo_scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_outline=show_outline,
            show_max_min=show_max_min,
        )
    else:
        plotbase.plot_peak_step(
            plotter,
            ele_tags=ele_tags,
            step=step,
            style=style,
            cpos=cpos,
            plot_model=show_model,
            defo_scale=defo_scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_outline=show_outline,
            show_max_min=show_max_min,
        )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)
    return plotbase._update_plotter(plotter, cpos)


def plot_unstruct_responses_animation(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    framerate: Optional[int] = None,
    ele_type: str = "Shell",
    resp_type: Optional[str] = None,
    resp_dof: Optional[str] = None,
    shell_fiber_loc: Optional[Union[str, int]] = "top",
    savefig: Optional[str] = None,
    off_screen: bool = True,
    style: str = "surface",
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_defo: bool = True,
    defo_scale: Union[float, int, bool] = 1.0,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_outline: bool = False,
    cpos: str = "iso",
    show_model: bool = True,
    show_max_min: bool = True,
    lazy_load: bool = False,
) -> pv.Plotter:
    """Unstructured element (Shell, Plane, Brick) response animation.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of truss elements to be visualized. If None, all truss elements are selected.
    ele_type: str, default: "Shell"
        Element type, optional, one of ["Shell", "Plane", "Solid"].
    framerate: int, default: None
        Framerate for the display, i.e., the number of frames per second.
    savefig: str, default: None
        Path to save the animation. The suffix can be ``.gif`` or ``.mp4``.
    off_screen: bool, default: True
        Off-screen rendering, i.e., not showing the rendering window.
        If False, the rendering window will be displayed.
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
    unit_factor: float, default: None
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    style: str, default: surface
        Visualization the mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    show_defo: bool, default: True
        Whether to display the deformed shape.
    defo_scale: float, default: 1.0
        Scales the size of the deformation presentation.
        If set to False, the deformed shape will not be scaled (original deformation).
        If set to True or "auto", the deformed shape will be scaled by the default scale (i.e., 1/20 of the maximum model dimensions).
        If set to a float or int, it will scale the deformed shape by that factor.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    show_model: bool, default: True
        Whether to plot the all model or not.
    show_max_min: bool, default: True
        Whether to show the maximum and minimum response value labels.
    lazy_load: bool, default: False
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
    if savefig is None:
        savefig = f"{ele_type.capitalize()}RespAnimation.gif"
    plotter = pv.Plotter(
        notebook=PLOT_ARGS.notebook,
        line_smoothing=PLOT_ARGS.line_smoothing,
        polygon_smoothing=PLOT_ARGS.polygon_smoothing,
        off_screen=off_screen,
    )
    plotbase = PlotUnstruResponse(odb_tag, lazy_load=lazy_load)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(
        ele_tags=ele_tags, ele_type=ele_type, resp_type=resp_type, component=resp_dof, fiber_point=shell_fiber_loc
    )
    plotbase.plot_anim(
        plotter,
        ele_tags=ele_tags,
        framerate=framerate,
        savefig=savefig,
        style=style,
        cpos=cpos,
        plot_model=show_model,
        defo_scale=defo_scale,
        show_defo=show_defo,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
        show_outline=show_outline,
        show_max_min=show_max_min,
    )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)
    print(f"Animation has been saved as {savefig}!")
    return plotbase._update_plotter(plotter, cpos)


def get_unstruct_responses_dataset(
    odb_tag: Union[int, str] = 1,
    ele_type: str = "Shell",
    ele_tags: Optional[Union[int, list]] = None,
    step: Union[int, str] = "absMax",
    resp_type: str = "sectionForces",
    resp_dof: str = "MXX",
    shell_fiber_loc: Optional[Union[str, int]] = "top",
    defo_scale: Union[float, int, bool] = 0.0,
    lazy_load: bool = False,
) -> pv.UnstructuredGrid:
    """Get unstructured element (Shell, Plane, Brick) Dataset (Pyvista).
    Data Model in PyVista can be found at `PyVista Data Model <https://docs.pyvista.org/user-guide/data_model>`_.

    Added in version 1.0.18+.

    .. note::
        This function is used to get the dataset for unstructured element responses.
        It is useful for further processing or visualization in PyVista.
        For more details, please refer to the
        `Pyvista Examples <https://docs.pyvista.org/examples/>`_.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of elements to be visualized.
        If None, all elements are selected.
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

        .. Note::
            The `resp_dof` here is consistent with stress-strain (force-deformation),
            and whether it is stress or strain depends on the parameter `resp_type`.

        #. For ``Shell`` elements, If resp_type is the section responses, one of ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"]. If resp_type is the stress or strain, one of ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"].
            If None, defaults to "MXX".
        #. For ``Plane`` elements, one of ["sigma11", "sigma22", "sigma12", "p1", "p2", "sigma_vm", "tau_max"].

            * "sigma11, sigma22, sigma12": Normal stress and shear stress (strain) in the x-y plane.
            * "p1, p2": Principal stresses (strains).
            * "sigma_vm": Von Mises stress.
            * "tau_max": Maximum shear stress (strains).
            * If None, defaults to "sigma_vm".

        #. For ``Brick`` or ``Solid`` elements, one of ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13", "p1", "p2", "p3", "sigma_vm", "tau_max", "sigma_oct", "tau_oct"]

            * "sigma11, sigma22, sigma33": Normal stress (strain) along x, y, z.
            * "sigma12, sigma23, sigma13": Shear stress (strain).
            * "p1, p2, p3": Principal stresses (strains).
            * "sigma_vm": Von Mises stress.
            * "tau_max": Maximum shear stress (strains).
            * "sigma_oct": Octahedral normal stress (strains).
            * "tau_oct": Octahedral shear stress (strains).
            * If None, defaults to "sigma_vm".

    shell_fiber_loc: Optional[Union[str, int]], default: "top", added in v1.0.16
        The location of the fiber point for shell elements.
        If str, one of ["top", "bottom", "middle"].
        If int, the index of the fiber point to be visualized, from 1 (bottom) to N (top).
        The fiber point is the fiber layer in the shell section.
        Note that this parameter is only valid for stresses and strains in shell elements.
    defo_scale: float, default: 0.0
        Scales the size of the deformation presentation.
        If set to False or 0.0, the deformed shape will not be scaled (original deformation).
        If set to True or "auto", the deformed shape will be scaled by the default scale (i.e., 1/20 of the maximum model dimensions).
        If set to a float or int, it will scale the deformed shape by that factor.
    lazy_load: bool, default: False
        Whether to lazily load the response data.
        If True, the response data will be loaded on demand when needed for plotting.
        This can save memory when dealing with large datasets.
        If False, all response data will be loaded into memory at once.
        If you encounter memory issues, consider setting this parameter to True, elsewise, set it to False for plotting in safety.

    Returns
    -------
    dataset: `pyvista.UnstructuredGrid <https://docs.pyvista.org/api/core/_autosummary/pyvista.unstructuredgrid#pyvista.UnstructuredGrid>`_.
        PyVista UnstructuredGrid dataset containing the response data for the specified elements and step.
        Scalars are stored in the ``resp_type`` field of the dataset.

        Data Model in PyVista can be found at `PyVista Data Model <https://docs.pyvista.org/user-guide/data_model>`_.
    """
    plotbase = PlotUnstruResponse(odb_tag, lazy_load=lazy_load)
    plotbase.refactor_resp_step(
        ele_tags=ele_tags, ele_type=ele_type, resp_type=resp_type, component=resp_dof, fiber_point=shell_fiber_loc
    )
    return plotbase.get_dataset(step, ele_tags, defo_scale=defo_scale)
