import numpy as np
import plotly.graph_objs as go

from .._plot_resp_base import PlotResponseBase
from .plot_utils import (
    PLOT_ARGS,
    _make_lines_plotly,
    _plot_all_mesh,
    _plot_lines,
    _VTKElementTriangulator,
)


class PlotResponsePlotlyBase(PlotResponseBase):
    def __init__(self, odb_tag, lazy_load: bool = True):
        super().__init__(odb_tag, lazy_load=lazy_load)

        # ----------------------------------------
        self.pargs = PLOT_ARGS
        self.FIGURE = go.Figure()
        self.title = ""

    @staticmethod
    def _set_txt_props(txt, color="blue", weight="bold"):
        return f'<span style="color:{color}; font-weight:{weight}">{txt}</span>'

    @staticmethod
    def _get_plotly_unstru_data(points, unstru_cell_types, unstru_cells, scalars=None, scalars_by_element=False):
        grid = _VTKElementTriangulator(points, scalars=scalars, scalars_by_element=scalars_by_element)
        for cell_type, cell in zip(unstru_cell_types, unstru_cells):
            grid.add_cell(cell_type, cell)
        return grid.get_results()

    @staticmethod
    def _get_plotly_line_data(points, line_cells, scalars=None):
        return _make_lines_plotly(points, line_cells, scalars=scalars)

    def _plot_bc(self, plotter, step: int, defo_scale: float, bc_scale: float):
        fixed_node_data = self._get_bc_da(step)
        if len(fixed_node_data) > 0:
            fix_tags = fixed_node_data["nodeTags"].values
            fixed_data = fixed_node_data.to_numpy()
            fixed_dofs = fixed_data[:, -6:].astype(int)
            if defo_scale == 0.0:
                node_deform_coords_da = self._get_node_da(step)
            else:
                node_deform_coords_da = self._get_defo_coord_da(step, defo_scale)
            fixed_coords = node_deform_coords_da.sel({"nodeTags": fix_tags}).to_numpy()
            s = (self.min_bound_size + self.max_bound_size) / 75 * bc_scale
            _plot_bc(
                plotter,
                fixed_dofs,
                fixed_coords,
                s,
                show_zaxis=self.show_zaxis,
                color=self.pargs.color_bc,
            )

    def _plot_mp_constraint(self, plotter, step: int, defo_scale):
        mp_constraint_data = self._get_mp_constraint_da(step)
        if len(mp_constraint_data) > 0:
            if defo_scale == 0.0:
                node_deform_coords = self._get_node_da(step).to_numpy()
            else:
                node_deform_coords = np.array(self._get_defo_coord_da(step, defo_scale))
            cells = mp_constraint_data.to_numpy()[:, :3].astype(int)
            _plot_mp_constraint(
                plotter,
                node_deform_coords,
                cells,
                None,
                self.pargs.line_width / 2,
                self.pargs.color_constraint,
                show_dofs=False,
            )

    def _plot_all_mesh(self, plotter, color="#738595", step=0):
        pos = self._get_node_da(step).to_numpy()
        line_cells, _ = self._get_line_cells(self._get_line_da(step))
        _, unstru_cell_types, unstru_cells = self._get_unstru_cells(self._get_unstru_da(step))
        output = self._get_plotly_unstru_data(pos, unstru_cell_types, unstru_cells, scalars=None)
        face_line_points = output[1]
        line_points, _ = self._get_plotly_line_data(pos, line_cells, scalars=None)
        _plot_all_mesh(plotter, line_points, face_line_points, color=color, width=1.5)

    def _update_slider_layout(self, ndatas: list, clim, showscale=True):
        for i in range(0, len(self.FIGURE.data) - ndatas[-1]):
            self.FIGURE.data[i].visible = False
        # Create and add slider
        steps = []
        idx_cum = 0
        for i in range(self.num_steps):
            txt = self._make_title(i) if not showscale else {"text": ""}
            txt["text"] = self.title["text"] + txt["text"]
            step = {
                "method": "update",
                "args": [{"visible": [False] * len(self.FIGURE.data)}, {"title": txt}],  # layout attribute
                "label": str(i),
            }
            step["args"][0]["visible"][idx_cum : idx_cum + ndatas[i]] = [True] * ndatas[i]
            # Toggle i'th trace to "visible"
            steps.append(step)
            idx_cum += ndatas[i]
        sliders = [
            {
                "active": self.num_steps,
                "currentvalue": {"prefix": "Step: "},
                "pad": {"t": 50},
                "steps": steps,
            }
        ]
        coloraxiss = {}
        for i in range(self.num_steps):
            txt = self._make_title(i)
            coloraxiss[f"coloraxis{i + 1}"] = {
                "colorscale": self.pargs.cmap,
                "cmin": clim[0],
                "cmax": clim[1],
                "showscale": showscale,
                "colorbar": {"tickfont": {"size": self.pargs.font_size - 2}, "title": txt},
            }
        self.FIGURE.update_layout(sliders=sliders, **coloraxiss)

    def _update_antimate_layout(self, duration, cbar_title="", is_response_step=True, showscale=True):
        if is_response_step:
            # Layout
            for i in range(len(self.FIGURE.frames)):
                txt = self._make_title(i, add_title=True)
                self.FIGURE.frames[i]["layout"].update(title=txt)
            coloraxiss = {}
            coloraxiss["coloraxis"] = {
                "colorscale": self.pargs.cmap,
                "cmin": self.clim[0],
                "cmax": self.clim[1],
                "showscale": showscale,
                "colorbar": {"tickfont": {"size": self.pargs.font_size - 2}, "title": cbar_title},
            }
            self.FIGURE.update_layout(**coloraxiss)

        def frame_args(duration):
            return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

        self.FIGURE.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                frame_args(duration),
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                frame_args(0),
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )

    def update_fig(self, show_outline: bool = False):
        if not self.show_zaxis:
            scene = self._get_plotly_dim_scene(mode="2d", show_outline=show_outline)
        else:
            scene = self._get_plotly_dim_scene(mode="3d", show_outline=show_outline)
        self.FIGURE.update_layout(
            template=self.pargs.theme,
            autosize=True,
            showlegend=False,
            scene=scene,
            width=self.pargs.window_size[0],
            height=self.pargs.window_size[1],
            font={"family": self.pargs.font_family},
            title=self.title,
        )

        return self.FIGURE

    def _get_plotly_dim_scene(self, mode="3d", show_outline=True):
        if show_outline:
            off_axis = {"showgrid": True, "zeroline": True, "visible": True}
        else:
            off_axis = {"showgrid": False, "zeroline": False, "visible": False}
        max_range = np.ptp(self.points, axis=0).max()  # ptp: max - mi
        if mode.lower() == "3d":
            eye = {
                "x": -max_range * 2,
                "y": -max_range * 2,
                "z": max_range * 5,
            }  # for 3D camera
            scene = {
                # "aspectratio": {"x": 1, "y": 1, "z": 1},
                "aspectmode": "data",
                "camera": {"eye": eye, "projection": {"type": "orthographic"}},
                "xaxis": off_axis,
                "yaxis": off_axis,
                "zaxis": off_axis,
            }
        elif mode.lower() == "2d":
            if show_outline:
                xaxis = {"showbackground": False}
                yaxis = {"showbackground": False}
                zaxis = {
                    "showbackground": True,
                    "showticklabels": False,
                    "showgrid": True,
                    "title": "",
                    "ticks": "",
                    "visible": True,
                }
            else:
                xaxis = off_axis
                yaxis = off_axis
                zaxis = off_axis
            eye = {
                "x": 0,
                "y": -0.1,
                "z": max_range * 1,
            }  # for 2D camera
            scene = {
                "camera": {"eye": eye},
                "aspectmode": "data",
                "dragmode": "pan",
                "xaxis": xaxis,
                "yaxis": yaxis,
                "zaxis": zaxis,
            }
        else:
            raise ValueError("mode must be '2d' or '3d'")  # noqa: TRY003
        return scene


def _plot_bc(plotter, fixed_dofs, fixed_coords, s, show_zaxis, color):
    bc_plot = None
    if len(fixed_coords) > 0:
        points = _get_bc_points(fixed_coords, fixed_dofs, s, show_zaxis)
        bc_plot = _plot_lines(
            plotter,
            points,
            width=1.0,
            name="BC",
            color=color,
            hoverinfo="skip",
        )
    # else:
    #     warnings.warn("Info:: Model has no fixed nodes!")
    return bc_plot


def _plot_mp_constraint(
    plotter,
    points,
    cells,
    dofs,
    lw,
    color,
    show_dofs=False,
):
    dofs = ["".join(map(str, row)) for row in dofs]
    if len(cells) > 0:
        line_points, line_mid_points = _make_lines_plotly(points, cells)
        x, y, z = line_points[:, 0], line_points[:, 1], line_points[:, 2]
        plotter.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                line={"color": color, "width": lw},
                mode="lines",
                name="mp constraint",
                connectgaps=False,
                hoverinfo="skip",
            )
        )
        if show_dofs:
            x, y, z = [line_mid_points[:, j] for j in range(3)]
            txt_plot = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                text=dofs,
                textfont={"color": color, "size": 12},
                mode="text",
                name="constraint dofs",
            )
            plotter.append(txt_plot)


def _make_lines_arrows(
    starts,
    lengths,
    xaxis,
    yaxis,
    zaxis,
    color,
    name,
    hovers,
    lw,
    arrow_height,
    arrow_width,
):
    coords = np.zeros_like(starts)
    for i, midpoint in enumerate(starts):
        coords[i] = midpoint + lengths[i] * xaxis[i]
    local_points = []
    labels = []
    for i, midpoint in enumerate(starts):
        local_points.append(midpoint)
        local_points.append(coords[i])
        local_points.append([np.nan, np.nan, np.nan])
        labels.extend([hovers[i]] * 3)
    local_points = np.array(local_points)
    line = go.Scatter3d(
        x=local_points[:, 0],
        y=local_points[:, 1],
        z=local_points[:, 2],
        line={"color": color, "width": lw},
        mode="lines",
        connectgaps=False,
        name=name,
        hovertemplate="<b>%{text}</b>",
        text=labels,
        # hoverinfo="skip",
    )
    # arrows
    angles = np.linspace(0, 2 * np.pi, 16)
    num = len(starts)
    points = []
    ijk = []
    labels = []
    for i in range(num):
        xs = (0.5 * arrow_width[i] * np.cos(angles)).reshape((-1, 1))
        ys = (0.5 * arrow_width[i] * np.sin(angles)).reshape((-1, 1))
        cen, ax, ay, az = coords[i], xaxis[i], yaxis[i], zaxis[i]
        tips = cen + arrow_height[i] * ax
        secs = xs @ np.reshape(ay, (1, 3)) + ys @ np.reshape(az, (1, 3))
        secs += cen
        for j in range(len(secs) - 1):
            ijk.append([len(points) + j, len(points) + j + 1, len(points) + len(secs)])
            ijk.append([len(points) + j, len(points) + j + 1, len(points) + len(secs) + 1])
        ijk.append([len(points) + len(secs) - 1, len(points), len(points) + len(secs)])
        ijk.append([len(points) + len(secs) - 1, len(points), len(points) + len(secs) + 1])
        points.extend(np.vstack([secs, cen, tips]))
        labels.extend([hovers[i]] * (len(secs) + 2))
    points = np.array(points)
    ijk = np.array(ijk)
    arrow = go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=ijk[:, 0],
        j=ijk[:, 1],
        k=ijk[:, 2],
        color=color,
        name=name,
        text=labels,
        hovertemplate="<b>%{text}",
    )
    return line, arrow


def _get_bc_points(fixed_coords, fixed_dofs, s, show_zaxis):
    if show_zaxis:
        return _get_bc_points_3d(fixed_coords, fixed_dofs, s)
    else:
        return _get_bc_points_2d(fixed_coords, fixed_dofs, s)


def _get_bc_points_3d(fixed_coords, fixed_dofs, s):
    fixed_dofs = ["".join(map(str, row)) for row in fixed_dofs]
    points = []
    for coord, dof in zip(fixed_coords, fixed_dofs):
        x, y, z = coord
        if dof[0] == "1":
            points.extend([
                [x, y - s / 2, z - s / 2],
                [x, y + s / 2, z - s / 2],
                [x, y + s / 2, z + s / 2],
                [x, y - s / 2, z + s / 2],
                [x, y - s / 2, z - s / 2],
                [np.nan, np.nan, np.nan],
            ])
        if dof[1] == "1":
            points.extend([
                [x - s / 2, y, z - s / 2],
                [x + s / 2, y, z - s / 2],
                [x + s / 2, y, z + s / 2],
                [x - s / 2, y, z + s / 2],
                [x - s / 2, y, z - s / 2],
                [np.nan, np.nan, np.nan],
            ])
        if dof[2] == "1":
            points.extend([
                [x - s / 2, y - s / 2, z],
                [x + s / 2, y - s / 2, z],
                [x + s / 2, y + s / 2, z],
                [x - s / 2, y + s / 2, z],
                [x - s / 2, y - s / 2, z],
                [np.nan, np.nan, np.nan],
            ])
    return np.array(points)


def _get_bc_points_2d(fixed_coords, fixed_dofs, s):
    fixed_dofs = ["".join(map(str, row)) for row in fixed_dofs]
    points = []
    for coord, dof in zip(fixed_coords, fixed_dofs):
        x, y, z = coord
        if dof[2] == "1":
            y -= s / 2
            points.extend([
                [x - s / 2, y - s / 2, z],
                [x + s / 2, y - s / 2, z],
                [x + s / 2, y + s / 2, z],
                [x - s / 2, y + s / 2, z],
                [x - s / 2, y - s / 2, z],
                [np.nan, np.nan, np.nan],
            ])
        elif dof[0] == "1" and dof[1] == "1":
            points.extend([
                [x - s * 0.5, y - s, z],
                [x + s * 0.5, y - s, z],
                [x, y, z],
                [x - s * 0.5, y - s, z],
                [np.nan, np.nan, np.nan],
            ])
        else:
            angles = np.linspace(0, 2 * np.pi, 21)
            coords = np.zeros((len(angles), 3))
            coords[:, 0] = 0.5 * s * np.cos(angles)
            coords[:, 1] = 0.5 * s * np.sin(angles)
            coords[:, 2] = z
            if dof[0] == "1":
                coords[:, 0] += x - s / 2
                coords[:, 1] += y
            elif dof[1] == "1":
                coords[:, 0] += x
                coords[:, 1] += y - s / 2
            points.extend(coords)
            points.append([np.nan, np.nan, np.nan])
    return np.array(points)
