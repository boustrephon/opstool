from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt

from .responses_data import get_element_responses, get_nodal_responses


def plot_nodal_responses(
    odb_tag: int | str,
    resp_type: Literal["disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure"],
    resp_dof: Literal["UX", "UY", "UZ", "RX", "RY", "RZ"] | None = None,
    node_tags: list[int] | tuple[int, ...] | int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Read nodal responses data from a file.

    Added in opstool v1.0.25.

    .. important::
        You can use :func:`opstool.post.get_nodal_responses_info` to get valid response types and DOFs.

    Parameters
    ----------
    odb_tag: int | str
        Tag of output databases (ODB) to be read.
    resp_type: Literal["disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure"]
        Type of response to be read.
        Optional:

        * "disp" - Displacement at the node.
        * "vel" - Velocity at the node.
        * "accel" - Acceleration at the node.
        * "reaction" - Reaction forces at the node.
        * "reactionIncInertia" - Reaction forces including inertial effects.
        * "rayleighForces" - Forces resulting from Rayleigh damping.
        * "pressure" - Pressure applied to the node.

        .. Note::
            If the nodes include fluid pressure dof,
            such as those used for ``**UP`` elements, the pore pressure should be extracted using ``resp_type="vel"``,
            and the value is placed in the degree of freedom ``RZ``.
    resp_dof : Literal["UX", "UY", "UZ", "RX", "RY", "RZ"]
        Degree of freedom of response to be read.
        Optional:

        * "UX", "UY", "UZ" - Displacement/Velocity/Acceleration in X, Y, Z directions.
        * "RX", "RY", "RZ" - Rotational displacement/Velocity/Acceleration about X, Y, Z axes.

        If None, no selection on DOF is applied. If resp_type is "pressure", this parameter is ignored.

    node_tags: list[int] | tuple[int, ...] | int | None = None
        Node tags to be read.
        Such as [1, 2, 3] or numpy.array([1, 2, 3]) or 1.
        If None, return all nodal responses.

        .. Note::
            If some nodes are deleted during the analysis,
            their response data will be filled with `numpy.nan`.

    ax: plt.Axes | None
        Matplotlib Axes object to plot on.
        If None, a new figure and axes will be created.

    Returns
    ---------
    plt.Axes
        Matplotlib Axes object containing the plot.

    """
    nodal_resp = get_nodal_responses(
        odb_tag=odb_tag,
        resp_type=resp_type,
        node_tags=node_tags,
        print_info=False,
        lazy_load=False,
    )

    if resp_dof != "pressure" and resp_dof is not None:
        nodal_resp = nodal_resp.sel(DOFs=resp_dof)

    node_tags = nodal_resp.coords["nodeTags"].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    for node_tag in node_tags:
        data = nodal_resp.sel(nodeTags=node_tag)
        ax.plot(data.coords["time"].values, data.values, label=f"Node {node_tag}")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{resp_type}-{resp_dof}")
    ax.legend()
    ax.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)
    return ax


def plot_element_responses(
    odb_tag: int | str,
    ele_type: Literal["Frame", "FiberSection", "FiberSec", "Truss", "Link", "Shell", "Plane", "Solid", "Contact"],
    resp_type: str,
    resp_dof: str | None = None,
    ele_tags: list[int] | tuple[int, ...] | int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot element responses over time.

    Added in opstool v1.0.25.

    .. important::
        You can use :func:`opstool.post.get_element_responses_info` to get valid response types and DOFs for each element type.

    Parameters
    ----------
    odb_tag : int | str
        ODB tag to read data from.
    ele_type : Literal["Frame", "FiberSection", "FiberSec", "Truss", "Link", "Shell", "Plane", "Solid", "Contact"]
        Element type to read data from.
    resp_type : str
        Response type to read data from.
        This should be one of the valid response types for the specified element type.
    resp_dof : str | None, optional
        Response degree of freedom to read data from.
        This should be one of the valid degrees of freedom for the specified response type.
        If None, no selection on DOF is applied.
    ele_tags : list[int] | tuple[int, ...] | int | None, optional
        Element tags to read data from, by default None
    ax : plt.Axes | None, optional
        Matplotlib Axes object to plot on, by default None

    Returns
    -------
    plt.Axes
        _description_
    """
    ele_resp = get_element_responses(
        odb_tag=odb_tag,
        ele_type=ele_type,
        resp_type=resp_type,
        ele_tags=ele_tags,
        print_info=False,
        lazy_load=False,
    )
    if resp_dof is not None:
        dim = ele_resp.dims[-1]
        ele_resp = ele_resp.sel({dim: resp_dof})

    ele_tags = ele_resp.coords["eleTags"].values
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    for ele_tag in ele_tags:
        data = ele_resp.sel(eleTags=ele_tag)
        ax.plot(data.coords["time"].values, data.values, label=f"Element {ele_tag}")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{resp_type}-{resp_dof}")
    ax.legend()
    ax.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)
    return ax
