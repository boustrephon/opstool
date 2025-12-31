from importlib.util import find_spec

from ...utils import make_dependency_missing

if not find_spec("pyvista"):
    extra = "pyvista"
    plot_model = make_dependency_missing("pyvista.plot_model", "pyvista", extra=extra)
    plot_eigen = make_dependency_missing("pyvista.plot_eigen", "pyvista", extra=extra)
    plot_eigen_animation = make_dependency_missing("pyvista.plot_eigen_animation", "pyvista", extra=extra)
    plot_frame_responses = make_dependency_missing("pyvista.plot_frame_responses", "pyvista", extra=extra)
    plot_frame_responses_animation = make_dependency_missing(
        "pyvista.plot_frame_responses_animation", "pyvista", extra=extra
    )
    plot_nodal_responses = make_dependency_missing("pyvista.plot_nodal_responses", "pyvista", extra=extra)
    plot_nodal_responses_animation = make_dependency_missing(
        "pyvista.plot_nodal_responses_animation", "pyvista", extra=extra
    )
    plot_truss_responses = make_dependency_missing("pyvista.plot_truss_responses", "pyvista", extra=extra)
    plot_truss_responses_animation = make_dependency_missing(
        "pyvista.plot_truss_responses_animation", "pyvista", extra=extra
    )
    plot_unstruct_responses = make_dependency_missing("pyvista.plot_unstruct_responses", "pyvista", extra=extra)
    plot_unstruct_responses_animation = make_dependency_missing(
        "pyvista.plot_unstruct_responses_animation", "pyvista", extra=extra
    )
    get_unstruct_responses_dataset = make_dependency_missing(
        "pyvista.get_unstruct_responses_dataset", "pyvista", extra=extra
    )
    set_plot_colors = make_dependency_missing("pyvista.set_plot_colors", "pyvista", extra=extra)
    set_plot_props = make_dependency_missing("pyvista.set_plot_props", "pyvista", extra=extra)
    reset_plot_props = make_dependency_missing("pyvista.reset_plot_props", "pyvista", extra=extra)
else:
    from .plot_utils import reset_plot_props, set_plot_colors, set_plot_props
    from .vis_eigen import plot_eigen, plot_eigen_animation
    from .vis_frame_resp import plot_frame_responses, plot_frame_responses_animation
    from .vis_model import plot_model
    from .vis_nodal_resp import get_nodal_responses_dataset, plot_nodal_responses, plot_nodal_responses_animation
    from .vis_truss_resp import plot_truss_responses, plot_truss_responses_animation
    from .vis_unstru_resp import (
        get_unstruct_responses_dataset,
        plot_unstruct_responses,
        plot_unstruct_responses_animation,
    )

reset_plot_props()

__all__ = [
    "get_nodal_responses_dataset",
    "get_unstruct_responses_dataset",
    # --------------------------------
    "plot_eigen",
    "plot_eigen_animation",
    # --------------------------------
    "plot_frame_responses",
    "plot_frame_responses_animation",
    # --------------------------------
    "plot_model",
    # --------------------------------
    "plot_nodal_responses",
    "plot_nodal_responses_animation",
    # --------------------------------
    "plot_truss_responses",
    "plot_truss_responses_animation",
    # --------------------------------
    "plot_unstruct_responses",
    "plot_unstruct_responses_animation",
    "reset_plot_props",
    # --------------------------------
    "set_plot_colors",
    "set_plot_props",
]
