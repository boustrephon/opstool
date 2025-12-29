from importlib.util import find_spec

from ...utils import make_dependency_missing

if not find_spec("plotly"):
    extra = "plotly"
    plot_model = make_dependency_missing("plotly.plot_model", "plotly", extra=extra)
    plot_eigen = make_dependency_missing("plotly.plot_eigen", "plotly", extra=extra)
    plot_eigen_animation = make_dependency_missing("plotly.plot_eigen_animation", "plotly", extra=extra)
    plot_eigen_table = make_dependency_missing("plotly.plot_eigen_table", "plotly", extra=extra)
    plot_frame_responses = make_dependency_missing("plotly.plot_frame_responses", "plotly", extra=extra)
    plot_frame_responses_animation = make_dependency_missing(
        "plotly.plot_frame_responses_animation", "plotly", extra=extra
    )
    plot_nodal_responses = make_dependency_missing("plotly.plot_nodal_responses", "plotly", extra=extra)
    plot_nodal_responses_animation = make_dependency_missing(
        "plotly.plot_nodal_responses_animation", "plotly", extra=extra
    )
    plot_truss_responses = make_dependency_missing("plotly.plot_truss_responses", "plotly", extra=extra)
    plot_truss_responses_animation = make_dependency_missing(
        "plotly.plot_truss_responses_animation", "plotly", extra=extra
    )
    plot_unstruct_responses = make_dependency_missing("plotly.plot_unstruct_responses", "plotly", extra=extra)
    plot_unstruct_responses_animation = make_dependency_missing(
        "plotly.plot_unstruct_responses_animation", "plotly", extra=extra
    )
    set_plot_colors = make_dependency_missing("plotly.set_plot_colors", "plotly", extra=extra)
    set_plot_props = make_dependency_missing("plotly.set_plot_props", "plotly", extra=extra)
    reset_plot_props = make_dependency_missing("plotly.reset_plot_props", "plotly", extra=extra)
else:
    from .plot_utils import reset_plot_props, set_plot_colors, set_plot_props
    from .vis_eigen import plot_eigen, plot_eigen_animation, plot_eigen_table
    from .vis_frame_resp import plot_frame_responses, plot_frame_responses_animation
    from .vis_model import plot_model
    from .vis_nodal_resp import plot_nodal_responses, plot_nodal_responses_animation
    from .vis_truss_resp import plot_truss_responses, plot_truss_responses_animation
    from .vis_unstru_resp import plot_unstruct_responses, plot_unstruct_responses_animation

reset_plot_props()

__all__ = [
    "plot_eigen",
    "plot_eigen_animation",
    "plot_eigen_table",
    # # --------------------------------
    "plot_frame_responses",
    "plot_frame_responses_animation",
    # --------------------------------
    "plot_model",
    # --------------------------------
    "plot_nodal_responses",
    "plot_nodal_responses_animation",
    # # --------------------------------
    "plot_truss_responses",
    "plot_truss_responses_animation",
    # # --------------------------------
    "plot_unstruct_responses",
    "plot_unstruct_responses_animation",
    "reset_plot_props",
    # --------------------------------
    "set_plot_colors",
    "set_plot_props",
]
