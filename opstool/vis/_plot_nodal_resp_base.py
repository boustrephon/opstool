import numpy as np

from ._plot_resp_base import PlotResponseBase


class PlotNodalResponseBase(PlotResponseBase):
    def __init__(
        self,
        model_info_steps,
        node_resp_steps,
        model_update,
    ):
        super().__init__(model_info_steps, node_resp_steps, model_update)
        self.resps_norm = None

    def set_comp_resp_type(self, resp_type, component):
        if resp_type.lower() in ["disp", "dispacement"]:
            self.resp_type = "disp"
        elif resp_type.lower() in ["vel", "velocity"]:
            self.resp_type = "vel"
        elif resp_type.lower() in ["accel", "acceleration"]:
            self.resp_type = "accel"
        elif resp_type.lower() in ["reaction", "reactionforce"]:
            self.resp_type = "reaction"
        elif resp_type.lower() in ["reactionincinertia", "reactionincinertiaforce"]:
            self.resp_type = "reactionIncInertia"
        elif resp_type.lower() in ["rayleighforces", "rayleigh"]:
            self.resp_type = "rayleighForces"
        elif resp_type.lower() in ["pressure"]:
            self.resp_type = "pressure"
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid response type: {resp_type}. "
                "Valid options are: disp, vel, accel, reaction, reactionIncInertia, rayleighForces, pressure."
            )
        if isinstance(component, str):
            self.component = component.upper()
        else:
            self.component = list(component)

    def _get_resp_clim_peak(self, idx=None):
        # If idx is None, get clim for all steps, else for specified step
        resps = []
        for i in range(self.num_steps):
            da = self._get_resp_da(i, self.resp_type, self.component)
            resps.append(da)
        if self.ModelUpdate:
            resps_norm = resps if resps[0].ndim == 1 else [np.linalg.norm(resp, axis=1) for resp in resps]
        else:
            resps_norm = resps if resps[0].ndim == 1 else np.linalg.norm(resps, axis=2)
        if isinstance(idx, str):
            if idx.lower() == "absmax":
                resp = [np.nanmax(np.abs(data)) for data in resps]
                step = np.argmax(resp)
            elif idx.lower() == "max":
                resp = [np.max(data) for data in resps]
                step = np.argmax(resp)
            elif idx.lower() == "absmin":
                resp = [np.nanmin(np.abs(data)) for data in resps]
                step = np.argmin(resp)
            elif idx.lower() == "min":
                resp = [np.nanmin(data) for data in resps]
                step = np.argmin(resp)
            else:
                raise ValueError("Invalid argument, one of [absMax, absMin, Max, Min]")  # noqa: TRY003
        elif isinstance(idx, (int, float)):
            step = int(idx)
        else:
            resp = [np.nanmax(np.abs(data)) for data in resps]  # absmax
            step = np.argmax(resp)
        if idx is None:
            max_resps = [np.nanmax(resp) for resp in resps_norm]
            min_resps = [np.nanmin(resp) for resp in resps_norm]
            cmin, cmax = np.nanmin(min_resps), np.nanmax(max_resps)
        else:
            cmin, cmax = np.nanmin(resps_norm[step]), np.nanmax(resps_norm[step])
        self.resps_norm = resps_norm
        self.clim = (cmin, cmax)
        return cmin, cmax, step

    def _make_title(self, *args, **kwargs):
        pass

    def _create_mesh(self, *args, **kwargs):
        pass

    def _update_mesh(self, *args, **kwargs):
        pass

    def plot_slide(self, *args, **kwargs):
        pass

    def plot_peak_step(self, *args, **kwargs):
        pass

    def plot_anim(self, *args, **kwargs):
        pass
