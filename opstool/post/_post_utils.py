from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ------------------------------------------------------------
# 3D Beam interpolator

Space = Literal["global", "local"]
NaNPolicy = Literal["ignore", "propagate"]


@dataclass
class Beam3DDispInterpolator:
    """Interpolate 3D beam element displacements from nodal/global to element/local and interior points."""

    node_coords: np.ndarray
    conn: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    one_based_node_id: bool = False

    _conn0: np.ndarray | None = None
    _Xi: np.ndarray | None = None
    _dX: np.ndarray | None = None
    _L: np.ndarray | None = None
    _R: np.ndarray | None = None

    _invalid_axes: np.ndarray | None = None
    _zero_length: np.ndarray | None = None  # zero-length elements mask

    _grid_cache: (
        dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None
    ) = None

    def __post_init__(self) -> None:
        self.node_coords = np.asarray(self.node_coords, dtype=float)
        self.conn = np.asarray(self.conn, dtype=np.int64)
        self.ex = np.asarray(self.ex, dtype=float)
        self.ey = np.asarray(self.ey, dtype=float)
        self.ez = np.asarray(self.ez, dtype=float)

        if self.node_coords.ndim != 2 or self.node_coords.shape[1] != 3:
            raise ValueError("node_coords must have shape (nNodes, 3).")  # noqa: TRY003
        if self.conn.ndim != 2 or self.conn.shape[1] != 2:
            raise ValueError("conn must have shape (nEles, 2).")  # noqa: TRY003
        nEles = self.conn.shape[0]
        if self.ex.shape != (nEles, 3) or self.ey.shape != (nEles, 3) or self.ez.shape != (nEles, 3):
            raise ValueError("ex/ey/ez must have shape (nEles, 3).")  # noqa: TRY003

        self._grid_cache = {}
        self._build_geometry_cache()
        self._build_invalid_axes_mask()

    # --------------------------
    # Public API (unchanged)
    # --------------------------
    def global_to_local_ends(
        self,
        nodal_global: np.ndarray,  # (..., nNodes, 6)
        *,
        nan_policy: NaNPolicy = "ignore",
    ) -> np.ndarray:
        g = np.asarray(nodal_global, dtype=float)
        if g.shape[-1] < 6:
            raise ValueError("nodal_global last dim must be >= 6.")  # noqa: TRY003
        if g.shape[-2] != self.node_coords.shape[0]:
            raise ValueError("nodal_global must have nNodes matching node_coords.")  # noqa: TRY003

        ni = self._conn0[:, 0]
        nj = self._conn0[:, 1]

        di_g = np.take(g[..., :, :6], ni, axis=-2)  # (..., nEles, 6)
        dj_g = np.take(g[..., :, :6], nj, axis=-2)

        if nan_policy == "ignore":
            di_g = np.nan_to_num(di_g, nan=0.0)
            dj_g = np.nan_to_num(dj_g, nan=0.0)
        elif nan_policy != "propagate":
            raise ValueError("nan_policy must be 'ignore' or 'propagate'.")  # noqa: TRY003

        invalid = self._invalid_axes  # (nEles,)
        valid = ~invalid

        # ---- Fast path: all valid
        if not np.any(invalid):
            di_l = self._rot6(di_g, R=self._R)
            dj_l = self._rot6(dj_g, R=self._R)
            return np.concatenate([di_l, dj_l], axis=-1)

        # allocate outputs
        di_l = np.empty_like(di_g)
        dj_l = np.empty_like(dj_g)

        # ---- valid rotate (only if any valid)
        if np.any(valid):
            Rv = self._R[valid]  # (nValid, 3, 3)
            di_l[..., valid, :] = self._rot6(di_g[..., valid, :], R=Rv)
            dj_l[..., valid, :] = self._rot6(dj_g[..., valid, :], R=Rv)

        # ---- invalid: store global translations directly; rotations set to 0
        if np.any(invalid):
            di_l[..., invalid, 0:3] = di_g[..., invalid, 0:3]
            dj_l[..., invalid, 0:3] = dj_g[..., invalid, 0:3]
            di_l[..., invalid, 3:6] = 0.0
            dj_l[..., invalid, 3:6] = 0.0

        return np.concatenate([di_l, dj_l], axis=-1)

    def interpolate(
        self,
        end_local: np.ndarray,  # (..., nEles, 12)
        *,
        npts_per_ele: int = 11,
        nan_policy: NaNPolicy = "ignore",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate local end vectors to interior points and return line data.

        Smart handling:
        - valid axes elements: sample m points and interpolate (Hermite for uy/uz)
        - invalid axes elements: no interpolation; only two end points + one segment cell.

        Returns
        -------
        points : (N, 3)
        response : (..., N, 3)
        cells : (M, 3) each row [2, idx_i, idx_j]
        """
        el = np.asarray(end_local, dtype=float)
        if el.shape[-1] != 12:
            raise ValueError("end_local last dim must be 12.")  # noqa: TRY003
        if el.shape[-2] != self.conn.shape[0]:
            raise ValueError("end_local must have nEles matching conn.")  # noqa: TRY003
        if npts_per_ele < 2:
            raise ValueError("npts_per_ele must be >= 2.")  # noqa: TRY003
        if nan_policy not in ("ignore", "propagate"):
            raise ValueError("nan_policy must be 'ignore' or 'propagate'.")  # noqa: TRY003

        nEles = self.conn.shape[0]
        m = int(npts_per_ele)

        invalid = self._invalid_axes
        valid = ~invalid

        # precompute shape functions (cached)
        s, L1, L2, N1, N2, N3, N4 = self._get_shapes(m)

        # ---- Case 1: all invalid -> only endpoints
        if not np.any(valid):
            return self._endpoints_only_from_endlocal(el, response_space="global", nan_policy=nan_policy)

        # ---- Build VALID part (sample + interpolate)
        # points_valid: (nValid*m, 3)
        Xi_v = self._Xi[valid]
        dX_v = self._dX[valid]
        points_valid = (Xi_v[:, None, :] + dX_v[:, None, :] * s[None, :, None]).reshape((-1, 3))

        # extract valid end_local
        el_v = el[..., valid, :]  # (..., nValid, 12)
        ui = el_v[..., :, 0:6]
        uj = el_v[..., :, 6:12]

        uxi = ui[..., :, 0]
        uyi = ui[..., :, 1]
        uzi = ui[..., :, 2]
        ryi = ui[..., :, 4]
        rzi = ui[..., :, 5]
        uxj = uj[..., :, 0]
        uyj = uj[..., :, 1]
        uzj = uj[..., :, 2]
        ryj = uj[..., :, 4]
        rzj = uj[..., :, 5]

        # local translations (..., nValid, m)
        ux_l = self._interp_linear(a=uxi, b=uxj, L1=L1, L2=L2, nan_policy=nan_policy)
        uy_l = self._interp_hermite_or_linear(
            u_i=uyi,
            th_i=rzi,
            u_j=uyj,
            th_j=rzj,
            L=self._L[valid],
            N1=N1,
            N2=N2,
            N3=N3,
            N4=N4,
            L1=L1,
            L2=L2,
            th_sign=+1.0,
            nan_policy=nan_policy,
        )
        uz_l = self._interp_hermite_or_linear(
            u_i=uzi,
            th_i=ryi,
            u_j=uzj,
            th_j=ryj,
            L=self._L[valid],
            N1=N1,
            N2=N2,
            N3=N3,
            N4=N4,
            L1=L1,
            L2=L2,
            th_sign=-1.0,
            nan_policy=nan_policy,  # duz/dx = -ry
        )

        disp_l = np.stack([ux_l, uy_l, uz_l], axis=-1)  # (..., nValid, m, 3)

        # valid axes global conversion
        nb = disp_l.ndim - 3
        exv = self._axes_broadcast(self.ex[valid], nb)
        eyv = self._axes_broadcast(self.ey[valid], nb)
        ezv = self._axes_broadcast(self.ez[valid], nb)
        resp_valid = (
            disp_l[..., :, :, 0:1] * exv + disp_l[..., :, :, 1:2] * eyv + disp_l[..., :, :, 2:3] * ezv
        )  # (..., nValid, m, 3)

        nValid = int(np.sum(valid))
        resp_valid = resp_valid.reshape((*resp_valid.shape[:-3], nValid * m, 3))  # (..., Nvalid, 3)

        # valid cells: segments inside each polyline
        idx0 = (np.arange(nValid, dtype=np.int64) * m)[:, None] + np.arange(m - 1, dtype=np.int64)[None, :]
        idx1 = idx0 + 1
        cells_valid = np.empty((nValid * (m - 1), 3), dtype=np.int64)
        cells_valid[:, 0] = 2
        cells_valid[:, 1] = idx0.reshape(-1)
        cells_valid[:, 2] = idx1.reshape(-1)

        # ---- Build INVALID part (endpoints-only)
        points_bad, resp_bad, cells_bad = self._endpoints_only_for_invalid_from_endlocal(
            el, response_space="global", nan_policy=nan_policy
        )

        # ---- Concatenate with reindexing
        points = np.concatenate([points_valid, points_bad], axis=0)

        # response: (..., N, 3)
        response = np.concatenate([resp_valid, resp_bad], axis=-2)

        # cells: offset bad indices
        offset = points_valid.shape[0]
        if cells_bad.size > 0:
            cells_bad = cells_bad.copy()
            cells_bad[:, 1:] += offset
        cells = np.concatenate([cells_valid, cells_bad], axis=0)

        return points, response, cells

    # --------------------------
    # Geometry cache
    # --------------------------
    def _build_geometry_cache(self, tol_len: float = 1e-14) -> None:
        self._conn0 = self.conn - 1 if self.one_based_node_id else self.conn.copy()

        ni = self._conn0[:, 0]
        nj = self._conn0[:, 1]
        Xi = self.node_coords[ni]
        Xj = self.node_coords[nj]
        dX = Xj - Xi
        L = np.linalg.norm(dX, axis=1)

        # ZERO LENGTH CHECK
        #    RECORD which elements are zero-length for later use in invalid axes mask
        self._zero_length = tol_len >= L

        self._Xi = Xi
        self._dX = dX
        self._L = L

        R = np.empty((self.conn.shape[0], 3, 3), dtype=float)
        R[:, 0, :] = self.ex
        R[:, 1, :] = self.ey
        R[:, 2, :] = self.ez
        self._R = R

    def _build_invalid_axes_mask(self, tol_axis: float = 1e-14) -> None:
        exn = np.linalg.norm(self.ex, axis=1)
        eyn = np.linalg.norm(self.ey, axis=1)
        ezn = np.linalg.norm(self.ez, axis=1)

        invalid_axis = (exn < tol_axis) & (eyn < tol_axis) & (ezn < tol_axis)

        # key fix: also treat zero-length elements as invalid
        zl = self._zero_length
        if zl is None:
            self._invalid_axes = invalid_axis
        else:
            self._invalid_axes = invalid_axis | zl

    # --------------------------
    # Shapes cache (only shapes; points/cells built per valid/invalid)
    # --------------------------
    def _get_shapes(self, npts_per_ele: int):
        cached = self._grid_cache.get(npts_per_ele)
        if cached is not None:
            return cached

        m = int(npts_per_ele)
        s = np.linspace(0.0, 1.0, m)
        L1 = 1.0 - s
        L2 = s
        s2 = s * s
        s3 = s2 * s
        N1 = 1.0 - 3.0 * s2 + 2.0 * s3
        N2 = s - 2.0 * s2 + s3
        N3 = 3.0 * s2 - 2.0 * s3
        N4 = -s2 + s3

        cached = (s, L1, L2, N1, N2, N3, N4)
        self._grid_cache[npts_per_ele] = cached
        return cached

    # --------------------------
    # Invalid elements: endpoints-only builders
    # --------------------------
    def _endpoints_only_from_endlocal(
        self,
        el: np.ndarray,
        *,
        response_space: Space,
        nan_policy: NaNPolicy,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All elements treated as invalid: endpoints-only output."""
        invalid = np.ones(self.conn.shape[0], dtype=bool)
        return self._endpoints_only_for_mask_from_endlocal(
            el, invalid, response_space=response_space, nan_policy=nan_policy
        )

    def _endpoints_only_for_invalid_from_endlocal(
        self,
        el: np.ndarray,
        *,
        response_space: Space,
        nan_policy: NaNPolicy,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Endpoints-only output for invalid elements only."""
        invalid = self._invalid_axes
        return self._endpoints_only_for_mask_from_endlocal(
            el, invalid, response_space=response_space, nan_policy=nan_policy
        )

    def _endpoints_only_for_mask_from_endlocal(
        self,
        el: np.ndarray,
        mask: np.ndarray,
        *,
        response_space: Space,
        nan_policy: NaNPolicy,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build endpoints-only points/response/cells for elements in mask.

        For invalid elements, end_local translational slots store GLOBAL XYZ translations
        (set in global_to_local_ends). For robustness, we simply read ux,uy,uz here.
        """
        ids = np.where(mask)[0]
        if ids.size == 0:
            points = np.empty((0, 3), dtype=float)
            cells = np.empty((0, 3), dtype=np.int64)
            # response with batch dims preserved: (..., 0, 3)
            resp = el[..., :0, :3]  # shape trick
            return points, resp, cells

        # endpoints geometry
        Xi = self._Xi[ids]
        Xj = self._Xi[ids] + self._dX[ids]
        points = np.empty((ids.size * 2, 3), dtype=float)
        points[0::2] = Xi
        points[1::2] = Xj

        # endpoints responses
        ui = el[..., ids, 0:6]
        uj = el[..., ids, 6:12]
        # for invalid: ui/uj translations are already global XYZ (by our smart global_to_local_ends)
        Ui = ui[..., 0:3]
        Uj = uj[..., 0:3]
        if nan_policy == "ignore":
            Ui = np.nan_to_num(Ui, nan=0.0)
            Uj = np.nan_to_num(Uj, nan=0.0)

        resp = np.empty((*Ui.shape[:-2], ids.size * 2, 3), dtype=float)
        resp[..., 0::2, :] = Ui
        resp[..., 1::2, :] = Uj

        # one segment per element
        base = np.arange(ids.size, dtype=np.int64) * 2
        cells = np.empty((ids.size, 3), dtype=np.int64)
        cells[:, 0] = 2
        cells[:, 1] = base
        cells[:, 2] = base + 1

        # If user asks response_space="local" for invalid, we still return these 3 comps.
        # (No meaningful local axes exist, so treat as stored comps.)
        _ = response_space
        return points, resp, cells

    # --------------------------
    # Rotation (global -> local)
    # --------------------------
    def _rot6(self, d6: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
        Ruse = self._R if R is None else R
        u = np.einsum("eab,...eb->...ea", Ruse, d6[..., 0:3], optimize=True)
        r = np.einsum("eab,...eb->...ea", Ruse, d6[..., 3:6], optimize=True)
        return np.concatenate([u, r], axis=-1)

    @staticmethod
    def _axes_broadcast(axis_e3: np.ndarray, nbatch: int) -> np.ndarray:
        if nbatch <= 0:
            return axis_e3[:, None, :]  # (nEles,1,3)
        return axis_e3[(None,) * nbatch + (slice(None), None, slice(None))]

    # --------------------------
    # Interpolation (NaN-robust)
    # --------------------------
    @staticmethod
    def _interp_linear(a, b, L1, L2, nan_policy: NaNPolicy) -> np.ndarray:
        if nan_policy == "propagate":
            return a[..., :, None] * L1 + b[..., :, None] * L2

        a_ok = np.isfinite(a)
        b_ok = np.isfinite(b)
        out = np.full((*a.shape, L1.size), np.nan, dtype=float)

        both = a_ok & b_ok
        if np.any(both):
            out[both] = a[both][..., None] * L1 + b[both][..., None] * L2

        only_a = a_ok & ~b_ok
        if np.any(only_a):
            out[only_a] = a[only_a][..., None]

        only_b = ~a_ok & b_ok
        if np.any(only_b):
            out[only_b] = b[only_b][..., None]

        return out

    @classmethod
    def _interp_hermite_or_linear(
        cls,
        *,
        u_i,
        th_i,
        u_j,
        th_j,
        L,
        N1,
        N2,
        N3,
        N4,
        L1,
        L2,
        th_sign: float,
        nan_policy: NaNPolicy,
    ) -> np.ndarray:
        """
        Hermite interpolation; if any required term missing, fallback to linear.

        u(x) = N1*u_i + N2*(L*th_i) + N3*u_j + N4*(L*th_j)
        """
        if nan_policy == "propagate":
            # broadcast L safely: (1,...,1,nEles) against (...,nEles)
            Lb = L.reshape((1,) * (u_i.ndim - 1) + (-1,))
            return (
                u_i[..., :, None] * N1
                + (Lb * (th_sign * th_i))[..., None] * N2
                + u_j[..., :, None] * N3
                + (Lb * (th_sign * th_j))[..., None] * N4
            )

        ui_ok = np.isfinite(u_i)
        uj_ok = np.isfinite(u_j)
        ti_ok = np.isfinite(th_i)
        tj_ok = np.isfinite(th_j)
        full = ui_ok & uj_ok & ti_ok & tj_ok

        out = np.full((*u_i.shape, N1.size), np.nan, dtype=float)

        if np.any(full):
            Lb = L.reshape((1,) * (u_i.ndim - 1) + (-1,))

            out[full] = (
                u_i[full][..., None] * N1
                + (Lb * (th_sign * th_i))[full][..., None] * N2
                + u_j[full][..., None] * N3
                + (Lb * (th_sign * th_j))[full][..., None] * N4
            )

        if np.any(~full):
            out[~full] = cls._interp_linear(u_i, u_j, L1, L2, nan_policy="ignore")[~full]

        return out


# -------------------------------------------------------------------------------------------
# -- Functions for estimating and generating chunk sizes for xarray/DataTree variables --#
# -------------------------------------------------------------------------------------------
def _estimate_chunk_size(shape, dtype, target_mb=10.0):
    """Estimate balanced chunk sizes aiming ~target_mb per chunk."""
    itemsize = np.dtype(dtype).itemsize or 1
    if not shape:
        return ()
    target_items = max(1, int((target_mb * 1024 * 1024) / itemsize))
    total_items = int(np.prod(shape))

    # Small arrays → use full shape
    if total_items <= target_items:
        return tuple(int(s) for s in shape)

    # 1D arrays
    if len(shape) == 1:
        return (min(int(shape[0]), target_items),)

    # Balanced per-dimension chunk target with a small floor
    per_dim = max(32, round(target_items ** (1.0 / len(shape))))
    return tuple(int(min(int(dim), per_dim)) for dim in shape)


def _sanitize_chunks_from_dask(dask_chunks, shape):
    """
    Convert dask's chunks (tuple of tuples) → per-dim tuple.
    Takes the first chunk per dim; falls back to full dim if None/invalid.
    Ensures ≥1 and ≤ dim length.
    """
    try:
        out = []
        for chks, dim in zip(dask_chunks, shape):
            c0 = int(dim) if chks is None or len(chks) == 0 or chks[0] is None else int(chks[0])
            c0 = max(1, min(int(dim), c0))
            out.append(c0)
        return tuple(out)
    except Exception:
        return None


def _make_var_chunks(var, target_chunk_mb):
    """Return a valid chunks tuple for a single xarray Variable/DataArray."""
    # Scalars: return None to indicate "do not set chunks"
    if getattr(var, "ndim", 0) == 0:
        return None

    # Try Dask
    data = var.data
    chunks = None
    if hasattr(data, "chunks") and data.chunks is not None:
        chunks = _sanitize_chunks_from_dask(data.chunks, var.shape)

    # Otherwise estimate
    if chunks is None:
        chunks = _estimate_chunk_size(var.shape, var.dtype, target_mb=target_chunk_mb)

    # Final sanity
    try:
        chunks = tuple(int(max(1, c)) for c in chunks)
    except Exception:
        return None
    return chunks


def generate_chunk_encoding_for_datatree(datatree, target_chunk_mb=10.0, include_coords=True):
    """
    Build encoding dict for DataTree.to_zarr() that ONLY sets 'chunks' and
    ensures no None is passed for non-scalar arrays. Works for both data_vars and coords.
    """
    encoding = {}

    # Iterate nodes
    try:
        nodes = datatree.subtree.items()
    except AttributeError:
        nodes = [(node.path, node) for node in datatree.subtree]

    for node_path, node in nodes:
        ds = getattr(node, "ds", None)
        if ds is None:
            continue

        group_encoding = {}

        # Data variables
        for name, var in ds.data_vars.items():
            chunks = _make_var_chunks(var, target_chunk_mb)
            if chunks is not None:
                group_encoding[name] = {"chunks": chunks}
            else:
                group_encoding[name] = {}  # scalar → leave unset

        # Coordinates (important: avoid chunks=None here)
        if include_coords:
            for name, var in ds.coords.items():
                chunks = _make_var_chunks(var, target_chunk_mb)
                if chunks is not None:
                    group_encoding[name] = {"chunks": chunks}
                else:
                    group_encoding[name] = {}  # scalar coord

        if group_encoding:
            encoding[node_path] = group_encoding

    return encoding


# ------------------------------------------------------------------------------
# DIMENSIONS AND ATTRIBUTES UTILITIES
# ------------------------------------------------------------------------------
_NODAL_RESP_HELPS = {
    "resp_type": ["disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure"],
    "resp_dim": {
        "disp": ["time", "nodeTags", "DOFs"],
        "vel": ["time", "nodeTags", "DOFs"],
        "accel": ["time", "nodeTags", "DOFs"],
        "reaction": ["time", "nodeTags", "DOFs"],
        "reactionIncInertia": ["time", "nodeTags", "DOFs"],
        "rayleighForces": ["time", "nodeTags", "DOFs"],
        "pressure": ["time", "nodeTags"],
    },
    "resp_dof": {
        "disp": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
        "vel": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
        "accel": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
        "reaction": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
        "reactionIncInertia": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
        "rayleighForces": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
        "pressure": None,
    },
}

_FRAME_ELE_RESP_HELPS = {
    "resp_type": [
        "localForces",
        "basicForces",
        "basicDeformations",
        "plasticDeformation",
        "sectionForces",
        "sectionDeformations",
        "sectionLocs",
    ],
    "resp_dim": {
        "localForces": ["time", "eleTags", "localDofs"],
        "basicForces": ["time", "eleTags", "basicDofs"],
        "basicDeformations": ["time", "eleTags", "basicDofs"],
        "plasticDeformation": ["time", "eleTags", "basicDofs"],
        "sectionForces": ["time", "eleTags", "secPoints", "secDofs"],
        "sectionDeformations": ["time", "eleTags", "secPoints", "secDofs"],
        "sectionLocs": ["time", "eleTags", "secPoints", "locs"],
    },
    "resp_dof": {
        "localForces": ["FX1", "FY1", "FZ1", "MX1", "MY1", "MZ1", "FX2", "FY2", "FZ2", "MX2", "MY2", "MZ2"],
        "basicForces": ["N", "MZ1", "MZ2", "MY1", "MY2", "T"],
        "basicDeformations": ["N", "MZ1", "MZ2", "MY1", "MY2", "T"],
        "plasticDeformation": ["N", "MZ1", "MZ2", "MY1", "MY2", "T"],
        "sectionForces": ["N", "MZ", "VY", "MY", "VZ", "T"],
        "sectionDeformations": ["N", "MZ", "VY", "MY", "VZ", "T"],
        "sectionLocs": ["alpha", "X", "Y", "Z"],
    },
}

_TRUSS_ELE_RESP_HELPS = {
    "resp_type": ["axialForce", "axialDefo", "Stress", "Strain"],
    "resp_dim": {
        "axialForce": ["time", "eleTags"],
        "axialDefo": ["time", "eleTags"],
        "Stress": ["time", "eleTags"],
        "Strain": ["time", "eleTags"],
    },
    "resp_dof": {
        "axialForce": None,
        "axialDefo": None,
        "Stress": None,
        "Strain": None,
    },
}

_FIBER_SEC_RESP_HELPS = {
    "resp_type": ["Stresses", "Strains", "secForce", "secDefo"],
    "resp_dim": {
        "Stresses": ["time", "eleTags", "secPoints", "fiberPoints"],
        "Strains": ["time", "eleTags", "secPoints", "fiberPoints"],
        "secForce": ["time", "eleTags", "secPoints", "DOFs"],
        "secDefo": ["time", "eleTags", "secPoints", "DOFs"],
    },
    "resp_dof": {
        "Stresses": None,
        "Strains": None,
        "secForce": ["P", "Mz", "My", "T"],
        "secDefo": ["P", "Mz", "My", "T"],
    },
}

_PLANE_RESPS_HELP = {
    "resp_type": [
        "Stresses",
        "Strains",
        "StressesAtNodes",
        "StressAtNodesErr",
        "StrainsAtNodes",
        "StrainsAtNodesErr",
        "PorePressureAtNodes",
    ],
    "resp_dim": {
        "Stresses": ["time", "eleTags", "GaussPoints", "stressDOFs"],
        "Strains": ["time", "eleTags", "GaussPoints", "strainDOFs"],
        "StressesAtNodes": ["time", "nodeTags", "stressDOFs"],
        "StressAtNodesErr": ["time", "nodeTags", "stressDOFs"],
        "StrainsAtNodes": ["time", "nodeTags", "strainDOFs"],
        "StrainsAtNodesErr": ["time", "nodeTags", "strainDOFs"],
        "PorePressureAtNodes": ["time", "nodeTags"],
    },
    "resp_dof": {
        "Stresses": ["sigma11", "sigma22", "sigma12", "sigma33"],
        "Strains": ["eps11", "eps22", "eps12"],
        "StressesAtNodes": ["sigma11", "sigma22", "sigma12", "sigma33"],
        "StressAtNodesErr": ["sigma11", "sigma22", "sigma12", "sigma33"],
        "StrainsAtNodes": ["eps11", "eps22", "eps12"],
        "StrainsAtNodesErr": ["eps11", "eps22", "eps12"],
        "PorePressureAtNodes": None,
    },
}

_SOLID_RESPS_HELP = {
    "resp_type": [
        "Stresses",
        "Strains",
        "StressesAtNodes",
        "StressAtNodesErr",
        "StrainsAtNodes",
        "StrainsAtNodesErr",
        "PorePressureAtNodes",
    ],
    "resp_dim": {
        "Stresses": ["time", "eleTags", "GaussPoints", "stressDOFs"],
        "Strains": ["time", "eleTags", "GaussPoints", "strainDOFs"],
        "StressesAtNodes": ["time", "nodeTags", "stressDOFs"],
        "StressAtNodesErr": ["time", "nodeTags", "stressDOFs"],
        "StrainsAtNodes": ["time", "nodeTags", "strainDOFs"],
        "StrainsAtNodesErr": ["time", "nodeTags", "strainDOFs"],
        "PorePressureAtNodes": ["time", "nodeTags"],
    },
    "resp_dof": {
        "Stresses": ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23"],
        "Strains": ["eps11", "eps22", "eps33", "eps12", "eps23", "eps13"],
        "StressesAtNodes": ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23"],
        "StressAtNodesErr": ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23"],
        "StrainsAtNodes": ["eps11", "eps22", "eps33", "eps12", "eps23", "eps13"],
        "StrainsAtNodesErr": ["eps11", "eps22", "eps33", "eps12", "eps23", "eps13"],
        "PorePressureAtNodes": None,
    },
}

_SHELL_RESPS_HELP = {
    "resp_type": [
        "sectionForces",
        "sectionDeformations",
        "Stresses",
        "Strains",
        "sectionForcesAtNodes",
        "sectionDeformationsAtNodes",
        "StressesAtNodes",
        "StrainsAtNodes",
    ],
    "resp_dim": {
        "sectionForces": ["time", "eleTags", "GaussPoints", "secDOFs"],
        "sectionDeformations": ["time", "eleTags", "GaussPoints", "secDOFs"],
        "Stresses": ["time", "eleTags", "GaussPoints", "fiberPoints", "stressDOFs"],
        "Strains": ["time", "eleTags", "GaussPoints", "fiberPoints", "stressDOFs"],
        "sectionForcesAtNodes": ["time", "nodeTags", "secDOFs"],
        "sectionDeformationsAtNodes": ["time", "nodeTags", "secDOFs"],
        "StressesAtNodes": ["time", "nodeTags", "fiberPoints", "stressDOFs"],
        "StrainsAtNodes": ["time", "nodeTags", "fiberPoints", "stressDOFs"],
    },
    "resp_dof": {
        "sectionForces": ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"],
        "sectionDeformations": ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"],
        "Stresses": ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"],
        "Strains": ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"],
        "sectionForcesAtNodes": ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"],
        "sectionDeformationsAtNodes": ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"],
        "StressesAtNodes": ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"],
        "StrainsAtNodes": ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"],
    },
}

_LINK_RESPS_HELP = {
    "resp_type": ["basicDeformation", "basicForce"],
    "resp_dim": {
        "basicDeformation": ["time", "eleTags", "DOFs"],
        "basicForce": ["time", "eleTags", "DOFs"],
    },
    "resp_dof": {
        "basicDeformation": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
        "basicForce": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
    },
}

_CONTACT_RESPS_HELP = {
    "resp_type": ["globalForces", "localForces", "localDisp", "slips"],
    "resp_dim": {
        "globalForces": ["time", "eleTags", "globalDOFs"],
        "localForces": ["time", "eleTags", "localDOFs"],
        "localDisp": ["time", "eleTags", "localDOFs"],
        "slips": ["time", "eleTags", "slipDOFs"],
    },
    "resp_dof": {
        "globalForces": ["Px", "Py", "Pz"],
        "localForces": ["N", "Tx", "Ty"],
        "localDisp": ["N", "Tx", "Ty"],
        "slips": ["Tx", "Ty"],
    },
}

_ELE_RESP_HELPS = {
    "frame": _FRAME_ELE_RESP_HELPS,
    "truss": _TRUSS_ELE_RESP_HELPS,
    "fibersection": _FIBER_SEC_RESP_HELPS,
    "plane": _PLANE_RESPS_HELP,
    "solid": _SOLID_RESPS_HELP,
    "shell": _SHELL_RESPS_HELP,
    "link": _LINK_RESPS_HELP,
    "contact": _CONTACT_RESPS_HELP,
}


def get_nodal_responses_info(print_help: bool = True) -> dict:
    """Get nodal response types, dimensions, and DOFs.

    Added in opstool v1.0.25.

    Parameters
    ----------
    print_help : bool, optional
        print help info, by default True

    Returns
    -------
    dict
        A dictionary containing response types, dimensions, and DOFs.
    """
    if print_help:
        print("Nodal Responses:")
        print("Available Response Types (resp_type):")
        for resp_type in _NODAL_RESP_HELPS["resp_type"]:
            print(f"  - {resp_type}")
            dims = _NODAL_RESP_HELPS["resp_dim"].get(resp_type, [])
            dofs = _NODAL_RESP_HELPS["resp_dof"].get(resp_type, [])
            print(f"    resp_dim: {dims}")
            print(f"    resp_dof: {dofs}")
        print()
    return _NODAL_RESP_HELPS


def get_element_responses_info(
    ele_type: Literal["Frame", "Truss", "FiberSection", "Plane", "Solid", "Shell", "Link", "Contact"],
    print_help: bool = True,
) -> dict:
    """
    Return a dictionary of response types, dimensions, and DOFs
    for the given element type.

    Added in opstool v1.0.25.

    Parameters
    ----------
    ele_type : str
        The type of element (e.g., "frame", "truss", "fibersection", "plane", "solid", "shell", "link", "contact").
    print_help : bool, optional
        If True, prints the available response types and their dimensions/DOFs. Default is True.

    Returns
    -------
    dict
        A dictionary containing response types, dimensions, and DOFs for the specified element type.
    """
    element_type = ele_type.lower()

    data = _ELE_RESP_HELPS.get(element_type)
    if data is None:
        raise ValueError(f"Unsupported element type: {ele_type}. Supported types are: {list(_ELE_RESP_HELPS.keys())}")  # noqa: TRY003

    if print_help:
        print(f"ele_type: {ele_type}")
        print("Available Response Types (resp_type):")
        for resp_type in data["resp_type"]:
            print(f"  - {resp_type}")
            dims = data["resp_dim"].get(resp_type, [])
            dofs = data["resp_dof"].get(resp_type, [])
            print(f"    resp_dim: {dims}")
            print(f"    resp_dof: {dofs}")
        print()

    return data


# -- End of opstool/post/_post_utils.py --#
