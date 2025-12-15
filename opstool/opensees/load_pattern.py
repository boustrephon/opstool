from __future__ import annotations

from collections.abc import Iterable
from typing import Literal


class pattern:
    """Load Pattern Commands in OpenSees."""

    _ops = None
    _ndm = None
    _ndf = None

    @classmethod
    def Plain(
        cls,
        patternTag: int,
        tsTag: int,
        factor: float | None = None,
    ) -> int | None:
        """This commnand allows the user to construct a _Plain_ _LoadPattern_ object.
        Each _plain_ load pattern is associated with a TimeSeries object and can contain multiple NodalLoads, ElementalLoads and SP_Constraint objects.

        Parameters
        ----------
        patternType : Literal["Plain"]
            Pattern type, here is _Plain_.
        patternTag : int
            Unique tag among load patterns.
        tsTag : int
            The tag of the time series to be used in the load pattern
        factor : Optional[float], optional
            Scaling factor for the load pattern, by default None

        Returns
        -------
        Union[int, None]
            O if successful, None otherwise.
        """
        args = ["Plain", int(patternTag), int(tsTag)]
        if factor is not None:
            args += ["-factor", float(factor)]
        return cls._ops.pattern(*args)

    @classmethod
    def UniformExcitation(
        cls,
        patternTag: int,
        direction: Literal[1, 2, 3, 4, 5, 6],
        dispSeriesTag: int | None,
        velSeriesTag: int | None,
        accelSeriesTag: int | None,
        initialVel: float | None = None,
        factor: float | None = None,
        integrator: Literal["Trapezoidal", "Simpson"] | None = None,
    ) -> int | None:
        """The UniformExcitation pattern allows the user to apply a *uniform excitation* to a model acting in a certain direction.

        .. note::
            * The responses obtained from the nodes for this type of excitation are **RELATIVE** values, and not the absolute values obtained from a multi-support case.
            * Must set one of the disp, vel or accel time series

        Parameters
        ----------
        patternType : Literal["UniformExcitation"]
            Pattern type, here is _UniformExcitation_.
        patternTag : int
            Unique tag among load patterns.
        direction : Literal[1, 2, 3, 4, 5, 6]
            Direction in which ground motion acts:
                * 1--corresponds to translation along the global X axis
                * 2--corresponds to translation along the global Y axis
                * 3--corresponds to translation along the global Z axis
                * 4--corresponds to rotation about the global X axis
                * 5--corresponds to rotation about the global Y axis
                * 6--corresponds to rotation about the global Z axis
        dispSeriesTag : Optional[int]
            Tag of the TimeSeries series defining the displacement history
        velSeriesTag : Optional[int]
            Tag of the TimeSeries series defining the velocity history
        accelSeriesTag : Optional[int]
            Tag of the TimeSeries series defining the acceleration history
        initialVel : Optional[float]
            Initial velocity, DEFAULT = 0.0
        factor : Optional[float], optional
            Scaling factor for the load pattern, by default 1.0.
        integrator : Optional[Literal["Trapezoidal", "Simpson"]], optional
            Type of time integrator to be used, by default "Trapezoidal".

        Returns
        -------
        Union[int, None]
            O if successful, None otherwise.
        """
        if dispSeriesTag is None and velSeriesTag is None and accelSeriesTag is None:
            raise ValueError("At least one of dispSeriesTag, velSeriesTag, or accelSeriesTag must be provided.")  # noqa: TRY003
        args = [
            "UniformExcitation",
            int(patternTag),
            int(direction),
        ]
        if dispSeriesTag is not None:
            args += ["-disp", int(dispSeriesTag)]
        if velSeriesTag is not None:
            args += ["-vel", int(velSeriesTag)]
        if accelSeriesTag is not None:
            args += ["-accel", int(accelSeriesTag)]
        if initialVel is not None:
            args += ["-initialVel", float(initialVel)]
        if factor is not None:
            args += ["-factor", float(factor)]
        if integrator is not None:
            args += ["-integrator", integrator]
        return cls._ops.pattern(*args)

    @classmethod
    def MultipleSupport(
        cls,
        patternTag: int,
    ) -> int | None:
        """The Multi-Support pattern allows similar or different prescribed ground motions to be input at various supports in the structure.
        In OpenSees, the prescribed motion is applied using single-point constraints,
        the single-point constraints taking their constraint value from user created ground motions.

        .. note::
            * The results for the responses at the nodes are the **ABSOLUTE** values, and not relative values as in the case of a UniformExcitation.
            * The non-homogeneous single point constraints require an appropriate choice of constraint handler.

        Parameters
        ----------
        patternType : Literal["MultipleSupport"]
            Pattern type, here is _MultipleSupport_.
        patternTag : int
            Unique tag among load patterns.

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = ["MultipleSupport", int(patternTag)]
        return cls._ops.pattern(*args)


class eleLoad:
    """Element Load Commands in OpenSees."""

    _ops = None
    _ndm = None
    _ndf = None

    @classmethod
    def _parse_ele_tags_and_range(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
    ) -> list:
        if eleTags is None and eleRange is None:
            raise ValueError("Either eleTags or eleRange must be provided.")  # noqa: TRY003
        args = []
        if eleTags is not None:
            args += ["-ele"] + [int(etag) for etag in eleTags]
        if eleRange is not None:
            if len(eleRange) != 2:
                raise ValueError("eleRange must have exactly two elements: [startTag, endTag].")  # noqa: TRY003
            args += ["-range"] + [int(etag) for etag in eleRange]
        return args

    @classmethod
    def beamUniform(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
        wy: float = 0.0,
        wz: float = 0.0,
        wx: float = 0.0,
    ) -> int | None:
        """
        Beam uniform distributed load.

        Parameters
        ----------
        loadType : Literal["beamUniform"]
            Type of elemental load to be created. Here is _beamUniform_.
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None.
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None.
        wy : float, optional
            Load intensity in the local Y direction, by default 0.0.
        wz : float, optional
            Load intensity in the local Z direction, by default 0.0.
        wx : float, optional
            Load intensity in the local X direction, by default 0.0.

        ..  note::
            * For 2D models, only `wy` and `wx` are used.
            * For 3D models, `wy`, `wz`, and `wx` are used.

        Returns
        -------
        Union[int, None]
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-beamUniform"]
        if cls._ndm == 3:
            args += [float(wy), float(wz), float(wx)]
        elif cls._ndm == 2:
            args += [float(wy), float(wx)]
        else:
            raise ValueError("beamUniform load is only applicable for 2D and 3D models.")  # noqa: TRY003
        return cls._ops.eleLoad(*args)

    @classmethod
    def beamPartialUniform(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
        wya: float = 0.0,
        wza: float = 0.0,
        wxa: float = 0.0,
        wyb: float | None = None,
        wzb: float | None = None,
        wxb: float | None = None,
        aL: float = 0.0,
        bL: float = 1.0,
    ) -> int | None:
        """Create a beam partial uniform distributed load.

        Parameters
        ----------
        loadType : Literal["beamPartialUniform"]
            Type of elemental load to be created. Here is _beamPartialUniform_.
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None.
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None.
        wya : int, optional
            Load intensity in the local Y direction at the start of the element, by default 0.0
        wza : int, optional
            Load intensity in the local Z direction at the start of the element, by default 0.0
        wxa : int, optional
            Load intensity in the local X direction at the start of the element, by default 0.0
        wyb : Optional[int], optional
            Load intensity in the local Y direction at the end of the element, by default None
        wzb : Optional[int], optional
            Load intensity in the local Z direction at the end of the element, by default None
        wxb : Optional[int], optional
            Load intensity in the local X direction at the end of the element, by default None
        aL : int, optional
            Start location of the load as a fraction of the element length, by default 0.0
        bL : int, optional
            End location of the load as a fraction of the element length, by default 1.0

        ..  note::
            * For 2D models, only `wya`, `wxa`, `wyb`, `wxb`, `aL`, and `bL` are used.
            * For 3D models, all parameters are used.

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-beamUniform"]
        if wyb is None:
            wyb = wya
        if wzb is None:
            wzb = wza
        if wxb is None:
            wxb = wxa
        if cls._ndm == 3:
            args += [
                float(wya),
                float(wza),
                float(wxa),
                float(aL),
                float(bL),
                float(wyb),
                float(wzb),
                float(wxb),
            ]
        elif cls._ndm == 2:
            args += [
                float(wya),
                float(wxa),
                float(aL),
                float(bL),
                float(wyb),
                float(wxb),
            ]
        else:
            raise ValueError("beamPartialUniform load is only applicable for 2D and 3D models.")  # noqa: TRY003
        return cls._ops.eleLoad(*args)

    @classmethod
    def beamPoint(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
        py: float = 0.0,
        pz: float = 0.0,
        px: float = 0.0,
        aL: float = 0.0,
    ) -> int | None:
        """Create a beam point load.

        Parameters
        ----------
        loadType : Literal["beamPoint"]
            Type of elemental load to be created. Here is _beamPoint_.
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None
        py : float, optional
            Point load in the local Y direction, by default 0.0
        pz : float, optional
            Point load in the local Z direction, by default 0.0
        px : float, optional
            Point load in the local X direction, by default 0.0
        aL : float, optional
            Location of the point load as a fraction of the element length, by default 0.0

        ..  note::
            * For 2D models, only `py`, `px`, and `aL` are used.
            * For 3D models, `py`, `pz`, `px`, and `aL` are used.

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-beamPoint"]
        if cls._ndm == 2:
            args += [float(py), float(aL), float(px)]
        elif cls._ndm == 3:
            args += [float(py), float(pz), float(aL), float(px)]
        else:
            raise ValueError("beamPoint load is only applicable for 2D and 3D models.")  # noqa: TRY003
        return cls._ops.eleLoad(*args)

    @classmethod
    def BrickW(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
    ) -> int | None:
        """Create a BrickW load.

        Parameters
        ----------
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-BrickW"]
        return cls._ops.eleLoad(*args)

    @classmethod
    def selfWeight(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
        xf: float = 0.0,
        yf: float = 0.0,
        zf: float = 0.0,
    ) -> int | None:
        """Create a selfWeight load.

        Parameters
        ----------
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None
        xf : float, optional
            Load factor in the X direction, by default 0.0
        yf : float, optional
            Load factor in the Y direction, by default 0.0
        zf : float, optional
            Load factor in the Z direction, by default 0.0

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-selfWeight", float(xf), float(yf), float(zf)]
        return cls._ops.eleLoad(*args)

    @classmethod
    def surfaceLoad(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
    ) -> int | None:
        """Create a surfaceLoad load.

        .. note::
            This command is applicable for ``SurfaceLoad`` and ``TriSurfaceLoad`` Element only.
            In other words, the actual load is applied through ``SurfaceLoad`` or ``TriSurfaceLoad``, but here it is simply added to the OpenSees domain.


        Parameters
        ----------
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-surfaceLoad"]
        return cls._ops.eleLoad(*args)

    @classmethod
    def IGAFollowerLoad(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
        xi: float = 0.0,
        eta: float = 0.0,
        xf: float = 0.0,
        yf: float = 0.0,
        zf: float = 0.0,
    ) -> int | None:
        """Create an IGAFollowerLoad load.

        Parameters
        ----------
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None
        xi : float, optional
            waiting for description, by default 0.0
        eta : float, optional
            waiting for description, by default 0.0
        xf : float, optional
            waiting for description, by default 0.0
        yf : float, optional
            waiting for description, by default 0.0
        zf : float, optional
            waiting for description, by default 0.0

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-IGAFollowerLoad", float(xi), float(eta), float(xf), float(yf), float(zf)]
        return cls._ops.eleLoad(*args)

    @classmethod
    def beamThermal(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
        data: Iterable[float] | None = None,
    ) -> int | None:
        """Create a beam thermal load.

        Parameters
        ----------
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None
        data : Optional[Iterable[float]], optional
            Temperature change data at different layers, by default None

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-beamThermal"]
        if data is not None:
            args += [float(d) for d in data]
        return cls._ops.eleLoad(*args)

    @classmethod
    def shellThermal(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
        data: Iterable[float] | None = None,
    ) -> int | None:
        """Create a shell thermal load.

        Parameters
        ----------
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None
        data : Optional[Iterable[float]], optional
            Temperature change data at different layers, by default None

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-shellThermal"]
        if data is not None:
            args += [float(d) for d in data]
        return cls._ops.eleLoad(*args)

    @classmethod
    def beamTemp(
        cls,
        eleTags: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
        data: Iterable[float] | None = None,
    ) -> int | None:
        """Create a beam temperature load.

        Parameters
        ----------
        eleTags : Optional[Iterable[int]], optional
            Element tags to which the load is applied, by default None
        eleRange : Optional[Iterable[int]], optional
            Range of element tags [startTag, endTag] to which the load is applied, by default None
        data : Optional[Iterable[float]], optional
            Temperature change data at different layers, by default None

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = cls._parse_ele_tags_and_range(eleTags, eleRange)
        args += ["-type", "-beamTemp"]
        if data is not None:
            args += [float(d) for d in data]
        return cls._ops.eleLoad(*args)


class groundMotion:
    """Ground Motion Commands in OpenSees."""

    _ops = None
    _ndm = None
    _ndf = None

    @classmethod
    def Plain(
        cls,
        gmTag: int,
        dispSeriesTag: int | None,
        velSeriesTag: int | None,
        accelSeriesTag: int | None,
        factor: float | None = None,
        integrator: Literal["Trapezoidal", "Simpson"] | None = None,
        dtIntegrator: float | None = None,
    ) -> int | None:
        """Create a ground motion of type Plain.

        Parameters
        ----------
        gmTag : int
            Tag of the ground motion.
        dispSeriesTag : Optional[int]
            Tag of the displacement time series.
        velSeriesTag : Optional[int]
            Tag of the velocity time series.
        accelSeriesTag : Optional[int]
            Tag of the acceleration time series.
        factor : Optional[float], optional
            Factor applied to the ground motion, by default None
        integrator : Optional[Literal["Trapezoidal", "Simpson"]], optional
            Integration method to be used, by default None
        dtIntegrator : Optional[float], optional
            Time step for the integrator, by default None

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = [int(gmTag), "Plain"]
        if dispSeriesTag is not None:
            args += ["-disp", int(dispSeriesTag)]
        if velSeriesTag is not None:
            args += ["-vel", int(velSeriesTag)]
        if accelSeriesTag is not None:
            args += ["-accel", int(accelSeriesTag)]
        if factor is not None:
            args += ["-factor", float(factor)]
        if integrator is not None:
            args += ["-integrator", integrator]
        if dtIntegrator is not None:
            args += ["-dtIntegrator", float(dtIntegrator)]
        return cls._ops.groundMotion(*args)

    @classmethod
    def Interpolated(
        cls,
        gmTag: int,
        gmTagsPool: Iterable[int],
        factors: Iterable[float],
    ) -> int | None:
        """Create a ground motion of type Interpolated.

        Parameters
        ----------
        gmType : Literal["Interpolated"]
            Type of ground motion. Here is _Interpolated_.
        gmTag : int
            Tag of the ground motion.
        gmTagsPool : Iterable[int]
            Pool of ground motion tags to be interpolated.
        factors : Iterable[float]
            Factors corresponding to each ground motion tag in the pool.

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = [int(gmTag), "Interpolated"]
        args += [int(tag) for tag in gmTagsPool]
        args += ["-fact"] + [float(factor) for factor in factors]
        return cls._ops.groundMotion(*args)
