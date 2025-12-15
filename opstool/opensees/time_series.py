from __future__ import annotations

from collections.abc import Iterable


class timeSeries:
    r"""
    Class for time series in OpenSees.
    This command is used to construct a TimeSeries object which represents the relationship between the time in the domain, :math:`t`,
    and the load factor applied to the loads, :math:`\lambda`,
    in the load pattern with which the TimeSeries object is associated, i.e. :math:`\lambda = F(t)`
    """

    _ops = None
    _ndm = None
    _ndf = None

    @classmethod
    def Constant(cls, tag: int, factor: float = 1.0) -> None | int:
        """
        Constant time series.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        factor : float, optional
            Constant factor, by default 1.0

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), "-factor", float(factor)]
        return cls._ops.timeSeries("Constant", *args)

    @classmethod
    def Linear(cls, tag: int, factor: float = 1.0) -> None | int:
        """
        Linear time series.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        factor : float, optional
            Slope of the linear time series, by default 1.0

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), "-factor", float(factor)]
        return cls._ops.timeSeries("Linear", *args)

    @classmethod
    def Path(
        cls,
        tag: int,
        dt: float | None = None,
        times: Iterable[float] | None = None,
        fileTime: str | None = None,
        values: Iterable[float] | None = None,
        fileValues: str | None = None,
        fileNmae: str | None = None,
        factor: float | None = None,
        useLast: bool | None = None,
        prependZero: bool | None = None,
        startTime: float | None = None,
    ) -> None | int:
        """
        User-defined path time series.

        Four different ways to define the time series are supported:
        1. By specifying a time interval `dt` and a list of `values`.
        2. By specifying a time interval `dt` and a file path `fileValues` to read the values from.
        3. By specifying a list of `times` and a list of `values`.
        4. By specifying file paths `fileTime` and `fileValues` to read the time points and values from.
        5. By specifying a single file path `fileNmae` to read both time points and values from.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        dt : float, optional
            Time interval between values, by default None.
        times : Iterable[float], optional
            A list of time points, by default None.
        fileTime : str, optional
            File path to read time points from, by default None.
        values : Iterable[float], optional
            A list of values, by default None.
        fileValues : str, optional
            File path to read values from, by default None.
        fileNmae : str, optional
            File path to read both time points and values from, by default None.
        factor : float, optional
            Scale factor for the time series, by default None.
        useLast : bool, optional
            Whether to use the last value for times beyond the last defined time, by default None.
            Only useful for ways ``1, 2, 5``.
        prependZero : bool, optional
            Whether to prepend a zero value to the series of load factors, by default None.
            Only useful for ways ``1, 2``.
        startTime : float, optional
            Time to start the time series, by default None.
            Only useful for ways ``1, 2``, i.e., when `dt` is specified.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag)]
        if dt is not None and values is not None:
            args += ["-dt", float(dt)] + ["-values"] + [float(v) for v in values]
        elif dt is not None and fileValues is not None:
            args += ["-dt", float(dt), "-filePath", str(fileValues)]
        elif times is not None and values is not None:
            args += ["-time"] + [float(t) for t in times] + ["-values"] + [float(v) for v in values]
        elif fileTime is not None and fileValues is not None:
            args += ["-fileTime", str(fileTime), "-filePath", str(fileValues)]
        elif fileNmae is not None:
            args += ["-file", str(fileNmae)]
        else:
            raise ValueError("Invalid combination of time and value inputs.")  # noqa: TRY003
        if useLast:
            args += ["-useLast"]
        if prependZero:
            args += ["-prependZero"]
        if startTime is not None:
            args += ["-startTime", float(startTime)]
        if factor is not None:
            args += ["-factor", float(factor)]
        return cls._ops.timeSeries("Path", *args)

    @classmethod
    def Ramp(
        cls,
        tag: int,
        tStart: float,
        tRamp: float,
        offset: float | None = None,
        smoothness: float | None = None,
        factor: float | None = None,
    ) -> None | int:
        """
        Ramp time series.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        tStart : float
            Time at which the ramp starts.
        tRamp : float
            Duration of the ramp.
        offset : float, optional
            Offset to be added to the ramp, by default None.
        smoothness : float, optional
            Smoothness factor for the ramp, by default None.
        factor : float, optional
            Scale factor for the time series, by default None.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(tStart), float(tRamp)]
        if offset is not None:
            args += ["-offset", float(offset)]
        if smoothness is not None:
            args += ["-smooth", float(smoothness)]
        if factor is not None:
            args += ["-factor", float(factor)]
        return cls._ops.timeSeries("Ramp", *args)

    @classmethod
    def Trig(
        cls,
        tag: int,
        tStart: float,
        tFinish: float,
        period: float,
        phaseShift: float | None = None,
        factor: float | None = None,
        zeroShift: float | None = None,
    ) -> None | int:
        """
        Trigonometric time series.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        tStart : float
            Time at which the trigonometric function starts.
        tFinish : float
            Time at which the trigonometric function ends.
        period : float
            Characteristic period of the trigonometric function.
        phaseShift : float, optional
            Phase shift of the trigonometric function, by default None.
        factor : float, optional
            Scale factor for the time series, by default None.
        zeroShift : float, optional
            Zero shift of the trigonometric function, by default None.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(tStart), float(tFinish), float(period)]
        if phaseShift is not None:
            args += ["-phaseShift", float(phaseShift)]
        if factor is not None:
            args += ["-factor", float(factor)]
        if zeroShift is not None:
            args += ["-zeroShift", float(zeroShift)]
        return cls._ops.timeSeries("Trig", *args)

    @classmethod
    def Rectangular(cls, tag: int, tStart: float, tFinish: float, factor: float | None = None) -> None | int:
        """
        Rectangular time series.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        tStart : float
            Time at which the rectangle starts.
        tFinish : float
            Time at which the rectangle ends.
        factor : float, optional
            Scale factor for the time series, by default None.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(tStart), float(tFinish)]
        if factor is not None:
            args += ["-factor", float(factor)]
        return cls._ops.timeSeries("Rectangular", *args)

    @classmethod
    def Pulse(
        cls,
        tag: int,
        tStart: float,
        tFinish: float,
        period: float,
        width: float | None = None,
        phaseShift: float | None = None,
        factor: float | None = None,
        zeroShift: float | None = None,
    ) -> None | int:
        """
        Pulse time series.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        tStart : float
            Time at which the pulse starts.
        tFinish : float
            Time at which the pulse ends.
        period : float
            Characteristic period of the pulse function.
        width : float, optional
            Width of the pulse, by default None.
        phaseShift : float, optional
            Phase shift of the pulse function, by default None.
        factor : float, optional
            Scale factor for the time series, by default None.
        zeroShift : float, optional
            Zero shift of the pulse function, by default None.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(tStart), float(tFinish), float(period)]
        if width is not None:
            args += ["-width", float(width)]
        if phaseShift is not None:
            args += ["-phaseShift", float(phaseShift)]
        if factor is not None:
            args += ["-factor", float(factor)]
        if zeroShift is not None:
            args += ["-zeroShift", float(zeroShift)]
        return cls._ops.timeSeries("Pulse", *args)

    @classmethod
    def Triangle(
        cls,
        tag: int,
        tStart: float,
        tFinish: float,
        period: float,
        phaseShift: float | None = None,
        factor: float | None = None,
        zeroShift: float | None = None,
    ) -> None | int:
        """
        Triangle time series.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        tStart : float
            Time at which the triangle starts.
        tFinish : float
            Time at which the triangle ends.
        period : float
            Characteristic period of the triangle function.
        phaseShift : float, optional
            Phase shift of the triangle function, by default None.
        factor : float, optional
            Scale factor for the time series, by default None.
        zeroShift : float, optional
            Zero shift of the triangle function, by default None.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(tStart), float(tFinish), float(period)]
        if phaseShift is not None:
            args += ["-phaseShift", float(phaseShift)]
        if factor is not None:
            args += ["-factor", float(factor)]
        if zeroShift is not None:
            args += ["-zeroShift", float(zeroShift)]
        return cls._ops.timeSeries("Triangle", *args)

    @classmethod
    def MPAcc(
        cls,
        tag: int,
        tStart: float,
        tFinish: float,
        period: float,
        AFactor: float | None = None,
        gammaMP: float | None = None,
        nuMP: float | None = None,
    ) -> None | int:
        """
        Multi-point acceleration time series.

        $tStart $tFinish $period <-AFactor> <-gammaMP> <-nuMP>

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        tStart : float
            Time at which the time series starts.
        tFinish : float
            Time at which the time series ends.
        period : float
            Characteristic period of the time series.
        AFactor : float, optional
            The M&P velocity amplificarion factor(optional,default=1.0).
        gammaMP : float, optional
            Factor in M&P pulse model, by default None.
        nuMP : float, optional
            Nu in degree in M&P pulse model., by default None.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(tStart), float(tFinish), float(period)]
        if AFactor is not None:
            args += ["-AFactor", float(AFactor)]
        if gammaMP is not None:
            args += ["-gammaMP", float(gammaMP)]
        if nuMP is not None:
            args += ["-nuMP", float(nuMP)]
        return cls._ops.timeSeries("MPAcc", *args)

    @classmethod
    def DiscretizedRandomProcess(
        cls,
        tag: int,
        mean: float,
        maxStdv: float,
        modTags: Iterable[int],
    ) -> None | int:
        """
        Discretized random process time series.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        mean : float
            Mean value of the random process.
        maxStdv : float
            Maximum standard deviation of the random process.
        modTags : Iterable[int]
            List of module tags defining the modulating function.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(mean), float(maxStdv)] + [int(mt) for mt in modTags]
        return cls._ops.timeSeries("DiscretizedRandomProcess", *args)

    @classmethod
    def SimulatedRandomProcess(
        cls,
        tag: int,
        spectrumTag: int,
        mean: float,
        NfreqIntervals: int,
    ) -> None | int:
        """
        Simulated random process time series.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the time series.
        spectrumTag : int
            Tag of the power spectral density function.
        mean : float
            Mean value of the random process.
        NfreqIntervals : int
            Number of frequency intervals used in the simulation.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), int(spectrumTag), float(mean), int(NfreqIntervals)]
        return cls._ops.timeSeries("SimulatedRandomProcess", *args)

    Sine = Trig
    Series = Path
