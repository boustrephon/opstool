from __future__ import annotations

from collections.abc import Iterable


class frictionModel:
    """
    Class for friction models in OpenSees.
    The frictionModel command is used to construct a friction model object,
    which specifies the behavior of the coefficient of friction in terms of the absolute sliding velocity and the pressure on the contact area.
    """

    _ops = None
    _ndm = None
    _ndf = None

    @classmethod
    def Coulomb(cls, tag: int, mu: float) -> None | int:
        """
        Coulomb friction model.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the friction model.
        mu : float
            Coefficient of friction.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(mu)]
        return cls._ops.frictionModel("Coulomb", *args)

    @classmethod
    def VelDependent(cls, tag: int, muSlow: float, muFast: float, transRate: float) -> None | int:
        """
        Velocity-dependent friction model.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the friction model.
        muSlow : float
            Coefficient of friction at low velocity.
        muFast : float
            Coefficient of friction at high velocity.
        transRate : float
            Transition rate from low to high velocity


        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(muSlow), float(muFast), float(transRate)]
        return cls._ops.frictionModel("VelDependent", *args)

    @classmethod
    def VelPressureDep(
        cls, tag: int, muSlow: float, muFast: float, A: float, deltaMu: float, alpha: float, transRate: float
    ) -> None | int:
        """
        Velocity and pressure-dependent friction model.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the friction model.
        muSlow : float
            Coefficient of friction at low velocity.
        muFast : float
            Coefficient of friction at high velocity.
        A : float
            Nominal contact area.
        deltaMu : float
            Pressure parameter calibrated from experimental data.
        alpha : float
            Pressure parameter calibrated from experimental data.
        transRate : float
            Transition rate from low to high velocity.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag), float(muSlow), float(muFast), float(A), float(deltaMu), float(alpha), float(transRate)]
        return cls._ops.frictionModel("VelPressureDep", *args)

    @classmethod
    def VelDepMultiLinear(
        cls, tag: int, velocityPoints: Iterable[float], frictionPoints: Iterable[float]
    ) -> None | int:
        """
        This command is used to construct a VelDepMultiLinear friction model object.
        The friction-velocity relationship is given by a multi-linear curve that is define by a set of points.
        The slope given by the last two specified points on the positive velocity axis is extrapolated to infinite positive velocities.
        Velocity and friction points need to be equal or larger than zero (no negative values should be defined).
        The number of provided velocity points needs to be equal to the number of provided friction points.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the friction model.
        velocityPoints : Iterable[float]
            A list of velocity points defining the friction-velocity curve.
        frictionPoints : Iterable[float]
            A list of friction points defining the friction-velocity curve.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag)]
        args += ["-vel"] + [float(v) for v in velocityPoints]
        args += ["-frn"] + [float(f) for f in frictionPoints]
        return cls._ops.frictionModel("VelDepMultiLinear", *args)

    @classmethod
    def VelNormalFrcDep(
        cls,
        tag: int,
        aSlow: float,
        nSlow: float,
        aFast: float,
        nFast: float,
        alpha0: float,
        alpha1: float,
        alpha2: float,
        maxMuFactor: float,
    ):
        """
        Velocity and Normal Force Dependent Friction Model.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the friction model.
        aSlow : float
            Constant for coefficient of friction at low velocity
        nSlow : float
            Exponent for coefficient of friction at low velocity
        aFast : float
            Constant for coefficient of friction at high velocity
        nFast : float
            Exponent for coefficient of friction at high velocity
        alpha0 : float
            Constant rate parameter coefficient
        alpha1 : float
            Linear rate parameter coefficient
        alpha2 : float
            Quadratic rate parameter coefficient
        maxMuFactor : float
            factor for determining the maximum coefficient of friction.
            This value prevents the friction coefficient from exceeding an unrealistic maximum value when the normal force becomes very small.
            The maximum friction coefficient is determined from μFast, for example μ ≤ $maxMuFactor*μFast.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [
            int(tag),
            float(aSlow),
            float(nSlow),
            float(aFast),
            float(nFast),
            float(alpha0),
            float(alpha1),
            float(alpha2),
            float(maxMuFactor),
        ]
        return cls._ops.frictionModel("VelNormalFrcDep", *args)
