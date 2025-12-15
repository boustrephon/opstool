from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType


class uniaxialMaterial:
    """
    Typed wrapper around `ops.uniaxialMaterial`.

    Example
    -------
    >>> import openseespy.opensees as ops
    >>> mat = UniaxialMaterial(ops)
    >>> mat.Elastic(tag=1, E=2.0e5)
    >>> mat.Steel01(tag=2, Fy=345.0, E0=2.0e5, b=0.01)
    """

    def __init__(self, ops_module: ModuleType) -> int | None:
        self._ops = ops_module

    def Elastic(self, tag: int, E: float, eta: float | None = None, Eneg: float | None = None) -> int:
        """
        Elastic uniaxial material.

        See `Elastic Material <https://opensees.berkeley.edu/wiki/index.php?title=Elastic_Uniaxial_Material`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        E : float
            Elastic modulus.
        eta : float | None, optional
            Viscous damping coefficient (default is 0.0).
        Eneg : float | None, optional
            Negative elastic modulus (default is E).

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(E)]
        if eta is not None:
            args.append(float(eta))
        if Eneg is not None:
            args.append(float(Eneg))
        return self._ops.uniaxialMaterial("Elastic", *args)

    def ElasticPP(
        self, tag: int, E: float, epsyPos: float, epsyNeg: float | None = None, epsIni: float | None = None
    ) -> int | None:
        """
        Elastic perfectly plastic uniaxial material.

        See `ElasticPP Material <https://opensees.berkeley.edu/wiki/index.php?title=Elastic-Perfectly_Plastic_Material>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        E : float, Positive
            Elastic modulus.
        epsyPos : float
            Positive yield strain.
        epsyNeg : float | None, optional
            Negative yield strain.
            If not provided, assumed to be equal in magnitude to `epsyPos`.
        epsIni : float | None, optional
            Initial strain.
            If not provided, assumed to be zero.

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(E), float(epsyPos)]
        if epsyNeg is not None:
            args.append(float(epsyNeg))
        if epsIni is not None:
            args.append(float(epsIni))
        return self._ops.uniaxialMaterial("ElasticPP", *args)

    def ElasticPPGap(
        self, tag: int, E: float, Fy: float, gap: float, eta: float | None = None, damage: bool | None = None
    ) -> int | None:
        """
        Elastic perfectly plastic uniaxial material with gap.

        See `ElasticPPGap Material <https://opensees.berkeley.edu/wiki/index.php?title=Elastic-Perfectly_Plastic_Gap_Material>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        E : float, Positive
            Elastic modulus.
        Fy : float
            Stress or force at which material reaches plastic state
        gap : float
            Gap length.
            Fy and gap must have the same units.
        eta : float | None, optional
            Hardening ratio (=Eh/E), which can be negative.
            Value of eta must be < 1.0.
        damage : bool | None, optional
            Damage flag (default is False).

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(E), float(Fy), float(gap)]
        if eta is not None:
            args.append(float(eta))
        if damage is not None and damage:
            args.append("-damage")
        return self._ops.uniaxialMaterial("ElasticPPGap", *args)

    def Parallel(self, tag: int, matTags: Iterable[int], factors: Iterable[float] | None = None) -> int | None:
        """
        Parallel uniaxial material.

        See `Parallel Material <https://opensees.berkeley.edu/wiki/index.php?title=Parallel_Material>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        matTags : Iterable[int]
            Tags of the uniaxial materials to be combined in parallel.
        factors : Iterable[float] | None, optional
            Scaling factors for each material.
            If not provided, all factors are assumed to be 1.0.

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag)] + [int(mat_tag) for mat_tag in matTags]
        if factors is not None:
            args += [float(factor) for factor in factors]
        return self._ops.uniaxialMaterial("Parallel", *args)

    def Series(self, tag: int, matTags: Iterable[int], iterParas: Iterable[int, float] | None = None) -> int | None:
        """
        Series uniaxial material.

        See `Series Material <https://opensees.berkeley.edu/wiki/index.php?title=Series_Material>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        matTags : Iterable[int]
            Tags of the uniaxial materials to be combined in series.
        iterParas : Iterable[float] | None, optional
            Iteration parameters, [maxIter, tol], i.e., maximum number of iterations and tolerance.
            If not provided, default values are used.

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag)] + [int(mat_tag) for mat_tag in matTags]
        if iterParas is not None:
            if len(iterParas) != 2:
                raise ValueError("iterParas must contain exactly two elements: [maxIter, tol]")  # noqa: TRY003
            args += [int(iterParas[0]), float(iterParas[1])]
        return self._ops.uniaxialMaterial("Series", *args)

    def ENT(self, tag: int, E: float, a: float | None = None, b: float | None = None) -> int | None:
        """
        This command is used to construct a uniaxial *elastic-no-tension* material object.

        See `ENT Material <https://opensees.berkeley.edu/wiki/index.php?title=Elastic-No_Tension_Material>`_ for more details.

        Under compression i.e., strain < 0, it behaves elastically such that
        stress = E*strain

        Under tension, it exhibits the following stress = a*E*(tanh(strain*b)) and tangent = a*E*b*(1-tanh(strain*b)*tanh(strain*b));

        Parameters
        ----------
        tag : int
            Material tag.
        E : float
            Elastic modulus.
        a : float, optional
            Parameter a for tension behavior.
            Default is 0.0.
        b : float, optional
            Parameter b for tension behavior.
            Default is 1.0.

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(E)]
        if a is not None:
            args.append(float(a))
        if b is not None:
            args.append(float(b))
        return self._ops.uniaxialMaterial("ENT", *args)

    def Steel01(
        self,
        tag: int,
        fy: float,
        E: float,
        b: float,
        a1: float | None = None,
        a2: float | None = None,
        a3: float | None = None,
        a4: float | None = None,
    ) -> int | None:
        """
        This command is used to construct a uniaxial Steel01 material object.

        See `Steel01 Material <https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/Steel01.html>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        fy : float
            Yield stress.
        E : float
            Initial elastic tangent.
        b : float
            Strain hardening ratio.
        a1 : float, optional
            Coefficient for isotropic hardening (default is 0.0).
        a2 : float, optional
            Coefficient for isotropic hardening (default is 55.0).
        a3 : float, optional
            Coefficient for kinematic hardening (default is 0.0).
        a4 : float, optional
            Coefficient for kinematic hardening (default is 55.0).

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(fy), float(E), float(b)]
        if a1 is not None and a2 is not None and a3 is not None and a4 is not None:
            args.append(float(a1))
            args.append(float(a2))
            args.append(float(a3))
            args.append(float(a4))
        return self._ops.uniaxialMaterial("Steel01", *args)

    def Steel02(
        self,
        tag: int,
        fy: float,
        E: float,
        b: float,
        R0: float = 15.0,
        cR1: float = 0.925,
        cR2: float = 0.15,
        a1: float = 0.0,
        a2: float = 1.0,
        a3: float = 0.0,
        a4: float = 1.0,
        sigInit: float = 0.0,
    ) -> int | None:
        """
        This command is used to construct a uniaxial Steel02 material object.

        See `Steel02 Material <https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/Steel02.html>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        fy : float
            Yield stress.
        E : float
            Initial elastic tangent.
        b : float
            Strain hardening ratio.
        R0 : float, optional
            Initial value of the curvature parameter (default is 15.0).
        cR1 : float, optional
            Coefficient for the curvature parameter (default is 0.925).
        cR2 : float, optional
            Coefficient for the curvature parameter (default is 0.15).
        a1 : float, optional
            Coefficient for isotropic hardening (default is 0.0).
        a2 : float, optional
            Coefficient for isotropic hardening (default is 1.0).
        a3 : float, optional
            Coefficient for kinematic hardening (default is 0.0).
        a4 : float, optional
            Coefficient for kinematic hardening (default is 1.0).
        sigInit : float, optional
            Initial stress (default is 0.0).

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(fy), float(E), float(b)]
        args += [float(R0), float(cR1), float(cR2), float(a1), float(a2), float(a3), float(a4), float(sigInit)]
        return self._ops.uniaxialMaterial("Steel02", *args)

    def Steel02Fatigue(
        self,
        tag: int,
        fy: float,
        E: float,
        b: float,
        Cd: float,
        Cf: float,
        alpha: float,
        beta: float,
        minStrain: float,
        maxStrain: float,
        R0: float = 15.0,
        cR1: float = 0.925,
        cR2: float = 0.15,
        a1: float = 0.0,
        a2: float = 1.0,
        a3: float = 0.0,
        a4: float = 1.0,
        sigInit: float = 0.0,
    ) -> int | None:
        """
        This command is used to construct a uniaxial Steel02Fatigue material object.

        Parameters
        ----------
        tag : int
            Material tag.
        fy : float
            Yield stress.
        E : float
            Initial elastic tangent.
        b : float
            Strain hardening ratio.
        Cd : float
            ...
        Cf : float
            ...
        alpha : float
            ...
        beta : float
            ...
        minStrain : float
            ...
        maxStrain : float
            ...
        R0 : float, optional
            Initial value of the curvature parameter (default is 15.0).
        cR1 : float, optional
            Coefficient for the curvature parameter (default is 0.925).
        cR2 : float, optional
            Coefficient for the curvature parameter (default is 0.15).
        a1 : float, optional
            Coefficient for isotropic hardening (default is 0.0).
        a2 : float, optional
            Coefficient for isotropic hardening (default is 1.0).
        a3 : float, optional
            Coefficient for kinematic hardening (default is 0.0).
        a4 : float, optional
            Coefficient for kinematic hardening (default is 1.0).
        sigInit : float, optional
            Initial stress (default is 0.0).

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(fy), float(E), float(b)]
        args += [float(Cd), float(Cf), float(alpha), float(beta), float(minStrain), float(maxStrain)]
        args += [float(R0), float(cR1), float(cR2)]
        args += [float(a1), float(a2), float(a3), float(a4), float(sigInit)]
        return self._ops.uniaxialMaterial("Steel02Fatigue", *args)

    def ASDSteel1D(
        self,
        tag: int,
        E: float,
        sy: float,
        su: float,
        eu: float,
        implex: bool | None = None,
        implexControl: Iterable[float] | None = None,
        autoRegularization: bool | None = None,
        bucklingControl: Iterable[float] | None = None,
        fractureControl: Iterable[float] | None = None,
        slipControl: Iterable[int, float] | None = None,
        Kalpha: float | None = None,
        maxIter: int | None = None,
        tolU: float | None = None,
        tolR: float | None = None,
    ) -> int | None:
        """The ASDSteel1D command is used to create a uniaxial material object that models the nonlinear response of reinforcing steel bars under monotonic and cyclic loading.
        The formulation integrates kinematic hardening plasticity, damage mechanics, buckling behavior and bond-slip mechanism.
        To improve numerical robustness and computational performance under severe nonlinearities an optional IMPL-EX integration scheme is available.
        Additionally, the entire model is regularized using the element’s characteristic length to guarantee mesh-independent results.

        See `ASDSteel1D Material <https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/ASDSteel1D.html>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        E : float
            Elastic modulus.
        sy : float
            Yield stress.
        su : float
            Ultimate stress.
        eu : float
            Ultimate strain.
        implex : bool | None, optional
            If true, the IMPL-EX integration will be used, otherwise the standard implicit integration will be used (default).
        implexControl : Iterable[float] | None, optional
            List of two values [implexErrorTolerance, implexTimeReductionLimit] controlling the IMPL-EX scheme (default is [0.05, 0.01]).
        autoRegularization : bool | None, optional
            If true, Activates automatic regularization based on the characteristic length of the finite element.
        bucklingControl : Iterable[float] | None, optional
            Enables buckling simulation using an RVE-based approach.
            Requires characteristic length $lch and optionally a section radius $r.
            It means [lch, r] or [lch] if r is not provided.
        fractureControl : Iterable[float] | None, optional
            Activates fracture modeling. Optionally specify the section radius $r.
            It means [r] or [] if r is not provided.
        slipControl : Iterable[int, float] | None, optional
            Activates slip modeling with a secondary uniaxial material ($matTag). Optionally specify the section radius $r.
            It means [matTag, r] or [matTag] if r is not provided.
        Kalpha : float | None, optional
            Defines the weight between the consistent elastoplastic tangent modulus and the purely elastic modulus (default = 0.5). Set to 1.0 for full consistent tangent, or 0.0 to use only the elastic modulus.
        maxIter : int | None, optional
            Maximum number of iterations for the global Newton-Raphson loop used in the RVE (default = 100).
        tolU : float | None, optional
            Tolerance on displacement increment convergence (default = 1e-6).
        tolR : float | None, optional
            Tolerance on residual force convergence (default = 1e-6).

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(E), float(sy), float(su), float(eu)]
        if implex is not None and implex:
            args.append("-implex")
        if implexControl is not None:
            args.append("-implexControl")
            args += [float(val) for val in implexControl]
        if autoRegularization is not None and autoRegularization:
            args.append("-auto_regularization")
        if bucklingControl is not None:
            args.append("-buckling")
            args += [float(val) for val in bucklingControl]
        if fractureControl is not None:
            args.append("-fracture")
            args += [float(val) for val in fractureControl]
        if slipControl is not None:
            args += ["-slip", int(slipControl[0])]
            if len(slipControl) > 1:
                args.append(float(slipControl[1]))
        if Kalpha is not None:
            args += ["-K_alpha", float(Kalpha)]
        if maxIter is not None:
            args += ["-max_iter", int(maxIter)]
        if tolU is not None:
            args += ["-tolU", float(tolU)]
        if tolR is not None:
            args += ["-tolR", float(tolR)]

        self._ops.uniaxialMaterial("ASDSteel1D", *args)

    def ASDConcrete1D(
        self,
        tag: int,
        E: float,
        fc: float | None = None,
        ft: float | None = None,
        Te: Iterable[float] | None = None,
        Ts: Iterable[float] | None = None,
        Td: Iterable[float] | None = None,
        Ce: Iterable[float] | None = None,
        Cs: Iterable[float] | None = None,
        Cd: Iterable[float] | None = None,
        implex: bool | None = None,
        implexControl: Iterable[float] | None = None,
        implexAlpha: float | None = None,
        eta: float | None = None,
        tangent: bool | None = None,
        autoRegularizationPara: float | None = None,
    ) -> int | None:
        """This command is used to construct an ASDConcrete1D material object, a plastic-damage model for concrete and masonry like materials.
        To improve robustness and convergence of the simulation in case of strain-softening, this model optionally allows to use the IMPL-EX integration scheme (a mixed IMPLicit EXplicit integration scheme).

        Parameters
        ----------
        tag : int
            _description_
        E : float
            _description_
        fc : float | None, optional
            The concrete compressive strength.
        ft : float | None, optional
            The concrete tension (rupture) strength.
        Te : Iterable[float] | None, optional
            A list of total-strain values for the tensile hardening-softening law. If not specified, $Te will be computed automatically from $fc and $ft. If specified, $Te will override $fc and $ft.
        Ts : Iterable[float] | None, optional
            A list of stress values for the tensile hardening-softening law. If not specified, $Ts will be computed automatically from $fc and $ft. If specified, $Ts will override $fc and $ft.
        Td : Iterable[float] | None, optional
            A list of damage values for the tensile hardening-softening law. If not defined, no stiffness degradation will be considered. If not specified, $Td will be computed automatically from $fc and $ft. If specified, $Td will override $fc and $ft.
        Ce : Iterable[float] | None, optional
            A list of total-strain values for the compressive hardening-softening law. If not specified, $Ce will be computed automatically from $fc and $ft. If specified, $Ce will override $fc and $ft.
        Cs : Iterable[float] | None, optional
            A list of stress values for the compressive hardening-softening law. If not specified, $Cs will be computed automatically from $fc and $ft. If specified, $Cs will override $fc and $ft.
        Cd : Iterable[float] | None, optional
            A list of damage values for the compressive hardening-softening law. If not defined, no stiffness degradation will be considered. If not specified, $Cd will be computed automatically from $fc and $ft. If specified, $Cd will override $fc and $ft.
        implex : bool | None, optional
            If True, the IMPL-EX integration will be used, otherwise the standard implicit integration will be used (default).
        implexControl : Iterable[float] | None, optional
            Activates the control of the IMPL-EX error, [implexErrorTolerance, implexTimeReductionLimit] (default is [0.05, 0.01]).
            implexErrorTolerance: Relative error tolerance.
            implexTimeReductionLimit: Minimum allowed relative reduction of the time-step.
            If the error introduced by the IMPL-EX algorithm is larger than implexErrorTolerance , the material will fail during the computation.
            The user can therfore use an adaptive time-step to reduce the time-step to keep the error under control.
            If the reduction of the time-step is smaller than implexTimeReductionLimit , the error control will be skipped.
            *Suggested values: -implexControl 0.05 0.01.*
        implexAlpha : float | None, optional
            Default = 1. The alpha coefficient for the explicit extrapolation of the internal variables in the IMPL-EX algorithm. It can range from 0 to 1.
        eta : float | None, optional
            If defined, the rate-dependent model is used (By default the model is rate-independent). -eta: Activates the rate-dependent model. eta: The viscosity parameter eta, representing the relaxation time of the viscoplastic system.
        tangent : bool | None, optional
            If True, the tangent constitutive matrix is used. By default, the secant stiffness is used.
        autoRegularizationPara : float | None, optional, ``lch_ref``
            Optional. If defined, and if the tensile and/or the compressive hardening-softening
            law has strain-softening, the area under the hardening-softening law is assumed to be
            a real fracture energy (``G_f`` with dimension :math:`F/L`), and the specific fracture
            energy :math:`g_f` (with dimension :math:`F/L^2`) is automatically computed as

            .. math::
                g_f = \frac{G_f}{l_{\\mathrm{ch}}},

            where :math:`l_{\\mathrm{ch}}` is the characteristic length of the Finite Element.

            In this case ``lch_ref`` is ``1``.
            If, instead, the area is a *specific fracture energy* (:math:`g_{f,\\mathrm{ref}}`
            with dimension :math:`F/L^2`), ``lch_ref`` should be set equal to the experimental
            size used to obtain the strain from the displacement jump.

            In this case, the regularization will be performed as

            .. math::
                g_f = \frac{G_f}{l_{\\mathrm{ch}}}
                    = g_{f,\\mathrm{ref}} \\cdot \frac{l_{\\mathrm{ch,ref}}}{l_{\\mathrm{ch}}}.

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(E)]
        if fc is not None:
            args += ["-fc", float(fc)]
        if ft is not None:
            args += ["-ft", float(ft)]
        if Te is not None:
            args += ["-Te"] + [float(val) for val in Te]
        if Ts is not None:
            args += ["-Ts"] + [float(val) for val in Ts]
        if Td is not None:
            args += ["-Td"] + [float(val) for val in Td]
        if Ce is not None:
            args += ["-Ce"] + [float(val) for val in Ce]
        if Cs is not None:
            args += ["-Cs"] + [float(val) for val in Cs]
        if Cd is not None:
            args += ["-Cd"] + [float(val) for val in Cd]
        if implex is not None and implex:
            args.append("-implex")
        if implexControl is not None:
            args.append("-implexControl")
            args += [float(val) for val in implexControl]
        if implexAlpha is not None:
            args += ["-implexAlpha", float(implexAlpha)]
        if eta is not None:
            args += ["-eta", float(eta)]
        if tangent is not None and tangent:
            args.append("-tangent")
        if autoRegularizationPara is not None:
            args += ["-autoRegularization", float(autoRegularizationPara)]
        return self._ops.uniaxialMaterial("ASDConcrete1D", *args)

    def Concrete01(
        self,
        tag: int,
        fc: float,
        ec: float,
        fcu: float,
        ecu: float,
    ) -> int | None:
        """
        This command is used to construct a uniaxial Concrete01 material object.

        See `Concrete01 Material <https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/Concrete01.html>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        fc : float
            Compressive strength.
        ec : float
            Strain at compressive strength.
        fcu : float
            Crushing strength.
        ecu : float
            Strain at crushing strength.

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(fc), float(ec), float(fcu), float(ecu)]
        return self._ops.uniaxialMaterial("Concrete01", *args)

    def Concrete02(
        self,
        tag: int,
        fc: float,
        ec: float,
        fcu: float,
        ecu: float,
        rat: float | None = None,
        ft: float | None = None,
        ets: float | None = None,
    ) -> int | None:
        """
        This command is used to construct a uniaxial Concrete02 material object.

        See `Concrete02 Material <https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/Concrete02.html>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        fc : float
            Compressive strength.
        ec : float
            Strain at compressive strength.
        fcu : float
            Crushing strength.
        ecu : float
            Strain at crushing strength.
        rat : float | None, optional
            ratio between unloading slope at $epscu and initial slope.
        ft : float | None, optional
            Tensile strength.
        ets : float | None, optional
            tension softening stiffness (absolute value) (slope of the linear tension softening branch)

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(fc), float(ec), float(fcu), float(ecu)]
        if rat is not None and ft is not None and ets is not None:
            args.extend([float(rat), float(ft), float(ets)])
        return self._ops.uniaxialMaterial("Concrete02", *args)

    def Concrete04(
        self,
        tag: int,
        fc: float,
        ec: float,
        ecu: float,
        Ec: float,
        ft: float | None = None,
        et: float | None = None,
        beta: float | None = None,
    ) -> int | None:
        """This command is used to construct a uniaxial Popovics concrete material object with degraded linear unloading/reloading stiffness according
        to the work of Karsan-Jirsa and tensile strength with exponential decay

        Parameters
        ----------
        tag : int
            Material tag.
        fc : float
            Compressive strength.
        ec : float
            Strain at compressive strength.
        ecu : float
            Concrete crushing strain.
        Ec : float
            Initial elastic modulus.
        ft : float | None, optional
            Tensile strength.
        et : float | None, optional
            Ultimate tensile strain of concrete
        beta : float | None, optional
            Exponential curve parameter to define the residual stress (as a factor of $ft) at $etu.

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(fc), float(ec), float(ecu), float(Ec)]
        if ft is not None and et is not None:
            args.extend([float(ft), float(et)])
            if beta is not None:
                args.append(float(beta))
        return self._ops.uniaxialMaterial("Concrete04", *args)

    def Hysteretic(
        self,
        tag: int,
        sPos: Iterable[float],
        ePos: Iterable[float],
        sNeg: Iterable[float],
        eNeg: Iterable[float],
        pinchX: float,
        pinchY: float,
        damfc1: float,
        damfc2: float,
        beta: float | None = None,
    ) -> int | None:
        """
        This command is used to construct a uniaxial Hysteretic material object.

        Parameters
        ----------
        tag : int
            Material tag.
        sPos : Iterable[float]
            Force or stress points of the envelope in the positive direction.
            2 or 3 elements required.
        ePos : Iterable[float]
            Deformation or strain points of the envelope in the positive direction.
            2 or 3 elements required. It's length must match that of `sPos`.
        sNeg : Iterable[float]
            Force or stress points of the envelope in the negative direction.
            2 or 3 elements required.
            It's length must match that of `sPos`.
        eNeg : Iterable[float]
            Deformation or strain points of the envelope in the negative direction.
            2 or 3 elements required.
            It's length must match that of `sPos`.
        pinchX : float
            Pinching factor for strain (or deformation) during reloading.
        pinchY : float
            Pinching factor for stress (or force) during reloading.
        damfc1 : float
            Damage due to ductility: D1(mu-1)
        damfc2 : float
            Damage due to energy: D2(Eii/Eult)
        beta : float, optional
            Power used to determine the degraded unloading stiffness based on ductility, mu-beta (optional, default=0.0).

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag)]
        args += [float(val) for pair in zip(sPos, ePos) for val in pair]
        args += [float(val) for pair in zip(sNeg, eNeg) for val in pair]
        args += [float(pinchX), float(pinchY), float(damfc1), float(damfc2)]
        if beta is not None:
            args.append(float(beta))
        return self._ops.uniaxialMaterial("Hysteretic", *args)

    def HystereticSM(
        self,
        tag: int,
        sPos: Iterable[float],
        ePos: Iterable[float],
        sNeg: Iterable[float],
        eNeg: Iterable[float],
        pinchX: float,
        pinchY: float,
        damfc1: float,
        damfc2: float,
        beta: float | None = None,
        degEnv: Iterable[float] | None = None,
        defoLimitStates: Iterable[float] | None = None,
        forceLimitStates: Iterable[float] | None = None,
        printInput: bool | None = None,
        XYorder: bool | None = None,
    ) -> int | None:
        """This command is used to construct a uniaxial multilinear hysteretic material object with pinching of force and deformation, damage due to ductility and energy, and degraded unloading stiffness based on ductility.

        - This material is an extension of the Hysteretic Material - the envelope can be defined 2,3, 4,5,6 or 7 points, while the original one only had 2 or 3.
        - The positive and negative backbone of this material do not need to have the same number of segments.
        - This material also has the option to degrade the envelope using the degEnv parameters - these parameters must be used in combination with the damage parameters
        - This material also has additional DCR-type recorder output (this is still a work in progress).

        Parameters
        ----------
        tag : int
            _description_
        sPos : Iterable[float]
            _description_
        ePos : Iterable[float]
            _description_
        sNeg : Iterable[float]
            _description_
        eNeg : Iterable[float]
            _description_
        pinchX : float
            Pinching factor for strain (or deformation) during reloading.
        pinchY : float
            Pinching factor for stress (or force) during reloading.
        damfc1 : float
            Damage due to ductility: D1(mu-1)
        damfc2 : float
            Damage due to energy: D2(Eii/Eult)
        beta : float, optional
            Power used to determine the degraded unloading stiffness based on ductility, mu-beta (optional, default=0.0).
        degEnv : Iterable[float] | None, optional
            [degEnvP] or [degEnvP, degEnvN].

            * degEnvP : envelope-degredation factor.
                This factor works with the damage parameters to degrade the POSITIVE envelope.
                A positive value degrades both strength and strain values, a negative values degrades only strength.
                The factor is applied to points 3+ (optional, default=0.0).
            * degEnvN : envelope-degredation factor.
                This factor works with the damage parameters to degrade the NEGATIVE envelope.
                A positive value degrades both strength and strain values, a negative values degrades only strength.
                The factor is applied to points 3+ (optional, default=degEnvP, if defined, =0. otherwise)
        defoLimitStates : Iterable[float] | None, optional
            List of user-defined strain/deformation limits for computing deformation DCRs
        forceLimitStates : Iterable[float] | None, optional
            List of user-defined stress/force limits for computing force DCRs (optional)
        printInput : bool | None, optional
            If true, program will output input-parameter values (optional)
        XYorder : bool | None, optional
            Invert backbone-envelope points to be strain-stress instead of stress-strain (optional).
            This flag has the same effect as using -posEnvXY and the optional -negEnvXY, so it should be used with the -posEnv and -negEnv flags., by default None

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag)]
        args += ["-posEnv"]
        args += [float(val) for pair in zip(sPos, ePos) for val in pair]
        args += ["-negEnv"]
        args += [float(val) for pair in zip(sNeg, eNeg) for val in pair]
        args += ["-pinch"]
        args += [float(pinchX), float(pinchY)]
        args += ["-damage"]
        args += [float(damfc1), float(damfc2)]
        if beta is not None:
            args += ["-beta"]
            args.append(float(beta))
        if degEnv is not None:
            args += ["-degEnv"]
            args += [float(val) for val in degEnv]
        if defoLimitStates is not None:
            args += ["-defoLimitStates"]
            args += [float(val) for val in defoLimitStates]
        if forceLimitStates is not None:
            args += ["-forceLimitStates"]
            args += [float(val) for val in forceLimitStates]
        if printInput is not None and printInput:
            args += ["-printInput"]
        if XYorder is not None and XYorder:
            args += ["-XYorder"]
        return self._ops.uniaxialMaterial("HystereticSM", *args)

    def HystereticPoly(
        self,
        tag: int,
        ka: float,
        kb: float,
        alpha: float,
        beta1: float,
        beta2: float,
        delta: float | None = None,
    ) -> int | None:
        """This command is used to construct the uniaxial HystereticPoly material producing smooth hysteretic loops and local maxima/minima. It is based on a polynomial formulation of its tangent stiffness.

        See `HystereticPoly Material <https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/HystereticPoly.html>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        ka : float
            Tangent stiffness of the initial “elastic” part of the loop.
        kb : float
            Tangent stiffness of the asymptotic part of the loop.
        alpha : float
            Parameter governing the amplitude of the loop.
        beta1 : float
            Parameter governing the shape of the asymptotic region of the loop.
        beta2 : float
            Parameter governing the shape of the asymptotic region of the loop.
        delta : float | None, optional
            Asymptotic tolerance (optional. Default 1.0e-20).

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(ka), float(kb), float(alpha), float(beta1), float(beta2)]
        if delta is not None:
            args.append(float(delta))
        return self._ops.uniaxialMaterial("HystereticPoly", *args)

    def HystereticSmooth(
        self, tag: int, ka: float, kb: float, fbar: float, beta: float, alpha: bool | None = None
    ) -> int | None:
        """
        This command is used to construct a uniaxial HystereticSmooth material object.

        See `HystereticSmooth Material <https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/HystereticSmooth.html>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        ka : float
            Tangent stiffness of the initial “elastic” part of the loop.
        kb : float
            Tangent stiffness at zero displacement/strain.
        fbar : float
            Hysteresys force/stress at zero displacement/strain.
        beta : float
            Parameter governing hardening/softening behavior.
        alpha : bool | None, optional
            If true, the 3rd parameter is “alpha” instead of “fbar”.

        Returns
        -------
        int | None
            0 if successful, None otherwise.

        """
        args = [int(tag), float(ka), float(kb), float(fbar), float(beta)]
        if alpha is not None and alpha:
            args.append("-alpha")
        return self._ops.uniaxialMaterial("HystereticSmooth", *args)

    def HystereticAsym(
        self,
        tag: int,
        ka: float,
        kb: float,
        fo: float,
        b1: float,
        b2: float,
        gamma: float,
        alpha: bool | None = None,
    ) -> int | None:
        """This command is used to construct the uniaxial HystereticAsym material proposed by Vaiana et al. [VaianaEtAl2021]. It produces smooth and asymmetric hysteretic loops with hardening-softening behavior.

        See `HystereticAsym Material <https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/HystereticAsym.html>`_ for more details.

        Parameters
        ----------
        tag : int
            Material tag.
        ka : float
            Tangent stiffness of the initial “elastic” part of the loop.
        kb : float
            Tangent stiffness at zero displacement/strain.
        fo : float
            Hysteresys force/stress at zero displacement/strain.
        b1 : float
            Parameter governing hardening/softening behavior and asymmetry.
        b2 : float
            Parameter governing hardening/softening behavior and asymmetry.
        gamma : float
            Parameter governing hardening/softening behavior and asymmetry.
        alpha : bool | None, optional
            If true, the 3rd parameter is “alpha” instead of “f0” .

        Returns
        -------
        int | None
            0 if successful, None otherwise.
        """
        args = [int(tag), float(ka), float(kb), float(fo), float(b1), float(b2), float(gamma)]
        if alpha is not None and alpha:
            args.append("-alpha")
        return self._ops.uniaxialMaterial("HystereticAsym", *args)
