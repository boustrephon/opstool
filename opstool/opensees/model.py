from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import openseespy.opensees as ops

from .friction_model import frictionModel
from .geom_transf import geomTransf
from .load_pattern import eleLoad, groundMotion, pattern
from .time_series import timeSeries
from .uniaxial_material import uniaxialMaterial


class Model:
    """A class to define and manage the structural model in OpenSees.

    Parameters
    ----------
    modelType : Literal["basic", "Basic", "BasicBuilder", "basicBuilder"]
        The type of model to be created.
    ndm : Literal[1, 2, 3], optional
        Number of dimensions of the model (1D, 2D, or 3D). Default is 3.
    ndf : Optional[int], optional
        Number of degrees of freedom per node. If None, it will be set based on ndm:
            - ndm = 1 -> ndf = 1
            - ndm = 2 -> ndf = 3
            - ndm = 3 -> ndf = 6
    """

    def __init__(
        self,
        modelType: Literal["basic", "Basic", "BasicBuilder", "basicBuilder"],
        ndm: Literal[1, 2, 3] = 3,
        ndf: int | None = None,
    ) -> int | None:
        if ndf is None:
            if ndm == 1:
                ndf = 1
            elif ndm == 2:
                ndf = 3
            elif ndm == 3:
                ndf = 6
        self.modelType: Literal["basic", "Basic", "BasicBuilder", "basicBuilder"] = modelType
        self.ndm: int = ndm
        self.ndf: int = ndf
        self._ops = ops

        # Initialize sub-modules
        self.timeSeries: timeSeries = timeSeries
        self.pattern: pattern = pattern
        self.eleLoad: eleLoad = eleLoad
        self.groundMotion: groundMotion = groundMotion
        self.uniaxialMaterial: uniaxialMaterial = uniaxialMaterial
        self.frictionModel: frictionModel = frictionModel
        self.geomTransf: geomTransf = geomTransf

        self.resetModel(self.modelType, self.ndm, self.ndf)

    def _set_subcls_ops_data(self) -> None:
        """Set the OpenSees module to be used."""
        for cls in [
            self.timeSeries,
            self.pattern,
            self.eleLoad,
            self.groundMotion,
            self.uniaxialMaterial,
            self.frictionModel,
            self.geomTransf,
        ]:
            cls._ops = self._ops
            cls._ndm = self.ndm
            cls._ndf = self.ndf

    def wipe(self) -> None:
        """Wipe the current model, removing all nodes, elements, and other definitions."""
        self._ops.wipe()

    def resetModel(
        self,
        modelType: Literal["basic", "Basic", "BasicBuilder", "basicBuilder"],
        ndm: Literal[1, 2, 3],
        ndf: int | None = None,
    ) -> None:
        """Reset the model with the specified type, number of dimensions, and degrees of freedom."""
        if ndf is None:
            if ndm == 1:
                ndf = 1
            elif ndm == 2:
                ndf = 3
            elif ndm == 3:
                ndf = 6
        self.modelType = modelType
        self.ndm = ndm
        self.ndf = ndf
        self._set_subcls_ops_data()  # Update sub-classes with new ops and data
        self._ops.model(self.modelType, "-ndm", self.ndm, "-ndf", self.ndf)

    def node(
        self,
        tag: int,
        coords: Iterable[float],
        disp: Iterable[float] | None = None,
        vel: Iterable[float] | None = None,
        mass: Iterable[float] | None = None,
        temp: Iterable[float] | None = None,
        dispLoc: Iterable[float] | None = None,
        ndf: int | None = None,
    ) -> int | None:
        """Create a node in the model.

        Parameters
        ----------
        tag : int
            Unique node tag.
        coords : Iterable[float]
            Coordinates of the node.
        disp : Optional[Iterable[float]], optional
            Initial displacements of the node, by default None
        vel : Optional[Iterable[float]], optional
            Initial velocities of the node, by default None
        mass : Optional[Iterable[float]], optional
            Masses at the node, by default None
        temp : Optional[Iterable[float]], optional
            Initial temperatures at the node, by default None
        dispLoc : Optional[Iterable[float]], optional
            Locations for prescribed displacements, by default None
        ndf : Optional[int], optional
            Number of degrees of freedom for the node, by default None

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        args = [int(tag), *[float(c) for c in coords]]
        if disp is not None:
            disp = [float(d) for d in disp]
            args += ["-disp", *disp]
        if vel is not None:
            vel = [float(v) for v in vel]
            args += ["-vel", *vel]
        if mass is not None:
            mass = [float(m) for m in mass]
            args += ["-mass", *mass]
        if temp is not None:
            temp = [float(t) for t in temp]
            args += ["-temp", *temp]
        if dispLoc is not None:
            dispLoc = [float(dl) for dl in dispLoc]
            args += ["-dispLoc", *dispLoc]
        if ndf is not None:
            args += ["-ndf", ndf]
        return self._ops.node(*args)

    def fix(self, nodeTag: int, constrValues: Iterable[Literal[0, 1]]) -> int | None:
        """Apply boundary conditions to a node.
        Parameters
        ----------
        nodeTag : int
            The tag of the node to which the boundary conditions are applied.
        constrValues : Iterable[Literal[0, 1]]
            An iterable of integers (0 or 1) indicating the constraint status for each degree of freedom.
        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        constrValues = [int(d) for d in constrValues]
        return self._ops.fix(nodeTag, *constrValues)

    def fixX(self, xLoc: int | float, constrValues: Iterable[Literal[0, 1]], tol: float = 1e-10) -> int | None:
        """Apply boundary conditions to nodes at a specific X location.

        Parameters
        ----------
        xLoc : int | float
            The X coordinate where the boundary conditions are applied.
        constrValues : Iterable[Literal[0, 1]]
            An iterable of integers (0 or 1) indicating the constraint status for each degree of freedom.
        tol : float, optional
            Tolerance for matching the X location, by default 1e-10

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        constrValues = [int(d) for d in constrValues]
        return self._ops.fixX(float(xLoc), *constrValues, "-tol", tol=tol)

    def fixY(self, yLoc: int | float, constrValues: Iterable[Literal[0, 1]], tol: float = 1e-10) -> int | None:
        """Apply boundary conditions to nodes at a specific Y location.

        Parameters
        ----------
        yLoc : int | float
            The Y coordinate where the boundary conditions are applied.
        constrValues : Iterable[Literal[0, 1]]
            An iterable of integers (0 or 1) indicating the constraint status for each degree of freedom.
        tol : float, optional
            Tolerance for matching the Y location, by default 1e-10

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        constrValues = [int(d) for d in constrValues]
        return self._ops.fixY(float(yLoc), *constrValues, "-tol", tol=tol)

    def fixZ(self, zLoc: int | float, constrValues: Iterable[Literal[0, 1]], tol: float = 1e-10) -> int | None:
        """Apply boundary conditions to nodes at a specific Z location.

        Parameters
        ----------
        zLoc : int | float
            The Z coordinate where the boundary conditions are applied.
        constrValues : Iterable[Literal[0, 1]]
            An iterable of integers (0 or 1) indicating the constraint status for each degree of freedom.
        tol : float, optional
            Tolerance for matching the Z location, by default 1e-10

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        constrValues = [int(d) for d in constrValues]
        return self._ops.fixZ(float(zLoc), *constrValues, "-tol", tol=tol)

    def equalDOF(self, rNodeTag: int, cNodeTag: int, dofs: Iterable[int] | None = None) -> int | None:
        """Constrain the degrees of freedom of two nodes to be equal.

        Parameters
        ----------
        rNodeTag : int
            Retained node tag.
        cNodeTag : int
            Constrained node tag.
        dofs : Iterable[int] | None, optional
            Degrees of freedom to be constrained, by default None

        Returns
        -------
        int | None
            0 if successful, or None if it fails.
        """
        args = [int(rNodeTag), int(cNodeTag)]
        if dofs is not None:
            dofs = [int(d) for d in dofs]
            args += dofs
        return self._ops.equalDOF(*args)

    def equalDOF_Mixed(
        self, rNodeTag: int, cNodeTag: int, numDof: int, rcDofs: Iterable[int] | None = None
    ) -> int | None:
        """Constrain the degrees of freedom of two nodes to be equal, with mixed specification.

        Parameters
        ----------
        rNodeTag : int
            Retained node tag.
        cNodeTag : int
            Constrained node tag.
        numDof : int
            Number of degrees of freedom to be constrained.
        rcDofs : Iterable[int] | None, optional
            List of degrees of freedom to be constrained, by default None.
            If None, all degrees of freedom up to numDof are constrained.

        Returns
        -------
        int | None
            0 if successful, or None if it fails.
        """

        args = [int(rNodeTag), int(cNodeTag), int(numDof)]
        if len(rcDofs) != numDof * 2:
            raise ValueError("Length of rcDofs must be equal to numDof*2.")  # noqa: TRY003
        rcDofs = [int(d) for d in rcDofs]
        args += rcDofs
        return self._ops.equalDOF_Mixed(*args)

    def rigidDiaphragm(self, perpDirn: Literal[1, 2, 3], rNodeTag: int, cNodeTags: Iterable[int]) -> int | None:
        """Create a rigid diaphragm constraint.

        Parameters
        ----------
        perpDirn : Literal[1, 2, 3]
            The direction perpendicular to the diaphragm plane (1 for X, 2 for Y, 3 for Z).
        rNodeTag : int
            The reference node tag.
        cNodeTags : Iterable[int]
            The constrained node tags.

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        cNodeTags = [int(d) for d in cNodeTags]
        return self._ops.rigidDiaphragm(int(perpDirn), int(rNodeTag), *cNodeTags)

    def rigidLink(self, linkType: Literal["beam", "bar"], rNodeTag: int, cNodeTag: int) -> int | None:
        """Create a rigid link between two nodes.

        Parameters
        ----------
        linkType : Literal["beam", "bar"]
            The type of rigid link.
        rNodeTag : int
            The reference node tag.
        cNodeTag : int
            The constrained node tag.

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        return self._ops.rigidLink(str(linkType), int(rNodeTag), int(cNodeTag))

    def mass(self, nodeTag: int, massValues: Iterable[float]) -> int | None:
        """Assign mass to a node.

        Parameters
        ----------
        nodeTag : int
            The tag of the node.
        massValues : Iterable[float]
            Mass values for each degree of freedom at the node.

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        args = [int(nodeTag)] + [float(m) for m in massValues]
        return self._ops.mass(*args)

    def region(
        self,
        regTag: int,
        eleTags: Iterable[int] | None = None,
        eleOnly: Iterable[int] | None = None,
        eleRange: Iterable[int] | None = None,
        eleOnlyRange: Iterable[int] | None = None,
        nodeTags: Iterable[int] | None = None,
        nodeOnly: Iterable[int] | None = None,
        nodeRange: Iterable[int] | None = None,
        nodeOnlyRange: Iterable[int] | None = None,
        rayleighParas: Iterable[float] | None = None,
        dampingTag: int | None = None,
    ) -> int | None:
        """The region command is used to label a group of nodes and elements.
        This command is also used to assign rayleigh damping parameters to the nodes and elements in this region.
        The region is specified by either elements or nodes, not both.
        If elements are defined, the region includes these elements and the all connected nodes,
        unless the -eleOnly option is used in which case only elements are included.
        If nodes are specified, the region includes these nodes and all elements of which all nodes are prescribed to be in the region,
        unless the -nodeOnly option is used in which case only the nodes are included.

        Parameters
        ----------
        regTag : int
            Unique region tag.
        eleTags : Optional[Iterable[int]], optional
            Element tags to be included in the region, by default None.
            If None, this option is ignored.
            If specified, all nodes connected to these elements are also included in the region.
        eleOnly : Optional[Iterable[int]], optional
            Element tags to be included in the region, by default None.
            If None, this option is ignored.
            If specified, only these elements are included in the region.
        eleRange : Optional[Iterable[int]], optional
            Element tag range [start, end] to be included in the region, by default None
            If None, this option is ignored.
            If specified, all nodes connected to these elements are also included in the region.
        eleOnlyRange : Optional[Iterable[int]], optional
            Element tag range [start, end] to be included in the region, by default None.
            If None, this option is ignored.
            If specified, only these elements are included in the region.
        nodeTags : Optional[Iterable[int]], optional
            Node tags to be included in the region, by default None.
            If None, this option is ignored.
            If specified, all elements of which all nodes are prescribed to be in the region are also included in the region.
        nodeOnly : Optional[Iterable[int]], optional
            Node tags to be included in the region, by default None.
            If None, this option is ignored.
            If specified, only these nodes are included in the region.
        nodeRange : Optional[Iterable[int]], optional
            Node tag range [start, end] to be included in the region, by default None.
            If None, this option is ignored.
            If specified, all elements of which all nodes are prescribed to be in the region are also included in the region.
        nodeOnlyRange : Optional[Iterable[int]], optional
            Node tag range [start, end] to be included in the region, by default None.
            If None, this option is ignored.
            If specified, only these nodes are included in the region.
        rayleighParas : Optional[Iterable[float]], optional
            Rayleigh damping parameters to be assigned to the region, by default None
            4 parameters are expected: [alphaM, betaK, betaKinit, betaKcomm].
            If None, this option is ignored.
        dampingTag : Optional[int], optional
            Damping tag to be assigned to the region, by default None.
            If None, this option is ignored.

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        args = [int(regTag)]
        if eleTags is not None:
            eleTags = [int(e) for e in eleTags]
            args += ["-ele", *eleTags]
        if eleOnly is not None:
            eleOnly = [int(eo) for eo in eleOnly]
            args += ["-eleOnly", *eleOnly]
        if eleRange is not None:
            eleRange = [int(er) for er in eleRange]
            args += ["-eleRange", *eleRange]
        if eleOnlyRange is not None:
            eleOnlyRange = [int(eor) for eor in eleOnlyRange]
            args += ["-eleOnlyRange", *eleOnlyRange]
        if nodeTags is not None:
            nodeTags = [int(n) for n in nodeTags]
            args += ["-node", *nodeTags]
        if nodeOnly is not None:
            nodeOnly = [int(no) for no in nodeOnly]
            args += ["-nodeOnly", *nodeOnly]
        if nodeRange is not None:
            nodeRange = [int(nr) for nr in nodeRange]
            args += ["-nodeRange", *nodeRange]
        if nodeOnlyRange is not None:
            nodeOnlyRange = [int(nor) for nor in nodeOnlyRange]
            args += ["-nodeOnlyRange", *nodeOnlyRange]
        if rayleighParas is not None:
            args += ["-rayleigh", *[float(r) for r in rayleighParas]]
        if dampingTag is not None:
            args += ["-damp", int(dampingTag)]
        return self._ops.region(*args)

    def rayleigh(
        self,
        alphaM: float,
        betaK: float,
        betaKinit: float,
        betaKcomm: float,
    ) -> int | None:
        """Assign Rayleigh damping parameters.

        Parameters
        ----------
        alphaM : float
            Factor applied to elements or nodes mass matrix
        betaK : float
            Factor applied to elements current stiffness matrix.
        betaKinit : float
            Factor applied to elements initial stiffness matrix.
        betaKcomm : float
            Factor applied to elements committed stiffness matrix.

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        return self._ops.rayleigh(float(alphaM), float(betaK), float(betaKinit), float(betaKcomm))

    def ShallowFoundationGen(self, tag: int, ConnectNode: int, InputFileName: str, FootingCondition: int) -> int | None:
        """Create a shallow foundation model.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the shallow foundation.
        ConnectNode : int
            Node tag where the shallow foundation is connected.
        InputFileName : str
            Name of file containing soil, footing, and mesh properties
        FootingCondition : int
            Integer code for footing condition.
            * FootingCondition = 1	Foundation is fixed
            * FootingCondition = 2	ZeroLength Elastic springs in the vertical direction, sliding restrained
            * FootingCondition = 3	ZeroLength Elastic springs in the vertical and horizontal direction
            * FootingCondition = 4	ZeroLength springs with QzSimple2 material in the vertical direction, sliding restrained
            * FootingCondition = 5	ZeroLength springs with QzSimple2 material in the vertical direction, two ZeroLength springs one with PxSimple1 material and the other with TxSimple1 material in the horizontal direction

        Returns
        -------
        Union[int, None]
            0 if successful, or None if it fails.
        """
        args = [int(tag), int(ConnectNode), str(InputFileName), int(FootingCondition)]
        return self._ops.ShallowFoundationGen(*args)

    def load(
        self,
        nodeTag: int,
        values: Iterable[float],
        const: bool | None = None,
        loadPatternTag: int | None = None,
    ) -> int | None:
        """Applies a nodal load to a node in the currently active load pattern.

        Parameters
        ----------
        nodeTag : int
            Tag of the node where the load is applied.
        values : Iterable[float]
            Load values to be applied at the node.
            It should contain as many values as the number of DOFs of the node.
        const : Optional[bool], optional
            Indicates if the load is constant, by default None
        loadPatternTag : Optional[int], optional
            Tag of the load pattern to which the load is applied, by default None
            If None, the load is applied to the currently active load pattern.

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = [int(nodeTag)] + [float(val) for val in values]
        if const is not None and const:
            args.append("-const")
        if loadPatternTag is not None:
            args += ["-pattern", int(loadPatternTag)]
        return self._ops.load(*args)

    def sp(
        self,
        nodeTag: int,
        dof: int,
        value: float,
        const: bool | None = None,
        subtractInit: bool | None = None,
        loadPatternTag: int | None = None,
    ) -> int | None:
        """Applies a single point constraint to a node in the currently active load pattern.

        Parameters
        ----------
        nodeTag : int
            Tag of the node where the single point constraint is applied.
        dof : int
            Degree of freedom where the single point constraint is applied.
        value : float
            Value of the single point constraint.
        const : Optional[bool], optional
            Indicates if the constraint is constant, by default None
        subtractInit : Optional[bool], optional
            Indicates if the initial value should be subtracted, by default None
        loadPatternTag : Optional[int], optional
            Tag of the load pattern to which the single point constraint is applied, by default None

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = [int(nodeTag), int(dof), float(value)]
        if const is not None and const:
            args.append("-const")
        if subtractInit is not None and subtractInit:
            args.append("-subtractInit")
        if loadPatternTag is not None:
            args += ["-pattern", int(loadPatternTag)]
        return self._ops.sp(*args)

    def imposedMotion(self, nodeTag: int, dof: int, gmTag: int, other: bool | None = None) -> int | None:
        """Applies an imposed motion to a node using a ground motion.

        Parameters
        ----------
        nodeTag : int
            Tag of the node where the imposed motion is applied.
        dof : int
            Degree of freedom where the imposed motion is applied.
        gmTag : int
            Tag of the ground motion to be used.
        other : Optional[bool], optional
            If True, indicates that the imposed motion is of type "other", by default None

        Returns
        -------
        Union[int, None]
            0 if successful, None otherwise.
        """
        args = [int(nodeTag), int(dof), int(gmTag)]
        if other is not None and other:
            args.append("-other")
        return self._ops.imposedMotion(*args)

    def block2D(
        nx: int,
        ny: int,
        startNode: int,
        startEle: int,
        eleType: str,
        eleArgs: Iterable[float | int],
        xs: Iterable[float],
        ys: Iterable[float],
        zs: Iterable[float] | None = None,
        nodeLocs: Iterable[int] | None = None,
    ) -> None | int:
        """Create a 2D mesh block of nodes and elements.

        Parameters
        ----------
        nx : int
            Number of elements in the X direction.
        ny : int
            Number of elements in the Y direction.
        startNode : int
            Starting node tag.
        startEle : int
            Starting element tag.
        eleType : str
            Type of element to be created, ('quad', "stdQuad", "shellMITC4", "shellNLDKGQ", "shellDKGQ", 'bbarQuad', "mixedQuad", 'enhancedQuad', or 'SSPquad')
        eleArgs : tuple
            Additional arguments for the element type.
        xs : Iterable[float]
            X coordinates of the block corners.
            Length must >=4.
        ys : Iterable[float]
            Y coordinates of the block corners.
            Length must >=4.
        zs : Optional[Iterable[float]], optional
            Z coordinates of the block corners, by default None.
            Length must >=4 if provided.
        nodeLocs : Optional[Iterable[int]], optional
            Locations of nodes along the edges. If None, nodes are evenly distributed, by default None.

            4 --- 7 --- 3
            |     |     |
            8 --- 9 --- 6
            |     |     |
            1 --- 5 --- 2

        Returns
        -------
        None | int
            0 if successful, None otherwise.
        """
        args = [int(nx), int(ny), int(startNode), int(startEle), str(eleType), *eleArgs]
        nodeLocs = [int(nl) for nl in nodeLocs] if nodeLocs is not None else [1 + i for i in range(len(xs))]
        crds = []
        if zs is None:
            for i in range(len(xs)):
                crds.extend([nodeLocs[i], float(xs[i]), float(ys[i])])
        else:
            for i in range(len(xs)):
                crds.extend([nodeLocs[i], float(xs[i]), float(ys[i]), float(zs[i])])
        args += crds
        return ops.block2D(*args)

    def block3D(
        nx: int,
        ny: int,
        nz: int,
        startNode: int,
        startEle: int,
        eleType: str,
        eleArgs: Iterable[float | int],
        xs: Iterable[float],
        ys: Iterable[float],
        zs: Iterable[float],
        nodeLocs: Iterable[int] | None = None,
    ) -> None | int:
        """Mesh a 3D block of nodes and elements.

        Parameters
        ----------
        nx : int
            Number of elements in the X direction.
        ny : int
            Number of elements in the Y direction.
        nz : int
            Number of elements in the Z direction.
        startNode : int
            Starting node tag.
        startEle : int
            Starting element tag.
        eleType : str
            Type of element to be created.
        eleArgs : Iterable[float  |  int]
            Additional arguments for the element type.
        xs : Iterable[float]
            X coordinates of the block corners.
            Length must be >=8.
        ys : Iterable[float]
            Y coordinates of the block corners.
            Length must be >=8.
        zs : Iterable[float]
            Z coordinates of the block corners.
            Length must be >=8.
        nodeLocs : Iterable[int] | None, optional
            Locations of nodes along the edges. If None, nodes are evenly distributed 1-length(xs), by default None.
            1--27.

        Returns
        -------
        None | int
            0 if successful, None otherwise.
        """
        args = [int(nx), int(ny), int(nz), int(startNode), int(startEle), str(eleType), *eleArgs]
        nodeLocs = [int(nl) for nl in nodeLocs] if nodeLocs is not None else [1 + i for i in range(len(xs))]
        crds = []
        for i in range(len(xs)):
            crds.extend([nodeLocs[i], float(xs[i]), float(ys[i]), float(zs[i])])
        args += crds
        return ops.block3D(*args)


# def frictionModel(self, *args):
#     src = _get_cmds(args, "frictionModel", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _geomTransf(self, *args):
#     src = _get_cmds(args, "geomTransf", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _region(self, *args):
#     src = _get_cmds(args, "region", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _uniaxialMaterial(self, *args):
#     src = _get_cmds(args, "uniaxialMaterial", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nDMaterial(self, *args):
#     src = _get_cmds(args, "nDMaterial", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _beamIntegration(self, *args):
#     src = _get_cmds(args, "beamIntegration", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _section(self, *args):
#     args, comments = _process_args(args, keep_comments=self.keep_comments)
#     if args[0] in (
#         "Fiber",
#         "fiberSec",
#         "FiberWarping",
#         "FiberAsym",
#         "FiberThermal",
#         "NDFiber",
#         "NDFiberWarping",
#     ):
#         if args[0] not in ["NDFiber", "NDFiberWarping"] and ("-GJ" not in args or "-torsion" not in args):
#             rprint(
#                 "[bold #d20962]Warning[/bold #d20962]: "
#                 "-GJ or -torsion not used for fiber section, GJ=100000000 is assumed!"
#             )
#             new_args = (args[0], args[1], "-GJ", 1.0e8)
#         else:
#             new_args = args[:4]
#         self.contents.append(f"{self.prefix}section{new_args}{comments}")
#         txt = args[-1]
#         txt.replace("\\n", "")
#         self.interp.eval(txt)
#     else:
#         self.contents.append(f"{self.prefix}section{args}{comments}")

# def _fiber(self, *args):
#     src = _get_cmds(args, "fiber", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _patch(self, *args):
#     src = _get_cmds(args, "patch", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _layer(self, *args):
#     src = _get_cmds(args, "layer", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _element(self, *args):
#     args, comments = _process_args(args, keep_comments=self.keep_comments)
#     if args[0] not in [
#         "nonlinearBeamColumn",
#         "forceBeamColumn",
#         "dispBeamColumn",
#         "forceBeamColumnCBDI",
#         "forceBeamColumnCSBDI",
#         "forceBeamColumnWarping",
#         "forceBeamColumnThermal",
#         "elasticForceBeamColumnWarping",
#         "dispBeamColumnNL",
#         "dispBeamColumnThermal",
#         "nonlinearBeamColumn",
#         "dispBeamColumnWithSensitivity",
#     ]:
#         self.contents.append(f"{self.prefix}element{args}{comments}")
#     else:
#         eleTag = args[1]
#         secTag = args[5]
#         if isinstance(secTag, int):
#             Np = args[4]
#             transfTag = args[6]
#             if args[0] == "dispBeamColumn":
#                 self.contents.append(f"{self.prefix}beamIntegration('Legendre', {eleTag}, {secTag}, {Np})")
#             else:
#                 self.contents.append(f"{self.prefix}beamIntegration('Lobatto', {eleTag}, {secTag}, {Np})")
#             idx = 7
#         elif secTag == "-sections":  # Handle variable section tags
#             Np = args[4]
#             sectags = args[6 : 6 + Np]
#             transfTag = args[6 + Np]
#             idx = 6 + Np + 1
#             if args[0] == "dispBeamColumn":
#                 self.contents.append(f"{self.prefix}beamIntegration('Legendre', {eleTag}, {Np}, *{sectags})")
#             else:
#                 self.contents.append(f"{self.prefix}beamIntegration('Lobatto', {eleTag}, {Np}, *{sectags})")
#         else:
#             transfTag = args[4]
#             interp_paras = []
#             idx = 6
#             for i, arg in enumerate(args[6:]):
#                 if not isinstance(arg, str):
#                     interp_paras.append(arg)
#                 else:
#                     idx += i
#                     break
#             self.contents.append(f"{self.prefix}beamIntegration('{args[5]}', {eleTag}, *{interp_paras})")
#         # write the element command
#         if args[0] == "nonlinearBeamColumn":
#             args[0] = "forceBeamColumn"
#         if "-mass" not in args and "-iter" not in args and "-cMass" not in args:
#             self.contents.append(
#                 f"{self.prefix}element('{args[0]}', {eleTag}, {args[2]}, {args[3]}, {transfTag}, {eleTag}){comments}"
#             )
#         else:
#             self.contents.append(
#                 f"{self.prefix}element('{args[0]}', {eleTag}, {args[2]}, "
#                 f"{args[3]}, {transfTag}, {eleTag}, *{args[idx:]}){comments}"
#             )

# def _timeSeries(self, *args):
#     args, comments = _process_args(args, keep_comments=self.keep_comments)
#     if args[0] in ["Path", "Series"]:
#         if ("-time" in args) or ("-values" in args):
#             time, values = None, None
#             if "-time" in args:
#                 idx = args.index("-time")
#                 time = list(args[idx + 1].split())
#                 time = [float(i) for i in time]
#                 args.pop(idx)
#                 args.pop(idx)
#             if "-values" in args:
#                 idx = args.index("-values")
#                 values = list(args[idx + 1].split())
#                 values = [float(i) for i in values]
#                 args.pop(idx)
#                 args.pop(idx)
#             if time and values:
#                 args = [*args[:2], "-time", *time, "-values", *values, *args[2:]]
#             elif values is None:
#                 args = [*args[:2], "-time", *time, *args[2:]]
#             else:
#                 args = [*args[:2], "-values", *values, *args[2:]]
#             txt = f"{self.prefix}timeSeries('Path', {args[1]}, *{args[2:]}){comments}"
#             self.contents.append(txt)
#         else:
#             self.contents.append(f"{self.prefix}timeSeries{tuple(args)}{comments}")
#     else:
#         self.contents.append(f"{self.prefix}timeSeries{tuple(args)}{comments}")


# def _block2D(self, *args):
#     args, comments = _process_args(args, keep_comments=self.keep_comments)
#     txt = args[-1]
#     txt = txt.replace("\n", "").replace("\t", " ")
#     crds = txt.split()
#     crds = [_type_convert(i) for i in crds]
#     self.contents.append(f"crds = {crds}")
#     if isinstance(args[-2], str):
#         eleargs = args[-2].split()
#         eleargs = [_type_convert(i) for i in eleargs]
#         args = args[:-2] + eleargs
#         args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args]
#         args.append("*crds")
#     else:
#         args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args[:-1]]
#         args.append("*crds")
#     txt = f"{self.prefix}block2D(" + ", ".join(args) + f"){comments}"
#     self.contents.append(txt)

# def _block3D(self, *args):
#     args, comments = _process_args(args, keep_comments=self.keep_comments)
#     txt = args[-1]
#     txt = txt.replace("\n", "").replace("\t", " ")
#     crds = txt.split()
#     crds = [_type_convert(i) for i in crds]
#     self.contents.append(f"crds = {crds}")
#     if isinstance(args[-2], str):
#         eleargs = args[-2].split()
#         eleargs = [_type_convert(i) for i in eleargs]
#         args = args[:-2] + eleargs
#         args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args]
#         args.append("*crds")
#     else:
#         args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args[:-1]]
#         args.append("*crds")
#     txt = f"{self.prefix}block3D(" + ", ".join(args) + f"){comments}"
#     self.contents.append(txt)

# def _ShallowFoundationGen(self, *args):
#     src = _get_cmds(args, "ShallowFoundationGen", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _constraints(self, *args):
#     src = _get_cmds(args, "constraints", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _numberer(self, *args):
#     src = _get_cmds(args, "numberer", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _system(self, *args):
#     src = _get_cmds(args, "system", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _test(self, *args):
#     src = _get_cmds(args, "test", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _algorithm(self, *args):
#     src = _get_cmds(args, "algorithm", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _integrator(self, *args):
#     src = _get_cmds(args, "integrator", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _analysis(self, *args):
#     src = _get_cmds(args, "analysis", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _eigen(self, *args):
#     src = _get_cmds(args, "eigen", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)
#     return None

# def _analyze(self, *args):
#     src = _get_cmds(args, "analyze", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)
#     return None

# def _modalProperties(self, *args):
#     src = _get_cmds(args, "modalProperties", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)
#     return None

# def _responseSpectrumAnalysis(self, *args):
#     src = _get_cmds(args, "responseSpectrumAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _recorder(self, *args):
#     src = _get_cmds(args, "recorder", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _record(self, *args):
#     src = _get_cmds(args, "record", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _print(self, *args):
#     src = _get_cmds(args, "printModel", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _printA(self, *args):
#     src = _get_cmds(args, "printA", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _logFile(self, *args):
#     src = _get_cmds(args, "logFile", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _remove(self, *args):
#     src = _get_cmds(args, "remove", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _loadConst(self, *args):
#     src = _get_cmds(args, "loadConst", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _wipeAnalysis(self, *args):
#     src = _get_cmds(args, "wipeAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _modalDamping(self, *args):
#     src = _get_cmds(args, "modalDamping", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _database(self, *args):
#     src = _get_cmds(args, "database", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getTime(self, *args):
#     src = _get_cmds(args, "getTime", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setTime(self, *args):
#     src = _get_cmds(args, "setTime", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _testUniaxialMaterial(self, *args):
#     src = _get_cmds(args, "testUniaxialMaterial", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setStrain(self, *args):
#     src = _get_cmds(args, "setStrain", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getStrain(self, *args):
#     src = _get_cmds(args, "getStrain", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getStress(self, *args):
#     src = _get_cmds(args, "getStress", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getTangent(self, *args):
#     src = _get_cmds(args, "getTangent", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getDampTangent(self, *args):
#     src = _get_cmds(args, "getDampTangent", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _reactions(self, *args):
#     src = _get_cmds(args, "reactions", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeReaction(self, *args):
#     src = _get_cmds(args, "nodeReaction", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeEigenvector(self, *args):
#     src = _get_cmds(args, "nodeEigenvector", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setCreep(self, *args):
#     src = _get_cmds(args, "setCreep", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _eleResponse(self, *args):
#     src = _get_cmds(args, "eleResponse", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _reset(self, *args):
#     src = _get_cmds(args, "reset", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _initialize(self, *args):
#     src = _get_cmds(args, "initialize", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getLoadFactor(self, *args):
#     src = _get_cmds(args, "getLoadFactor", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _build(self, *args):
#     src = _get_cmds(args, "build", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _printGID(self, *args):
#     src = _get_cmds(args, "printGID", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getCTestNorms(self, *args):
#     src = _get_cmds(args, "getCTestNorms", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getCTestIter(self, *args):
#     src = _get_cmds(args, "getCTestIter", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _save(self, *args):
#     src = _get_cmds(args, "save", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _restore(self, *args):
#     src = _get_cmds(args, "restore", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _eleForce(self, *args):
#     src = _get_cmds(args, "eleForce", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _eleDynamicalForce(self, *args):
#     src = _get_cmds(args, "eleDynamicalForce", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeUnbalance(self, *args):
#     src = _get_cmds(args, "nodeUnbalance", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeDisp(self, *args):
#     src = _get_cmds(args, "nodeDisp", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setNodeDisp(self, *args):
#     src = _get_cmds(args, "setNodeDisp", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeVel(self, *args):
#     src = _get_cmds(args, "nodeVel", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setNodeVel(self, *args):
#     src = _get_cmds(args, "setNodeVel", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeAccel(self, *args):
#     src = _get_cmds(args, "nodeAccel", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setNodeAccel(self, *args):
#     src = _get_cmds(args, "setNodeAccel", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeResponse(self, *args):
#     src = _get_cmds(args, "nodeResponse", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeCoord(self, *args):
#     src = _get_cmds(args, "nodeCoord", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setNodeCoord(self, *args):
#     src = _get_cmds(args, "setNodeCoord", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _updateElementDomain(self, *args):
#     src = _get_cmds(args, "updateElementDomain", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getNDMM(self, *args):
#     src = _get_cmds(args, "getNDM", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getNDFF(self, *args):
#     src = _get_cmds(args, "getNDF", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _eleNodes(self, *args):
#     src = _get_cmds(args, "eleNodes", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _eleType(self, *args):
#     src = _get_cmds(args, "eleType", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeDOFs(self, *args):
#     src = _get_cmds(args, "nodeDOFs", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeMass(self, *args):
#     src = _get_cmds(args, "nodeMass", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodePressure(self, *args):
#     src = _get_cmds(args, "nodePressure", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setNodePressure(self, *args):
#     src = _get_cmds(args, "setNodePressure", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _nodeBounds(self, *args):
#     src = _get_cmds(args, "nodeBounds", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _startTimer(self, *args):
#     src = _get_cmds(args, "start", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _stopTimer(self, *args):
#     src = _get_cmds(args, "stop", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _modalDampingQ(self, *args):
#     src = _get_cmds(args, "modalDampingQ", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setElementRayleighDampingFactors(self, *args):
#     src = _get_cmds(args, "setElementRayleighDampingFactors", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setPrecision(self, *args):
#     src = _get_cmds(args, "setPrecision", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _searchPeerNGA(self, *args):
#     src = _get_cmds(args, "searchPeerNGA", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _domainChange(self, *args):
#     src = _get_cmds(args, "domainChange", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _defaultUnits(self, *args):
#     src = _get_cmds(args, "defaultUnits", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _stripXML(self, *args):
#     src = _get_cmds(args, "stripXML", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _convertBinaryToText(self, *args):
#     src = _get_cmds(args, "convertBinaryToText", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _convertTextToBinary(self, *args):
#     src = _get_cmds(args, "convertTextToBinary", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getEleTags(self, *args):
#     src = _get_cmds(args, "getEleTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getCrdTransfTags(self, *args):
#     src = _get_cmds(args, "getCrdTransfTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getNodeTags(self, *args):
#     src = _get_cmds(args, "getNodeTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getParamTags(self, *args):
#     src = _get_cmds(args, "getParamTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getParamValue(self, *args):
#     src = _get_cmds(args, "getParamValue", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sectionForce(self, *args):
#     src = _get_cmds(args, "sectionForce", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sectionDeformation(self, *args):
#     src = _get_cmds(args, "sectionDeformation", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sectionStiffness(self, *args):
#     src = _get_cmds(args, "sectionStiffness", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sectionFlexibility(self, *args):
#     src = _get_cmds(args, "sectionFlexibility", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sectionLocation(self, *args):
#     src = _get_cmds(args, "sectionLocation", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sectionWeight(self, *args):
#     src = _get_cmds(args, "sectionWeight", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sectionTag(self, *args):
#     src = _get_cmds(args, "sectionTag", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sectionDisplacement(self, *args):
#     src = _get_cmds(args, "sectionDisplacement", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _cbdiDisplacement(self, *args):
#     src = _get_cmds(args, "cbdiDisplacement", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _basicDeformation(self, *args):
#     src = _get_cmds(args, "basicDeformation", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _basicForce(self, *args):
#     src = _get_cmds(args, "basicForce", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _basicStiffness(self, *args):
#     src = _get_cmds(args, "basicStiffness", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _InitialStateAnalysis(self, *args):
#     src = _get_cmds(args, "InitialStateAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _totalCPU(self, *args):
#     src = _get_cmds(args, "totalCPU", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _solveCPU(self, *args):
#     src = _get_cmds(args, "solveCPU", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _accelCPU(self, *args):
#     src = _get_cmds(args, "accelCPU", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _numFact(self, *args):
#     src = _get_cmds(args, "numFact", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _numIter(self, *args):
#     src = _get_cmds(args, "numIter", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _systemSize(self, *args):
#     src = _get_cmds(args, "systemSize", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _version(self, *args):
#     src = _get_cmds(args, "version", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setMaxOpenFiles(self, *args):
#     src = _get_cmds(args, "setMaxOpenFiles", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _limitCurve(self, *args):
#     src = _get_cmds(args, "limitCurve", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setElementRayleighFactors(self, *args):
#     src = _get_cmds(args, "setElementRayleighFactors", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _mesh(self, *args):
#     src = _get_cmds(args, "mesh", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _remesh(self, *args):
#     src = _get_cmds(args, "remesh", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _parameter(self, *args):
#     src = _get_cmds(args, "parameter", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _addToParameter(self, *args):
#     src = _get_cmds(args, "addToParameter", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _updateParameter(self, *args):
#     src = _get_cmds(args, "updateParameter", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setParameter(self, *args):
#     src = _get_cmds(args, "setParameter", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getPID(self, *args):
#     src = _get_cmds(args, "getPID", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getNP(self, *args):
#     src = _get_cmds(args, "getNP", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _barrier(self, *args):
#     src = _get_cmds(args, "barrier", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _send(self, *args):
#     src = _get_cmds(args, "send", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _recv(self, *args):
#     src = _get_cmds(args, "recv", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _Bcast(self, *args):
#     src = _get_cmds(args, "Bcast", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _computeGradients(self, *args):
#     src = _get_cmds(args, "computeGradients", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sensitivityAlgorithm(self, *args):
#     src = _get_cmds(args, "sensitivityAlgorithm", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sensNodeDisp(self, *args):
#     src = _get_cmds(args, "sensNodeDisp", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sensNodeVel(self, *args):
#     src = _get_cmds(args, "sensNodeVel", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sensNodeAccel(self, *args):
#     src = _get_cmds(args, "sensNodeAccel", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sensLambda(self, *args):
#     src = _get_cmds(args, "sensLambda", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sensSectionForce(self, *args):
#     src = _get_cmds(args, "sensSectionForce", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sensNodePressure(self, *args):
#     src = _get_cmds(args, "sensNodePressure", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getNumElements(self, *args):
#     src = _get_cmds(args, "getNumElements", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getEleClassTags(self, *args):
#     src = _get_cmds(args, "getEleClassTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getEleLoadClassTags(self, *args):
#     src = _get_cmds(args, "getEleLoadClassTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getEleLoadTags(self, *args):
#     src = _get_cmds(args, "getEleLoadTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getEleLoadData(self, *args):
#     src = _get_cmds(args, "getEleLoadData", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getNodeLoadTags(self, *args):
#     src = _get_cmds(args, "getNodeLoadTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getNodeLoadData(self, *args):
#     src = _get_cmds(args, "getNodeLoadData", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _randomVariable(self, *args):
#     src = _get_cmds(args, "randomVariable", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getRVTags(self, *args):
#     src = _get_cmds(args, "getRVTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getRVMean(self, *args):
#     src = _get_cmds(args, "getRVMean", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getRVStdv(self, *args):
#     src = _get_cmds(args, "getRVStdv", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getRVPDF(self, *args):
#     src = _get_cmds(args, "getRVPDF", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getRVCDF(self, *args):
#     src = _get_cmds(args, "getRVCDF", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getRVInverseCDF(self, *args):
#     src = _get_cmds(args, "getRVInverseCDF", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _addCorrelate(self, *args):
#     src = _get_cmds(args, "addCorrelate", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _correlate(self, *args):
#     src = _get_cmds(args, "correlate", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _performanceFunction(self, *args):
#     src = _get_cmds(args, "performanceFunction", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _gradPerformanceFunction(self, *args):
#     src = _get_cmds(args, "gradPerformanceFunction", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _transformUtoX(self, *args):
#     src = _get_cmds(args, "transformUtoX", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _wipeReliability(self, *args):
#     src = _get_cmds(args, "wipeReliability", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _updateMaterialStage(self, *args):
#     src = _get_cmds(args, "updateMaterialStage", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _sdfResponse(self, *args):
#     src = _get_cmds(args, "sdfResponse", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _probabilityTransformation(self, *args):
#     src = _get_cmds(args, "probabilityTransformation", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _startPoint(self, *args):
#     src = _get_cmds(args, "startPoint", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _randomNumberGenerator(self, *args):
#     src = _get_cmds(args, "randomNumberGenerator", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _reliabilityConvergenceCheck(self, *args):
#     src = _get_cmds(args, "reliabilityConvergenceCheck", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _searchDirection(self, *args):
#     src = _get_cmds(args, "searchDirection", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _meritFunctionCheck(self, *args):
#     src = _get_cmds(args, "meritFunctionCheck", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _stepSizeRule(self, *args):
#     src = _get_cmds(args, "stepSizeRule", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _rootFinding(self, *args):
#     src = _get_cmds(args, "rootFinding", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _functionEvaluator(self, *args):
#     src = _get_cmds(args, "functionEvaluator", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _gradientEvaluator(self, *args):
#     src = _get_cmds(args, "gradientEvaluator", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _runFOSMAnalysis(self, *args):
#     src = _get_cmds(args, "runFOSMAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _findDesignPoint(self, *args):
#     src = _get_cmds(args, "findDesignPoint", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _runFORMAnalysis(self, *args):
#     src = _get_cmds(args, "runFORMAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getLSFTags(self, *args):
#     src = _get_cmds(args, "getLSFTags", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _runImportanceSamplingAnalysis(self, *args):
#     src = _get_cmds(args, "runImportanceSamplingAnalysis", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _IGA(self, *args):
#     src = _get_cmds(args, "IGA", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _NDTest(self, *args):
#     src = _get_cmds(args, "NDTest", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _getNumThreads(self, *args):
#     src = _get_cmds(args, "getNumThreads", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setNumThreads(self, *args):
#     src = _get_cmds(args, "setNumThreads", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _setStartNodeTag(self, *args):
#     src = _get_cmds(args, "setStartNodeTag", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _hystereticBackbone(self, *args):
#     src = _get_cmds(args, "hystereticBackbone", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _stiffnessDegradation(self, *args):
#     src = _get_cmds(args, "stiffnessDegradation", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _strengthDegradation(self, *args):
#     src = _get_cmds(args, "strengthDegradation", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _unloadingRule(self, *args):
#     src = _get_cmds(args, "unloadingRule", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _partition(self, *args):
#     src = _get_cmds(args, "partition", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _pc(self, *args):
#     src = _get_cmds(args, "pressureConstraint", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# def _domainCommitTag(self, *args):
#     src = _get_cmds(args, "domainCommitTag", prefix=self.prefix, keep_comments=self.keep_comments)
#     self.contents.append(src)

# @staticmethod
# def _display(*args):
#     print(f"This <display {args}> function will be ignored!")

# @staticmethod
# def _prp(*args):
#     print(f"This display <prp {args}> function will be ignored!")

# @staticmethod
# def _vup(*args):
#     print(f"This display <vup {args}> function will be ignored!")

# @staticmethod
# def _vpn(*args):
#     print(f"This display <vpn {args}> function will be ignored!")

# @staticmethod
# def _vrp(*args):
#     print(f"This display <vrp {args}> function will be ignored!")
