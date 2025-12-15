from __future__ import annotations

from collections.abc import Iterable


class geomTransf:
    """
    Class for geometric transformations in OpenSees.
    The geomTransf command is used to construct a geometric transformation object,
    which defines how the local element coordinate system is oriented with respect to the global coordinate system.
    """

    _ops = None
    _ndm = None
    _ndf = None

    @classmethod
    def Linear(
        cls,
        tag: int,
        vecxz: Iterable[float] | None = None,
        jntOffsetI: Iterable[float] | None = None,
        jntOffsetJ: Iterable[float] | None = None,
    ) -> None | int:
        """
        Linear geometric transformation.
        The element coordinate system is specified as follows:

        The x-axis is the axis connecting the two element nodes;
        the y- and z-axes are then defined using a vector that lies on a plane parallel to the local x-z plane -- vecxz.
        The local y-axis is defined by taking the cross product of the vecxz vector and the x-axis.
        The z-axis by taking the cross-product of x and y vectors.
        The section is attached to the element such that the y-z coordinate system used to specify the section corresponds to the y-z axes of the element.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the geometric transformation.
        vecxz : Iterable[float] | None, optional
            The vector is specified in the global-coordinate system X,Y,Z and defines a vector that is in a plane parallel to the x-z plane of the local-coordinate system.
            [x_component, y_component, z_component] for 3D only.
        jntOffsetI : Iterable[float] | None, optional
            Joint offset at the I-end of the element in the global-coordinate system X,Y,Z.
            [x_offset, y_offset, z_offset (3D only)]
        jntOffsetJ : Iterable[float] | None, optional
            Joint offset at the J-end of the element in the global-coordinate system X,Y,Z.
            [x_offset, y_offset, z_offset (3D only)]

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag)]
        if vecxz is not None:
            args += [float(v) for v in vecxz]
        if jntOffsetI is not None and jntOffsetJ is not None:
            args += ["-jntOffset"]
            args += [float(v) for v in jntOffsetI]
            args += [float(v) for v in jntOffsetJ]
        return cls._ops.geomTransf("Linear", *args)

    @classmethod
    def PDelta(
        cls,
        tag: int,
        vecxz: Iterable[float] | None = None,
        jntOffsetI: Iterable[float] | None = None,
        jntOffsetJ: Iterable[float] | None = None,
    ) -> None | int:
        """
        P-Delta geometric transformation.
        The element coordinate system is specified as follows:

        The x-axis is the axis connecting the two element nodes;
        the y- and z-axes are then defined using a vector that lies on a plane parallel to the local x-z plane -- vecxz.
        The local y-axis is defined by taking the cross product of the vecxz vector and the x-axis.
        The z-axis by taking the cross-product of x and y vectors.
        The section is attached to the element such that the y-z coordinate system used to specify the section corresponds to the y-z axes of the element.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the geometric transformation.
        vecxz : Iterable[float] | None, optional
            The vector is specified in the global-coordinate system X,Y,Z and defines a vector that is in a plane parallel to the x-z plane of the local-coordinate system.
            [x_component, y_component, z_component] for 3D only.
        jntOffsetI : Iterable[float] | None, optional
            Joint offset at the I-end of the element in the global-coordinate system X,Y,Z.
            [x_offset, y_offset, z_offset (3D only)]
        jntOffsetJ : Iterable[float] | None, optional
            Joint offset at the J-end of the element in the global-coordinate system X,Y,Z.
            [x_offset, y_offset, z_offset (3D only)]

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag)]
        if vecxz is not None:
            args += [float(v) for v in vecxz]
        if jntOffsetI is not None and jntOffsetJ is not None:
            args += ["-jntOffset"]
            args += [float(v) for v in jntOffsetI]
            args += [float(v) for v in jntOffsetJ]
        return cls._ops.geomTransf("PDelta", *args)

    @classmethod
    def Corotational(
        cls,
        tag: int,
        vecxz: Iterable[float] | None = None,
        jntOffsetI: Iterable[float] | None = None,
        jntOffsetJ: Iterable[float] | None = None,
    ) -> None | int:
        """
        P-Delta geometric transformation.
        The element coordinate system is specified as follows:

        The x-axis is the axis connecting the two element nodes;
        the y- and z-axes are then defined using a vector that lies on a plane parallel to the local x-z plane -- vecxz.
        The local y-axis is defined by taking the cross product of the vecxz vector and the x-axis.
        The z-axis by taking the cross-product of x and y vectors.
        The section is attached to the element such that the y-z coordinate system used to specify the section corresponds to the y-z axes of the element.

        Parameters
        ----------
        tag : int
            Unique integer tag identifying the geometric transformation.
        vecxz : Iterable[float] | None, optional
            The vector is specified in the global-coordinate system X,Y,Z and defines a vector that is in a plane parallel to the x-z plane of the local-coordinate system.
            [x_component, y_component, z_component] for 3D only.
        jntOffsetI : Iterable[float] | None, optional
            Joint offset at the I-end of the element in the global-coordinate system X,Y,Z.
            [x_offset, y_offset] for 2D only in current implementation.
        jntOffsetJ : Iterable[float] | None, optional
            Joint offset at the J-end of the element in the global-coordinate system X,Y,Z.
            [x_offset, y_offset] for 2D only in current implementation.

        Returns
        -------
        None | int
            0 if successful, otherwise an error code.
        """
        args = [int(tag)]
        if vecxz is not None:
            args += [float(v) for v in vecxz]
        if jntOffsetI is not None and jntOffsetJ is not None:
            args += ["-jntOffset"]
            args += [float(v) for v in jntOffsetI]
            args += [float(v) for v in jntOffsetJ]
        return cls._ops.geomTransf("Corotational", *args)
