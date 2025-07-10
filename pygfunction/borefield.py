# -*- coding: utf-8 -*-
from typing import Union, List, Dict, Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import Self  # for compatibility with Python <= 3.10

from .boreholes import Borehole
from .utilities import _initialize_figure, _format_axes, _format_axes_3d


class Borefield:
    """
    Contains information regarding the dimensions and positions of boreholes within a borefield.

    Attributes
    ----------
    H : float or (nBoreholes,) array
        Borehole lengths (in meters).
    D : float or (nBoreholes,) array
        Borehole buried depths (in meters).
    r_b : float or (nBoreholes,) array
        Borehole radii (in meters).
    x : float or (nBoreholes,) array
        Position (in meters) of the head of the boreholes along the x-axis.
    y : float or (nBoreholes,) array
        Position (in meters) of the head of the boreholes along the y-axis.
    tilt : float or (nBoreholes,) array, optional
        Angle (in radians) from vertical of the axis of the boreholes.
        Default is 0.
    orientation : float or (nBoreholes,) array, optional
        Direction (in radians) of the tilt of the boreholes. Defaults to zero
        if the borehole is vertical.
        Default is 0.

    Notes
    -----
    Parameters that are equal for all boreholes can be provided as scalars.
    These parameters are then broadcasted to (nBoreholes,) arrays.

    """
    def __init__(
            self, H: npt.ArrayLike, D: npt.ArrayLike, r_b: npt.ArrayLike,
            x: npt.ArrayLike, y: npt.ArrayLike, tilt: npt.ArrayLike = 0.,
            orientation: npt.ArrayLike = 0.):
        # Convert x and y coordinates to arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        self.nBoreholes = np.maximum(len(x), len(y))

        # Broadcast all variables to arrays of length `nBoreholes`
        self.H = np.broadcast_to(H, self.nBoreholes)
        self.D = np.broadcast_to(D, self.nBoreholes)
        self.r_b = np.broadcast_to(r_b, self.nBoreholes)
        self.x = np.broadcast_to(x, self.nBoreholes)
        self.y = np.broadcast_to(y, self.nBoreholes)
        self.tilt = np.broadcast_to(tilt, self.nBoreholes)

        # Identify tilted boreholes
        self._is_tilted = np.broadcast_to(
            np.greater(np.abs(tilt), 1e-6),
            self.nBoreholes)
        # Vertical boreholes default to an orientation of zero
        if not np.any(self._is_tilted):
            self.orientation = np.broadcast_to(0., self.nBoreholes)
        elif np.all(self._is_tilted):
            self.orientation = np.broadcast_to(orientation, self.nBoreholes)
        else:
            self.orientation = np.multiply(orientation, self._is_tilted)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            # Returns a borehole object if only one borehole is indexed
            output_class = Borehole
        else:
            # Returns a borefield object for slices and lists of indexes
            output_class = Borefield
        return output_class(
            self.H[key], self.D[key], self.r_b[key], self.x[key], self.y[key],
            tilt=self.tilt[key], orientation=self.orientation[key])

    def __eq__(
            self, other_field: Union[Borehole, List[Borehole], Self]) -> bool:
        """Return True if other_field is the same as self."""
        # Convert other_field into Borefield object
        if isinstance(other_field, (Borehole, list)):
            other_field = Borefield.from_boreholes(other_field)
        check = bool(
            self.nBoreholes == other_field.nBoreholes
            and np.allclose(self.H, other_field.H)
            and np.allclose(self.D, other_field.D)
            and np.allclose(self.r_b, other_field.r_b)
            and np.allclose(self.x, other_field.x)
            and np.allclose(self.y, other_field.y)
            and np.allclose(self.tilt, other_field.tilt)
            and np.allclose(self.orientation, other_field.orientation)
            )
        return check

    def __len__(self) -> int:
        """Return the number of boreholes."""
        return self.nBoreholes

    def __ne__(
            self, other_field: Union[Borehole, List[Borehole], Self]) -> bool:
        """Return True if other_field is not the same as self."""
        check = not self == other_field
        return check

    def __add__(self,
                other_field: Union[Borehole, List[Borehole], Self]) -> Self:
        """Add two borefields together"""
        if not isinstance(other_field, (Borehole, list, self.__class__)):
            raise TypeError(
                f'Expected Borefield, list or Borehole input;'
                f' got {other_field}'
                )
        # List of boreholes
        field = self.to_boreholes()
        # Convert other_field to a list if it is a Borehole
        if isinstance(other_field, Borehole):
            other_field = [other_field]
        # Convert borefield to a list if it is a Borefield
        if isinstance(other_field, self.__class__):
            other_field = other_field.to_boreholes()
        return Borefield.from_boreholes(field + other_field)

    def __radd__(self,
                other_field: Union[Borehole, List[Borehole], Self]) -> Self:
        """Add two borefields together"""
        if not isinstance(other_field, (Borehole, list, self.__class__)):
            raise TypeError(
                f'Expected Borefield, list or Borehole input;'
                f' got {other_field}'
                )
        # List of boreholes
        field = self.to_boreholes()
        # Convert other_field to a list if it is a Borehole
        if isinstance(other_field, Borehole):
            other_field = [other_field]
        # Convert borefield to a list if it is a Borefield
        if isinstance(other_field, self.__class__):
            other_field = other_field.to_boreholes()
        return Borefield.from_boreholes(other_field + field)

    def evaluate_g_function(
            self,
            alpha: float,
            time: npt.ArrayLike,
            method: str = "equivalent",
            boundary_condition: str = "UBWT",
            options: Union[Dict[str, str], None] = None):
        """
        Evaluate the g-function of the bore field.

        Parameters
        ----------
        alpha : float
            Soil thermal diffusivity (in m2/s).
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        method : str, optional
            Method for the evaluation of the g-function. Should be one of

                - 'similarities' :
                    The accelerated method of Cimmino (2018)
                    [#borefield-Cimmin2018]_, using similarities in the bore
                    field to decrease the number of evaluations of the FLS
                    solution.
                - 'detailed' :
                    The classical superposition of the FLS solution. The FLS
                    solution is evaluated for all pairs of segments in the bore
                    field.
                - 'equivalent' :
                    The equivalent borehole method of Prieto and Cimmino (2021)
                    [#borefield-PriCim2021]_. Boreholes are assembled into
                    groups of boreholes that share similar borehole wall
                    temperatures and heat extraction rates. Each group is
                    represented by an equivalent borehole and the
                    group-to-group thermal interactions are calculated by the
                    FLS solution. This is an approximation of the
                    'similarities' method.

            Default is 'equivalent'.
        boundary_condition : str, optional
            Boundary condition for the evaluation of the g-function. Should be
            one of

                - 'UHTR' :
                    **Uniform heat transfer rate**. This corresponds to
                    boundary condition *BC-I* as defined by Cimmino and Bernier
                    (2014) [#borefield-CimBer2014]_.
                - 'UBWT' :
                    **Uniform borehole wall temperature**. This corresponds to
                    boundary condition *BC-III* as defined by Cimmino and
                    Bernier (2014) [#borefield-CimBer2014]_.

            Default is 'UBWT'.
        options : dict, optional
            A dictionary of solver options. All methods accept the following
            generic options:

                nSegments : int or list, optional
                    Number of line segments used per borehole, or list of
                    number of line segments used for each borehole.
                    Default is 8.
                segment_ratios : array, list of arrays, or callable, optional
                    Ratio of the borehole length represented by each segment.
                    The sum of ratios must be equal to 1. The shape of the
                    array is of (nSegments,) or list of (nSegments[i],). If
                    segment_ratios==None, segments of equal lengths are
                    considered. If a callable is provided, it must return an
                    array of size (nSegments,) when provided with nSegments (of
                    type int) as an argument, or an array of size
                    (nSegments[i],) when provided with an element of nSegments
                    (of type list).
                    Default is :func:`utilities.segment_ratios`.
                approximate_FLS : bool, optional
                    Set to true to use the approximation of the FLS solution of
                    Cimmino (2021) [#borefield-Cimmin2021]_. This approximation
                    does not require the numerical evaluation of any integral.
                    When using the 'equivalent' solver, the approximation is
                    only applied to the thermal response at the borehole
                    radius. Thermal interaction between boreholes is evaluated
                    using the FLS solution.
                    Default is False.
                nFLS : int, optional
                    Number of terms in the approximation of the FLS solution.
                    This parameter is unused if `approximate_FLS` is set to
                    False.
                    Default is 10. Maximum is 25.
                mQuad : int, optional
                    Number of Gauss-Legendre sample points for the integral
                    over :math:`u` in the inclined FLS solution.
                    Default is 11.
                linear_threshold : float, optional
                    Threshold time (in seconds) under which the g-function is
                    linearized. The g-function value is then interpolated
                    between 0 and its value at the threshold. If
                    `linear_threshold==None`, the g-function is linearized for
                    times `t < r_b**2 / (25 * self.alpha)`.
                    Default is None.
                disp : bool, optional
                    Set to true to print progression messages.
                    Default is False.
                kind : str, optional
                    Interpolation method used for segment-to-segment thermal
                    response factors. See documentation for
                    scipy.interpolate.interp1d.
                    Default is 'linear'.
                dtype : numpy dtype, optional
                    numpy data type used for matrices and vectors. Should be
                    one of numpy.single or numpy.double.
                    Default is numpy.double.

            The 'similarities' solver accepts the following method-specific
            options:

                disTol : float, optional
                    Relative tolerance on radial distance. Two distances
                    (d1, d2) between two pairs of boreholes are considered
                    equal if the difference between the two distances
                    (abs(d1-d2)) is below tolerance.
                    Default is 0.01.
                tol : float, optional
                    Relative tolerance on length and depth. Two lengths H1, H2
                    (or depths D1, D2) are considered equal if
                    abs(H1 - H2)/H2 < tol.
                    Default is 1.0e-6.

            The 'equivalent' solver accepts the following method-specific
            options:

                disTol : float, optional
                    Relative tolerance on radial distance. Two distances
                    (d1, d2) between two pairs of boreholes are considered
                    equal if the difference between the two distances
                    (abs(d1-d2)) is below tolerance.
                    Default is 0.01.
                tol : float, optional
                    Relative tolerance on length and depth. Two lengths H1, H2
                    (or depths D1, D2) are considered equal if
                    abs(H1 - H2)/H2 < tol.
                    Default is 1.0e-6.
                kClusters : int, optional
                    Increment on the minimum number of equivalent boreholes
                    determined by cutting the dendrogram of the bore field
                    given by the hierarchical agglomerative clustering method.
                    Increasing the value of this parameter increases the
                    accuracy of the method.
                    Default is 1.

        Notes
        -----
        - The 'equivalent' solver does not support inclined boreholes.
        - The g-function is linearized for times
          `t < r_b**2 / (25 * self.alpha)`. The g-function value is then
          interpolated between 0 and its value at the threshold.
        - This method only returns the values of the g-functions. The
          :class:`gfunction.gFunction` class provides additional
          capabilities for visualization boundary conditions.

        References
        ----------
        .. [#borefield-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
           semi-analytical method to generate g-functions for geothermal bore
           fields. International Journal of Heat and Mass Transfer, 70,
           641-650.
        .. [#borefield-Cimmin2018] Cimmino, M. (2018). Fast calculation of the
           g-functions of geothermal borehole fields using similarities in the
           evaluation of the finite line source solution. Journal of Building
           Performance Simulation, 11 (6), 655-668.
        .. [#borefield-PriCim2021] Prieto, C., & Cimmino, M.
           (2021). Thermal interactions in large irregular fields of geothermal
           boreholes: the method of equivalent borehole. Journal of Building
           Performance Simulation, 14 (4), 446-460.
        .. [#borefield-Cimmin2021] Cimmino, M. (2021). An approximation of the
           finite line source solution to model thermal interactions between
           geothermal boreholes. International Communications in Heat and Mass
           Transfer, 127, 105496.

        Returns
        -------
        gFunction : float or array
            Values of the g-function.

        """
        from .gfunction import gFunction
        if options is None:
            options = {}

        gfunc = gFunction(
            self,
            alpha,
            time=time,
            method=method,
            boundary_condition=boundary_condition,
            options=options,
        )

        return gfunc.gFunc

    def visualize_field(
            self, viewTop: bool = True, view3D: bool = True,
            labels: bool = True, showTilt: bool = True):
        """
        Plot the top view and 3D view of borehole positions.

        Parameters
        ----------
        viewTop : bool, optional
            Set to True to plot top view.
            Default is True
        view3D : bool, optional
            Set to True to plot 3D view.
            Default is True
        labels : bool, optional
            Set to True to annotate borehole indices to top view plot.
            Default is True
        showTilt : bool, optional
            Set to True to show borehole inclination on top view plot.
            Default is True

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        from ._mpl import plt

        # Configure figure and axes
        fig = _initialize_figure()
        if viewTop and view3D:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, projection='3d')
        elif viewTop:
            ax1 = fig.add_subplot(111)
        elif view3D:
            ax2 = fig.add_subplot(111, projection='3d')
        if viewTop:
            ax1.set_xlabel(r'$x$ [m]')
            ax1.set_ylabel(r'$y$ [m]')
            ax1.axis('equal')
            _format_axes(ax1)
        if view3D:
            ax2.set_xlabel(r'$x$ [m]')
            ax2.set_ylabel(r'$y$ [m]')
            ax2.set_zlabel(r'$z$ [m]')
            _format_axes_3d(ax2)
            ax2.invert_zaxis()

        # Bottom end of boreholes
        x_H = self.x + self.H * np.sin(self.tilt) * np.cos(self.orientation)
        y_H = self.y + self.H * np.sin(self.tilt) * np.sin(self.orientation)
        z_H = self.D + self.H * np.cos(self.tilt)

        # -------------------------------------------------------------------------
        # Top view
        # -------------------------------------------------------------------------
        if viewTop:
            if showTilt:
                ax1.plot(
                    np.stack((self.x, x_H), axis=0),
                    np.stack((self.y, y_H), axis=0),
                    'k--')
            ax1.plot(self.x, self.y, 'ko')
            if labels:
                for i, borehole in enumerate(self):
                    ax1.text(
                        borehole.x,
                        borehole.y,
                        f' {i}',
                        ha="left",
                        va="bottom")

        # -------------------------------------------------------------------------
        # 3D view
        # -------------------------------------------------------------------------
        if view3D:
            ax2.plot(self.x, self.y, self.D, 'ko')
            for i in range(self.nBoreholes):
                ax2.plot(
                    (self.x[i], x_H[i]),
                    (self.y[i], y_H[i]),
                    (self.D[i], z_H[i]),
                    'k-')

        if viewTop and view3D:
            plt.tight_layout(rect=[0, 0.0, 0.90, 1.0])
        else:
            plt.tight_layout()

        return fig

    def to_boreholes(self) -> List[Borehole]:
        """
        Return a list of boreholes in the bore field.

        Returns
        -------
        boreholes : list of Borehole objects
            List of boreholes in the bore field.

        """
        return list(self)

    def to_file(self, filename: str):
        """
        Save the bore field into a text file.

        Parameters
        ----------
        filename : str
            The filename in which to save the bore field.

        """
        data = np.stack(
            (self.x,
             self.y,
             self.H,
             self.D,
             self.r_b,
             self.tilt,
             self.orientation),
            axis=-1)
        np.savetxt(
            filename,
            data,
            delimiter='\t',
            header='x\ty\tH\tD\tr_b\ttilt\torientation')

    @classmethod
    def from_boreholes(
            cls, boreholes: Union[Borehole, List[Borehole]]) -> Self:
        """
        Build a borefield given a list of Borehole objects.

        Parameters
        ----------
        boreholes : list of Borehole objects
            List of boreholes in the bore field.

        Returns
        -------
        borefield : Borefield object
            Borefield object.

        """
        if isinstance(boreholes, Borehole):
            boreholes = [boreholes]
        # Build parameter arrays from borehole objects
        H = np.array([b.H for b in boreholes])
        D = np.array([b.D for b in boreholes])
        r_b = np.array([b.r_b for b in boreholes])
        tilt = np.array([b.tilt for b in boreholes])
        orientation = np.array([b.orientation for b in boreholes])
        x = np.array([b.x for b in boreholes])
        y = np.array([b.y for b in boreholes])
        # Create the borefield object
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Build a bore field given coordinates and dimensions provided in a text file.

        Parameters
        ----------
        filename : str
            Absolute path to the text file.

        Returns
        -------
        borefield : Borefield object
            Borefield object.

        Notes
        -----
        The text file should be formatted as follows::

            # x   y     H     D     r_b     tilt   orientation
            0.    0.    100.  2.5   0.075   0.     0.
            5.    0.    100.  2.5   0.075   0.     0.
            0.    5.    100.  2.5   0.075   0.     0.
            0.    10.   100.  2.5   0.075   0.     0.
            0.    20.   100.  2.5   0.075   0.     0.

        """
        # Load data from file
        data = np.loadtxt(filename, ndmin=2)
        # Build the bore field
        x = data[:, 0]
        y = data[:, 1]
        H = data[:, 2]
        D = data[:, 3]
        r_b = data[:, 4]
        if np.shape(data)[1] == 7:
            tilt = data[:, 5]
            orientation = data[:, 6]
        else:
            tilt = 0.
            orientation = 0.
        # Create the bore field
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield

    @classmethod
    def rectangle_field(
            cls, N_1: int, N_2: int, B_1: float, B_2: float, H: float,
            D: float, r_b: float, tilt: float = 0.,
            origin: Union[Tuple[float, float], None] = None) -> Self:
        """
        Build a bore field in a rectangular configuration.

        Parameters
        ----------
        N_1 : int
            Number of boreholes in the x direction.
        N_2 : int
            Number of boreholes in the y direction.
        B_1 : float
            Distance (in meters) between adjacent boreholes in the x direction.
        B_2 : float
            Distance (in meters) between adjacent boreholes in the y direction.
        H : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        tilt : float, optional
            Angle (in radians) from vertical of the axis of the borehole. The
            orientation of the tilt is orthogonal to the origin coordinate.
            Default is 0.
        origin : tuple, optional
            A coordinate indicating the origin of reference for orientation of
            boreholes.
            Default is the center of the rectangle.

        Returns
        -------
        borefield : Borefield object
            The rectangular bore field.

        Notes
        -----
        Boreholes located at the origin will remain vertical.

        Examples
        --------
        >>> borefield = gt.borefield.Borefield.rectangle_field(
            N_1=3, N_2=2, B_1=5., B_2=5., H=100., D=2.5, r_b=0.05)

        The bore field is constructed line by line. For N_1=3 and N_2=2, the
        bore field layout is as follows::

         3   4   5

         0   1   2

        """
        if origin is None:
            # When no origin is supplied, compute the origin to be at the
            # center of the rectangle
            x0 = (N_1 - 1) / 2 * B_1
            y0 = (N_2 - 1) / 2 * B_2
        else:
            x0, y0 = origin

        # Borehole positions and orientation
        x = np.tile(np.arange(N_1), N_2) * B_1
        y = np.repeat(np.arange(N_2), N_1) * B_2
        orientation = np.arctan2(y - y0, x - x0)
        # Boreholes are inclined only if they do not lie on the origin
        dis0 = np.sqrt((x - x0)**2 + (y - y0)**2)
        if np.any(dis0 < r_b):
            tilt = np.full(N_1 * N_2, tilt)
            tilt[dis0 < r_b] = 0.

        # Create the bore field
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield

    @classmethod
    def staggered_rectangle_field(
            cls, N_1: int, N_2: int, B_1: float, B_2: float, H: float,
            D: float, r_b: float, include_last_borehole: bool,
            tilt: float = 0.,
            origin: Union[Tuple[float, float], None] = None) -> Self:
        """
        Build a bore field in a staggered rectangular bore field configuration.

        Parameters
        ----------
        N_1 : int
            Number of borehole in the x direction.
        N_2 : int
            Number of borehole in the y direction.
        B_1 : float
            Distance (in meters) between adjacent boreholes in the x direction.
        B_2 : float
            Distance (in meters) between adjacent boreholes in the y direction.
        H : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        include_last_borehole : bool
            If True, then each row of boreholes has equal numbers of boreholes.
            If False, then the staggered rows have one borehole less so they
            are contained within the imaginary 'box' around the borefield.
        tilt : float, optional
            Angle (in radians) from vertical of the axis of the borehole. The
            orientation of the tilt is orthogonal to the origin coordinate.
            Default is 0.
        origin : tuple, optional
            A coordinate indicating the origin of reference for orientation of
            boreholes.
            Default is the center of the rectangle.

        Returns
        -------
        borefield : Borefield object
            The staggered rectangular bore field.

        Notes
        -----
        Boreholes located at the origin will remain vertical.

        Examples
        --------
        >>> borefield = gt.borefield.Borefield.staggered_rectangle_field(
            N_1=3, N_2=2, B_1=5., B_2=5., H=100., D=2.5, r_b=0.05,
            include_last_borehole=True)

        The bore field is constructed line by line. For N_1=3 and N_2=3, the
        bore field layout is as follows, if `include_last_borehole` is True::

         6    7    8
           3    4    5
         0    1    2

        and if `include_last_borehole` is False::

         5    6    7
           3    4
         0    1    2

        """
        if N_1 == 1 or N_2 == 1:
            borefield = cls.rectangle_field(
                N_1, N_2, B_1, B_2, H, D, r_b, tilt, origin)
            return borefield

        if origin is None:
            # When no origin is supplied, compute the origin to be at the
            # center of the rectangle
            if include_last_borehole:
                x0 = (N_1 - 0.5) / 2 * B_1
                y0 = (N_2 - 1) / 2 * B_2
            else:
                x0 = (N_1 - 1) / 2 * B_1
                y0 = (N_2 - 1) / 2 * B_2
        else:
            x0, y0 = origin

        # Borehole positions and orientation
        x = np.array(
            [i + (0.5 if j % 2 == 1 else 0.)
             for j in range(N_2)
             for i in range(N_1)
             if i < (N_1 - 1) or include_last_borehole or (j % 2 == 0)]) * B_1
        y = np.array(
            [j
             for j in range(N_2)
             for i in range(N_1)
             if i < (N_1 - 1) or include_last_borehole or (j % 2 == 0)]) * B_2
        orientation = np.arctan2(y - y0, x - x0)
        nBoreholes = len(x)
        # Boreholes are inclined only if they do not lie on the origin
        dis0 = np.sqrt((x - x0)**2 + (y - y0)**2)
        if np.any(dis0 < r_b):
            tilt = np.full(nBoreholes, tilt)
            tilt[dis0 < r_b] = 0.

        # Create the bore field
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield

    @classmethod
    def dense_rectangle_field(
            cls, N_1: int, N_2: int, B: float, H: float, D: float,
            r_b: float, include_last_borehole: bool, tilt: float = 0.,
            origin: Union[Tuple[float, float], None] = None) -> Self:
        """
        Build a bore field in a dense rectangular bore field configuration.

        Parameters
        ----------
        N_1 : int
            Number of borehole in the x direction.
        N_2 : int
            Number of borehole in the y direction.
        B : float
            Distance (in meters) between adjacent boreholes.
        H : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        include_last_borehole : bool
            If True, then each row of boreholes has equal numbers of boreholes.
            If False, then the staggered rows have one borehole less so they
            are contained within the imaginary 'box' around the borefield.
        tilt : float, optional
            Angle (in radians) from vertical of the axis of the borehole. The
            orientation of the tilt is orthogonal to the origin coordinate.
            Default is 0.
        origin : tuple, optional
            A coordinate indicating the origin of reference for orientation of
            boreholes.
            Default is the center of the rectangle.

        Returns
        -------
        borefield : Borefield object
            The dense rectangular bore field.

        Notes
        -----
        Boreholes located at the origin will remain vertical.

        Examples
        --------
        >>> borefield = gt.borefield.Borefield.dense_rectangle_field(
                N_1=3, N_2=2, B=5., H=100., D=2.5, r_b=0.05,
                include_last_borehole=True)

        The bore field is constructed line by line. For N_1=3 and N_2=3, the
        bore field layout is as follows, if `include_last_borehole` is True::

         6    7    8
           3    4    5
         0    1    2

        and if `include_last_borehole` is False::

         5    6    7
           3    4
         0    1    2

        """
        if N_1 > 1:
            B_2 = np.sqrt(3)/2 * B
        else:
            B_2 = B
        borefield = cls.staggered_rectangle_field(
            N_1, N_2, B, B_2, H, D, r_b, include_last_borehole,
            tilt=tilt, origin=origin)
        return borefield

    @classmethod
    def L_shaped_field(
            cls, N_1: int, N_2: int, B_1: float, B_2: float, H: float,
            D: float, r_b: float, tilt: float = 0.,
            origin: Union[Tuple[float, float], None] = None) -> Self:
        """
        Build a bore field in a L-shaped configuration.

        Parameters
        ----------
        N_1 : int
            Number of borehole in the x direction.
        N_2 : int
            Number of borehole in the y direction.
        B_1 : float
            Distance (in meters) between adjacent boreholes in the x direction.
        B_2 : float
            Distance (in meters) between adjacent boreholes in the y direction.
        H : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        tilt : float, optional
            Angle (in radians) from vertical of the axis of the borehole. The
            orientation of the tilt is orthogonal to the origin coordinate.
            Default is 0.
        origin : tuple, optional
            A coordinate indicating the origin of reference for orientation of
            boreholes.
            Default is origin is placed at the center of an assumed rectangle.

        Returns
        -------
        borefield : Borefield object
            The L-shaped bore field.

        Notes
        -----
        Boreholes located at the origin will remain vertical.

        Examples
        --------
        >>> borefield = gt.borefield.Borefield.L_shaped_field(
            N_1=3, N_2=2, B_1=5., B_2=5., H=100., D=2.5, r_b=0.05)

        The bore field is constructed line by line. For N_1=3 and N_2=2, the
        bore field layout is as follows::

         3

         0   1   2

        """
        if origin is None:
            # When no origin is supplied, compute the origin to be at the
            # center of the rectangle
            x0 = (N_1 - 1) / 2 * B_1
            y0 = (N_2 - 1) / 2 * B_2
        else:
            x0, y0 = origin

        # Borehole positions and orientation
        x = np.concatenate((np.arange(N_1), np.zeros(N_2 - 1))) * B_1
        y = np.concatenate((np.zeros(N_1), np.arange(1, N_2))) * B_2
        orientation = np.arctan2(y - y0, x - x0)
        nBoreholes = len(x)
        # Boreholes are inclined only if they do not lie on the origin
        dis0 = np.sqrt((x - x0)**2 + (y - y0)**2)
        if np.any(dis0 < r_b):
            tilt = np.full(nBoreholes, tilt)
            tilt[dis0 < r_b] = 0.

        # Create the bore field
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield

    @classmethod
    def U_shaped_field(
            cls, N_1: int, N_2: int, B_1: float, B_2: float, H: float,
            D: float, r_b: float, tilt: float = 0.,
            origin: Union[Tuple[float, float], None] = None) -> Self:
        """
        Build a bore field in a U-shaped configuration.

        Parameters
        ----------
        N_1 : int
            Number of borehole in the x direction.
        N_2 : int
            Number of borehole in the y direction.
        B_1 : float
            Distance (in meters) between adjacent boreholes in the x direction.
        B_2 : float
            Distance (in meters) between adjacent boreholes in the y direction.
        H : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        tilt : float, optional
            Angle (in radians) from vertical of the axis of the borehole. The
            orientation of the tilt is orthogonal to the origin coordinate.
            Default is 0.
        origin : tuple, optional
            A coordinate indicating the origin of reference for orientation of
            boreholes.
            Default is the center considering an outer rectangle.

        Returns
        -------
        borefield : Borefield object
            The U-shaped bore field.

        Notes
        -----
        Boreholes located at the origin will remain vertical.

        Examples
        --------
        >>> boreField = gt.borefield.Borefield.U_shaped_field(
            N_1=3, N_2=2, B_1=5., B_2=5., H=100., D=2.5, r_b=0.05)

        The bore field is constructed line by line. For N_1=3 and N_2=2, the
        bore field layout is as follows::

         3       4

         0   1   2

        """
        if origin is None:
            # When no origin is supplied, compute the origin to be at the
            # center of the rectangle
            x0 = (N_1 - 1) / 2 * B_1
            y0 = (N_2 - 1) / 2 * B_2
        else:
            x0, y0 = origin

        # Borehole positions and orientation
        n_vertical = np.minimum(N_1, 2)
        x = np.concatenate(
            (np.arange(N_1),
             np.tile(np.arange(n_vertical), N_2 - 1) * (N_1 - 1)
            )) * B_1
        y = np.concatenate(
            (np.zeros(N_1),
             np.repeat(np.arange(1, N_2), n_vertical)
            )) * B_2
        orientation = np.arctan2(y - y0, x - x0)
        nBoreholes = len(x)
        # Boreholes are inclined only if they do not lie on the origin
        dis0 = np.sqrt((x - x0)**2 + (y - y0)**2)
        if np.any(dis0 < r_b):
            tilt = np.full(nBoreholes, tilt)
            tilt[dis0 < r_b] = 0.

        # Create the bore field
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield

    @classmethod
    def box_shaped_field(
            cls, N_1: int, N_2: int, B_1: float, B_2: float, H: float,
            D: float, r_b: float, tilt: float = 0.,
            origin: Union[Tuple[float, float], None] = None) -> Self:
        """
        Build a bore field in a box-shaped configuration.

        Parameters
        ----------
        N_1 : int
            Number of borehole in the x direction.
        N_2 : int
            Number of borehole in the y direction.
        B_1 : float
            Distance (in meters) between adjacent boreholes in the x direction.
        B_2 : float
            Distance (in meters) between adjacent boreholes in the y direction.
        H : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        tilt : float, optional
            Angle (in radians) from vertical of the axis of the borehole. The
            orientation of the tilt is orthogonal to the origin coordinate.
            Default is 0.
        origin : tuple, optional
            A coordinate indicating the origin of reference for orientation of
            boreholes.
            Default is the center of the box.

        Returns
        -------
        borefield : Borefield object
            The box-shaped bore field.

        Notes
        -----
        Boreholes located at the origin will remain vertical.

        Examples
        --------
        >>> boreField = gt.borefield.Borefield.box_shaped_field(
            N_1=4, N_2=3, B_1=5., B_2=5., H=100., D=2.5, r_b=0.05)

        The bore field is constructed line by line. For N_1=4 and N_2=3, the
        bore field layout is as follows::

         6   7   8   9

         4           5

         0   1   2   3

        """
        if origin is None:
            # When no origin is supplied, compute the origin to be at the
            # center of the rectangle
            x0 = (N_1 - 1) / 2 * B_1
            y0 = (N_2 - 1) / 2 * B_2
        else:
            x0, y0 = origin

        # Borehole positions and orientation
        n_vertical = np.minimum(N_1, 2)
        n_middle = np.maximum(0, N_2 - 2)
        n_top = N_1 if N_2 > 1 else 0
        x = np.concatenate(
            (np.arange(N_1),
             np.tile(np.arange(n_vertical), n_middle) * (N_1 - 1),
             np.arange(n_top)
            )) * B_1
        y = np.concatenate(
            (np.zeros(N_1),
             np.repeat(np.arange(1, N_2 - 1), n_vertical),
             np.full(n_top, N_2 - 1)
            )) * B_2
        orientation = np.arctan2(y - y0, x - x0)
        nBoreholes = len(x)
        # Boreholes are inclined only if they do not lie on the origin
        dis0 = np.sqrt((x - x0)**2 + (y - y0)**2)
        if np.any(dis0 < r_b):
            tilt = np.full(nBoreholes, tilt)
            tilt[dis0 < r_b] = 0.

        # Create the bore field
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield

    @classmethod
    def circle_field(
            cls, N: int, R: float, H: float, D: float, r_b: float,
            tilt: float = 0.,
            origin: Union[Tuple[float, float], None] = None) -> Self:
        """
        Build a list of boreholes in a circular field configuration.

        Parameters
        ----------
        N : int
            Number of boreholes in the bore field.
        R : float
            Distance (in meters) of the boreholes from the center of the field.
        H : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        tilt : float, optional
            Angle (in radians) from vertical of the axis of the borehole. The
            orientation of the tilt is towards the exterior of the bore field.
            Default is 0.
        origin : tuple
            A coordinate indicating the origin of reference for orientation of
            boreholes.
            Default is the origin (0, 0).

        Returns
        -------
        borefield : Borefield object
            The circular shaped bore field.

        Notes
        -----
        Boreholes located at the origin will remain vertical.

        Examples
        --------
        >>> boreField = gt.borefield.Borefield.circle_field(
            N=8, R = 5., H=100., D=2.5, r_b=0.05)

        The bore field is constructed counter-clockwise. For N=8, the bore
        field layout is as follows::

               2
           3       1

         4           0

           5       7
               6

        """
        if origin is None:
            # When no origin is supplied, the origin is the center of the
            # circle
            x0 = 0.
            y0 = 0.
        else:
            x0, y0 = origin

        # Borehole positions and orientation
        x = R * np.cos(2 * np.pi * np.arange(N) / N)
        y = R * np.sin(2 * np.pi * np.arange(N) / N)
        orientation = np.arctan2(y - y0, x - x0)
        nBoreholes = len(x)
        # Boreholes are inclined only if they do not lie on the origin
        dis0 = np.sqrt((x - x0)**2 + (y - y0)**2)
        if np.any(dis0 < r_b):
            tilt = np.full(nBoreholes, tilt)
            tilt[dis0 < r_b] = 0.

        # Create the bore field
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield
