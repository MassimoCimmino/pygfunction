# -*- coding: utf-8 -*-
from typing import Union, List, Dict, Self

import numpy as np
import numpy.typing as npt

from .boreholes import Borehole

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
            np.greater(tilt, 1e-6),
            self.nBoreholes)
        # Vertical boreholes default to an orientation of zero
        if not np.any(self._is_tilted):
            self.orientation = np.broadcast_to(0., self.nBoreholes)
        elif np.all(self._is_tilted):
            self.orientation = np.broadcast_to(orientation, self.nBoreholes)
        else:
            self.orientation = np.multiply(orientation, self._is_tilted)
        pass

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

    def __len__(self):
        """Returns the number of boreholes."""
        return self.nBoreholes

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Build a borefield given coordinates and dimensions provided in a text file.

        Parameters
        ----------
        filename : str
            Absolute path to text file.

        Returns
        -------
        boreField : Borefield object
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
        borefield = []
        for line in data:
            x = line[0]
            y = line[1]
            H = line[2]
            D = line[3]
            r_b = line[4]
            # Previous versions of pygfunction only required up to line[4].
            # Now check to see if tilt and orientation exist.
            if len(line) == 7:
                tilt = line[5]
                orientation = line[6]
            else:
                tilt = 0.
                orientation = 0.
        # Create the borefield object
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield

    @classmethod
    def from_boreholes(
            cls, boreholes: Union[Borehole, List[Borehole]]) -> Self:
        """
        Build a borefield given coordinates and dimensions provided in a text file.

        Parameters
        ----------
        boreholes : list of Borehole objects
            List of boreholes in the bore field.

        Returns
        -------
        boreField : Borefield object
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
                    Cimmino (2021) [#gFunction-Cimmin2021]_. This approximation
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
