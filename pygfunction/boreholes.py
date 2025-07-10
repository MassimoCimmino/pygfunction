# -*- coding: utf-8 -*-
from typing import Union
import warnings

import numpy as np
from scipy.spatial.distance import pdist
from typing_extensions import Self  # for compatibility with Python <= 3.10

from .utilities import _initialize_figure, _format_axes, _format_axes_3d


class Borehole(object):
    """
    Contains information regarding the dimensions and position of a borehole.

    Attributes
    ----------
    H : float
        Borehole length (in meters).
    D : float
        Borehole buried depth (in meters).
    r_b : float
        Borehole radius (in meters).
    x : float
        Position (in meters) of the head of the borehole along the x-axis.
    y : float
        Position (in meters) of the head of the borehole along the y-axis.
    tilt : float, optional
        Angle (in radians) from vertical of the axis of the borehole.
        Default is 0.
    orientation : float, optional
        Direction (in radians) of the tilt of the borehole. Defaults to zero
        if the borehole is vertical.
        Default is 0.

    """
    def __init__(self, H, D, r_b, x, y, tilt=0., orientation=0.):
        self.H = float(H)      # Borehole length
        self.D = float(D)      # Borehole buried depth
        self.r_b = float(r_b)  # Borehole radius
        self.x = float(x)      # Borehole x coordinate position
        self.y = float(y)      # Borehole y coordinate position
        # Borehole inclination
        self.tilt = float(tilt)
        # Check if borehole is inclined
        self._is_tilted = np.abs(self.tilt) > 1.0e-6
        # Borehole orientation
        if self._is_tilted:
            self.orientation = float(orientation)
        else:
            self.orientation = 0.

    def __repr__(self):
        s = (f'Borehole(H={self.H}, D={self.D}, r_b={self.r_b}, x={self.x},'
             f' y={self.y}, tilt={self.tilt},'
             f' orientation={self.orientation})')
        return s

    def __add__(self, other: Union[Self, list]):
        """
        Adds two boreholes together to form a borefield
        """
        if not isinstance(other, (self.__class__, list)):
            # Check if other is a borefield and try the operation using
            # other.__radd__
            try:
                field = other.__radd__(self)
            except:
                # Invalid input
                raise TypeError(
                    f'Expected Borefield, list or Borehole input;'
                    f' got {other}'
                    )
        elif isinstance(other, list):
            # Create a borefield from the borehole and a list
            from .borefield import Borefield
            field = Borefield.from_boreholes([self] + other)
        else:
            # Create a borefield from the two boreholes
            from .borefield import Borefield
            field = Borefield.from_boreholes([self, other])
        return field

    def __radd__(self, other: Union[Self, list]):
        """
        Adds two boreholes together to form a borefield
        """
        if not isinstance(other, (self.__class__, list)):
            # Check if other is a borefield and try the operation using
            # other.__radd__
            try:
                field = other.__add__(self)
            except:
                # Invalid input
                raise TypeError(
                    f'Expected Borefield, list or Borehole input;'
                    f' got {other}'
                    )
        elif isinstance(other, list):
            # Create a borefield from the borehole and a list
            from .borefield import Borefield
            field = Borefield.from_boreholes(other + [self])
        else:
            # Create a borefield from the two boreholes
            from .borefield import Borefield
            field = Borefield.from_boreholes([other, self])
        return field

    def distance(self, target):
        """
        Evaluate the distance between the current borehole and a target
        borehole.

        Parameters
        ----------
        target : Borehole object
            Target borehole for which the distance is evaluated.

        Returns
        -------
        dis : float
            Distance (in meters) between current borehole and target borehole.

        .. Note::
           The smallest distance returned is equal to the borehole radius.
           This means that the distance between a borehole and itself is
           equal to r_b.

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
        >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
        >>> b1.distance(b2)
        5.0

        """
        dis = max(self.r_b,
                  np.sqrt((self.x - target.x)**2 + (self.y - target.y)**2))
        return dis

    def is_tilted(self):
        """
        Returns true if the borehole is inclined.

        Returns
        -------
        bool
            True if borehole is inclined.

        """
        return self._is_tilted

    def is_vertical(self):
        """
        Returns true if the borehole is vertical.

        Returns
        -------
        bool
            True if borehole is vertical.

        """
        return not self._is_tilted

    def position(self):
        """
        Returns the position of the borehole.

        Returns
        -------
        pos : tuple
            Position (x, y) (in meters) of the borehole.

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
        >>> b1.position()
        (5.0, 0.0)

        """
        pos = (self.x, self.y)
        return pos

    def segments(self, nSegments, segment_ratios=None):
        """
        Split a borehole into segments.

        Parameters
        ----------
        nSegments : int
            Number of segments.
        segment_ratios : array, optional
            Ratio of the borehole length represented by each segment. The
            sum of ratios must be equal to 1. The shape of the array is of
            (nSegments,). If segment_ratios==None, segments of equal lengths
            are considered.
            Default is None.

        Returns
        -------
        boreSegments : list
            List of borehole segments.

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
        >>> b1.segments(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(nSegments, 1. / nSegments)
        z = self._segment_edges(nSegments, segment_ratios=segment_ratios)[:-1]
        boreSegments = []
        for z_i, ratios in zip(z, segment_ratios):
            # Divide borehole into segments of equal length
            H = ratios * self.H
            # Buried depth of the i-th segment
            D = self.D + z_i * np.cos(self.tilt)
            # x-position
            x = self.x + z_i * np.sin(self.tilt) * np.cos(self.orientation)
            # y-position
            y = self.y + z_i * np.sin(self.tilt) * np.sin(self.orientation)
            # Add to list of segments
            boreSegments.append(
                Borehole(H, D, self.r_b, x, y,
                         tilt=self.tilt,
                         orientation=self.orientation))
        return boreSegments

    def _segment_edges(self, nSegments, segment_ratios=None):
        """
        Linear coordinates of the segment edges.

        Parameters
        ----------
        nSegments : int
            Number of segments.
        segment_ratios : array, optional
            Ratio of the borehole length represented by each segment. The
            sum of ratios must be equal to 1. The shape of the array is of
            (nSegments,). If segment_ratios==None, segments of equal lengths
            are considered.
            Default is None.

        Returns
        -------
        z : array
            Coordinates of the segment edges, with z=0. corresponding to the
            borehole head and z=H corresponding to the bottom edge of the
            borehole. The shape of the array is of (nSegments + 1,).

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
        >>> b1._segment_edges(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(nSegments, 1. / nSegments)
        z = np.concatenate(([0.], np.cumsum(segment_ratios))) * self.H
        return z

    def _segment_midpoints(self, nSegments, segment_ratios=None):
        """
        Linear coordinates of the segment midpoints.

        Parameters
        ----------
        nSegments : int
            Number of segments.
        segment_ratios : array, optional
            Ratio of the borehole length represented by each segment. The
            sum of ratios must be equal to 1. The shape of the array is of
            (nSegments,). If segment_ratios==None, segments of equal lengths
            are considered.
            Default is None.

        Returns
        -------
        z : array
            Coordinates of the segment midpoints, with z=0. corresponding to
            the borehole head and z=H corresponding to the bottom edge of the
            borehole. The shape of the array is of (nSegments,).

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
        >>> b1._segment_midpoints(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(nSegments, 1. / nSegments)
        z = self._segment_edges(nSegments, segment_ratios=segment_ratios)[:-1] \
            + segment_ratios * self.H / 2
        return z


class _EquivalentBorehole(object):
    """
    Contains information regarding the dimensions and position of an equivalent
    borehole.

    An equivalent borehole is meant to be representative of a group of
    boreholes, under the assumption that boreholes in this group share similar
    borehole wall temperatures and heat extraction rates. A methodology to
    identify equivalent boreholes is introduced in Prieto & Cimmino (2021)
    [#EqBorehole-PriCim2021].

    Parameters
    ----------
    boreholes : list of Borehole objects, or tuple
        Boreholes to be represented by the equivalent borehole. Alternatively,
        tuple of attributes (H, D, r_b, x, y, tilt=0., orientation=0.).

    Attributes
    ----------
    H : float
        Borehole length (in meters).
    D : float
        Borehole buried depth (in meters).
    r_b : float
        Borehole radius (in meters).
    x : (nBoreholes,) array
        Position (in meters) of the head of each borehole along the x-axis.
    y : (nBoreholes,) array
        Position (in meters) of the head of each borehole along the y-axis.
    tilt : float
        Angle (in radians) from vertical of the axis of the borehole.
    orientation : (nBoreholes,) array
        Direction (in radians) of the tilt of the borehole.
    nBoreholes : int
        Number of boreholes represented by the equivalent borehole.

    References
    ----------
    .. [#EqBorehole-PriCim2021] Prieto, C., & Cimmino, M., 2021. Thermal
        interactions in large irregular fields of geothermal boreholes: the
        method of equivalent borehole. Journal of Building Performance
        Simulation, 14 (4), 446-460.

    """
    def __init__(self, boreholes):
        if isinstance(boreholes[0], Borehole):
            self.H = boreholes[0].H
            self.D = boreholes[0].D
            self.r_b = boreholes[0].r_b
            self.x = np.array([b.x for b in boreholes])
            self.y = np.array([b.y for b in boreholes])
            self.tilt = boreholes[0].tilt
            self.orientation = np.array([b.orientation for b in boreholes])
        elif isinstance(boreholes[0], _EquivalentBorehole):
            self.H = boreholes[0].H
            self.D = boreholes[0].D
            self.r_b = boreholes[0].r_b
            self.x = np.concatenate([b.x for b in boreholes])
            self.y = np.concatenate([b.y for b in boreholes])
            self.tilt = boreholes[0].tilt
            self.orientation = np.concatenate(
                [b.orientation for b in boreholes])
        elif type(boreholes) is tuple:
            self.H, self.D, self.r_b, self.x, self.y = boreholes[:5]
            self.x = np.atleast_1d(self.x)
            self.y = np.atleast_1d(self.y)
            if len(boreholes)==7:
                self.tilt, self.orientation = boreholes[5:]

        self.nBoreholes = len(self.x)
        # Check if borehole is inclined
        self._is_tilted = np.abs(self.tilt) > 1.0e-6

    def distance(self, target):
        """
        Evaluate the distance between the current borehole and a target
        borehole.

        Parameters
        ----------
        target : _EquivalentBorehole object
            Target borehole for which the distances are evaluated.

        Returns
        -------
        dis : (nBoreholes_target, nBoreholes,) array
            Distances (in meters) between the boreholes represented by the
            equivalent borehole and the boreholes represented by another
            equivalent borehole.

        .. Note::
           The smallest distance returned is equal to the borehole radius.

        Examples
        --------
        >>> b1 = gt.boreholes._EquivalentBorehole((150., 4., 0.075, np.array([0., 5., 10]), np.array([0., 0., 0.])))
        >>> b2 = gt.boreholes._EquivalentBorehole((150., 4., 0.075, np.array([0.]), np.array([5.])))
        >>> b1.distance(b2)
        array([[ 5., 7.07106781, 11.18033989]])

        """
        dis = np.maximum(
            np.sqrt(
                np.add.outer(target.x, -self.x)**2 + np.add.outer(target.y, -self.y)**2),
            self.r_b)
        return dis

    def is_tilted(self):
        """
        Returns true if the borehole is inclined.

        Returns
        -------
        bool
            True if borehole is inclined.

        """
        return self._is_tilted

    def is_vertical(self):
        """
        Returns true if the borehole is vertical.

        Returns
        -------
        bool
            True if borehole is vertical.

        """
        return not self._is_tilted

    def position(self):
        """
        Returns the position of the boreholes represented by the equivalent
        borehole.

        Returns
        -------
        pos : tuple
            Positions (x, y) (in meters) of the borehole.

        Examples
        --------
        >>> b1 = gt.boreholes._EquivalentBorehole((150., 4., 0.075, np.array([0., 5., 10]), np.array([0., 0., 0.])))
        >>> b1.position()
        (array([ 0., 5., 10.]), array([0., 0., 0.]))

        """
        return (self.x, self.y)

    def segments(self, nSegments, segment_ratios=None):
        """
        Split an equivalent borehole into segments.

        Parameters
        ----------
        nSegments : int
            Number of segments.
        segment_ratios : array, optional
            Ratio of the borehole length represented by each segment. The
            sum of ratios must be equal to 1. The shape of the array is of
            (nSegments,). If segment_ratios==None, segments of equal lengths
            are considered.
            Default is None.
        Returns
        -------
        boreSegments : list
            List of borehole segments.

        Examples
        --------
        >>> b1 = gt.boreholes._EquivalentBorehole((150., 4., 0.075, np.array([0., 5., 10]), np.array([0., 0., 0.])))
        >>> b1.segments(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(nSegments, 1. / nSegments)
        z = self._segment_edges(nSegments, segment_ratios=segment_ratios)[:-1]
        segments = [_EquivalentBorehole(
            (ratios * self.H,
             self.D + z_i * np.cos(self.tilt),
             self.r_b,
             self.x + z_i * np.sin(self.tilt) * np.cos(self.orientation),
             self.y + z_i * np.sin(self.tilt) * np.sin(self.orientation),
             self.tilt,
             self.orientation)
            ) for z_i, ratios in zip(z, segment_ratios)]
        return segments

    def unique_distance(self, target, disTol=0.01):
        """
        Find unique distances between pairs of boreholes for a pair of
        equivalent boreholes.

        Parameters
        ----------
        target : _EquivalentBorehole object
            Target borehole for which the distances are evaluated.
        disTol : float, optional
            Relative tolerance on radial distance. Two distances
            (d1, d2) between two pairs of boreholes are considered equal if the
            difference between the two distances (abs(d1-d2)) is below tolerance.
            Default is 0.01.

        Returns
        -------
        dis : array
            Unique distances (in meters) between the boreholes represented by
            the equivalent borehole and the boreholes represented by another
            equivalent borehole.
        wDis : array
            Number of instances each of the unique distances arise.

        .. Note::
           The smallest distance returned is equal to the borehole radius.

        Examples
        --------
        >>> b1 = gt.boreholes._EquivalentBorehole((150., 4., 0.075, np.array([0., 5., 10]), np.array([0., 0., 0.])))
        >>> b2 = gt.boreholes._EquivalentBorehole((150., 4., 0.075, np.array([0., 5.]), np.array([5., 5.])))
        >>> b1.unique_distance(b2)
        (array([5., 7.07106781, 11.18033989]), array([2, 3, 1], dtype=int64))

        """
        # Find all distances between the boreholes, sorted and flattened
        all_dis = np.sort(self.distance(target).flatten())
        nDis = len(all_dis)

        # Find unique distances within tolerance
        dis = []
        wDis = []
        # Start the search at the first distance
        j0 = 0
        j1 = 1
        while j0 < nDis and j1 > 0:
            # Find the index of the first distance for which the distance is
            # outside tolerance to the current distance
            j1 = np.argmax(all_dis >= (1 + disTol) * all_dis[j0])
            if j1 > j0:
                # Add the average of the distances within tolerance to the
                # list of unique distances and store the number of distances
                dis.append(np.mean(all_dis[j0:j1]))
                wDis.append(j1-j0)
            else:
                # All remaining distances are within tolerance
                dis.append(np.mean(all_dis[j0:]))
                wDis.append(nDis-j0)
            j0 = j1

        return np.array(dis), np.array(wDis)

    def _segment_edges(self, nSegments, segment_ratios=None):
        """
        Linear coordinates of the segment edges.

        Parameters
        ----------
        nSegments : int
            Number of segments.
        segment_ratios : array, optional
            Ratio of the borehole length represented by each segment. The
            sum of ratios must be equal to 1. The shape of the array is of
            (nSegments,). If segment_ratios==None, segments of equal lengths
            are considered.
            Default is None.

        Returns
        -------
        z : array
            Coordinates of the segment edges, with z=0. corresponding to the
            borehole head and z=H corresponding to the bottom edge of the
            borehole. The shape of the array is of (nSegments + 1,).

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
        >>> b1._segment_edges(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(nSegments, 1. / nSegments)
        z = np.concatenate(([0.], np.cumsum(segment_ratios))) * self.H
        return z

    def _segment_midpoints(self, nSegments, segment_ratios=None):
        """
        Linear coordinates of the segment midpoints.

        Parameters
        ----------
        nSegments : int
            Number of segments.
        segment_ratios : array, optional
            Ratio of the borehole length represented by each segment. The
            sum of ratios must be equal to 1. The shape of the array is of
            (nSegments,). If segment_ratios==None, segments of equal lengths
            are considered.
            Default is None.

        Returns
        -------
        z : array
            Coordinates of the segment midpoints, with z=0. corresponding to
            the borehole head and z=H corresponding to the bottom edge of the
            borehole. The shape of the array is of (nSegments,).

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
        >>> b1._segment_midpoints(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(nSegments, 1. / nSegments)
        z = self._segment_edges(nSegments, segment_ratios=segment_ratios)[:-1] \
            + segment_ratios * self.H / 2
        return z


def find_duplicates(boreField, disp=False):
    """
    The distance method :func:`Borehole.distance` is utilized to find all
    duplicate boreholes in a boreField.
    This function considers a duplicate to be any pair of points that fall
    within each others radius. The lower index (i) is always stored in the
    0 position of the tuple, while the higher index (j) is stored in the 1
    position.

    Parameters
    ----------
    boreField : list
        A list of :class:`Borehole` objects
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    duplicate_pairs : list
        A list of tuples where the tuples are pairs of duplicates
    """
    # Number of boreholes
    n = len(boreField)
    # Max. borehole radius
    r_b = np.max([b.r_b for b in boreField])
    # Array of coordinates
    coordinates = np.array([[b.x, b.y] for b in boreField])
    # Find distance between each pair of boreholes
    distances = pdist(coordinates, 'euclidean')
    # Find duplicate boreholes
    duplicate_index = np.argwhere(distances < r_b)
    i = n - 2 - np.floor(
        np.sqrt(-8 * duplicate_index + 4 * n * (n - 1) - 7) / 2 - 0.5)
    j = duplicate_index + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2
    duplicate_pairs = [(int(ii), int(jj)) for (ii, jj) in zip(i, j)]

    if disp:
        print(' gt.boreholes.find_duplicates() '.center(50, '-'))
        print(f'The duplicate pairs of boreholes found:\n{duplicate_pairs}')
    return duplicate_pairs


def remove_duplicates(boreField, disp=False):
    """
    Removes all of the duplicates found from the duplicate pairs returned in
    :func:`check_duplicates`.

    For each pair of duplicates, the first borehole (with the lower index) is
    kept and the other (with the higher index) is removed.

    Parameters
    ----------
    boreField : list
        A list of :class:`Borehole` objects
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    new_boreField : list
        A boreField without duplicates
    """
    # Find duplicate pairs
    duplicate_pairs = find_duplicates(boreField, disp=disp)

    # Boreholes not to be included
    duplicate_bores = [pair[1] for pair in duplicate_pairs]
    # Initialize new borefield
    new_boreField = [b for i, b in enumerate(boreField) if i not in duplicate_bores]

    if disp:
        print(' gt.boreholes.remove_duplicates() '.center(50, '-'))
        n_duplicates = len(boreField) - len(new_boreField)
        print(f'The number of duplicates removed: {n_duplicates}')

    return new_boreField


def rectangle_field(N_1, N_2, B_1, B_2, H, D, r_b, tilt=0., origin=None):
    """
    Build a list of boreholes in a rectangular bore field configuration.

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
        Default is the center of the rectangle.

    Returns
    -------
    boreField : list of Borehole objects
        List of boreholes in the rectangular bore field.

    Notes
    -----
    Boreholes located at the origin will remain vertical.

    Examples
    --------
    >>> boreField = gt.boreholes.rectangle_field(N_1=3, N_2=2, B_1=5., B_2=5.,
                                                 H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=2, the bore
    field layout is as follows::

     3   4   5

     0   1   2

    """
    # This function is deprecated as of v2.3. It will be removed in v3.0.
    warnings.warn("`pygfunction.boreholes.rectangle_field` is "
                  "deprecated as of v2.3. It will be removed in v3.0. "
                  "Use the `pygfunction.borefield.Borefield` class instead.",
                  DeprecationWarning)
    borefield = []

    if origin is None:
        # When no origin is supplied, compute the origin to be at the center of
        # the rectangle
        x0 = (N_1 - 1) / 2 * B_1
        y0 = (N_2 - 1) / 2 * B_2
    else:
        x0, y0 = origin

    for j in range(N_2):
        for i in range(N_1):
            x = i * B_1
            y = j * B_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(Borehole(H, D, r_b, x, y, tilt=tilt,
                                          orientation=orientation))
            else:
                borefield.append(Borehole(H, D, r_b, x, y))

    return borefield


def staggered_rectangle_field(
        N_1, N_2, B_1, B_2, H, D, r_b, include_last_borehole, tilt=0.,
        origin=None):
    """
    Build a list of boreholes in a rectangular bore field configuration, with
    boreholes placed in a staggered configuration.

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
        If False, then the staggered rows have one borehole less so they are
        contained within the imaginary 'box' around the borefield.
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
    boreField : list of Borehole objects
        List of boreholes in the rectangular bore field.

    Notes
    -----
    Boreholes located at the origin will remain vertical.

    Examples
    --------
    >>> boreField = gt.boreholes.rectangle_field(N_1=3, N_2=2, B_1=5., B_2=5.,
                                                 H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=3, the bore
    field layout is as follows, if `include_last_borehole` is True::

     6    7    8
       3    4    5
     0    1    2

    and if `include_last_borehole` is False::

     5    6    7
       3    4
     0    1    2

    """
    # This function is deprecated as of v2.3. It will be removed in v3.0.
    warnings.warn("`pygfunction.boreholes.staggered_rectangle_field` is "
                  "deprecated as of v2.3. It will be removed in v3.0. "
                  "Use the `pygfunction.borefield.Borefield` class instead.",
                  DeprecationWarning)
    borefield = []

    if N_1 == 1 or N_2 == 1:
        return rectangle_field(N_1, N_2, B_1, B_2, H, D, r_b, tilt, origin)

    if origin is None:
        # When no origin is supplied, compute the origin to be at the center of
        # the rectangle
        if include_last_borehole:
            x0 = (N_1 - 1) / 2 * B_1
            y0 = (N_2 - 1) / 2 * B_2
        else:
            x0 = (N_1 - 0.5) / 2 * B_1
            y0 = (N_2 - 0.5) / 2 * B_2
    else:
        x0, y0 = origin

    for j in range(N_2):
        for i in range(N_1):
            x = i * B_1 + (B_1 / 2 if j % 2 == 1 else 0)
            y = j * B_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                if i < (N_1 - 1) or include_last_borehole or (j % 2 == 0):
                    borefield.append(
                        Borehole(
                            H, D, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                if i < (N_1 - 1) or include_last_borehole or (j % 2 == 0):
                    borefield.append(Borehole(H, D, r_b, x, y))

    return borefield


def dense_rectangle_field(
        N_1, N_2, B, H, D, r_b, include_last_borehole, tilt=0., origin=None):
    """
    Build a list of boreholes in a rectangular bore field configuration, with
    boreholes placed in a staggered configuration with uniform spacing between
    boreholes.

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
        If False, then the staggered rows have one borehole less so they are
        contained within the imaginary 'box' around the borefield.
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
    boreField : list of Borehole objects
        List of boreholes in the rectangular bore field.

    Notes
    -----
    Boreholes located at the origin will remain vertical.

    Examples
    --------
    >>> boreField = gt.boreholes.rectangle_field(
            N_1=3, N_2=2, B_1=5., B_2=5., H=100., D=2.5, r_b=0.05,
            include_last_borehole=True)

    The bore field is constructed line by line. For N_1=3 and N_2=3, the bore
    field layout is as follows, if `include_last_borehole` is True::

     6    7    8
       3    4    5
     0    1    2

    and if `include_last_borehole` is False::

     5    6    7
       3    4
     0    1    2

    """
    # This function is deprecated as of v2.3. It will be removed in v3.0.
    warnings.warn("`pygfunction.boreholes.dense_rectangle_field` is "
                  "deprecated as of v2.3. It will be removed in v3.0. "
                  "Use the `pygfunction.borefield.Borefield` class instead.",
                  DeprecationWarning)
    if N_1 == 1:
        # line field
        return rectangle_field(N_1, N_2, B, B, H, D, r_b, tilt, origin)
    return staggered_rectangle_field(
        N_1, N_2, B, np.sqrt(3)/2 * B, H, D, r_b, include_last_borehole,
        tilt=tilt, origin=origin)


def L_shaped_field(N_1, N_2, B_1, B_2, H, D, r_b, tilt=0., origin=None):
    """
    Build a list of boreholes in a L-shaped bore field configuration.

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
    boreField : list of Borehole objects
        List of boreholes in the L-shaped bore field.

    Notes
    -----
    Boreholes located at the origin will remain vertical.

    Examples
    --------
    >>> boreField = gt.boreholes.L_shaped_field(N_1=3, N_2=2, B_1=5., B_2=5.,
                                                H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=2, the bore
    field layout is as follows::

     3

     0   1   2

    """
    # This function is deprecated as of v2.3. It will be removed in v3.0.
    warnings.warn("`pygfunction.boreholes.L_shaped_field` is "
                  "deprecated as of v2.3. It will be removed in v3.0. "
                  "Use the `pygfunction.borefield.Borefield` class instead.",
                  DeprecationWarning)
    borefield = []

    if origin is None:
        # When no origin is supplied, compute the origin to be at the center of
        # the rectangle
        x0 = (N_1 - 1) / 2 * B_1
        y0 = (N_2 - 1) / 2 * B_2
    else:
        x0, y0 = origin

    for i in range(N_1):
        x = i * B_1
        y = 0.
        # The borehole is inclined only if it does not lie on the origin
        if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
            orientation = np.arctan2(y - y0, x - x0)
            borefield.append(
                Borehole(
                    H, D, r_b, x, y, tilt=tilt, orientation=orientation))
        else:
            borefield.append(Borehole(H, D, r_b, x, y))
    for j in range(1, N_2):
        x = 0.
        y = j * B_2
        # The borehole is inclined only if it does not lie on the origin
        if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
            orientation = np.arctan2(y - y0, x - x0)
            borefield.append(
                Borehole(
                    H, D, r_b, x, y, tilt=tilt, orientation=orientation))
        else:
            borefield.append(Borehole(H, D, r_b, x, y))

    return borefield


def U_shaped_field(N_1, N_2, B_1, B_2, H, D, r_b, tilt=0., origin=None):
    """
    Build a list of boreholes in a U-shaped bore field configuration.

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
    boreField : list of Borehole objects
        List of boreholes in the U-shaped bore field.

    Notes
    -----
    Boreholes located at the origin will remain vertical.

    Examples
    --------
    >>> boreField = gt.boreholes.U_shaped_field(N_1=3, N_2=2, B_1=5., B_2=5.,
                                                H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=2, the bore
    field layout is as follows::

     3       4

     0   1   2

    """
    # This function is deprecated as of v2.3. It will be removed in v3.0.
    warnings.warn("`pygfunction.boreholes.U_shaped_field` is "
                  "deprecated as of v2.3. It will be removed in v3.0. "
                  "Use the `pygfunction.borefield.Borefield` class instead.",
                  DeprecationWarning)
    borefield = []

    if origin is None:
        # When no origin is supplied, compute the origin to be at the center of
        # the rectangle
        x0 = (N_1 - 1) / 2 * B_1
        y0 = (N_2 - 1) / 2 * B_2
    else:
        x0, y0 = origin

    if N_1 > 2 and N_2 > 1:
        for i in range(N_1):
            x = i * B_1
            y = 0.
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        H, D, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(H, D, r_b, x, y))
        for j in range(1, N_2):
            x = 0.
            y = j * B_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        H, D, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(H, D, r_b, x, y))
            x = (N_1 - 1) * B_1
            y = j * B_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        H, D, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(H, D, r_b, x, y))
    else:
        borefield = rectangle_field(
            N_1, N_2, B_1, B_2, H, D, r_b, tilt=tilt, origin=origin)

    return borefield


def box_shaped_field(N_1, N_2, B_1, B_2, H, D, r_b, tilt=0, origin=None):
    """
    Build a list of boreholes in a box-shaped bore field configuration.

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
    boreField : list of Borehole objects
        List of boreholes in the box-shaped bore field.

    Notes
    -----
    Boreholes located at the origin will remain vertical.

    Examples
    --------
    >>> boreField = gt.boreholes.box_shaped_field(N_1=4, N_2=3, B_1=5., B_2=5.,
                                                  H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=4 and N_2=3, the bore
    field layout is as follows::

     6   7   8   9

     4           5

     0   1   2   3

    """
    # This function is deprecated as of v2.3. It will be removed in v3.0.
    warnings.warn("`pygfunction.boreholes.box_shaped_field` is "
                  "deprecated as of v2.3. It will be removed in v3.0. "
                  "Use the `pygfunction.borefield.Borefield` class instead.",
                  DeprecationWarning)
    borefield = []

    if origin is None:
        # When no origin is supplied, compute the origin to be at the center of
        # the rectangle
        x0 = (N_1 - 1) / 2 * B_1
        y0 = (N_2 - 1) / 2 * B_2
    else:
        x0, y0 = origin

    if N_1 > 2 and N_2 > 2:
        for i in range(N_1):
            x = i * B_1
            y = 0.
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        H, D, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(H, D, r_b, x, y))
            x = i * B_1
            y = (N_2 - 1) * B_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        H, D, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(H, D, r_b, x, y))
        for j in range(1, N_2-1):
            x = 0.
            y = j * B_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        H, D, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(H, D, r_b, x, y))
            x = (N_1 - 1) * B_1
            y = j * B_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        H, D, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(H, D, r_b, x, y))
    else:
        borefield = rectangle_field(
            N_1, N_2, B_1, B_2, H, D, r_b, tilt=tilt, origin=origin)

    return borefield


def circle_field(N, R, H, D, r_b, tilt=0., origin=None):
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
    boreField : list of Borehole objects
        List of boreholes in the circular shaped bore field.

    Notes
    -----
    Boreholes located at the origin will remain vertical.

    Examples
    --------
    >>> boreField = gt.boreholes.circle_field(N=8, R = 5., H=100., D=2.5,
                                              r_b=0.05)

    The bore field is constructed counter-clockwise. For N=8, the bore
    field layout is as follows::

           2
       3       1

     4           0

       5       7
           6

    """
    # This function is deprecated as of v2.3. It will be removed in v3.0.
    warnings.warn("`pygfunction.boreholes.circle_field` is "
                  "deprecated as of v2.3. It will be removed in v3.0. "
                  "Use the `pygfunction.borefield.Borefield` class instead.",
                  DeprecationWarning)
    borefield = []

    if origin is None:
        # When no origin is supplied, compute the origin to be at the center of
        # the rectangle
        x0 = 0.
        y0 = 0.
    else:
        x0, y0 = origin

    for i in range(N):
        x = R * np.cos(2 * np.pi * i / N)
        y = R * np.sin(2 * np.pi * i / N)
        orientation = np.arctan2(y - y0, x - x0)
        # The borehole is inclined only if it does not lie on the origin
        if np.sqrt((x - x0)**2 + (y - y0)**2) > r_b:
            orientation = np.arctan2(y - y0, x - x0)
            borefield.append(
                Borehole(
                    H, D, r_b, x, y, tilt=tilt, orientation=orientation))
        else:
            borefield.append(Borehole(H, D, r_b, x, y))

    return borefield


def field_from_file(filename):
    """
    Build a list of boreholes given coordinates and dimensions provided in a
    text file.

    Parameters
    ----------
    filename : str
        Absolute path to text file.

    Returns
    -------
    boreField : list of Borehole objects
        List of boreholes in the bore field.

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
        borefield.append(
            Borehole(H, D, r_b, x=x, y=y, tilt=tilt, orientation=orientation))

    return borefield


def visualize_field(
        borefield, viewTop=True, view3D=True, labels=True, showTilt=True):
    """
    Plot the top view and 3D view of borehole positions.

    Parameters
    ----------
    borefield : list
        List of boreholes in the bore field.
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
    # This function is deprecated as of v2.3. It will be removed in v3.0.
    warnings.warn("`pygfunction.boreholes.visualize_field` is "
                  "deprecated as of v2.3. It will be removed in v3.0. "
                  "Use the `pygfunction.borefield.Borefield` class instead.",
                  DeprecationWarning)

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

    # -------------------------------------------------------------------------
    # Top view
    # -------------------------------------------------------------------------
    if viewTop:
        i = 0   # Initialize borehole index
        for borehole in borefield:
            # Extract borehole parameters
            (x, y) = borehole.position()
            H = borehole.H
            tilt = borehole.tilt
            orientation = borehole.orientation
            # Add current borehole to the figure
            if showTilt:
                ax1.plot(
                    [x, x + H * np.sin(tilt) * np.cos(orientation)],
                    [y, y + H * np.sin(tilt) * np.sin(orientation)],
                    'k--')
            ax1.plot(x, y, 'ko')
            if labels: ax1.text(x, y,
                                f' {i}',
                                ha="left", va="bottom")
            i += 1  # Increment borehole index

    # -------------------------------------------------------------------------
    # 3D view
    # -------------------------------------------------------------------------
    if view3D:
        for borehole in borefield:
            # Position of head of borehole
            (x, y) = borehole.position()
            # Position of bottom of borehole
            x_H = x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)
            y_H = y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)
            z_H = borehole.D + borehole.H*np.cos(borehole.tilt)
            # Add current borehole to the figure
            ax2.plot(np.atleast_1d(x),
                     np.atleast_1d(y),
                     np.atleast_1d(borehole.D),
                     'ko')
            ax2.plot(np.array([x, x_H]),
                     np.array([y, y_H]),
                     np.array([borehole.D, z_H]),
                     'k-')


    if viewTop and view3D:
        plt.tight_layout(rect=[0, 0.0, 0.90, 1.0])
    else:
        plt.tight_layout()

    return fig
