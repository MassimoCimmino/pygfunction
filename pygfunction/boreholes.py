# -*- coding: utf-8 -*-
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.spatial.distance import pdist
from typing import Tuple, List, Union, Optional

from .utilities import _initialize_figure, _format_axes, _format_axes_3d


class Borehole:
    """
    Contains information regarding the dimensions and position of a borehole.

    Attributes
    ----------
    h : float
        Borehole length (in meters).
    d : float
        Borehole buried depth (in meters).
    r_b : float
        Borehole radius (in meters).
    x : float
        Position (in meters) of the head of the borehole along the x-axis.
    y : float
        Position (in meters) of the head of the borehole along the y-axis.
    tilt : float
        Angle (in radians) from vertical of the axis of the borehole.
    orientation : float
        Direction (in radians) of the tilt of the borehole.

    """

    def __init__(self, h: float, d: float, r_b: float, x: float, y: float, tilt: float = 0., orientation: Optional[float] = None):
        self.h: float = float(h)  # Borehole length
        self.d: float = float(d)  # Borehole buried depth
        self.r_b: float = float(r_b)  # Borehole radius
        self.x: float = float(x)  # Borehole x coordinate position
        self.y: float = float(y)  # Borehole y coordinate position
        # Borehole inclination
        self.tilt: float = float(tilt)
        # Borehole orientation
        self.orientation: float = float(orientation) if orientation is not None else 0.
        # Check if borehole is inclined
        self._is_tilted: bool = np.abs(self.tilt) > 1.0e-6

    def __repr__(self):
        s = (f'Borehole(H={self.h}, D={self.d}, r_b={self.r_b}, x={self.x},'
             f' y={self.y}, tilt={self.tilt},'
             f' orientation={self.orientation})')
        return s

    def distance(self, target: Borehole) -> float:
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
        >>> b1 = gt.boreholes.Borehole(h=150., d=4., r_b=0.075, x=0., y=0.)
        >>> b2 = gt.boreholes.Borehole(h=150., d=4., r_b=0.075, x=5., y=0.)
        >>> b1.distance(b2)
        5.0

        """
        return max(self.r_b, np.sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2))

    def is_tilted(self) -> bool:
        """
        Returns true if the borehole is inclined.

        Returns
        -------
        bool
            True if borehole is inclined.

        """
        return self._is_tilted

    def is_vertical(self) -> bool:
        """
        Returns true if the borehole is vertical.

        Returns
        -------
        bool
            True if borehole is vertical.

        """
        return not self._is_tilted

    def position(self) -> Tuple[float, float]:
        """
        Returns the position of the borehole.

        Returns
        -------
        pos : tuple
            Position (x, y) (in meters) of the borehole.

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(h=150., d=4., r_b=0.075, x=5., y=0.)
        >>> b1.position()
        (5.0, 0.0)

        """
        return self.x, self.y

    def segments(self, n_segments: int, segment_ratios: Optional[NDArray[np.float64]] = None) -> List[Borehole]:
        """
        Split a borehole into segments.

        Parameters
        ----------
        n_segments : int
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
        >>> b1 = gt.boreholes.Borehole(h=150., d=4., r_b=0.075, x=5., y=0.)
        >>> b1.segments(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(n_segments, 1. / n_segments)
        z = self._segment_edges(n_segments, segment_ratios=segment_ratios)[:-1]
        bore_segments = []
        for z_i, ratios in zip(z, segment_ratios):
            # Divide borehole into segments of equal length
            H = ratios * self.h
            # Buried depth of the i-th segment
            D = self.d + z_i * np.cos(self.tilt)
            # x-position
            x = self.x + z_i * np.sin(self.tilt) * np.cos(self.orientation)
            # y-position
            y = self.y + z_i * np.sin(self.tilt) * np.sin(self.orientation)
            # Add to list of segments
            bore_segments.append(
                Borehole(H, D, self.r_b, x, y,
                         tilt=self.tilt,
                         orientation=self.orientation))
        return bore_segments

    def _segment_edges(self, n_segments: int, segment_ratios: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        """
        Linear coordinates of the segment edges.

        Parameters
        ----------
        n_segments : int
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
        >>> b1 = gt.boreholes.Borehole(h=150., d=4., r_b=0.075, x=5., y=0.)
        >>> b1._segment_edges(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(n_segments, 1. / n_segments)
        z = np.concatenate(([0.], np.cumsum(segment_ratios))) * self.h
        return z

    def _segment_midpoints(self, n_segments: int, segment_ratios: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        """
        Linear coordinates of the segment midpoints.

        Parameters
        ----------
        n_segments : int
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
        >>> b1 = gt.boreholes.Borehole(h=150., d=4., r_b=0.075, x=5., y=0.)
        >>> b1._segment_midpoints(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(n_segments, 1. / n_segments)
        z = self._segment_edges(n_segments, segment_ratios=segment_ratios)[:-1] \
            + segment_ratios * self.h / 2
        return z


class _EquivalentBorehole:
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

    def __init__(self, boreholes: Union[List[Borehole], List[_EquivalentBorehole], tuple]):
        if isinstance(boreholes[0], Borehole):
            self.h: float = boreholes[0].h
            self.d: float = boreholes[0].d
            self.r_b: float = boreholes[0].r_b
            self.x: NDArray[np.float64] = np.array([b.x for b in boreholes])
            self.y: NDArray[np.float64] = np.array([b.y for b in boreholes])
            self.tilt: float = boreholes[0].tilt
            self.orientation = np.array([b.orientation for b in boreholes])
        elif isinstance(boreholes[0], _EquivalentBorehole):
            self.h = boreholes[0].h
            self.d = boreholes[0].d
            self.r_b = boreholes[0].r_b
            self.x = np.concatenate([b.x for b in boreholes])
            self.y = np.concatenate([b.y for b in boreholes])
            self.tilt = boreholes[0].tilt
            self.orientation = np.concatenate(
                [b.orientation for b in boreholes])
        elif type(boreholes) is tuple:
            self.h, self.d, self.r_b, self.x, self.y = boreholes[:5]
            self.x = np.atleast_1d(self.x)
            self.y = np.atleast_1d(self.y)
            if len(boreholes) == 7:
                self.tilt, self.orientation = boreholes[5:]

        self.nBoreholes: int = len(self.x)
        # Check if borehole is inclined
        self._is_tilted: bool = np.abs(self.tilt) > 1.0e-6

    def distance(self, target: _EquivalentBorehole) -> NDArray[np.float64]:
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
        return np.maximum(np.sqrt(np.add.outer(target.x, -self.x) ** 2 + np.add.outer(target.y, -self.y) ** 2), self.r_b)

    def is_tilted(self) -> bool:
        """
        Returns true if the borehole is inclined.

        Returns
        -------
        bool
            True if borehole is inclined.

        """
        return self._is_tilted

    def is_vertical(self) -> bool:
        """
        Returns true if the borehole is vertical.

        Returns
        -------
        bool
            True if borehole is vertical.

        """
        return not self._is_tilted

    def position(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
        return self.x, self.y

    def segments(self, n_segments: int, segment_ratios: Optional[NDArray[np.float64]] = None) -> List[_EquivalentBorehole]:
        """
        Split an equivalent borehole into segments.

        Parameters
        ----------
        n_segments : int
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
            segment_ratios = np.full(n_segments, 1. / n_segments)
        z = self._segment_edges(n_segments, segment_ratios=segment_ratios)[:-1]
        segments = [_EquivalentBorehole(
            (ratios * self.h,
             self.d + z_i * np.cos(self.tilt),
             self.r_b,
             self.x + z_i * np.sin(self.tilt) * np.cos(self.orientation),
             self.y + z_i * np.sin(self.tilt) * np.sin(self.orientation),
             self.tilt,
             self.orientation)
        ) for z_i, ratios in zip(z, segment_ratios)]
        return segments

    def unique_distance(self, target: _EquivalentBorehole, dis_tol: float = 0.01) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Find unique distances between pairs of boreholes for a pair of
        equivalent boreholes.

        Parameters
        ----------
        target : _EquivalentBorehole object
            Target borehole for which the distances are evaluated.
        dis_tol : float, optional
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
        n_dis = len(all_dis)

        # Find unique distances within tolerance
        dis: List[float] = []
        w_dis: List[float] = []
        # Start the search at the first distance
        j0 = 0
        j1 = 1
        while j0 < n_dis and j1 > 0:
            # Find the index of the first distance for which the distance is
            # outside tolerance to the current distance
            j1 = np.argmax(all_dis >= (1 + dis_tol) * all_dis[j0])
            if j1 > j0:
                # Add the average of the distances within tolerance to the
                # list of unique distances and store the number of distances
                dis.append(np.mean(all_dis[j0:j1])[0])
                w_dis.append(j1 - j0)
            else:
                # All remaining distances are within tolerance
                dis.append(np.mean(all_dis[j0:])[0])
                w_dis.append(n_dis - j0)
            j0 = j1

        return np.array(dis), np.array(w_dis)

    def _segment_edges(self, n_segments: int, segment_ratios: Optional[np.float64] = None) -> float:
        """
        Linear coordinates of the segment edges.

        Parameters
        ----------
        n_segments : int
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
        >>> b1 = gt.boreholes.Borehole(h=150., d=4., r_b=0.075, x=5., y=0.)
        >>> b1._segment_edges(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(n_segments, 1. / n_segments)
        return np.concatenate(([0.], np.cumsum(segment_ratios))) * self.h

    def _segment_midpoints(self, n_segments: int, segment_ratios: Optional[np.float64] = None) -> float:
        """
        Linear coordinates of the segment midpoints.

        Parameters
        ----------
        n_segments : int
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
        >>> b1 = gt.boreholes.Borehole(h=150., d=4., r_b=0.075, x=5., y=0.)
        >>> b1._segment_midpoints(5)

        """
        if segment_ratios is None:
            segment_ratios = np.full(n_segments, 1. / n_segments)
        return self._segment_edges(n_segments, segment_ratios=segment_ratios)[:-1] + segment_ratios * self.h / 2


def find_duplicates(bore_field: List[Borehole], disp: bool = False):
    """
    The distance method :func:`Borehole.distance` is utilized to find all
    duplicate boreholes in a boreField.
    This function considers a duplicate to be any pair of points that fall
    within each others radius. The lower index (i) is always stored in the
    0 position of the tuple, while the higher index (j) is stored in the 1
    position.

    Parameters
    ----------
    bore_field : list
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
    n = len(bore_field)
    # Max. borehole radius
    r_b = np.max([b.r_b for b in bore_field])
    # Array of coordinates
    coordinates = np.array([[b.x, b.y] for b in bore_field])
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


def remove_duplicates(bore_field: List[Borehole], disp=False):
    """
    Removes all of the duplicates found from the duplicate pairs returned in
    :func:`check_duplicates`.

    For each pair of duplicates, the first borehole (with the lower index) is
    kept and the other (with the higher index) is removed.

    Parameters
    ----------
    bore_field : list
        A list of :class:`Borehole` objects
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    new_bore_field : list
        A boreField without duplicates
    """
    # Find duplicate pairs
    duplicate_pairs = find_duplicates(bore_field, disp=disp)

    # Boreholes not to be included
    duplicate_bores = [pair[1] for pair in duplicate_pairs]
    # Initialize new borefield
    new_bore_field = [b for i, b in enumerate(bore_field) if i not in duplicate_bores]

    if disp:
        print(' gt.boreholes.remove_duplicates() '.center(50, '-'))
        n_duplicates = len(bore_field) - len(new_bore_field)
        print(f'The number of duplicates removed: {n_duplicates}')

    return new_bore_field


def _orientation(x: float, x0: float, y: float, y0: float, r_b: float) -> Optional[float]:
    dx = x - x0
    dy = y - y0
    return np.arctan2(dy, dx) if np.sqrt(dx * dx + dy * dy) > r_b else None


def rectangle_field(n_1: int, n_2: int, b_1: float, b_2: float, h: float, d: float, r_b: float, tilt: float = 0.,
                    origin: Optional[Tuple[float, float]] = None) -> List[Borehole]:
    """
    Build a list of boreholes in a rectangular bore field configuration.

    Parameters
    ----------
    n_1 : int
        Number of borehole in the x direction.
    n_2 : int
        Number of borehole in the y direction.
    b_1 : float
        Distance (in meters) between adjacent boreholes in the x direction.
    b_2 : float
        Distance (in meters) between adjacent boreholes in the y direction.
    h : float
        Borehole length (in meters).
    d : float
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
    >>> boreField = gt.boreholes.rectangle_field(n_1=3, n_2=2, b_1=5., b_2=5.,
                                                 H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=2, the bore
    field layout is as follows::

     3   4   5

     0   1   2

    """

    x0, y0 = origin if origin is not None else ((n_1 - 1) / 2 * b_1, (n_2 - 1) / 2 * b_2)

    return [Borehole(h, d, r_b, i * b_1, j * b_2, tilt=tilt, orientation=_orientation(i * b_1, x0, j * b_2, y0, r_b)) for j in range(n_2) for i in range(n_1)]


def l_shaped_field(n_1: int, n_2: int, b_1: float, b_2: float, h: float, d: float, r_b: float, tilt: float = 0.,
                   origin: Optional[Tuple[float, float]] = None) -> List[Borehole]:
    """
    Build a list of boreholes in a L-shaped bore field configuration.

    Parameters
    ----------
    n_1 : int
        Number of borehole in the x direction.
    n_2 : int
        Number of borehole in the y direction.
    b_1 : float
        Distance (in meters) between adjacent boreholes in the x direction.
    b_2 : float
        Distance (in meters) between adjacent boreholes in the y direction.
    h : float
        Borehole length (in meters).
    d : float
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
    >>> boreField = gt.boreholes.l_shaped_field(n_1=3, n_2=2, b_1=5., b_2=5.,
                                                H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=2, the bore
    field layout is as follows::

     3

     0   1   2

    """

    x0, y0 = origin if origin is not None else ((n_1 - 1) / 2 * b_1, (n_2 - 1) / 2 * b_2)

    borefield: List[Borehole] = [Borehole(h, d, r_b, i * b_1, 0, tilt=tilt, orientation=_orientation(i * b_1, x0, 0, y0, r_b)) for i in range(n_1)]

    borefield += [Borehole(h, d, r_b, 0, j * b_2, tilt=tilt, orientation=_orientation(0, x0, j * b_2, y0, r_b)) for j in range(1, n_2)]

    return borefield


def u_shaped_field(n_1: int, n_2: int, b_1: float, b_2: float, h: float, d: float, r_b: float, tilt: float = 0.,
                   origin: Optional[Tuple[float, float]] = None) -> List[Borehole]:
    """
    Build a list of boreholes in a U-shaped bore field configuration.

    Parameters
    ----------
    n_1 : int
        Number of borehole in the x direction.
    n_2 : int
        Number of borehole in the y direction.
    b_1 : float
        Distance (in meters) between adjacent boreholes in the x direction.
    b_2 : float
        Distance (in meters) between adjacent boreholes in the y direction.
    h : float
        Borehole length (in meters).
    d : float
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
    >>> boreField = gt.boreholes.u_shaped_field(n_1=3, n_2=2, b_1=5., b_2=5.,
                                                H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=2, the bore
    field layout is as follows::

     3       4

     0   1   2

    """
    borefield: List[Borehole] = []

    x0, y0 = origin if origin is not None else ((n_1 - 1) / 2 * b_1, (n_2 - 1) / 2 * b_2)

    if n_1 > 2 and n_2 > 1:
        for i in range(n_1):
            x = i * b_1
            y = 0.
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        h, d, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(h, d, r_b, x, y))
        for j in range(1, n_2):
            x = 0.
            y = j * b_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        h, d, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(h, d, r_b, x, y))
            x = (n_1 - 1) * b_1
            y = j * b_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        h, d, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(h, d, r_b, x, y))
        return borefield
    return rectangle_field(n_1, n_2, b_1, b_2, h, d, r_b, tilt=tilt, origin=origin)


def box_shaped_field(n_1: int, n_2: int, b_1: float, b_2: float, h: float, d: float, r_b: float, tilt: float = 0.,
                     origin: Optional[Tuple[float, float]] = None) -> List[Borehole]:
    """
    Build a list of boreholes in a box-shaped bore field configuration.

    Parameters
    ----------
    n_1 : int
        Number of borehole in the x direction.
    n_2 : int
        Number of borehole in the y direction.
    b_1 : float
        Distance (in meters) between adjacent boreholes in the x direction.
    b_2 : float
        Distance (in meters) between adjacent boreholes in the y direction.
    h : float
        Borehole length (in meters).
    d : float
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
    >>> boreField = gt.boreholes.box_shaped_field(n_1=4, n_2=3, b_1=5., b_2=5.,
                                                  H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=4 and N_2=3, the bore
    field layout is as follows::

     6   7   8   9

     4           5

     0   1   2   3

    """
    borefield = []

    x0, y0 = origin if origin is not None else ((n_1 - 1) / 2 * b_1, (n_2 - 1) / 2 * b_2)

    if n_1 > 2 and n_2 > 2:
        for i in range(n_1):
            x = i * b_1
            y = 0.
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        h, d, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(h, d, r_b, x, y))
            x = i * b_1
            y = (n_2 - 1) * b_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        h, d, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(h, d, r_b, x, y))
        for j in range(1, n_2 - 1):
            x = 0.
            y = j * b_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        h, d, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(h, d, r_b, x, y))
            x = (n_1 - 1) * b_1
            y = j * b_2
            # The borehole is inclined only if it does not lie on the origin
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > r_b:
                orientation = np.arctan2(y - y0, x - x0)
                borefield.append(
                    Borehole(
                        h, d, r_b, x, y, tilt=tilt, orientation=orientation))
            else:
                borefield.append(Borehole(h, d, r_b, x, y))
        return borefield
    return rectangle_field(n_1, n_2, b_1, b_2, h, d, r_b, tilt=tilt, origin=origin)


def circle_field(n: int, r: float, h: float, d: float, r_b: float, tilt: float = 0., origin: Optional[Tuple[float, float]] = None) -> List[Borehole]:
    """
    Build a list of boreholes in a circular field configuration.

    Parameters
    ----------
    n : int
        Number of boreholes in the bore field.
    r : float
        Distance (in meters) of the boreholes from the center of the field.
    h : float
        Borehole length (in meters).
    d : float
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
    >>> boreField = gt.boreholes.circle_field(n=8, r = 5., h=100., d=2.5, r_b=0.05)

    The bore field is constructed counter-clockwise. For N=8, the bore
    field layout is as follows::

           2
       3       1

     4           0

       5       7
           6

    """

    x0, y0 = origin if origin is not None else (0, 0)

    return [Borehole(h, d, r_b, r * np.cos(2 * pi * i / n), r * np.sin(2 * pi * i / n), tilt=tilt,
                     orientation=_orientation(r * np.cos(2 * pi * i / n), x0, r * np.sin(2 * pi * i / n), y0, r_b)) for i in range(n)]


def field_from_file(filename: str) -> List[Borehole]:
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

        # x   y     h     d     r_b     tilt   orientation
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
        h = line[2]
        d = line[3]
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
            Borehole(h, d, r_b, x=x, y=y, tilt=tilt, orientation=orientation))

    return borefield


def visualize_field(borefield: List[Borehole], view_top: bool = True, view_3d: bool = True, labels: bool = True, show_tilt: bool = True) -> plt.figure:
    """
    Plot the top view and 3D view of borehole positions.

    Parameters
    ----------
    borefield : list
        List of boreholes in the bore field.
    view_top : bool, optional
        Set to True to plot top view.
        Default is True
    view_3d : bool, optional
        Set to True to plot 3D view.
        Default is True
    labels : bool, optional
        Set to True to annotate borehole indices to top view plot.
        Default is True
    show_tilt : bool, optional
        Set to True to show borehole inclination on top view plot.
        Default is True

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    from mpl_toolkits.mplot3d import Axes3D

    # Configure figure and axes
    fig = _initialize_figure()
    if view_top and view_3d:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')
    elif view_top:
        ax1 = fig.add_subplot(111)
    elif view_3d:
        ax2 = fig.add_subplot(111, projection='3d')
    if view_top:
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        _format_axes(ax1)
    if view_3d:
        ax2.set_xlabel(r'$x$ [m]')
        ax2.set_ylabel(r'$y$ [m]')
        ax2.set_zlabel(r'$z$ [m]')
        _format_axes_3d(ax2)
        ax2.invert_zaxis()

    # -------------------------------------------------------------------------
    # Top view
    # -------------------------------------------------------------------------
    if view_top:
        i = 0  # Initialize borehole index
        for borehole in borefield:
            # Extract borehole parameters
            (x, y) = borehole.position()
            H = borehole.h
            tilt = borehole.tilt
            orientation = borehole.orientation
            # Add current borehole to the figure
            if show_tilt:
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
    if view_3d:
        for borehole in borefield:
            # Position of head of borehole
            (x, y) = borehole.position()
            # Position of bottom of borehole
            x_H = x + borehole.h * np.sin(borehole.tilt) * np.cos(borehole.orientation)
            y_H = y + borehole.h * np.sin(borehole.tilt) * np.sin(borehole.orientation)
            z_H = borehole.d + borehole.h * np.cos(borehole.tilt)
            # Add current borehole to the figure
            ax2.plot(np.atleast_1d(x),
                     np.atleast_1d(y),
                     np.atleast_1d(borehole.d),
                     'ko')
            ax2.plot(np.array([x, x_H]),
                     np.array([y, y_H]),
                     np.array([borehole.d, z_H]),
                     'k-')

    if view_top and view_3d:
        plt.tight_layout(rect=[0, 0.0, 0.90, 1.0])
    else:
        plt.tight_layout()

    return fig
