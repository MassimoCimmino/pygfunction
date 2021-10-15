# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi

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
    tilt : float
        Angle (in radians) from vertical of the axis of the borehole.
    orientation : float
        Direction (in radians) of the tilt of the borehole.

    """
    def __init__(self, H, D, r_b, x, y, tilt=0., orientation=0.):
        self.H = float(H)      # Borehole length
        self.D = float(D)      # Borehole buried depth
        self.r_b = float(r_b)  # Borehole radius
        self.x = float(x)      # Borehole x coordinate position
        self.y = float(y)      # Borehole y coordinate position
        self.tilt = float(tilt)
        self.orientation = float(orientation)

    def __repr__(self):
        s = ('Borehole(H={self.H}, D={self.D}, r_b={self.r_b}, x={self.x},'
             ' y={self.y}, tilt={self.tilt},'
             ' orientation={self.orientation})').format(self=self)
        return s

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
        for i in range(nSegments):
            # Divide borehole into segments of equal length
            H = segment_ratios[i] * self.H
            # Buried depth of the i-th segment
            D = self.D + z[i] * np.cos(self.tilt)
            # x-position
            x = self.x + z[i] * np.sin(self.tilt) * np.cos(self.orientation)
            # y-position
            y = self.y + z[i] * np.sin(self.tilt) * np.sin(self.orientation)
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
        tuple of attributes (H, D, r_b, x, y).

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
        elif isinstance(boreholes[0], _EquivalentBorehole):
            self.H = boreholes[0].H
            self.D = boreholes[0].D
            self.r_b = boreholes[0].r_b
            self.x = np.concatenate([b.x for b in boreholes])
            self.y = np.concatenate([b.y for b in boreholes])
        elif type(boreholes) is tuple:
            self.H, self.D, self.r_b, self.x, self.y = boreholes
            self.x = np.atleast_1d(self.x)
            self.y = np.atleast_1d(self.y)

        self.nBoreholes = len(self.x)

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
        return [_EquivalentBorehole((segment_ratios[i] * self.H, z[i] + self.D, self.r_b, self.x, self.y)) for i in range(nSegments)]

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

    duplicate_pairs = []   # define an empty list to be appended to
    for i in range(len(boreField)):
        borehole_1 = boreField[i]
        for j in range(i, len(boreField)):  # only loop unique interactions
            borehole_2 = boreField[j]
            if i == j:  # skip the borehole itself
                continue
            else:
                dist = borehole_1.distance(borehole_2)
            if abs(dist - borehole_1.r_b) < borehole_1.r_b:
                duplicate_pairs.append((i, j))
    if disp:
        print(' gt.boreholes.find_duplicates() '.center(50, '-'))
        print('The duplicate pairs of boreholes found: {}'\
              .format(duplicate_pairs))
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
    # get a list of tuple
    duplicate_pairs = find_duplicates(boreField, disp=disp)

    new_boreField = []

    # values not to be included
    duplicate_bores = []
    for i in range(len(duplicate_pairs)):
        duplicate_bores.append(duplicate_pairs[i][1])

    for i in range(len(boreField)):
        if i in duplicate_bores:
            continue
        else:
            new_boreField.append(boreField[i])
    if disp:
        print(' gt.boreholes.find_duplicates() '.center(50, '-'))
        n_duplicates = len(boreField) - len(new_boreField)
        print('The number of duplicates removed: {}'.format(n_duplicates))

    return new_boreField


def rectangle_field(N_1, N_2, B_1, B_2, H, D, r_b):
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

    Returns
    -------
    boreField : list of Borehole objects
        List of boreholes in the rectangular bore field.

    Examples
    --------
    >>> boreField = gt.boreholes.rectangle_field(N_1=3, N_2=2, B_1=5., B_2=5.,
                                                 H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=2, the bore
    field layout is as follows::

     3   4   5

     0   1   2

    """
    borefield = []

    for j in range(N_2):
        for i in range(N_1):
            borefield.append(Borehole(H, D, r_b, x=i*B_1, y=j*B_2))

    return borefield


def L_shaped_field(N_1, N_2, B_1, B_2, H, D, r_b):
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

    Returns
    -------
    boreField : list of Borehole objects
        List of boreholes in the L-shaped bore field.

    Examples
    --------
    >>> boreField = gt.boreholes.L_shaped_field(N_1=3, N_2=2, B_1=5., B_2=5.,
                                                H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=2, the bore
    field layout is as follows::

     3

     0   1   2

    """
    borefield = []

    for i in range(N_1):
        borefield.append(Borehole(H, D, r_b, x=i*B_1, y=0.))
    for j in range(1, N_2):
        borefield.append(Borehole(H, D, r_b, x=0., y=j*B_2))

    return borefield


def U_shaped_field(N_1, N_2, B_1, B_2, H, D, r_b):
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

    Returns
    -------
    boreField : list of Borehole objects
        List of boreholes in the U-shaped bore field.

    Examples
    --------
    >>> boreField = gt.boreholes.U_shaped_field(N_1=3, N_2=2, B_1=5., B_2=5.,
                                                H=100., D=2.5, r_b=0.05)

    The bore field is constructed line by line. For N_1=3 and N_2=2, the bore
    field layout is as follows::

     3       4

     0   1   2

    """
    borefield = []

    if N_1 > 2 and N_2 > 1:
        for i in range(N_1):
            borefield.append(Borehole(H, D, r_b, x=i*B_1, y=0.))
        for j in range(1, N_2):
            borefield.append(Borehole(H, D, r_b, x=0, y=j*B_2))
            borefield.append(Borehole(H, D, r_b, x=(N_1-1)*B_1, y=j*B_2))
    else:
        borefield = rectangle_field(N_1, N_2, B_1, B_2, H, D, r_b)

    return borefield


def box_shaped_field(N_1, N_2, B_1, B_2, H, D, r_b):
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

    Returns
    -------
    boreField : list of Borehole objects
        List of boreholes in the box-shaped bore field.

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
    borefield = []


    if N_1 > 2 and N_2 > 2:
        for i in range(N_1):
            borefield.append(Borehole(H, D, r_b, x=i*B_1, y=0.))
        for j in range(1, N_2-1):
            borefield.append(Borehole(H, D, r_b, x=0., y=j*B_2))
            borefield.append(Borehole(H, D, r_b, x=(N_1-1)*B_1, y=j*B_2))
        for i in range(N_1):
            borefield.append(Borehole(H, D, r_b, x=i*B_1, y=(N_2-1)*B_2))
    else:
        borefield = rectangle_field(N_1, N_2, B_1, B_2, H, D, r_b)

    return borefield


def circle_field(N, R, H, D, r_b):
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

    Returns
    -------
    boreField : list of Borehole objects
        List of boreholes in the circular shaped bore field.

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
    borefield = []

    for i in range(N):
        borefield.append(Borehole(H, D, r_b, x=R*np.cos(2*pi*i/N),
                                  y=R*np.sin(2*pi*i/N)))

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

        # x   y     H     D     r_b
        0.    0.    100.  2.5   0.075
        5.    0.    100.  2.5   0.075
        0.    5.    100.  2.5   0.075
        0.    10.   100.  2.5   0.075
        0.    20.   100.  2.5   0.075

    """
    # Load data from file
    data = np.loadtxt(filename)
    # Build the bore field
    borefield = []
    for line in data:
        x = line[0]
        y = line[1]
        H = line[2]
        D = line[3]
        r_b = line[4]
        borefield.append(Borehole(H, D, r_b, x=x, y=y))

    return borefield


def visualize_field(borefield, viewTop=True, view3D=True, labels=True):
    """
    Plot the top view and 3D view of borehole positions.

    Parameters
    ----------
    borefield : list
        List of boreholes in the bore field.
    viewTop : bool
        Set to True to plot top view.
        Default is True
    view3D : bool
        Set to True to plot 3D view.
        Default is True
    labels : bool
        Set to True to annotate borehole indices to top view plot.
        Default is True

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    from mpl_toolkits.mplot3d import Axes3D

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
            (x, y) = borehole.position()    # Extract borehole position
            # Add current borehole to the figure
            ax1.plot(x, y, 'ko')
            if labels: ax1.text(x, y,
                                ' {}'.format(i),
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
