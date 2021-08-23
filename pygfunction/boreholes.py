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

        Raises
        ------
        SomeError

        See Also
        --------
        OtherModules

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
        >>> b1.position()
        (5.0, 0.0)

        """
        pos = (self.x, self.y)
        return pos

    def segments(self, nSegments):
        """
        Split a borehole into segments.

        Parameters
        ----------
        nSegments : int
            Number of segments.

        Returns
        -------
        boreSegments : list
            List of borehole segments.

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
        >>> b1.segments(5)

        """
        boreSegments = []
        for i in range(nSegments):
            # Divide borehole into segments of equal length
            H = self.H / nSegments
            # Buried depth of the i-th segment
            D = self.D + i * self.H / nSegments * np.cos(self.tilt)
            # x-position
            x = self.x + i * self.H / nSegments * np.sin(self.tilt) * np.cos(self.orientation)
            # y-position
            y = self.y + i * self.H / nSegments * np.sin(self.tilt) * np.sin(self.orientation)
            # Add to list of segments
            boreSegments.append(
                Borehole(H, D, self.r_b, x, y,
                         tilt=self.tilt,
                         orientation=self.orientation))
        return boreSegments


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
