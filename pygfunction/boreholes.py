from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.constants import pi


class Borehole(object):
    """
    Contains information regarding the dimensions and position of a borehole.

    Attributes
    ----------
    H : float
        Borehole length (in meters).
    D : float
        Borehole burried depth (in meters).
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
        Borehole burried depth (in meters).
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
        Borehole burried depth (in meters).
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
        Borehole burried depth (in meters).
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
        Borehole burried depth (in meters).
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
        Borehole burried depth (in meters).
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

    The text file should be formatted as follows:

    .. code-block:: none

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
