from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.constants import pi
from math import sin, cos
import matplotlib.pyplot as plt


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

    def draw_borehole(self, pos_pipes: list, rp_in: float, rp_out: float, show_plot: bool = False,
                      save_plot: bool = False, plot_name: str = 'Plot', ):
        """
        Draw the borehole from a top view
        Parameters
        ----------
        pos_pipes : list
            A list of the pipe positions
        rp_in : float
            The outer pipe radius
        rp_out : float
            The inner pipe radius
        show_plot : bool
            Whether or not to display the plot in a window with plt.show(), will need a GUI enabled backend
        save_plot : bool
            Whether or not to save the plot, saves plot as pdf
        plot_name : str
            If the plot is going to be saved, a plot name can be described, can be a plot path

        Returns
        -------
        None

        Examples
        --------
        >>> b1 = gt.boreholes.Borehole(H=150.0, D=4.0, r_b=0.075, x=0., y=0.)
        >>> rp_out = 0.0211
        >>> rp_in = 0.0147
        >>> D_s = 0.052
        >>> pos_pipes = [(-D_s, 0.), (D_s, 0.)]
        >>> b1.draw_borehole(pos_pipes, rp_in, rp_out, show_plot=True, save_plot=True)
        """

        def plot_list_tuples(list_of_tuples: list, ax_inst: object, color: str = 'C2'):
            # the outer pipe and inner pipe points are stored in the same format, but are going to be plotted with
            # different colors, this function plots that data and allows a color option for the plot
            for j in range(len(list_of_tuples)):
                x_points, y_points = list_of_tuples[j]  # separate out x and y points into individual tuples
                ax_inst.plot(x_points, y_points, c=color)  # plot the points with a given color
            return

        def draw_circle(origin: tuple, radius: float) -> list:
            # this function creates a list of points in a circle given the center (origin) and the radius of the circle
            x_origin = origin[0]
            y_origin = origin[1]

            angles = [k for k in range(361)]  # 0 to 360 needs to range(361)

            def degrees_to_radians(theta: float) -> float:
                # Take in an angle theta in Degrees and return the angle in radians
                return theta * pi / 180

            points = [(x_origin + radius * cos(degrees_to_radians(angles[k])),
                       y_origin + radius * sin(degrees_to_radians(angles[k])))
                      for k in range(len(angles))]

            return list(zip(*points))  # return [(x_points), (y_points)]

        bh_origin = (self.x, self.y)  # get outer borehole radius points
        radius = self.r_b
        bh_x_points, bh_y_points = draw_circle(bh_origin, radius)

        rp_out_points_list = []
        rp_in_points_list = []
        for i in range(len(pos_pipes)):
            pipe_diff_x, pipe_diff_y = pos_pipes[i]  # loop through the different pipe locations
            pipe_origin = (bh_origin[0] + pipe_diff_x, bh_origin[1] + pipe_diff_y)  # offset the origin of the pipes
            rp_out_points_list.append(draw_circle(pipe_origin, rp_out))  # save all rp_out points to a list
            rp_in_points_list.append(draw_circle(pipe_origin, rp_in))    # save all rp_in points to a list

        if show_plot is False:
            plt.switch_backend('agg')  # make compatible for non display CPU's (ie. Linux cluster)

        fig, ax = plt.subplots()
        ax.plot(bh_x_points, bh_y_points, c='C0')

        plot_list_tuples(rp_out_points_list, ax, color='C1')  # plot the outer pipe circle
        plot_list_tuples(rp_in_points_list, ax, color='C2')   # plot the inner pipe circle

        plt.gca().set_aspect('equal', adjustable='box')  # make the circle look round

        if save_plot is True:
            fig.savefig(plot_name + '.pdf')

        if show_plot is True:
            fig.show()

        plt.close(fig)  # close out the figure, make nullptr
        return


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


def visualize_field(borefield):
    """
    Plot the top view and 3D view of borehole positions.

    Parameters
    ----------
    borefield : list
        List of boreholes in the bore field.

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from mpl_toolkits.mplot3d import Axes3D
    # -------------------------------------------------------------------------
    # Initialize figure
    # -------------------------------------------------------------------------
    LW = 1.5    # Line width
    bbox_props = dict(boxstyle="circle,pad=0.3", fc="white", ec="b", lw=LW)

    plt.rc('figure', figsize=(160.0/25.4, 80.0*4.0/4.0/25.4))
    fig = plt.figure()

    # -------------------------------------------------------------------------
    # Top view
    # -------------------------------------------------------------------------
    i = 0   # Initialize borehole index
    ax0 = fig.add_subplot(121)

    for borehole in borefield:
        i += 1  # Increment borehole index
        (x, y) = borehole.position()    # Extract borehole position
        # Add current borehole to the figure
        ax0.plot(x, y, 'k.')
        ax0.text(x, y, i, ha="center", va="center", size=9, bbox=bbox_props)

    # Configure figure axes
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    ax0.set_title('Top view')
    plt.axis('equal')
    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax0.yaxis.set_minor_locator(AutoMinorLocator())

    # -------------------------------------------------------------------------
    # 3D view
    # -------------------------------------------------------------------------
    i = 0   # Initialize borehole index
    ax1 = fig.add_subplot(122, projection='3d')

    for borehole in borefield:
        i += 1  # Increment borehole index
        # Position of head of borehole
        (x, y) = borehole.position()
        # Position of bottom of borehole
        x_H = x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)
        y_H = y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)
        z_H = borehole.D + borehole.H*np.cos(borehole.tilt)
        # Add current borehole to the figure
        ax1.plot(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(borehole.D),
                 'ko')
        ax1.plot(np.array([x, x_H]),
                 np.array([y, y_H]),
                 -np.array([borehole.D, z_H]), 'k-')

    # Configure figure axes
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.set_title('3D view')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.zaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout(rect=[0, 0.0, 0.95, 1.0])

    return fig
