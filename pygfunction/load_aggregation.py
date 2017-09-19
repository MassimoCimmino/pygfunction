# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.interpolate as interp


class _LoadAggregation(object):
    """
    Base class for load aggregation schemes.
    """
    def __init__(self, dt, tmax, nSources=1):
        self.dt = dt                # Simulation time step
        self.tmax = tmax            # Maximum simulation time
        self.nSources = nSources    # Number of heat sources
        return

    def initialize(self, g):
        raise NotImplementedError(
            'initialize class method not implemented, this '
            'method should do the start of simulation operations.')

    def get_times_for_simulation(self):
        raise NotImplementedError(
            'get_times_for_simulation class method not implemented, this '
            'method should return a list of time values at which the '
            'thermal response factors are needed.')

    def next_time_step(self, time):
        raise NotImplementedError(
            'next_time_step class method not implemented, this '
            'method should do the start of time step operations.')

    def set_current_load(self, Q):
        raise NotImplementedError(
            'set_current_load class method not implemented, this '
            'method should do the operations needed when setting current '
            'loads.')

    def temporal_superposition(self):
        raise NotImplementedError(
            'temporal_superposition class method not implemented, this '
            'method should return the borehole wall tempreatures at the '
            'current time step.')


class ClaessonJaved(_LoadAggregation):
    """
    Load aggregation algorithm of Claesson and Javed [#ClaessonJaved2012]_.

    Attributes
    ----------
    dt : float
        Simulation time step (in seconds).
    tmax : float
        Maximum simulation time (in seconds).
    nSources : int, optional
        Number of heat sources with independent load histories.
        Default is 1.
    cells_per_level : int, optional
        Number of load aggregation cells per level of aggregation. Cell widths
        double every cells_per_level cells.
        Default is 6.

    References
    ----------
    .. [#ClaessonJaved2012] Claesson, J., & Javed, S. (2011). A
       load-aggregation method to calculate extraction temperatures of
       borehole heat exchangers. ASHRAE Transactions, 118 (1): 530–539.
    """
    def __init__(self, dt, tmax, nSources=1, cells_per_level=6, **kwargs):
        self.dt = dt                # Simulation time step
        self.tmax = tmax            # Maximum simulation time
        self.nSources = nSources    # Number of heat sources
        # Initialize load aggregation cells
        self._build_cells(dt, tmax, nSources, cells_per_level)

    def initialize(self, g_d):
        """
        Initialize the thermal aggregation scheme.

        Creates a matrix of thermal response factor increments
        for later use in temporal superposition.

        Parameters
        ----------
        g_d : array
            Matrix of **dimensional** thermal response factors for temporal
            superposition (:math:`g/(2 \pi k_s)`).
            The expected size is (nSources, nSources, Nt), where Nt is the
            number of time values at which the thermal response factors are
            required. The time values are returned by
            :func:`~load_aggregation.ClaessonJaved.get_times_for_simulation`.

        Examples
        --------
        >>> 

        """
        # Build matrix of thermal response factor increments
        self.dg = np.zeros_like(g_d)
        self.dg[:,:,0] = g_d[:,:,0]
        for i in range(1, len(self._time)):
            self.dg[:,:,i] = g_d[:,:,i] - g_d[:,:,i-1]

    def next_time_step(self, time):
        """
        Shifts aggregated loads by one time step.

        Parameters
        ----------
        time : float
            Current value of time (in seconds).

        """
        for i in range(len(self._time)-2,-1,-1):
            # If the current time is greater than the time of cell (i+1),
            # remove one unit from cell (i+1) and add one unit of cell (i)
            # into cell (i+1).
            if time > self._time[i+1]:
                self.Q[:,i+1] = ((self._width[i+1]-1)*self.Q[:,i+1]
                            + self.Q[:,i])/self._width[i+1]
            # If the current time is greater than the time of cell (i) but less
            # than the time of cell (i+1), add one unit of cell (i) into cell
            # (i+1).
            elif time > self._time[i]:
                self.Q[:,i+1] = (self._width[i+1]*self.Q[:,i+1] + self.Q[:,i])\
                            /self._width[i+1]
        # Set the aggregated load of cell (0) to zero.
        self.Q[:,0:1] = 0.

    def get_times_for_simulation(self):
        """
        Returns a vector of time values at which the thermal response factors
        are required.

        Returns
        -------
        time_req : array
            Time values at which the thermal response factors are required
            (in seconds).

        Examples
        --------
        >>> 

        """
        return self._time

    def set_current_load(self, Q):
        """
        Set the load at the current time step.

        Parameters
        ----------
        Q_l : array
            Current value of heat extraction rates per unit borehole length
            (in watts per meter).

        """
        self.Q[:,0:1] = Q

    def temporal_superposition(self):
        """
        Returns the borehole wall temperature variations at the current time
        step from the temporal superposition of past loads.

        Returns
        -------
        deltaT : array
            Values of borehole wall temperature drops at the current time step
            (in degC).

        .. Note::
           *pygfunction* assumes positive values for heat
           **extraction** and for borehole wall temperature **drops**. The
           borehole wall temperature are thus given by :
           :math:`T_b = T_g - \Delta T_b`.

        """
        deltaT = self.dg[:,:,0].dot(self.Q[:,0])
        for i in range(1, len(self._time)):
            deltaT += (self.dg[:,:,i]).dot(self.Q[:,i])
        return np.reshape(deltaT, (self.nSources, 1))

    def _build_cells(self, dt, tmax, nSources, cells_per_level):
        """
        Initializes load aggregation cells.

        Parameters
        ----------
        dt : float
            Simulation time step (in seconds).
        tmax : float
            Maximum simulation time (in seconds).
        nSources : int
            Number of heat sources with independent load histories.
        cells_per_level : int
            Number of load aggregation cells per level of aggregation. Cell
            widths double every cells_per_level cells.

        """
        time = 0.0
        i = 0
        # Calculate time of load aggregation cells
        self._time = []
        self._width = []
        while time < tmax:
            i += 1
            v = np.ceil(i / cells_per_level)
            width = 2.0**(v-1)
            time += width*float(dt)
            self._time.append(time)
            self._width.append(width)
        # Initialize aggregated loads
        self.Q = np.zeros((nSources, len(self._time)))
        self._time = np.array(self._time)
