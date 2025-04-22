# -*- coding: utf-8 -*-
import numpy as np

from . import utilities


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
            'method should return the borehole wall temperatures at the '
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
        Default is 5.

    References
    ----------
    .. [#ClaessonJaved2012] Claesson, J., & Javed, S. (2012). A
       load-aggregation method to calculate extraction temperatures of
       borehole heat exchangers. ASHRAE Transactions, 118 (1): 530–539.
    """
    def __init__(self, dt, tmax, nSources=1, cells_per_level=5, **kwargs):
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
            superposition (:math:`g/(2 \\pi k_s)`).
            The expected size is (nSources, nSources, Nt), where Nt is the
            number of time values at which the thermal response factors are
            required. The time values are returned by
            :func:`~load_aggregation.ClaessonJaved.get_times_for_simulation`.
            If nSources=1, g_d can be 1 dimensional.

        """
        if g_d.ndim == 1:
            self._scalar_output = True
        else:
            self._scalar_output = False
        if self.nSources==1:
            g_d = g_d.reshape(1, 1, -1)
        # Build matrix of thermal response factor increments
        self.dg = np.zeros_like(g_d)
        self.dg[:,:,0] = g_d[:,:,0]
        self.dg[:,:,1:] = g_d[:,:,1:] - g_d[:,:,:-1]

    def next_time_step(self, time):
        """
        Shifts aggregated loads by one time step.

        Parameters
        ----------
        time : float
            Current value of time (in seconds).

        """
        self.q_b = self.q_b @ self.A

    def get_thermal_response_factor_increment(self):
        """
        Returns an array of the **dimensional** thermal response factors.

        Returns
        -------
        dg : array
            Array of **dimensional** thermal response factor increments used
            for temporal superposition
            (:math:`g(t_{i+1})/(2 \\pi k_s) - g(t_{i})/(2 \\pi k_s)`),
            in correspondence with the initialized values of the thermal
            response factors in
            :func:`~load_aggregation.ClaessonJaved.initialize`.
            The output size of the array is (nSources, nSources, Nt) if
            nSources>1. If nSources=1, then the method returns a 1d array.

        """
        if self.nSources == 1:
            dg = self.dg.flatten()
        else:
            dg = self.dg
        return dg

    def get_times_for_simulation(self):
        """
        Returns a vector of time values at which the thermal response factors
        are required.

        Returns
        -------
        time_req : array
            Time values at which the thermal response factors are required
            (in seconds).

        """
        return self._time

    def set_current_load(self, q_b):
        """
        Set the load at the current time step.

        Parameters
        ----------
        q_b : float or array
            Current value of heat extraction rates per unit borehole length
            (in watts per meter).

        """
        self.q_b[:,0] = q_b

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
           :math:`T_b = T_g - \\Delta T_b`.

        """
        # Use numpy.einsum for spatial and temporal superposition
        # This is equivalent to :
        #    deltaT = self.dg[:,:,0].dot(self.q_b[:,0])
        #    for i in range(1, len(self._time)):
        #        deltaT += (self.dg[:,:,i]).dot(self.q_b[:,i])

        deltaT = np.einsum('ijk,jk', self.dg, self.q_b)
        if self._scalar_output:
            deltaT = deltaT.item()
        return deltaT

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
        self._time = utilities.time_ClaessonJaved(
                dt, tmax, cells_per_level=cells_per_level)
        self._width = np.hstack((1,
                                 (self._time[1:] - self._time[:-1])/dt))
        # Initialize aggregated loads
        nt = len(self._time)
        self.q_b = np.zeros((nSources, nt))
        # Matrix for time shifting of aggregated loads. For two consecutive
        # time steps : Q(t+1) = Q(t) @ A
        self.A = (1. - 1./self._width) * np.eye(nt) \
            + np.diag(1./self._width[1:], k=1)



class MLAA(_LoadAggregation):
    """
    Multiple load aggregation algorithm (MLAA) of Bernier et al.
    [#Bernieretal2004]_.

    Attributes
    ----------
    dt : float
        Simulation time step (in seconds).
    tmax : float
        Maximum simulation time (in seconds).
    nSources : int, optional
        Number of heat sources with independent load histories.
        Default is 1.
    N0 : int, optional
        Number of non-aggregated loads.
        Default is 12.
    N1 : int, optional
        Number of time steps in first aggregation cell.
        Default is 48.
    N2 : int, optional
        Number of time steps in second aggregation cell.
        Default is 168.
    N3 : int, optional
        Number of time steps in third aggregation cell.
        Default is 360.

    References
    ----------
    .. [#Bernieretal2004] Bernier, M., Pinel, P., Labib, R. and Paillot, R.
       (2004). A multiple load aggregation algorithm for annual hourly
       simulations of GCHP systems. HVAC&R Research 10 (4): 471–487.
    """
    def __init__(self, dt, tmax, nSources=1,
                 N0=12, N1=48, N2=168, N3=360, **kwargs):
        self.dt = dt                # Simulation time step
        self.tmax = tmax            # Maximum simulation time
        self.nSources = nSources    # Number of heat sources
        # Initialize load aggregation cells
        self._build_cells(dt, tmax, nSources, N0, N1, N2, N3)

    def initialize(self, g_d):
        """
        Initialize the load aggregation scheme.

        Creates a matrix of thermal response factor
        for later use in temporal superposition.

        Parameters
        ----------
        g_d : array
            Matrix of **dimensional** thermal response factors for temporal
            superposition (:math:`g/(2 \\pi k_s)`).
            The expected size is (nSources, nSources, Nt), where Nt is the
            number of time values at which the thermal response factors are
            required. The time values are returned by
            :func:`~load_aggregation.ClaessonJaved.get_times_for_simulation`.
            If nSources=1, g_d can be 1 dimensional.

        """
        if self.nSources==1:
            g_d = g_d.reshape(1, 1, -1)
        self.g_d = g_d

    def next_time_step(self, time):
        """
        Shifts aggregated loads by one time step.

        Parameters
        ----------
        time : float
            Current value of time (in seconds).

        """
        self._nt += 1

        # The number of non-aggregated time steps is equal to the number of
        # time steps until it reaches the predefined number of non-aggregated
        # time steps
        if self._nt < self.N0:
            self._n0 = self._nt
        else:
            self._n0 = self.N0
        # The non-aggregated loads are the n0 latest loads
        self.Q0[:, -self._n0:] = self.q_b[:, self._nt-self._n0:self._nt]

        # Once the number of time steps reaches (N0 + N1), the first load
        # aggregation cell is created
        if self._nt >= self.N0 + self.N1:
            self._n1 = self.N1
            # Averaged loads in first cell
            start = self._nt - (self._n0 + self._n1)
            end = self._nt - self._n0
            self.Q1 = np.mean(self.q_b[:, start:end], axis=1)
        else:
            self._n1 = 0

        # Once the number of time steps reaches (N0 + N1 + N2), the second load
        # aggregation cell is created
        if self._nt >= self.N0 + self.N1 + self.N2:
            self._n2 = self.N2
            # Averaged loads in second cell
            start = self._nt - (self._n0 + self._n1 + self._n2)
            end = self._nt - (self._n0 + self._n1)
            self.Q2 = np.mean(self.q_b[:, start:end], axis=1)
        else:
            self._n2 = 0

        # Once the number of time steps reaches (N0 + N1 + N2 + N3), the third
        # load aggregation cell is created
        if self._nt >= self.N0 + self.N1 + self.N2 + self.N3:
            self._n3 = self.N3
            # Averaged loads in third cell
            start = self._nt - (self._n0 + self._n1 + self._n2 + self._n3)
            end = self._nt - (self._n0 + self._n1 + self._n2)
            self.Q3 = np.mean(self.q_b[:, start:end], axis=1)
        else:
            self._n3 = 0

        # The number of time steps in the fourth aggregation cell is equal to
        # the reminder of the load history
        self._n4 = self._nt - (self._n0 + self._n1 + self._n2 + self._n3)
        # Averaged loads in fourth cell
        if self._n4 > 0:
            start = 0
            end = self._n4
            self.Q4 = np.mean(self.q_b[:, start:end], axis=1)

    def get_times_for_simulation(self):
        """
        Returns a vector of time values at which the thermal response factors
        are required.

        Returns
        -------
        time_req : array
            Time values at which the thermal response factors are required
            (in seconds).

        """
        return np.array([(i+1)*self.dt for i in range(self.Nt)])

    def set_current_load(self, q_b):
        """
        Set the load at the current time step.

        Parameters
        ----------
        q_b : array
            Current value of heat extraction rates per unit borehole length
            (in watts per meter).

        """
        self.q_b[:,self._nt-1] = q_b
        self.Q0[:,-1] = q_b

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
           :math:`T_b = T_g - \\Delta T_b`.

        """
        # Non-aggregated loads
        deltaT = 0.
        for i_t in range(self._n0-1):
            g = self.g_d[:,:,i_t]
            deltaT += g.dot(self.Q0[:,-(i_t+1)]
                            - self.Q0[:,-(i_t+2)])
        g = self.g_d[:,:,self._n0-1]
        deltaT += g.dot(self.Q0[:,-self._n0] - self.Q1)

        # First load aggregation cell
        i_t = self._n0 + self._n1
        deltaT += self.g_d[:,:,i_t-1].dot(self.Q1 - self.Q2)

        # Second load aggregation cell
        i_t = self._n0 + self._n1 + self._n2
        deltaT += self.g_d[:,:,i_t-1].dot(self.Q2 - self.Q3)

        # Third load aggregation cell
        i_t = self._n0 + self._n1 + self._n2 + self._n3
        deltaT += self.g_d[:,:,i_t-1].dot(self.Q3 - self.Q4)

        # Fourth load aggregation cell
        i_t = self._n0 + self._n1 + self._n2 + self._n3 + self._n4
        deltaT += self.g_d[:,:,i_t-1].dot(self.Q4)

        return np.reshape(deltaT, (self.nSources, 1))

    def _build_cells(self, dt, tmax, nSources, N0, N1, N2, N3):
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
        N0 : int
            Number of non-aggregated time steps.
        N1 : int
            Number of time steps in first aggregation cell.
        N2 : int
            Number of time steps in second aggregation cell.
        N3 : int
            Number of time steps in third aggregation cell.

        """
        self.Nt = int(np.ceil(tmax/dt))  # Total number of time steps
        self._nt = 0    # Current time step
        self._n0 = 0    # Current number of non-aggregated time steps
        self._n1 = 0    # Current number of time steps in first cell
        self._n2 = 0    # Current number of time steps in second cell
        self._n3 = 0    # Current number of time steps in third cell
        self._n4 = 0    # Current number of time steps in fourth cell
        self.N0 = N0
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        # Initialize non-aggregated loads
        self.q_b = np.zeros((nSources, self.Nt))
        self.Q0 = np.zeros((nSources, self.N0))
        # Initialize aggregated loads
        self.Q1 = np.zeros(nSources)
        self.Q2 = np.zeros(nSources)
        self.Q3 = np.zeros(nSources)
        self.Q4 = np.zeros(nSources)


class Liu(_LoadAggregation):
    """
    Hierarchical load aggregation algorithm of Liu [#Liu2005]_.

    Attributes
    ----------
    dt : float
        Simulation time step (in seconds).
    tmax : float
        Maximum simulation time (in seconds).
    nSources : int, optional
        Number of heat sources with independent load histories.
        Default is 1.
    N1 : int, optional
        Size (number of time steps) of small load aggregation cell.
        Default is 24.
    N2 : int, optional
        Size (number of small cells) of medium load aggregation cell.
        Default is 5.
    N3 : int, optional
        Size (number of medium cells) of large load aggregation cell.
        Default is 73.
    W1 : int, optional
        Waiting period (number of time steps) for the creation of a small
        load aggregation cell.
        Default is 12.
    W2 : int, optional
        Waiting period (number of small cells) for the creation of a medium
        load aggregation cell.
        Default is 3.
    W3 : int, optional
        Waiting period (number of medium cells) for the creation of a large
        load aggregation cell.
        Default is 40.

    References
    ----------
    .. [#Liu2005] Liu, X. (2005). Development and experimental validation
       of simulation of hydronic snow melting systems for bridges. Ph.D.
       Thesis. Oklahoma State University.
    """
    def __init__(self, dt, tmax, nSources=1,
                 N1=24, N2=5, N3=73, W1=12, W2=3, W3=40):
        self.dt = dt                # Simulation time step
        self.tmax = tmax            # Maximum simulation time
        self.nSources = nSources    # Number of heat sources
        # Initialize load aggregation cells
        self._build_cells(dt, tmax, nSources, N1, N2, N3, W1, W2, W3)

    def initialize(self, g_d):
        """
        Initialize the load aggregation scheme.

        Creates a matrix of thermal response factor
        for later use in temporal superposition.

        Parameters
        ----------
        g_d : array
            Matrix of **dimensional** thermal response factors for temporal
            superposition (:math:`g/(2 \\pi k_s)`).
            The expected size is (nSources, nSources, Nt), where Nt is the
            number of time values at which the thermal response factors are
            required. The time values are returned by
            :func:`~load_aggregation.ClaessonJaved.get_times_for_simulation`.
            If nSources=1, g_d can be 1 dimensional.

        """
        if self.nSources==1:
            g_d = g_d.reshape(1, 1, -1)
        self.g_d = g_d

    def next_time_step(self, time):
        """
        Shifts aggregated loads by one time step.

        Parameters
        ----------
        time : float
            Current value of time (in seconds).

        """
        self._nt += 1   # Increment current time step
        self._n0 += 1   # Increment current number of small cells

        # Shift non-aggregated loads by one
        for i in range(self._n0-1,0,-1):
            self.Q0[:,i] = self.Q0[:,i-1]
        # Current load is zero
        self.Q0[:,0] = 0.0

        # If the number of non-aggregated loads is equal to (N1 + W1),
        # create a small aggregation cell
        if self._n0 >= self.N1 + self.W1:
            self._n1 += 1   # Increment number of small cells
            # Shift small cells by one
            for i in range(self._n1-1,0,-1):
                self.Q1[:,i] = self.Q1[:,i-1]
            # The value of the first small cell is equal to the average of the
            # N1 last non-aggregated loads
            self.Q1[:,0] = np.mean(self.Q0[:,self._n0-self.N1:self._n0],
                                   axis=1)
            # Remove the N1 latest non-aggregated loads
            self._n0 += -self.N1

            # If the number of small cells is equal to (N2 + W2),
            # create a medium aggregation cell
            if self._n1 >= self.N2 + self.W2:
                self._n2 += 1   # Increment number of medium cells
                # Shift medium cells by one
                for i in range(self._n2-1,0,-1):
                    self.Q2[:,i] = self.Q2[:,i-1]
                # The value of the first medium cell is equal to the average of
                # the N2 last non-aggregated loads
                self.Q2[:,0] = np.mean(self.Q1[:,self._n1-self.N2:self._n1],
                                       axis=1)
                # Remove the N2 latest small cells
                self._n1 += -self.N2

                # If the number of medium cells is equal to (N3 + W3),
                # create a large aggregation cell
                if self._n2 >= self.N3 + self.W3:
                    self._n3 += 1   # Increment number of large cells
                    # Shift large cells by one
                    for i in range(self._n3-1,0,-1):
                        self.Q3[:,i] = self.Q3[:,i-1]
                    # The value of the first large cell is equal to the average
                    # of the N3 last non-aggregated loads
                    self.Q3[:,0] = np.mean(self.Q2[:,self._n2-self.N3:self._n2], axis=1)
                    # Remove the N3 latest medium cells
                    self._n2 += -self.N3

    def get_times_for_simulation(self):
        """
        Returns a vector of time values at which the thermal response factors
        are required.

        Returns
        -------
        time_req : array
            Time values at which the thermal response factors are required
            (in seconds).

        """
        return np.array([(i+1)*self.dt for i in range(self.Nt)])

    def set_current_load(self, q_b):
        """
        Set the load at the current time step.

        Parameters
        ----------
        q_b : array
            Current value of heat extraction rates per unit borehole length
            (in watts per meter).

        """
        self.Q0[:,0] = q_b

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
           :math:`T_b = T_g - \\Delta T_b`.

        """
        deltaT = 0.
        for i in range(0, self._n0-1):
            deltaT += self.g_d[:,:,i].dot(self.Q0[:,i] - self.Q0[:,i+1])
        deltaT += self.g_d[:,:,self._n0-1].dot(self.Q0[:,self._n0-1] - self.Q1[:,0])
        for i in range(0, self._n1-1):
            i2 = self._n0 + (i+1)*self.N1 - 1
            deltaT += self.g_d[:,:,i2].dot(self.Q1[:,i] - self.Q1[:,i+1])
        i2 = self._n0 + self._n1*self.N1 - 1
        deltaT += self.g_d[:,:,i2].dot(self.Q1[:,self._n1-1] - self.Q2[:,0])
        for i in range(0, self._n2-1):
            i2 = self._n0 + self._n1*self.N1 + (i+1)*self.N2*self.N1 - 1
            deltaT += self.g_d[:,:,i2].dot(self.Q2[:,i] - self.Q2[:,i+1])
        i2 = self._n0 + self._n1*self.N1 + self._n2*self.N1*self.N2 - 1
        deltaT += self.g_d[:,:,i2].dot(self.Q2[:,self._n2-1] - self.Q3[:,0])
        for i in range(0, self._n3-1):
            i2 = self._n0 + self._n1*self.N1 + self._n2*self.N2*self.N1 + (i+1)*self.N3*self.N2*self.N1 - 1
            deltaT += self.g_d[:,:,i2].dot(self.Q3[:,i] - self.Q3[:,i+1])
        i2 = self._n0 - 1 + (self._n1*self.N1
                             + self._n2*self.N1*self.N2
                             + self._n3*self.N1*self.N2*self.N3)
        deltaT += self.g_d[:,:,i2].dot(self.Q3[:,self._n3-1])

        return deltaT

    def _build_cells(self, dt, tmax, nSources, N1, N2, N3, W1, W2, W3):
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
        N1 : int
            Size (number of time steps) of small load aggregation cell.
        N2 : int
            Size (number of small cells) of medium load aggregation cell.
        N3 : int
            Size (number of medium cells) of large load aggregation cell.
        W1 : int
            Waiting period (number of time steps) for the creation of a small
            load aggregation cell.
        W2 : int
            Waiting period (number of small cells) for the creation of a medium
            load aggregation cell.
        W3 : int
            Waiting period (number of medium cells) for the creation of a large
            load aggregation cell.

        """
        self.Nt = int(np.ceil(tmax/dt))  # Total number of time steps
        self._nt = 0    # Current time step
        self._n0 = 0    # Current number of non-aggregated time steps
        self._n1 = 0    # Current number of small cells
        self._n2 = 0    # Current number of medium cells
        self._n3 = 0    # Current number of large cells
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        # Initialize non-aggregated loads
        self.Q0 = np.zeros((nSources, self.N1+self.W1))
        # Initialize aggregated loads
        self.Q1 = np.zeros((nSources, self.N2+self.W2))
        self.Q2 = np.zeros((nSources, self.N3+self.W3))
        N4 = int(np.ceil(self.Nt/(N1*N2*N3)))
        self.Q3 = np.zeros((nSources, N4))
