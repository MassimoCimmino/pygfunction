from __future__ import absolute_import, division, print_function

import time as tim
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi
from scipy.interpolate import interp1d as interp1d

from .boreholes import Borehole, find_duplicates
from .heat_transfer import finite_line_source
from .networks import Network, network_thermal_resistance


class gFunction(object):
    """
    Class for the calculation and visualization of the g-functions of
    geothermal bore fields.

    This class superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#gFunction-CimBer2014]_. Different boundary conditions and solution
    methods are implemented.
    

    Attributes
    ----------
    boreholes_or_network : list of Borehole objects or Network object
        List of boreholes included in the bore field, or network of boreholes
        and pipes.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    time : float or array, optional
        Values of time (in seconds) for which the g-function is evaluated. The
        g-function is only evaluated at initialization if a value is provided.
    method : str, optional
        Method for the evaluation of the g-function. Should be one of

            - 'similarities' :
                The accelerated method of Cimmino (2018)
                [#gFunction-Cimmin2018]_, using similarities in the bore field
                to decrease the number of evaluations of the FLS solution.
            - 'detailed' :
                The classical superposition of the FLS solution. The FLS
                solution is evaluated for all pairs of segments in the bore
                field.

        Default is 'similarities'.
    boundary_condition : str, optional
        Boundary condition for the evaluation of the g-function. Should be one
        of

            - 'UHTR' :
                **Uniform heat transfer rate**. This is corresponds to boundary
                condition *BC-I* as defined by Cimmino and Bernier (2014)
                [#gFunction-CimBer2014]_.
            - 'UBWT' :
                **Uniform borehole wall temperature**. This is corresponds to
                boundary condition *BC-III* as defined by Cimmino and Bernier
                (2014) [#gFunction-CimBer2014]_.
            - 'MIFT' :
                **Mixed inlet fluid temperatures**. This boundary condition was
                introduced by Cimmino (2015) [#gFunction-Cimmin2015]_ for
                parallel-connected boreholes and extended to mixed
                configurations by Cimmino (2019) [#gFunction-Cimmin2019]_.

        If not given, chosen to be 'UBWT' if a list of boreholes is provided
        or 'MIFT' if a Network object is provided.
    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            nSegments : int, optional
                Number of line segments used per borehole.
                Default is 12.
            processes : int, optional
                Number of processors to use in calculations. If the value is
                set to None, a number of processors equal to cpu_count() is
                used.
                Default is None.
            disp : bool, optional
                Set to true to print progression messages.
                Default is False.
            profiles : bool, optional
                Set to true to keep in memory the temperatures and heat
                extraction rates.
                Default is False.
            kind : str, optional
                Interpolation method used for segment-to-segment thermal
                response factors. See documentation for
                scipy.interpolate.interp1d.
                Default is linear.
            dtype : numpy dtype, optional
                numpy data type used for matrices and vectors.
                Default is numpy.double.

        The 'similarities' solver accepts the following method-specific
        options:

            disTol : float, optional
                Relative tolerance on radial distance. Two distances
                (d1, d2) between two pairs of boreholes are considered equal if
                the difference between the two distances (abs(d1-d2)) is below
                tolerance.
                Default is 0.01.
            tol : float, optional
                Relative tolerance on length and depth. Two lengths H1, H2
                (or depths D1, D2) are considered equal if
                abs(H1 - H2)/H2 < tol.
                Default is 1.0e-6.

    References
    ----------
    .. [#gFunction-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#gFunction-Cimmin2015] Cimmino, M. (2015). The effects of borehole
       thermal resistances and fluid flow rate on the g-functions of geothermal
       bore fields. International Journal of Heat and Mass Transfer, 91,
       1119-1127.
    .. [#gFunction-Cimmin2018] Cimmino, M. (2018). Fast calculation of the
       g-functions of geothermal borehole fields using similarities in the
       evaluation of the finite line source solution. Journal of Building
       Performance Simulation, 11 (6), 655-668.
    .. [#gFunction-Cimmin2019] Cimmino, M. (2019). Semi-analytical method for
       g-function calculation of bore fields with series- and
       parallel-connected boreholes. Science and Technology for the Built
       Environment, 25 (8), 1007-1022.

    """
    def __init__(self, boreholes_or_network, alpha, time=None,
                 method='similarities', boundary_condition=None, options={}):
        self.alpha = alpha
        self.time = time
        self.method = method
        self.boundary_condition = boundary_condition
        self.options = options

        # Format inputs and assign default values where needed
        self._format_inputs(boreholes_or_network)
        # Check the validity of inputs
        self._check_inputs()

        # Load the chosen solver
        if self.method.lower()=='similarities':
            self.solver = Similarities(
                self.boreholes, self.network, self.time,
                self.boundary_condition, **self.options)
        elif self.method.lower()=='detailed':
            self.solver = Detailed(
                self.boreholes, self.network, self.time,
                self.boundary_condition, **self.options)
        else:
            raise ValueError('\'{}\' is not a valid method.'.format(method))

        # If a time vector is provided, evaluate the g-function
        if self.time is not None:
            self.gFunc = self.evaluate_g_function(self.time)

    def evaluate_g_function(self, time):
        """
        Evaluate the g-function.

        Parameters
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.

        Returns
        -------
        gFunction : float or array
            Values of the g-function
        """
        assert np.all(time[:-1] <= time[1:]), \
            "Time values must be provided in increasing order."
        # Save time values
        self.time = time
        if self.solver.disp:
            print(60*'-')
            print('Calculating g-function for boundary condition : \'{}\''.format(
                self.boundary_condition))
            print(60*'-')
        # Initialize chrono
        tic = tim.time()

        # Evaluate g-function
        self.gFunc = self.solver.solve(time, self.alpha)
        toc = tim.time()

        if self.solver.disp:
            print('Total time for g-function evaluation: {} sec'.format(
                toc - tic))
            print(60*'-')
        return self.gFunc

    def visualize_g_function(self):
        """
        Plot the g-function of the borefield.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # Configure figure and axes
        fig = self._initialize_figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'ln$(t/t_s)$')
        ax.set_ylabel(r'$g$-function')
        self._format_axes(ax)

        # Borefield characteristic time
        ts = np.mean([b.H for b in self.boreholes])**2/(9.*self.alpha)
        # Dimensionless time (log)
        lntts = np.log(self.time/ts)
        # Draw g-function
        ax.plot(lntts, self.gFunc)

        # Adjust figure to window
        plt.tight_layout()
        return fig

    def visualize_heat_extraction_rates(self, iBoreholes=None):
        """
        Plot the time-variation of the average heat extraction rates.

        Parameters
        ----------
        iBoreholes : list of int
            Borehole indices to plot heat extraction rates.
            If iBoreholes is None, heat extraction rates are plotted for all
            boreholes.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.boreholes))
        # Import heat extraction rates
        Q = self._heat_extraction_rates(iBoreholes)

        # Configure figure and axes
        fig = self._initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        self._format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'ln$(t/t_s)$')
        ax2.set_ylabel(r'$\bar{Q}_b$')
        self._format_axes(ax2)

        # Borefield characteristic time
        ts = np.mean([b.H for b in self.boreholes])**2/(9.*self.alpha)
        # Dimensionless time (log)
        lntts = np.log(self.time/ts)
        # Plot curves for requested boreholes
        for (i, Qi) in zip(iBoreholes, Q):
            line = ax2.plot(lntts, Qi)
            color = line[-1]._color
            ax1.plot(self.boreholes[i].x,
                     self.boreholes[i].y,
                     marker='o',
                     color=color)
        # Draw positions of other boreholes
        for i in range(len(self.boreholes)):
            if i not in iBoreholes:
                ax1.plot(self.boreholes[i].x,
                         self.boreholes[i].y,
                         marker='o',
                         color='k')

        # Adjust figure to window
        plt.tight_layout()
        return fig

    def visualize_heat_extraction_rate_profiles(
            self, time=None, iBoreholes=None):
        """
        Plot the heat extraction rate profiles at chosen time.

        Parameters
        ----------
        time : float
            Values of time (in seconds) to plot heat extraction rate profiles.
            If time is None, heat extraction rates are plotted at the last
            time step.
        iBoreholes : list of int
            Borehole indices to plot heat extraction rate profiles.
            If iBoreholes is None, heat extraction rates are plotted for all
            boreholes.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.boreholes))
        # Import heat extraction rate profiles
        z, Q = self._heat_extraction_rate_profiles(time, iBoreholes)

        # Configure figure and axes
        fig = self._initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        self._format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'$Q_b$')
        ax2.set_ylabel(r'$z$ [m]')
        ax2.invert_yaxis()
        self._format_axes(ax2)

        # Plot curves for requested boreholes
        for (i, zi, Qi) in zip(iBoreholes, z, Q):
            line = ax2.plot(Qi, zi)
            color = line[-1]._color
            ax1.plot(self.boreholes[i].x,
                     self.boreholes[i].y,
                     marker='o',
                     color=color)
        # Draw positions of other boreholes
        for i in range(len(self.boreholes)):
            if i not in iBoreholes:
                ax1.plot(self.boreholes[i].x,
                         self.boreholes[i].y,
                         marker='o',
                         color='k')

        plt.tight_layout()
        return fig

    def visualize_temperatures(self, iBoreholes=None):
        """
        Plot the time-variation of the average borehole wall temperatures.

        Parameters
        ----------
        iBoreholes : list of int
            Borehole indices to plot temperatures.
            If iBoreholes is None, temperatures are plotted for all boreholes.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.boreholes))
        # Import temperatures
        Tb = self._temperatures(iBoreholes)

        # Configure figure and axes
        fig = self._initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        self._format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'ln$(t/t_s)$')
        ax2.set_ylabel(r'$\bar{T}_b$')
        self._format_axes(ax2)

        # Borefield characteristic time
        ts = np.mean([b.H for b in self.boreholes])**2/(9.*self.alpha)
        # Dimensionless time (log)
        lntts = np.log(self.time/ts)
        # Plot curves for requested boreholes
        for (i, Tbi) in zip(iBoreholes, Tb):
            line = ax2.plot(lntts, Tbi)
            color = line[-1]._color
            ax1.plot(self.boreholes[i].x,
                     self.boreholes[i].y,
                     marker='o',
                     color=color)
        # Draw positions of other boreholes
        for i in range(len(self.boreholes)):
            if i not in iBoreholes:
                ax1.plot(self.boreholes[i].x,
                         self.boreholes[i].y,
                         marker='o',
                         color='k')

        # Adjust figure to window
        plt.tight_layout()
        return fig

    def visualize_temperature_profiles(self, time=None, iBoreholes=None):
        """
        Plot the borehole wall temperature profiles at chosen time.

        Parameters
        ----------
        time : float
            Values of time (in seconds) to plot temperature profiles.
            If time is None, temperatures are plotted at the last time step.
        iBoreholes : list of int
            Borehole indices to plot temperature profiles.
            If iBoreholes is None, temperatures are plotted for all boreholes.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.boreholes))
        # Import temperature profiles
        z, Tb = self._temperature_profiles(time, iBoreholes)

        # Configure figure and axes
        fig = self._initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        self._format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'$T_b$')
        ax2.set_ylabel(r'$z$ [m]')
        ax2.invert_yaxis()
        self._format_axes(ax2)

        # Plot curves for requested boreholes
        for (i, zi, Tbi) in zip(iBoreholes, z, Tb):
            line = ax2.plot(Tbi, zi)
            color = line[-1]._color
            ax1.plot(self.boreholes[i].x,
                     self.boreholes[i].y,
                     marker='o',
                     color=color)
        # Draw positions of other boreholes
        for i in range(len(self.boreholes)):
            if i not in iBoreholes:
                ax1.plot(self.boreholes[i].x,
                         self.boreholes[i].y,
                         marker='o',
                         color='k')

        plt.tight_layout()
        return fig

    def _heat_extraction_rates(self, iBoreholes):
        """
        Extract and format heat extraction rates for plotting.

        Parameters
        ----------
        iBoreholes : list of int
            Borehole indices to extract heat extration rates.

        Returns
        -------
        Q : list of array
            Heat extraction rates (dimensionless).

        """
        # Initialize list
        Q = []
        for i in iBoreholes:
            if self.boundary_condition == 'UHTR':
                # For the UHTR boundary condition, the heat extraction rate is
                # constant. A vector of length len(self.solver.time) is
                # required for the time-dependent figure.
                Q.append(self.solver.Q*np.ones(len(self.solver.time)))
            else:
                # For other boundary conditions, evaluate the average
                # heat extraction rate.
                i0 = i*self.solver.nSegments
                i1 = i0 + self.solver.nSegments
                Q.append(np.mean(self.solver.Q[i0:i1,:], axis=0))
        return Q

    def _heat_extraction_rate_profiles(self, time, iBoreholes):
        """
        Extract and format heat extraction rate profiles for plotting.

        Parameters
        ----------
        time : float
            Values of time (in seconds) to extract heat extraction rate
            profiles. If time is None, heat extraction rates are extracted at
            the last time step.
        iBoreholes : list of int
            Borehole indices to extract heat extraction rate profiles.

        Returns
        -------
        z : list of array
            Depths (in meters) corresponding to heat extraction rates.
        Q : list of array
            Heat extraction rates (dimensionless).

        """
        # Initialize lists
        z = []
        Q = []
        for i in iBoreholes:
            if self.boundary_condition == 'UHTR':
                # For the UHTR boundary condition, the solver only returns one
                # heat extraction rate (uniform and equal for all boreholes).
                # The heat extraction rate is duplicated to draw from
                # z = D to z = D + H.
                z.append(
                    np.array([self.boreholes[i].D,
                              self.boreholes[i].D + self.boreholes[i].H]))
                Q.append(np.array(2*[self.solver.Q]))
            else:
                i0 = i*self.solver.nSegments
                i1 = i0 + self.solver.nSegments
                if time is None:
                    # If time is None, heat extraction rates are extracted at
                    # the last time step.
                    Qi = self.solver.Q[i0:i1,-1].flatten()
                else:
                    # Otherwise, heat extraction rates are interpolated.
                    Qi = interp1d(self.time, self.solver.Q[i0:i1,:],
                                  kind='linear',
                                  copy=False,
                                  axis=1)(time).flatten()
                if self.solver.nSegments > 1:
                    # Borehole length ratio at the mid-depth of each segment
                    z_ratio = np.linspace(start=0.5/self.solver.nSegments,
                                          stop=1-0.5/self.solver.nSegments,
                                          num=self.solver.nSegments)
                    z.append(self.boreholes[i].D + self.boreholes[i].H*z_ratio)
                    Q.append(Qi)
                else:
                    # If there is only one segment, the heat extraction rate is
                    # duplicated to draw from z = D to z = D + H.
                    z.append(
                        np.array([self.boreholes[i].D,
                                  self.boreholes[i].D + self.boreholes[i].H]))
                    Q.append(np.array(2*[np.asscalar(Qi)]))
        return z, Q

    def _temperatures(self, iBoreholes):
        """
        Extract and format borehole wall temperatures for plotting.

        Parameters
        ----------
        iBoreholes : list of int
            Borehole indices to extract temperatures.

        Returns
        -------
        Tb : list of array
            Borehole wall temperatures (dimensionless).

        """
        # Initialize list
        Tb = []
        for i in iBoreholes:
            if self.boundary_condition == 'UBWT':
                # For the UBWT boundary condition, the solver only returns one
                # borehole wall temperature (uniform and equal for all
                # boreholes).
                Tb.append(self.solver.Tb)
            else:
                # For other boundary conditions, evaluate the average
                # borehole wall temperature.
                i0 = i*self.solver.nSegments
                i1 = i0 + self.solver.nSegments
                Tb.append(np.mean(self.solver.Tb[i0:i1,:], axis=0))
        return Tb

    def _temperature_profiles(self, time, iBoreholes):
        """
        Extract and format borehole wall temperature profiles for plotting.

        Parameters
        ----------
        time : float
            Values of time (in seconds) to extract temperature profiles.
            If time is None, temperatures are extracted at the last time step.
        iBoreholes : list of int
            Borehole indices to extract temperature profiles.

        Returns
        -------
        z : list of array
            Depths (in meters) corresponding to borehole wall temperatures.
        Tb : list of array
            Borehole wall temperatures (dimensionless).

        """
        # Initialize lists
        z = []
        Tb = []
        for i in iBoreholes:
            if self.boundary_condition == 'UBWT':
                # For the UBWT boundary condition, the solver only returns one
                # borehole wall temperature (uniform and equal for all
                # boreholes). The temperature is duplicated to draw from
                # z = D to z = D + H.
                z.append(
                    np.array([self.boreholes[i].D,
                              self.boreholes[i].D + self.boreholes[i].H]))
                if time is None:
                    # If time is None, temperatures are extracted at the last
                    # time step.
                    Tbi = np.asscalar(self.solver.Tb[-1])
                else:
                    # Otherwise, temperatures are interpolated.
                    Tbi = np.asscalar(
                        interp1d(self.time,
                                 self.solver.Tb[:],
                                 kind='linear',
                                 copy=False)(time))
                Tb.append(np.array(2*[Tbi]))
            else:
                i0 = i*self.solver.nSegments
                i1 = i0 + self.solver.nSegments
                if time is None:
                    # If time is None, temperatures are extracted at the last
                    # time step.
                    Tbi = self.solver.Tb[i0:i1,-1].flatten()
                else:
                    # Otherwise, temperatures are interpolated.
                    Tbi = interp1d(self.time,
                                   self.solver.Tb[i0:i1,:],
                                   kind='linear',
                                   copy=False,
                                   axis=1)(time).flatten()
                if self.solver.nSegments > 1:
                    # Borehole length ratio at the mid-depth of each segment
                    z_ratio = np.linspace(start=0.5/self.solver.nSegments,
                                          stop=1-0.5/self.solver.nSegments,
                                          num=self.solver.nSegments)
                    z.append(self.boreholes[i].D + self.boreholes[i].H*z_ratio)
                    Tb.append(Tbi)
                else:
                    # If there is only one segment, the temperature is
                    # duplicated to draw from z = D to z = D + H.
                    z.append(
                        np.array([self.boreholes[i].D,
                                  self.boreholes[i].D + self.boreholes[i].H]))
                    Tb.append(np.array(2*[np.asscalar(Tbi)]))
        return z, Tb

    def _initialize_figure(self):
        """
        Initialize a matplotlib figure object with overwritten default
        parameters.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        plt.rc('font', size=9)
        plt.rc('xtick', labelsize=9)
        plt.rc('ytick', labelsize=9)
        plt.rc('lines', lw=1.5, markersize=5.0)
        plt.rc('savefig', dpi=500)
        fig = plt.figure()
        return fig

    def _format_axes(self, ax):
        """
        Adjust axis parameters.

        Parameters
        ----------
        ax : axis
            Axis object (amtplotlib).

        """
        from matplotlib.ticker import AutoMinorLocator
        # Draw major and minor tick marks inwards
        ax.tick_params(
            axis='both', which='both', direction='in',
            bottom=True, top=True, left=True, right=True)
        # Auto-adjust minor tick marks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return

    def _format_inputs(self, boreholes_or_network):
        """
        Process and format the inputs to the gFunction class.

        """
        # Convert borehole to a list if a single borehole is provided
        if isinstance(boreholes_or_network, Borehole):
            boreholes_or_network = [boreholes_or_network]
        # Check if a borefield or a network is provided as an input and
        # correctly assign the variables self.boreholes and self.network
        if isinstance(boreholes_or_network, Network):
            self.network = boreholes_or_network
            self.boreholes = boreholes_or_network.b
            # If a network is provided and no boundary condition is provided,
            # use 'MIFT'
            if self.boundary_condition is None:
                self.boundary_condition = 'MIFT'
        else:
            self.network = None
            self.boreholes = boreholes_or_network
            # If a borefield is provided and no boundary condition is provided,
            # use 'UBWT'
            if self.boundary_condition is None:
                self.boundary_condition = 'UBWT'
        return

    def _check_inputs(self):
        """
        This method ensures that the instances filled in the gFunction object
        are what is expected.

        """
        assert isinstance(self.boreholes, list), \
            "Boreholes must be provided in a list."
        assert len(self.boreholes) > 0, \
            "The list of boreholes is empty."
        assert np.all([isinstance(b, Borehole) for b in self.boreholes]), \
            "The list of boreholes contains elements that are not Borehole objects."
        assert not find_duplicates(self.boreholes), \
            "There are duplicate boreholes in the borefield."
        assert (self.network is None and not self.boundary_condition=='MIFT') or isinstance(self.network, Network), \
            "The network is not a valid 'Network' object."
        assert self.network is None or (self.network.m_flow is not None and self.network.cp is not None), \
            "The mass flow rate 'm_flow' and heat capacity 'cp' must be " \
            "provided at the instanciation of the 'Network' object."
        assert type(self.time) is np.ndarray or type(self.time) is float or self.time is None, \
            "Time should be a float or an array."
        assert type(self.alpha) is float, \
            "The thermal diffusivity 'alpha' should be a float or an array."
        acceptable_boundary_conditions = ['UHTR', 'UBWT', 'MIFT']
        assert type(self.boundary_condition) is str and self.boundary_condition in acceptable_boundary_conditions, \
            "Boundary condition \'{}\' is not an acceptable boundary condition. \n" \
            "Please provide one of the following inputs : {}".format(self.boundary_condition, acceptable_boundary_conditions)
        acceptable_methods = ['detailed', 'similarities']
        assert type(self.method) is str and self.method in acceptable_methods, \
            "Method \'{}\' is not an acceptable method. \n" \
            "Please provide one of the following inputs : {}".format(self.boundary_condition, acceptable_methods)
        return


class _BaseSolver(object):
    """
    Template for solver classes.

    Solver classes inherit from this class.

    Attributes
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    network : network object
        Model of the network.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    boundary_condition : str
        Boundary condition for the evaluation of the g-function. Should be one
        of

            - 'UHTR' :
                Uniform heat transfer rate.
            - 'UBWT' :
                Uniform borehole wall temperature.
            - 'MIFT' :
                Mixed inlet fluid temperatures.

    nSegments : int, optional
        Number of line segments used per borehole.
        Default is 12.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    profiles : bool, optional
        Set to true to keep in memory the temperatures and heat extraction
        rates.
        Default is False.
    kind : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is linear.

    """
    def __init__(self, boreholes, network, time, boundary_condition,
                 nSegments=12, processes=None, disp=False, profiles=False,
                 kind='linear', dtype=np.double, **other_options):
        self.boreholes = boreholes
        self.network = network
        # Convert time to a 1d array
        self.time = np.atleast_1d(time).flatten()
        self.boundary_condition = boundary_condition
        self.nSegments = nSegments
        self.processes = processes
        self.disp = disp
        self.profiles = profiles
        self.kind = kind
        self.dtype = dtype
        # Check the validity of inputs
        self._check_inputs()
        # Initialize the solver with solver-specific options
        self.nSources = self.initialize(**other_options)

        return

    def initialize(self, *kwargs):
        """
        Perform any calculation required at the initialization of the solver
        and returns the number of finite line heat sources in the borefield.

        Raises
        ------
        NotImplementedError

        Returns
        -------
        nSources : int
            Number of finite line heat sources in the borefield used to
            initialize the matrix of segment-to-segment thermal response
            factors (of size: nSources x nSources).

        """
        raise NotImplementedError(
            'initialize class method not implemented, this method should '
            'return the number of finite line heat sources in the borefield '
            'used to initialize the matrix of segment-to-segment thermal '
            'response factors (of size: nSources x nSources)')
        return None

    def solve(self, time, alpha):
        """
        Build and solve the system of equations.

        Parameters
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).

        Returns
        -------
        gFunc : float or array
            Values of the g-function

        """
        # Number of time values
        nt = len(self.time)
        # Initialize g-function
        gFunc = np.zeros(nt)
        # Initialize segment heat extraction rates
        if self.boundary_condition == 'UHTR':
            Q = 1
        else:
            Q = np.zeros((self.nSources, nt), dtype=self.dtype)
        if self.boundary_condition == 'UBWT':
            Tb = np.zeros(nt, dtype=self.dtype)
        else:
            Tb = np.zeros((self.nSources, nt), dtype=self.dtype)
        # Calculate segment to segment thermal response factors
        h_ij = self.thermal_response_factors(time, alpha, kind=self.kind)
        # Segment lengths
        Hb = self.segment_lengths()
        Htot = np.sum(Hb)
        if self.disp: print('Building and solving the system of equations ...')
        # Initialize chrono
        tic = tim.time()

        # Build and solve the system of equations at all times
        for p in range(nt):
            if self.boundary_condition == 'UHTR':
                # Evaluate the g-function with uniform heat extraction along
                # boreholes

                # Thermal response factors evaluated at time t[p]
                h_dt = h_ij.y[:,:,p+1]
                # Borehole wall temperatures are calculated by the sum of
                # contributions of all segments
                Tb[:,p] = np.sum(h_dt, axis=1)
                # The g-function is the average of all borehole wall
                # temperatures
                gFunc[p] = np.sum(Tb[:,p]*Hb)/Htot
            else:
                # Current thermal response factor matrix
                if p > 0:
                    dt = self.time[p] - self.time[p-1]
                else:
                    dt = self.time[p]
                # Thermal response factors evaluated at t=dt
                h_dt = h_ij(dt)
                # Reconstructed load history
                Q_reconstructed = self.load_history_reconstruction(
                    self.time[0:p+1], Q[:,0:p+1])
                # Borehole wall temperature for zero heat extraction at
                # current step
                Tb_0 = self.temporal_superposition(
                    h_ij.y[:,:,1:], Q_reconstructed)

                if self.boundary_condition == 'UBWT':
                    # Evaluate the g-function with uniform borehole wall
                    # temperature
                    # ---------------------------------------------------------
                    # Build a system of equation [A]*[X] = [B] for the
                    # evaluation of the g-function. [A] is a coefficient
                    # matrix, [X] = [Qb,Tb] is a state space vector of the
                    # borehole heat extraction rates and borehole wall
                    # temperature (equal for all segments), [B] is a
                    # coefficient vector.
                    #
                    # Spatial superposition: [Tb] = [Tb0] + [h_ij_dt]*[Qb]
                    # Energy conservation: sum([Q*Hb]) = sum([Hb])
                    # ---------------------------------------------------------
                    A = np.block([[h_dt, -np.ones((self.nSources, 1), dtype=self.dtype)],
                                  [Hb, 0.]])
                    B = np.hstack((-Tb_0, Htot))
                    # Solve the system of equations
                    X = np.linalg.solve(A, B)
                    # Store calculated heat extraction rates
                    Q[:,p] = X[0:self.nSources]
                    # The borehole wall temperatures are equal for all segments
                    Tb[p] = X[-1]
                    gFunc[p] = Tb[p]
                elif self.boundary_condition == 'MIFT':
                    # Evaluate the g-function with mixed inlet fluid
                    # temperatures
                    # ---------------------------------------------------------
                    # Build a system of equation [A]*[X] = [B] for the
                    # evaluation of the g-function. [A] is a coefficient
                    # matrix, [X] = [Qb,Tb,Tf_in] is a state space vector of
                    # the borehole heat extraction rates, borehole wall
                    # temperatures and inlet fluid temperature (into the bore
                    # field), [B] is a coefficient vector.
                    #
                    # Spatial superposition: [Tb] = [Tb0] + [h_ij_dt]*[Qb]
                    # Heat transfer inside boreholes:
                    # [Q_{b,i}] = [a_in]*[T_{f,in}] + [a_{b,i}]*[T_{b,i}]
                    # Energy conservation: sum([Q*Hb]) = sum([Hb])
                    # ---------------------------------------------------------
                    a_in, a_b = self.network.coefficients_borehole_heat_extraction_rate(
                            self.network.m_flow, self.network.cp, self.nSegments)
                    k_s = self.network.p[0].k_s
                    A = np.block([[h_dt, -np.eye(self.nSources, dtype=self.dtype), np.zeros((self.nSources, 1), dtype=self.dtype)],
                                  [np.eye(self.nSources, dtype=self.dtype), a_b/(2.0*pi*k_s*np.atleast_2d(Hb).T), a_in/(2.0*pi*k_s*np.atleast_2d(Hb).T)],
                                  [Hb, np.zeros(self.nSources + 1, dtype=self.dtype)]])
                    B = np.hstack((-Tb_0, np.zeros(self.nSources, dtype=self.dtype), Htot))
                    # Solve the system of equations
                    X = np.linalg.solve(A, B)
                    # Store calculated heat extraction rates
                    Q[:,p] = X[0:self.nSources]
                    Tb[:,p] = X[self.nSources:2*self.nSources]
                    Tf_in = X[-1]
                    # The gFunction is equal to the effective borehole wall
                    # temperature
                    # Outlet fluid temperature
                    Tf_out = Tf_in - 2*pi*self.network.p[0].k_s*Htot/(
                        self.network.m_flow*self.network.cp)
                    # Average fluid temperature
                    Tf = 0.5*(Tf_in + Tf_out)
                    # Borefield thermal resistance
                    Rfield = network_thermal_resistance(
                        self.network, self.network.m_flow, self.network.cp)
                    # Effective borehole wall temperature
                    Tb_eff = Tf - 2*pi*self.network.p[0].k_s*Rfield
                    gFunc[p] = Tb_eff
            # Store temperature and heat extraction rate profiles
            if self.profiles:
                self.Q = Q
                self.Tb = Tb
        toc = tim.time()
        if self.disp: print('{} sec'.format(toc - tic))
        return gFunc

    def segment_lengths(self):
        """
        Return the length of all segments in the bore field.

        The segments lengths are used for the energy balance in the calculation
        of the g-function.

        Returns
        -------
        H : array
            Array of segment lengths (in m).

        """
        # Borehole lengths
        H = np.array([b.H for b in self.boreSegments], dtype=self.dtype)
        return H

    def borehole_segments(self):
        """
        Split boreholes into segments.

        This function goes through the list of boreholes and builds a new list,
        with each borehole split into nSegments of equal lengths.

        Returns
        -------
        boreSegments : list
            List of borehole segments.

        """
        boreSegments = []
        for b in self.boreholes:
            for i in range(self.nSegments):
                # Divide borehole into segments of equal length
                H = b.H / self.nSegments
                # Buried depth of the i-th segment
                D = b.D + i * b.H / self.nSegments
                # Add to list of segments
                boreSegments.append(Borehole(H, D, b.r_b, b.x, b.y))
        return boreSegments

    def temporal_superposition(self, h_ij, Q_reconstructed):
        """
        Temporal superposition for inequal time steps.

        Parameters
        ----------
        h_ij : array
            Values of the segment-to-segment thermal response factor increments
            at the given time step.
        Q_reconstructed : array
            Reconstructed heat extraction rates of all segments at all times.

        Returns
        -------
        Tb_0 : array
            Current values of borehole wall temperatures assuming no heat
            extraction during current time step.

        """
        # Number of heat sources
        nSources = Q_reconstructed.shape[0]
        # Number of time steps
        nt = Q_reconstructed.shape[1]
        # Borehole wall temperature
        Tb_0 = np.zeros(nSources)
        # Spatial and temporal superpositions
        dQ = np.concatenate((Q_reconstructed[:,0:1],
                             Q_reconstructed[:,1:]-Q_reconstructed[:,0:-1]),
                            axis=1)
        for it in range(nt):
            Tb_0 += h_ij[:,:,it].dot(dQ[:,nt-it-1])
        return Tb_0

    def load_history_reconstruction(self, time, Q):
        """
        Reconstructs the load history.

        This function calculates an equivalent load history for an inverted
        order of time step sizes.

        Parameters
        ----------
        time : array
            Values of time (in seconds) in the load history.
        Q : array
            Heat extraction rates (in Watts) of all segments at all times.

        Returns
        -------
        Q_reconstructed : array
            Reconstructed load history.

        """
        # Number of heat sources
        nSources = Q.shape[0]
        # Time step sizes
        dt = np.hstack((time[0], time[1:]-time[:-1]))
        # Time vector
        t = np.hstack((0., time, time[-1] + time[0]))
        # Inverted time step sizes
        dt_reconstructed = dt[::-1]
        # Reconstructed time vector
        t_reconstructed = np.hstack((0., np.cumsum(dt_reconstructed)))
        # Accumulated heat extracted
        f = np.hstack((np.zeros((nSources, 1), dtype=self.dtype), np.cumsum(Q*dt, axis=1)))
        f = np.hstack((f, f[:,-1:]))
        # Create interpolation object for accumulated heat extracted
        sf = interp1d(t, f, kind='linear', axis=1)
        # Reconstructed load history
        Q_reconstructed = (sf(t_reconstructed[1:]) - sf(t_reconstructed[:-1])) \
            / dt_reconstructed

        return Q_reconstructed

    def _check_inputs(self):
        """
        This method ensures that the instances filled in the Solver object
        are what is expected.

        """
        assert isinstance(self.boreholes, list), \
            "Boreholes must be provided in a list."
        assert len(self.boreholes) > 0, \
            "The list of boreholes is empty."
        assert np.all([isinstance(b, Borehole) for b in self.boreholes]), \
            "The list of boreholes contains elements that are not Borehole objects."
        assert self.network is None or isinstance(self.network, Network), \
            "The network is not a valid 'Network' object."
        assert self.network is None or (self.network.m_flow is not None and self.network.cp is not None), \
            "The mass flow rate 'm_flow' and heat capacity 'cp' must be " \
            "provided at the instanciation of the 'Network' object."
        assert type(self.time) is np.ndarray or type(self.time) is float or self.time is None, \
            "Time should be a float or an array."
        assert type(self.nSegments) is int and self.nSegments >= 1, \
            "The number of segments 'nSegments' should be a positive int (>= 1)."
        acceptable_boundary_conditions = ['UHTR', 'UBWT', 'MIFT']
        assert type(self.boundary_condition) is str and self.boundary_condition in acceptable_boundary_conditions, \
            "Boundary condition \'{}\' is not an acceptable boundary condition. \n" \
            "Please provide one of the following inputs : {}".format(self.boundary_condition, acceptable_boundary_conditions)
        assert (type(self.processes) is int and self.processes >= 1) or self.processes is None, \
            "The number of processes 'processes' should be a positive int (>= 1)."
        assert type(self.disp) is bool, \
            "The option 'disp' should be set to True or False."
        assert type(self.profiles) is bool, \
            "The option 'profiles' should be set to True or False."
        assert type(self.kind) is str, \
            "The option 'kind' should be set to a valid interpolation kind " \
            "in accordance with scipy.interpolate.interp1d options."
        return


class Detailed(_BaseSolver):
    """
    Detailed solver for the evaluation of the g-function.

    This solver superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#Detailed-CimBer2014]_.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    network : network object
        Model of the network.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    boundary_condition : str
        Boundary condition for the evaluation of the g-function. Should be one
        of

            - 'UHTR' :
                **Uniform heat transfer rate**. This is corresponds to boundary
                condition *BC-I* as defined by Cimmino and Bernier (2014)
                [#Detailed-CimBer2014]_.
            - 'UBWT' :
                **Uniform borehole wall temperature**. This is corresponds to
                boundary condition *BC-III* as defined by Cimmino and Bernier
                (2014) [#Detailed-CimBer2014]_.
            - 'MIFT' :
                **Mixed inlet fluid temperatures**. This boundary condition was
                introduced by Cimmino (2015) [#gFunction-Cimmin2015]_ for
                parallel-connected boreholes and extended to mixed
                configurations by Cimmino (2019) [#Detailed-Cimmin2019]_.

    nSegments : int, optional
        Number of line segments used per borehole.
        Default is 12.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    profiles : bool, optional
        Set to true to keep in memory the temperatures and heat extraction
        rates.
        Default is False.
    kind : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is linear.

    References
    ----------
    .. [#Detailed-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#Detailed-Cimmin2019] Cimmino, M. (2019). Semi-analytical method for
       g-function calculation of bore fields with series- and
       parallel-connected boreholes. Science and Technology for the Built
       Environment, 25 (8), 1007-1022.

    """
    def initialize(self, **kwargs):
        """
        Split boreholes into segments.

        Returns
        -------
        nSources : int
            Number of finite line heat sources in the borefield used to
            initialize the matrix of segment-to-segment thermal response
            factors (of size: nSources x nSources).

        """
        # Split boreholes into segments
        self.boreSegments = self.borehole_segments()
        nSources = len(self.boreSegments)
        return nSources

    def thermal_response_factors(self, time, alpha, kind='linear'):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments in the borefield at all time steps using the finite line
        source solution.

        This method returns a scipy.interpolate.interp1d object of the matrix
        of thermal response factors, containing a copy of the matrix accessible
        by h_ij.y[:nSources,:nSources,:nt+1]. The first index along the
        third axis corresponds to time t=0. The interp1d object can be used to
        obtain thermal response factors at any intermediat time by
        h_ij(t)[:nSources,:nSources].

        Attributes
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).
        kind : string, optional
            Interpolation method used for segment-to-segment thermal response
            factors. See documentation for scipy.interpolate.interp1d.
            Default is linear.

        Returns
        -------
        h_ij : interp1d
            interp1d object (scipy.interpolate) of the matrix of
            segment-to-segment thermal response factors.

        """
        if self.disp:
            print('Calculating segment to segment response factors ...')
        # Number of time values
        nt = len(np.atleast_1d(time))
        # Prepare pool of workers for parallel computation
        pool = Pool(processes=self.processes)
        # Initialize chrono
        tic = tim.time()
        # Initialize segment-to-segment response factors
        h_ij = np.zeros((self.nSources, self.nSources, nt), dtype=self.dtype)

        for i in range(self.nSources):
            # Segment to same-segment thermal response factor
            # FLS solution for combined real and image sources
            b2 = self.boreSegments[i]
            func = partial(finite_line_source,
                           alpha=alpha, borehole1=b2, borehole2=b2)
            # Evaluate the FLS solution at all times in parallel
            h = np.array(pool.map(func, time), dtype=self.dtype)
            h_ij[i, i, :] = h

            # Segment to other segments thermal response factor
            for j in range(i+1, self.nSources):
                b1 = self.boreSegments[j]
                # Evaluate the FLS solution at all times in parallel
                func = partial(finite_line_source,
                               alpha=alpha, borehole1=b1, borehole2=b2)
                h = np.array(pool.map(func, time), dtype=self.dtype)
                h_ij[i, j, :] = h
                h_ij[j, i, :] = b2.H / b1.H * h_ij[i, j, :]

        # Close pool of workers
        pool.close()
        pool.join()

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,0]

        # Interp1d object for thermal response factors
        h_ij = interp1d(
            np.hstack((0., time)),
            np.dstack((np.zeros((self.nSources,self.nSources), dtype=self.dtype), h_ij)),
            kind=kind, copy=True, axis=2)
        toc = tim.time()
        if self.disp: print('{} sec'.format(toc - tic))

        return h_ij


class Similarities(_BaseSolver):
    """
    Detailed solver for the evaluation of the g-function.

    This solver superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#Similarities-CimBer2014]_. The number of evaluations of the FLS solution
    is decreased by identifying similar pairs of boreholes, for which the same
    FLS value can be applied [#Similarities-Cimmin2018]_.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    network : network object
        Model of the network.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    boundary_condition : str
        Boundary condition for the evaluation of the g-function. Should be one
        of

            - 'UHTR' :
                **Uniform heat transfer rate**. This is corresponds to boundary
                condition *BC-I* as defined by Cimmino and Bernier (2014)
                [#Similarities-CimBer2014]_.
            - 'UBWT' :
                **Uniform borehole wall temperature**. This is corresponds to
                boundary condition *BC-III* as defined by Cimmino and Bernier
                (2014) [#Similarities-CimBer2014]_.
            - 'MIFT' :
                **Mixed inlet fluid temperatures**. This boundary condition was
                introduced by Cimmino (2015) [#Similarities-Cimmin2015]_ for
                parallel-connected boreholes and extended to mixed
                configurations by Cimmino (2019) [#Similarities-Cimmin2019]_.

    nSegments : int, optional
        Number of line segments used per borehole.
        Default is 12.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    profiles : bool, optional
        Set to true to keep in memory the temperatures and heat extraction
        rates.
        Default is False.
    kind : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is linear.
    disTol : float, optional
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.

    References
    ----------
    .. [#Similarities-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#Similarities-Cimmin2018] Cimmino, M. (2018). Fast calculation of the
       g-functions of geothermal borehole fields using similarities in the
       evaluation of the finite line source solution. Journal of Building
       Performance Simulation, 11 (6), 655-668.
    .. [#Similarities-Cimmin2019] Cimmino, M. (2019). Semi-analytical method
       for g-function calculation of bore fields with series- and
       parallel-connected boreholes. Science and Technology for the Built
       Environment, 25 (8), 1007-1022.

    """
    def initialize(self, disTol=0.01, tol=1.0e-6, **kwargs):
        """
        Split boreholes into segments and identify similarities in the
        borefield.

        Returns
        -------
        nSources : int
            Number of finite line heat sources in the borefield used to
            initialize the matrix of segment-to-segment thermal response
            factors (of size: nSources x nSources).

        """
        self.disTol = disTol
        self.tol = tol
        # Check the validity of inputs
        self._check_solver_specific_inputs()
        # Real and image FLS solutions are only split for numbers of segments
        # greater than 1
        self.splitRealAndImage = self.nSegments > 1
        # Split boreholes into segments
        self.boreSegments = self.borehole_segments()
        # Initialize similarities
        self.find_similarities()
        return len(self.boreSegments)

    def thermal_response_factors(self, time, alpha, kind='linear'):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments in the borefield at all time steps using the finite line
        source solution.

        This method returns a scipy.interpolate.interp1d object of the matrix
        of thermal response factors, containing a copy of the matrix accessible
        by h_ij.y[:nSources,:nSources,:nt+1]. The first index along the
        third axis corresponds to time t=0. The interp1d object can be used to
        obtain thermal response factors at any intermediat time by
        h_ij(t)[:nSources,:nSources].

        Attributes
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).
        kind : string, optional
            Interpolation method used for segment-to-segment thermal response
            factors. See documentation for scipy.interpolate.interp1d.
            Default is linear.

        Returns
        -------
        h_ij : interp1d
            interp1d object (scipy.interpolate) of the matrix of
            segment-to-segment thermal response factors.

        """
        if self.disp:
            print('Calculating segment to segment response factors ...')
        # Number of time values
        nt = len(np.atleast_1d(time))
        # Prepare pool of workers for parallel computation
        pool = Pool(processes=self.processes)
        # Initialize chrono
        tic = tim.time()
        # Initialize segment-to-segment response factors
        h_ij = np.zeros((self.nSources, self.nSources, nt), dtype=self.dtype)

        # Similarities for real sources
        for s in range(self.nSimPos):
            n1 = self.simPos[s][0][0]
            n2 = self.simPos[s][0][1]
            b1 = self.boreSegments[n1]
            b2 = self.boreSegments[n2]
            if self.splitRealAndImage:
                # FLS solution for real source only
                func = partial(finite_line_source,
                               alpha=alpha, borehole1=b1, borehole2=b2,
                               reaSource=True, imgSource=False)
            else:
                # FLS solution for combined real and image sources
                func = partial(finite_line_source,
                               alpha=alpha, borehole1=b1, borehole2=b2,
                               reaSource=True, imgSource=True)
            # Evaluate the FLS solution at all times in parallel
            hPos = np.array(pool.map(func, np.atleast_1d(time)), dtype=self.dtype)
            # Assign thermal response factors to similar segment pairs
            for (i, j) in self.simPos[s]:
                h_ij[j, i, :] = hPos
                h_ij[i, j, :] = b2.H/b1.H * hPos

        # Similarities for image sources (only if splitRealAndImage=True)
        if self.splitRealAndImage:
            for s in range(self.nSimNeg):
                n1 = self.simNeg[s][0][0]
                n2 = self.simNeg[s][0][1]
                b1 = self.boreSegments[n1]
                b2 = self.boreSegments[n2]
                # FLS solution for image source only
                func = partial(finite_line_source,
                               alpha=alpha, borehole1=b1, borehole2=b2,
                               reaSource=False, imgSource=True)
                # Evaluate the FLS solution at all times in parallel
                hNeg = np.array(pool.map(func, time), dtype=self.dtype)
                # Assign thermal response factors to similar segment pairs
                for (i, j) in self.simNeg[s]:
                    h_ij[j, i, :] = h_ij[j, i, :] + hNeg
                    h_ij[i, j, :] = b2.H/b1.H * h_ij[j, i, :]

        # Close pool of workers
        pool.close()
        pool.join()

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,0]

        # Interp1d object for thermal response factors
        h_ij = interp1d(np.hstack((0., time)),
                             np.dstack((np.zeros((self.nSources,self.nSources), dtype=self.dtype), h_ij)),
                             kind=kind, copy=True, axis=2)
        toc = tim.time()
        if self.disp: print('{} sec'.format(toc - tic))

        return h_ij

    def find_similarities(self):
        """
        Find similarities in the FLS solution for groups of boreholes.

        This function identifies pairs of boreholes for which the evaluation
        of the Finite Line Source (FLS) solution is equivalent.

        """
        if self.disp: print('Identifying similarities ...')
        # Initialize chrono
        tic = tim.time()
        # Initialize pool of workers
        pool = Pool(processes=self.processes)

        # Group pairs of boreholes by radial distance
        (nDis, disPairs, nPairs, pairs) = self.group_by_distance()

        # If real and image parts of the FLS are split, evaluate real and image
        # similarities separately:
        if self.splitRealAndImage:
            # Evaluate similarities for each distance in parallel
            func = partial(self.find_similarities_one_distance, kind='real')
            realSims = pool.map(func, pairs)

            # Evaluate similarities for each distance in parallel
            func = partial(self.find_similarities_one_distance, kind='image')
            imageSims = pool.map(func, pairs)

        # Otherwise, evaluate the combined real+image FLS similarities
        else:
            func = partial(
                self.find_similarities_one_distance, kind='realandimage')
            # Evaluate symmetries for each distance in parallel
            realSims = pool.map(func, pairs)

        # Close pool of workers
        pool.close()
        pool.join()

        # Aggregate real similarities for all distances
        self.nSimPos = 0
        self.simPos = []
        self.HSimPos = []
        self.DSimPos = []
        self.disSimPos = []
        for i in range(nDis):
            realSim = realSims[i]
            nSim = realSim[0]
            self.nSimPos += nSim
            self.disSimPos += [disPairs[i] for _ in range(nSim)]
            self.simPos += realSim[1]
            self.HSimPos += realSim[2]
            self.DSimPos += realSim[3]

        # Aggregate image similarities for all distances
        self.nSimNeg = 0
        self.simNeg = []
        self.HSimNeg = []
        self.DSimNeg = []
        self.disSimNeg = []
        if self.splitRealAndImage:
            for i in range(nDis):
                imageSim = imageSims[i]
                nSim = imageSim[0]
                self.nSimNeg += nSim
                self.disSimNeg += [disPairs[i] for _ in range(nSim)]
                self.simNeg += imageSim[1]
                self.HSimNeg += imageSim[2]
                self.DSimNeg += imageSim[3]

        # Stop chrono
        toc = tim.time()
        if self.disp: print('{} sec'.format(toc - tic))

        return

    def find_similarities_one_distance(self, pairs, kind):
        """
        Evaluate similarities for all pairs of boreholes separated by the same
        radial distance.

        Parameters
        ----------
        pairs : list
            List of tuples of the borehole indices of borehole pairs at each
            radial distance.
        kind : string
            Type of similarity to be evaluated
                - 'real' : similarity in real sources
                - 'image' : similarity in image sources
                - 'realandimage' : similarity for combined real and image
                    sources.

        Returns
        -------
        nSim : int
            Number of similarities.
        sim : list
            For each similarity, a list of pairs (tuple) of borehole indices
            is returned.
        HSim : list
            List of lengths (tuple) of the pairs of boreholes in each
            similarity.
        DSim : list
            List of depths (tuple) of the pairs of boreholes in each
            similarity.

        """
        # Condition for equivalence of the real part of the FLS solution
        def compare_real_segments(H1a, H1b, H2a, H2b, D1a,
                                  D1b, D2a, D2b, tol):
            if (abs((H1a-H1b)/H1a) < tol and
                abs((H2a-H2b)/H2a) < tol and
                abs(((D2a-D1a)-(D2b-D1b))/(D2a-D1a+1e-30)) < tol):
                similarity = True
            else:
                similarity = False
            return similarity

        # Condition for equivalence of the image part of the FLS solution
        def compare_image_segments(H1a, H1b, H2a, H2b,
                                   D1a, D1b, D2a, D2b, tol):
            if (abs((H1a-H1b)/H1a) < tol and
                abs((H2a-H2b)/H2a) < tol and
                abs(((D2a+D1a)-(D2b+D1b))/(D2a+D1a+1e-30)) < tol):
                similarity = True
            else:
                similarity = False
            return similarity

        # Condition for equivalence of the full FLS solution
        def compare_realandimage_segments(H1a, H1b, H2a, H2b,
                                          D1a, D1b, D2a, D2b,
                                          tol):
            if (abs((H1a-H1b)/H1a) < tol and
                abs((H2a-H2b)/H2a) < tol and
                abs((D1a-D1b)/(D1a+1e-30)) < tol and
                abs((D2a-D2b)/(D2a+1e-30)) < tol):
                similarity = True
            else:
                similarity = False
            return similarity

        # Initialize comparison function based on input argument
        if kind.lower() == 'real':
            # Check real part of FLS
            compare_segments = compare_real_segments
        elif kind.lower() == 'image':
            # Check image part of FLS
            compare_segments = compare_image_segments
        elif kind.lower() == 'realandimage':
            # Check full real+image FLS
            compare_segments = compare_realandimage_segments
        else:
            raise NotImplementedError(
                "Error: '{}' not implemented.".format(kind.lower()))

        # Initialize symmetries
        nSim = 1
        pair0 = pairs[0]
        i0 = pair0[0]
        j0 = pair0[1]
        sim = [[pair0]]
        HSim = [(self.boreSegments[i0].H, self.boreSegments[j0].H)]
        DSim = [(self.boreSegments[i0].D, self.boreSegments[j0].D)]

        # Cycle through all pairs of boreholes for the given distance
        for pair in pairs[1:]:
            ibor = pair[0]
            jbor = pair[1]
            b1 = self.boreSegments[ibor]
            b2 = self.boreSegments[jbor]
            # Verify if the current pair should be included in the
            # previously identified symmetries
            for k in range(nSim):
                H1 = HSim[k][0]
                H2 = HSim[k][1]
                D1 = DSim[k][0]
                D2 = DSim[k][1]
                if compare_segments(H1, b1.H, H2, b2.H,
                                    D1, b1.D, D2, b2.D, self.tol):
                    sim[k].append((ibor, jbor))
                    break
                elif compare_segments(H1, b2.H, H2, b1.H,
                                      D1, b2.D, D2, b1.D, self.tol):
                    sim[k].append((jbor, ibor))
                    break

            else:
                # Add symmetry to list if no match was found
                nSim += 1
                sim.append([pair])
                HSim.append((b1.H, b2.H))
                DSim.append((b1.D, b2.D))
        return nSim, sim, HSim, DSim

    def group_by_distance(self):
        """
        Group pairs of boreholes by radial distance between borehole.
    
        Returns
        -------
        nDis : int
            Number of unique radial distances between pairs of borehole.
        disPairs : list
            List of radial distances.
        nPairs : list
            List of number of pairs for each radial distance.
        pairs : list
            List of tuples of the borehole indices of borehole pairs at each
            radial distance.
    
        """
        # Initialize lists
        nPairs = [1]
        pairs = [[(0, 0)]]
        disPairs = [self.boreSegments[0].r_b]
        nDis = 1

        nb = len(self.boreSegments)
        for i in range(nb):
            b1 = self.boreSegments[i]
            # The first segment does not have to be compared to itself
            if i == 0:
                i2 = i + 1
            else:
                i2 = i

            for j in range(i2, nb):
                b2 = self.boreSegments[j]
                # Distance between current pair of boreholes
                dis = b1.distance(b2)
                if i == j:
                    # The relative tolerance is used for same-borehole
                    # distances
                    rTol = self.tol * b1.r_b
                else:
                    rTol = self.disTol*dis
                # Verify if the current pair should be included in the
                # previously identified symmetries
                for k in range(nDis):
                    if abs(disPairs[k] - dis) < rTol:
                        pairs[k].append((i, j))
                        nPairs[k] += 1
                        break

                else:
                    # Add symmetry to list if no match was found
                    nDis += 1
                    disPairs.append(dis)
                    pairs.append([(i, j)])
                    nPairs.append(1)
        return nDis, disPairs, nPairs, pairs

    def _check_solver_specific_inputs(self):
        """
        This method ensures that solver specific inputs to the Solver object
        are what is expected.

        """
        assert type(self.disTol) is float and self.disTol > 0., \
            "The distance tolerance 'disTol' should be a positive float."
        assert type(self.tol) is float and self.tol > 0., \
            "The relative tolerance 'tol' should be a positive float."
        return


def uniform_heat_extraction(boreholes, time, alpha, use_similarities=True,
                            disTol=0.01, tol=1.0e-6, processes=None,
                            disp=False, **kwargs):
    """
    Evaluate the g-function with uniform heat extraction along boreholes.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. This boundary
    condition correponds to *BC-I*, as defined by [#UHTR-CimBer2014]_.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    use_similarities : bool, optional
        True if similarities are used to limit the number of FLS evaluations.
        Default is True.
    disTol : float, optional
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    gFunction : float or array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = np.array([1.0*10**i for i in range(4, 12)])
    >>> gt.gfunction.uniform_heat_extraction([b1, b2], time, alpha)
    array([ 0.75978163,  1.84860837,  2.98861057,  4.33496051,  6.29199383,
        8.13636888,  9.08401497,  9.20736188])

    References
    ----------
    .. [#UHTR-CimBer2014] Cimmino, M., & Bernier, M. (2014). A semi-analytical
       method to generate g-functions for geothermal bore fields. International
       Journal of Heat and Mass Transfer, 70, 641-650.

    """
    boundary_condition = 'UHTR'
    # Build options dict
    options = {'nSegments':1,
               'disp':disp,
               'disTol':disTol,
               'tol':tol,
               'processes':processes,
               'disp':disp}
    # Select the correct solver:
    if use_similarities:
        method='similarities'
    else:
        method='detailed'
    # Evaluate g-function
    gFunc = gFunction(
        boreholes, alpha, time=time, method=method,
        boundary_condition=boundary_condition, options=options)

    return gFunc.gFunc


def uniform_temperature(boreholes, time, alpha, nSegments=12, kind='linear',
                        use_similarities=True, disTol=0.01, tol=1.0e-6,
                        processes=None, disp=False, **kwargs):
    """
    Evaluate the g-function with uniform borehole wall temperature.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments. This boundary
    condition correponds to *BC-III*, as defined by [#UBWT-CimBer2014]_.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSegments : int, optional
        Number of line segments used per borehole.
        Default is 12.
    kind : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is linear.
    use_similarities : bool, optional
        True if similarities are used to limit the number of FLS evaluations.
        Default is True.
    disTol : float, optional
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    gFunction : float or array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = np.array([1.0*10**i for i in range(4, 12)])
    >>> gt.gfunction.uniform_temperature([b1, b2], time, alpha)
    array([ 0.75978079,  1.84859851,  2.98852756,  4.33406497,  6.27830732,
        8.05746656,  8.93697282,  9.04925079])

    References
    ----------
    .. [#UBWT-CimBer2014] Cimmino, M., & Bernier, M. (2014). A semi-analytical
       method to generate g-functions for geothermal bore fields. International
       Journal of Heat and Mass Transfer, 70, 641-650.

    """
    boundary_condition = 'UBWT'
    # Build options dict
    options = {'nSegments':nSegments,
               'disp':disp,
               'kind':kind,
               'disTol':disTol,
               'tol':tol,
               'processes':processes,
               'disp':disp}
    # Select the correct solver:
    if use_similarities:
        method='similarities'
    else:
        method='detailed'
    # Evaluate g-function
    gFunc = gFunction(
        boreholes, alpha, time=time, method=method,
        boundary_condition=boundary_condition, options=options)

    return gFunc.gFunc


def equal_inlet_temperature(boreholes, UTubes, m_flow, cp, time, alpha,
                            kind='linear', nSegments=12,
                            use_similarities=True, disTol=0.01, tol=1.0e-6,
                            processes=None, disp=False, **kwargs):
    """
    Evaluate the g-function with equal inlet fluid temperatures.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#EIFT-Cimmin2015]_.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    UTubes : list of pipe objects
        Model for pipes inside each borehole.
    m_flow : float or array
        Fluid mass flow rate per borehole (in kg/s).
    cp : float
        Fluid specific isobaric heat capacity (in J/kg.K).
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSegments : int, optional
        Number of line segments used per borehole.
        Default is 12.
    kind : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is 'linear'.
    use_similarities : bool, optional
        True if similarities are used to limit the number of FLS evaluations.
        Default is True.
    disTol : float, optional
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    gFunction : float or array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = np.array([1.0*10**i for i in range(4, 12)])
    >>> gt.gfunction.uniform_temperature([b1, b2], time, alpha)
    array([ 0.75978079,  1.84859851,  2.98852756,  4.33406497,  6.27830732,
        8.05746656,  8.93697282,  9.04925079])

    References
    ----------
    .. [#EIFT-Cimmin2015] Cimmino, M. (2015). The effects of borehole thermal
       resistances and fluid flow rate on the g-functions of geothermal bore
       fields. International Journal of Heat and Mass Transfer, 91, 1119-1127.

    """
    network = Network(boreholes, UTubes, m_flow=m_flow*len(boreholes), cp=cp,
                      nSegments=nSegments)
    boundary_condition = 'MIFT'
    # Build options dict
    options = {'nSegments':nSegments,
                'disp':disp,
                'kind':kind,
                'disTol':disTol,
                'tol':tol,
                'processes':processes,
                'disp':disp}
    # Select the correct solver:
    if use_similarities:
        method='similarities'
    else:
        method='detailed'
    # Evaluate g-function
    gFunc = gFunction(
        network, alpha, time=time, method=method,
        boundary_condition=boundary_condition, options=options)

    return gFunc.gFunc


def mixed_inlet_temperature(network, m_flow, cp,
                            time, alpha, kind='linear', nSegments=12,
                            use_similarities=True, disTol=0.01, tol=1.0e-6,
                            processes=None, disp=False, **kwargs):
    """
    Evaluate the g-function with mixed inlet fluid temperatures.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#MIFT-Cimmin2019]_. The piping configurations between boreholes can be any
    combination of series and parallel connections.

    Parameters
    ----------
    network : Network objects
        List of boreholes included in the bore field.
    m_flow : float or array
        Total mass flow rate into the network or inlet mass flow rates
        into each circuit of the network (in kg/s). If a float is supplied,
        the total mass flow rate is split equally into all circuits.
    cp : float or array
        Fluid specific isobaric heat capacity (in J/kg.degC).
        Must be the same for all circuits (a single float can be supplied).
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSegments : int, optional
        Number of line segments used per borehole.
        Default is 12.
    kind : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is 'linear'.
    use_similarities : bool, optional
        True if similarities are used to limit the number of FLS evaluations.
        Default is True.
    disTol : float, optional
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    gFunction : float or array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> Utube1 = gt.pipes.SingleUTube(pos=[(-0.05, 0), (0, -0.05)],
                                      r_in=0.015, r_out=0.02,
                                      borehole=b1,k_s=2, k_g=1, R_fp=0.1)
    >>> Utube2 = gt.pipes.SingleUTube(pos=[(-0.05, 0), (0, -0.05)],
                                      r_in=0.015, r_out=0.02,
                                      borehole=b1,k_s=2, k_g=1, R_fp=0.1)
    >>> bore_connectivity = [-1, 0]
    >>> network = gt.networks.Network([b1, b2], [Utube1, Utube2], bore_connectivity)
    >>> time = np.array([1.0*10**i for i in range(4, 12)])
    >>> m_flow = 0.25
    >>> cp = 4000.
    >>> alpha = 1.0e-6
    >>> gt.gfunction.mixed_inlet_temperature(network, m_flow, cp, time, alpha)
    array([0.63782415, 1.63304116, 2.72191316, 4.04091713, 5.98240458,
       7.77216202, 8.66195828, 8.77567215])

    References
    ----------
    .. [#MIFT-Cimmin2019] Cimmino, M. (2019). Semi-analytical method for
       g-function calculation of bore fields with series- and
       parallel-connected boreholes. Science and Technology for the Built
       Environment, 25 (8), 1007-1022

    """
    boundary_condition = 'MIFT'
    # Build options dict
    options = {'nSegments':nSegments,
                'disp':disp,
                'kind':kind,
                'disTol':disTol,
                'tol':tol,
                'processes':processes,
                'disp':disp}
    # Select the correct solver:
    if use_similarities:
        method='similarities'
    else:
        method='detailed'
    # Evaluate g-function
    gFunc = gFunction(
        network, alpha, time=time, method=method,
        boundary_condition=boundary_condition, options=options)

    return gFunc.gFunc
