# -*- coding: utf-8 -*-
import time as tim

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi
from scipy.interpolate import interp1d as interp1d

from .boreholes import Borehole, find_duplicates
from .heat_transfer import finite_line_source, finite_line_source_vectorized
from .networks import Network, network_thermal_resistance
from .utilities import _initialize_figure, _format_axes


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
        Default is None.
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

            nSegments : int or list, optional
                Number of line segments used per borehole, or list of number of
                line segments used for each borehole.
                Default is 12.
            segmentLengths : list, optional
                This is a 2D list that defines the segment lengths of borehole
                i. Each of the lists must add up to the total height of borehole
                i.
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
                Default is 'linear'.
            dtype : numpy dtype, optional
                numpy data type used for matrices and vectors. Should be one of
                numpy.single or numpy.double.
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
            self.solver = _Similarities(
                self.boreholes, self.network, self.time,
                self.boundary_condition, **self.options)
        elif self.method.lower()=='detailed':
            self.solver = _Detailed(
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
        time = np.atleast_1d(time)
        assert len(time) == 1 or np.all(time[:-1] <= time[1:]), \
            "Time values must be provided in increasing order."
        # Save time values
        self.time = time
        if self.solver.disp:
            print(60*'-')
            print('Calculating g-function for boundary condition : \'{}\''.format(
                self.boundary_condition).center(60))
            print(60*'-')
        # Initialize chrono
        tic = tim.time()

        # Evaluate g-function
        self.gFunc = self.solver.solve(time, self.alpha)
        toc = tim.time()

        if self.solver.disp:
            print('Total time for g-function evaluation: {:.3f} sec'.format(
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
        fig = _initialize_figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'ln$(t/t_s)$')
        ax.set_ylabel(r'$g$-function')
        _format_axes(ax)

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
            Default is None.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.boreholes))
        # Import heat extraction rates
        Q_t = self._heat_extraction_rates(iBoreholes)

        # Configure figure and axes
        fig = _initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        _format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'ln$(t/t_s)$')
        ax2.set_ylabel(r'$\bar{Q}_b$')
        _format_axes(ax2)

        # Borefield characteristic time
        ts = np.mean([b.H for b in self.boreholes])**2/(9.*self.alpha)
        # Dimensionless time (log)
        lntts = np.log(self.time/ts)
        # Plot curves for requested boreholes
        for (i, Q_ti) in zip(iBoreholes, Q_t):
            line = ax2.plot(lntts, Q_ti)
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
            Default is None.
        iBoreholes : list of int
            Borehole indices to plot heat extraction rate profiles.
            If iBoreholes is None, heat extraction rates are plotted for all
            boreholes.
            Default is None.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.boreholes))
        # Import heat extraction rate profiles
        z, Q_b = self._heat_extraction_rate_profiles(time, iBoreholes)

        # Configure figure and axes
        fig = _initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        _format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'$Q_b$')
        ax2.set_ylabel(r'$z$ [m]')
        ax2.invert_yaxis()
        _format_axes(ax2)

        # Plot curves for requested boreholes
        for (i, zi, Q_bi) in zip(iBoreholes, z, Q_b):
            line = ax2.plot(Q_bi, zi)
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
            Default is None.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.boreholes))
        # Import temperatures
        T_b = self._temperatures(iBoreholes)

        # Configure figure and axes
        fig = _initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        _format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'ln$(t/t_s)$')
        ax2.set_ylabel(r'$\bar{T}_b$')
        _format_axes(ax2)

        # Borefield characteristic time
        ts = np.mean([b.H for b in self.boreholes])**2/(9.*self.alpha)
        # Dimensionless time (log)
        lntts = np.log(self.time/ts)
        # Plot curves for requested boreholes
        for (i, T_bi) in zip(iBoreholes, T_b):
            line = ax2.plot(lntts, T_bi)
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
            Default is None.
        iBoreholes : list of int
            Borehole indices to plot temperature profiles.
            If iBoreholes is None, temperatures are plotted for all boreholes.
            Default is None.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.boreholes))
        # Import temperature profiles
        z, T_b = self._temperature_profiles(time, iBoreholes)

        # Configure figure and axes
        fig = _initialize_figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'$x$ [m]')
        ax1.set_ylabel(r'$y$ [m]')
        ax1.axis('equal')
        _format_axes(ax1)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'$T_b$')
        ax2.set_ylabel(r'$z$ [m]')
        ax2.invert_yaxis()
        _format_axes(ax2)

        # Plot curves for requested boreholes
        for (i, zi, T_bi) in zip(iBoreholes, z, T_b):
            line = ax2.plot(T_bi, zi)
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
        Q_t : list of array
            Heat extraction rates (dimensionless).

        """
        # Initialize list
        Q_t = []
        for i in iBoreholes:
            if self.boundary_condition == 'UHTR':
                # For the UHTR boundary condition, the heat extraction rate is
                # constant. A vector of length len(self.solver.time) is
                # required for the time-dependent figure.
                Q_t.append(self.solver.Q_b*np.ones(len(self.solver.time)))
            else:
                # For other boundary conditions, evaluate the average
                # heat extraction rate.
                i0 = self.solver._i0Segments[i]
                i1 = self.solver._i1Segments[i]
                Q_t.append(np.mean(self.solver.Q_b[i0:i1,:], axis=0))
        return Q_t

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
        Q_b : list of array
            Heat extraction rates (dimensionless).

        """
        # Initialize lists
        z = []
        Q_b = []
        for i in iBoreholes:
            if self.boundary_condition == 'UHTR':
                # For the UHTR boundary condition, the solver only returns one
                # heat extraction rate (uniform and equal for all boreholes).
                # The heat extraction rate is duplicated to draw from
                # z = D to z = D + H.
                z.append(
                    np.array([self.boreholes[i].D,
                              self.boreholes[i].D + self.boreholes[i].H]))
                Q_b.append(np.array(2*[self.solver.Q_b]))
            else:
                i0 = self.solver._i0Segments[i]
                i1 = self.solver._i1Segments[i]
                if time is None:
                    # If time is None, heat extraction rates are extracted at
                    # the last time step.
                    Q_bi = self.solver.Q_b[i0:i1,-1].flatten()
                else:
                    # Otherwise, heat extraction rates are interpolated.
                    Q_bi = interp1d(self.time, self.solver.Q_b[i0:i1,:],
                                    kind='linear',
                                    copy=False,
                                    axis=1)(time).flatten()
                if self.solver.nBoreSegments[i] > 1:
                    # Borehole length ratio at the mid-depth of each segment
                    z_ratio = np.linspace(
                        start=0.5/self.solver.nBoreSegments[i],
                        stop=1-0.5/self.solver.nBoreSegments[i],
                        num=self.solver.nBoreSegments[i])
                    z.append(self.boreholes[i].D + self.boreholes[i].H*z_ratio)
                    Q_b.append(Q_bi)
                else:
                    # If there is only one segment, the heat extraction rate is
                    # duplicated to draw from z = D to z = D + H.
                    z.append(
                        np.array([self.boreholes[i].D,
                                  self.boreholes[i].D + self.boreholes[i].H]))
                    Q_b.append(np.array(2*[np.asscalar(Q_bi)]))
        return z, Q_b

    def _temperatures(self, iBoreholes):
        """
        Extract and format borehole wall temperatures for plotting.

        Parameters
        ----------
        iBoreholes : list of int
            Borehole indices to extract temperatures.

        Returns
        -------
        T_b : list of array
            Borehole wall temperatures (dimensionless).

        """
        # Initialize list
        T_b = []
        for i in iBoreholes:
            if self.boundary_condition == 'UBWT':
                # For the UBWT boundary condition, the solver only returns one
                # borehole wall temperature (uniform and equal for all
                # boreholes).
                T_b.append(self.solver.T_b)
            else:
                # For other boundary conditions, evaluate the average
                # borehole wall temperature.
                i0 = self.solver._i0Segments[i]
                i1 = self.solver._i1Segments[i]
                T_b.append(np.mean(self.solver.T_b[i0:i1,:], axis=0))
        return T_b

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
        T_b : list of array
            Borehole wall temperatures (dimensionless).

        """
        # Initialize lists
        z = []
        T_b = []
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
                    T_bi = np.asscalar(self.solver.T_b[-1])
                else:
                    # Otherwise, temperatures are interpolated.
                    T_bi = np.asscalar(
                        interp1d(self.time,
                                 self.solver.T_b[:],
                                 kind='linear',
                                 copy=False)(time))
                T_b.append(np.array(2*[T_bi]))
            else:
                i0 = self.solver._i0Segments[i]
                i1 = self.solver._i1Segments[i]
                if time is None:
                    # If time is None, temperatures are extracted at the last
                    # time step.
                    T_bi = self.solver.T_b[i0:i1,-1].flatten()
                else:
                    # Otherwise, temperatures are interpolated.
                    T_bi = interp1d(self.time,
                                    self.solver.T_b[i0:i1,:],
                                    kind='linear',
                                    copy=False,
                                    axis=1)(time).flatten()
                if self.solver.nBoreSegments[i] > 1:
                    # Borehole length ratio at the mid-depth of each segment
                    z_ratio = np.linspace(
                        start=0.5/self.solver.nBoreSegments[i],
                        stop=1-0.5/self.solver.nBoreSegments[i],
                        num=self.solver.nBoreSegments[i])
                    z.append(self.boreholes[i].D + self.boreholes[i].H*z_ratio)
                    T_b.append(T_bi)
                else:
                    # If there is only one segment, the temperature is
                    # duplicated to draw from z = D to z = D + H.
                    z.append(
                        np.array([self.boreholes[i].D,
                                  self.boreholes[i].D + self.boreholes[i].H]))
                    T_b.append(np.array(2*[np.asscalar(T_bi)]))
        return z, T_b

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
            "The list of boreholes contains elements that are not Borehole " \
            "objects."
        assert not find_duplicates(self.boreholes), \
            "There are duplicate boreholes in the borefield."
        assert (self.network is None and not self.boundary_condition=='MIFT') or isinstance(self.network, Network), \
            "The network is not a valid 'Network' object."
        assert self.network is None or (self.network.m_flow_network is not None and self.network.cp_f is not None), \
            "The mass flow rate 'm_flow_network' and heat capacity 'cp_f' must " \
            "be provided at the instanciation of the 'Network' object."
        assert type(self.time) is np.ndarray or isinstance(self.time, (np.floating, float)) or self.time is None, \
            "Time should be a float or an array."
        assert isinstance(self.alpha, (np.floating, float)), \
            "The thermal diffusivity 'alpha' should be a float or an array."
        acceptable_boundary_conditions = ['UHTR', 'UBWT', 'MIFT']
        assert type(self.boundary_condition) is str and self.boundary_condition in acceptable_boundary_conditions, \
            "Boundary condition \'{}\' is not an acceptable boundary condition. \n" \
            "Please provide one of the following inputs : {}".format(
                self.boundary_condition, acceptable_boundary_conditions)
        acceptable_methods = ['detailed', 'similarities']
        assert type(self.method) is str and self.method in acceptable_methods, \
            "Method \'{}\' is not an acceptable method. \n" \
            "Please provide one of the following inputs : {}".format(
                self.boundary_condition, acceptable_methods)
        return


def uniform_heat_extraction(boreholes, time, alpha, use_similarities=True,
                            disTol=0.01, tol=1.0e-6, dtype=np.double,
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
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.
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
               'dtype':dtype,
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
                        dtype=np.double, disp=False, **kwargs):
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
    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 12.
    segmentLengths : list, optional
        This is a 2D list that defines the segment lengths of borehole i. Each
        of the lists must add up to the total height of borehole i.
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
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.
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
               'dtype':dtype,
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


def equal_inlet_temperature(
        boreholes, UTubes, m_flow_borehole, cp_f, time, alpha,
        kind='linear', nSegments=12, segmentLengths=None,
        use_similarities=True, disTol=0.01, tol=1.0e-6, dtype=np.double,
        disp=False, **kwargs):
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
    m_flow_borehole : float or array
        Fluid mass flow rate per borehole (in kg/s).
    cp_f : float
        Fluid specific isobaric heat capacity (in J/kg.K).
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 12.
    segmentLengths : list, optional
        This is a 2D list that defines the segment lengths of borehole i. Each
        of the lists must add up to the total height of borehole i.
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
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    gFunction : float or array
        Values of the g-function

    References
    ----------
    .. [#EIFT-Cimmin2015] Cimmino, M. (2015). The effects of borehole thermal
       resistances and fluid flow rate on the g-functions of geothermal bore
       fields. International Journal of Heat and Mass Transfer, 91, 1119-1127.

    """
    network = Network(
        boreholes, UTubes, m_flow_network=m_flow_borehole*len(boreholes),
        cp_f=cp_f, nSegments=nSegments)
    boundary_condition = 'MIFT'
    # Build options dict
    options = {'nSegments':nSegments,
               'segmentLengths': segmentLengths,
               'disp':disp,
               'kind':kind,
               'disTol':disTol,
               'tol':tol,
               'dtype':dtype,
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


def mixed_inlet_temperature(network, m_flow_network, cp_f,
                            time, alpha, kind='linear', nSegments=12,
                            segmentLengths=None, use_similarities=True,
                            disTol=0.01, tol=1.0e-6, dtype=np.double,
                            disp=False, **kwargs):
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
    m_flow_network : float or array
        Total mass flow rate into the network or inlet mass flow rates
        into each circuit of the network (in kg/s). If a float is supplied,
        the total mass flow rate is split equally into all circuits.
    cp_f : float or array
        Fluid specific isobaric heat capacity (in J/kg.degC).
        Must be the same for all circuits (a single float can be supplied).
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 12.
    segmentLengths : list, optional
        This is a 2D list that defines the segment lengths of borehole i. Each
        of the lists must add up to the total height of borehole i.
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
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.
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
    >>> m_flow_network = 0.25
    >>> cp_f = 4000.
    >>> alpha = 1.0e-6
    >>> gt.gfunction.mixed_inlet_temperature(network, m_flow_network, cp_f, time, alpha)
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
               'segmentLengths': segmentLengths,
               'disp':disp,
               'kind':kind,
               'disTol':disTol,
               'tol':tol,
               'dtype':dtype,
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

    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 12.
    segmentLengths : list, optional
        This is a 2D list that defines the segment lengths of borehole i. Each
        of the lists must add up to the total height of borehole i.
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
        Default is 'linear'.
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.

    """
    def __init__(self, boreholes, network, time, boundary_condition,
                 nSegments=12, segmentLengths=None, disp=False, profiles=False,
                 kind='linear', dtype=np.double, **other_options):
        self.boreholes = boreholes
        self.network = network
        # Convert time to a 1d array
        self.time = np.atleast_1d(time).flatten()
        self.boundary_condition = boundary_condition
        # Find indices of first and last segments along boreholes
        nBoreholes = len(self.boreholes)
        if type(nSegments) is int:
            self.nBoreSegments = [nSegments] * nBoreholes
        else:
            self.nBoreSegments = nSegments
        if type(segmentLengths) is list:
            # Flatten the list for use inside of the solver
            self.segmentLengths = [Lq for sub in segmentLengths for Lq in sub]
        else:
            # Segment lengths will be equal for borehole i
            self.segmentLengths = [boreholes[i].H / self.nBoreSegments[i]
                                   for i in range(len(boreholes))
                                   for j in range(self.nBoreSegments[i])]
        self._i0Segments = [sum(self.nBoreSegments[0:i])
                            for i in range(nBoreholes)]
        self._i1Segments = [sum(self.nBoreSegments[0:(i + 1)])
                            for i in range(nBoreholes)]
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
        self.time = time
        nt = len(self.time)
        # Initialize g-function
        gFunc = np.zeros(nt)
        # Initialize segment heat extraction rates
        if self.boundary_condition == 'UHTR':
            Q_b = 1
        else:
            Q_b = np.zeros((self.nSources, nt), dtype=self.dtype)
        if self.boundary_condition == 'UBWT':
            T_b = np.zeros(nt, dtype=self.dtype)
        else:
            T_b = np.zeros((self.nSources, nt), dtype=self.dtype)
        # Calculate segment to segment thermal response factors
        h_ij = self.thermal_response_factors(time, alpha, kind=self.kind)
        # Segment lengths
        H_b = self.segment_lengths()
        H_tot = np.sum(H_b)
        if self.disp: print('Building and solving the system of equations ...',
                            end='')
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
                T_b[:,p] = np.sum(h_dt, axis=1)
                # The g-function is the average of all borehole wall
                # temperatures
                gFunc[p] = np.sum(T_b[:,p]*H_b)/H_tot
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
                    self.time[0:p+1], Q_b[:,0:p+1])
                # Borehole wall temperature for zero heat extraction at
                # current step
                T_b0 = self.temporal_superposition(
                    h_ij.y[:,:,1:], Q_reconstructed)

                if self.boundary_condition == 'UBWT':
                    # Evaluate the g-function with uniform borehole wall
                    # temperature
                    # ---------------------------------------------------------
                    # Build a system of equation [A]*[X] = [B] for the
                    # evaluation of the g-function. [A] is a coefficient
                    # matrix, [X] = [Q_b,T_b] is a state space vector of the
                    # borehole heat extraction rates and borehole wall
                    # temperature (equal for all segments), [B] is a
                    # coefficient vector.
                    #
                    # Spatial superposition: [T_b] = [T_b0] + [h_ij_dt]*[Q_b]
                    # Energy conservation: sum([Q_b*Hb]) = sum([Hb])
                    # ---------------------------------------------------------
                    A = np.block([[h_dt, -np.ones((self.nSources, 1),
                                                  dtype=self.dtype)],
                                  [H_b, 0.]])
                    B = np.hstack((-T_b0, H_tot))
                    # Solve the system of equations
                    X = np.linalg.solve(A, B)
                    # Store calculated heat extraction rates
                    Q_b[:,p] = X[0:self.nSources]
                    # The borehole wall temperatures are equal for all segments
                    T_b[p] = X[-1]
                    gFunc[p] = T_b[p]
                elif self.boundary_condition == 'MIFT':
                    # Evaluate the g-function with mixed inlet fluid
                    # temperatures
                    # ---------------------------------------------------------
                    # Build a system of equation [A]*[X] = [B] for the
                    # evaluation of the g-function. [A] is a coefficient
                    # matrix, [X] = [Q_b,T_b,Tf_in] is a state space vector of
                    # the borehole heat extraction rates, borehole wall
                    # temperatures and inlet fluid temperature (into the bore
                    # field), [B] is a coefficient vector.
                    #
                    # Spatial superposition: [T_b] = [T_b0] + [h_ij_dt]*[Q_b]
                    # Heat transfer inside boreholes:
                    # [Q_{b,i}] = [a_in]*[T_{f,in}] + [a_{b,i}]*[T_{b,i}]
                    # Energy conservation: sum([Q_b*H_b]) = sum([H_b])
                    # ---------------------------------------------------------
                    a_in, a_b = self.network.coefficients_borehole_heat_extraction_rate(
                            self.network.m_flow_network,
                            self.network.cp_f,
                            self.nBoreSegments)
                    k_s = self.network.p[0].k_s
                    A = np.block(
                        [[h_dt,
                          -np.eye(self.nSources, dtype=self.dtype),
                          np.zeros((self.nSources, 1), dtype=self.dtype)],
                         [np.eye(self.nSources, dtype=self.dtype),
                          a_b/(2.0*pi*k_s*np.atleast_2d(H_b).T),
                          a_in/(2.0*pi*k_s*np.atleast_2d(H_b).T)],
                         [H_b, np.zeros(self.nSources + 1, dtype=self.dtype)]])
                    B = np.hstack(
                        (-T_b0,
                         np.zeros(self.nSources, dtype=self.dtype),
                         H_tot))
                    # Solve the system of equations
                    X = np.linalg.solve(A, B)
                    # Store calculated heat extraction rates
                    Q_b[:,p] = X[0:self.nSources]
                    T_b[:,p] = X[self.nSources:2*self.nSources]
                    T_f_in = X[-1]
                    # The gFunction is equal to the effective borehole wall
                    # temperature
                    # Outlet fluid temperature
                    T_f_out = T_f_in - 2*pi*self.network.p[0].k_s*H_tot/(
                        self.network.m_flow_network*self.network.cp_f)
                    # Average fluid temperature
                    T_f = 0.5*(T_f_in + T_f_out)
                    # Borefield thermal resistance
                    R_field = network_thermal_resistance(
                        self.network, self.network.m_flow_network,
                        self.network.cp_f)
                    # Effective borehole wall temperature
                    T_b_eff = T_f - 2*pi*self.network.p[0].k_s*R_field
                    gFunc[p] = T_b_eff
        # Store temperature and heat extraction rate profiles
        if self.profiles:
            self.Q_b = Q_b
            self.T_b = T_b
        toc = tim.time()
        if self.disp: print(' {:.3f} sec'.format(toc - tic))
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
        H_b = np.array(self.segmentLengths, dtype=self.dtype)
        return H_b

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
        boreSegments = []  # list for storage of boreSegments
        for j in range(len(self.boreholes)):
            for k in range(self.nBoreSegments[j]):
                # Get index for current bore-segment
                idx = sum(self.nBoreSegments[0:j]) + k
                # Current borehole j
                b = self.boreholes[j]
                # Segment length, could be equal or unequal
                H_b = self.segmentLengths[idx]
                # Burial depth based on previous depth
                D = b.D + sum(self.segmentLengths[0:idx])
                boreSegments.append(Borehole(H_b, D, b.r_b, b.x, b.y))

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
        T_b0 : array
            Current values of borehole wall temperatures assuming no heat
            extraction during current time step.

        """
        # Number of heat sources
        nSources = Q_reconstructed.shape[0]
        # Number of time steps
        nt = Q_reconstructed.shape[1]
        # Spatial and temporal superpositions
        dQ = np.concatenate(
            (Q_reconstructed[:,0:1],
             Q_reconstructed[:,1:] - Q_reconstructed[:,0:-1]), axis=1)[:,::-1]
        # Borehole wall temperature
        T_b0 = np.einsum('ijk,jk', h_ij[:,:,:nt], dQ)
        
        return T_b0

    def load_history_reconstruction(self, time, Q_b):
        """
        Reconstructs the load history.

        This function calculates an equivalent load history for an inverted
        order of time step sizes.

        Parameters
        ----------
        time : array
            Values of time (in seconds) in the load history.
        Q_b : array
            Heat extraction rates (in Watts) of all segments at all times.

        Returns
        -------
        Q_reconstructed : array
            Reconstructed load history.

        """
        # Number of heat sources
        nSources = Q_b.shape[0]
        # Time step sizes
        dt = np.hstack((time[0], time[1:]-time[:-1]))
        # Time vector
        t = np.hstack((0., time, time[-1] + time[0]))
        # Inverted time step sizes
        dt_reconstructed = dt[::-1]
        # Reconstructed time vector
        t_reconstructed = np.hstack((0., np.cumsum(dt_reconstructed)))
        # Accumulated heat extracted
        f = np.hstack(
            (np.zeros((nSources, 1), dtype=self.dtype),
             np.cumsum(Q_b*dt, axis=1)))
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
            "The list of boreholes contains elements that are not Borehole " \
            "objects."
        assert np.all(np.abs([b.tilt for b in self.boreholes]) <= 1e-6), \
            "The current version of pygfunction only supports vertical " \
            "boreholes."
        assert self.network is None or isinstance(self.network, Network), \
            "The network is not a valid 'Network' object."
        assert self.network is None or (self.network.m_flow_network is not None and self.network.cp_f is not None), \
            "The mass flow rate 'm_flow_network' and heat capacity 'cp_f' must be " \
            "provided at the instanciation of the 'Network' object."
        assert type(self.time) is np.ndarray or isinstance(self.time, (float, np.floating)) or self.time is None, \
            "Time should be a float or an array."
        # self.nSegments can now be an int or list
        assert type(self.nBoreSegments) is list and len(self.nBoreSegments) == \
               len(self.nBoreSegments) and min(self.nBoreSegments) >= 1, \
            "The argument for number of segments `nSegments` should be " \
            "of type int or a list of integers. If passed as a list, the " \
            "length of the list should be equal to the number of boreholes" \
            "in the borefield. nSegments >= 1 is/are required."
        acceptable_boundary_conditions = ['UHTR', 'UBWT', 'MIFT']
        assert type(self.boundary_condition) is str and self.boundary_condition in acceptable_boundary_conditions, \
            "Boundary condition \'{}\' is not an acceptable boundary " \
            "condition. \n" \
            "Please provide one of the following inputs : {}".format(
                self.boundary_condition, acceptable_boundary_conditions)
        assert type(self.disp) is bool, \
            "The option 'disp' should be set to True or False."
        assert type(self.profiles) is bool, \
            "The option 'profiles' should be set to True or False."
        assert type(self.kind) is str, \
            "The option 'kind' should be set to a valid interpolation kind " \
            "in accordance with scipy.interpolate.interp1d options."
        acceptable_dtypes = (np.single, np.double)
        assert np.any([self.dtype is dtype for dtype in acceptable_dtypes]), \
            "Data type \'{}\' is not an acceptable data type. \n" \
            "Please provide one of the following inputs : {}".format(
                self.dtype, acceptable_dtypes)
        # Check to make sure the segment lengths for each borehole add up to the
        # total borehole length
        for j in range(len(self.boreholes)):
            previous_sum = sum(self.nBoreSegments[0:j])
            total_length_j = \
                sum(self.segmentLengths[
                    previous_sum:self.nBoreSegments[j] + previous_sum])
            error = abs(total_length_j - self.boreholes[j].H)
            assert(error < 1.0e-01), \
                "Defined segment lengths must add up to within a tenth " \
                "of a meter of the total borehole length, check borehole " \
                + str(j)

        return


class _Detailed(_BaseSolver):
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

    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 12.
    segmentLengths : list, optional
        This is a 2D list that defines the segment lengths of borehole i. Each
        of the lists must add up to the total height of borehole i.
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
        Default is 'linear'.
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.

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
            Default is 'linear'.

        Returns
        -------
        h_ij : interp1d
            interp1d object (scipy.interpolate) of the matrix of
            segment-to-segment thermal response factors.

        """
        if self.disp:
            print('Calculating segment to segment response factors ...',
                  end='')
        # Number of time values
        nt = len(np.atleast_1d(time))
        # Initialize chrono
        tic = tim.time()
        # Initialize segment-to-segment response factors
        h_ij = np.zeros((self.nSources, self.nSources, nt+1), dtype=self.dtype)
        nBoreholes = len(self.boreholes)
        segment_lengths = self.segment_lengths()

        for i in range(nBoreholes):
            # Segments of the receiving borehole
            b2 = self.boreholes[i].segments(self.nBoreSegments[i])
            # -----------------------------------------------------------------
            # Segment-to-segment thermal response factors for same-borehole
            # thermal interactions
            # -----------------------------------------------------------------
            b1 = b2
            h = finite_line_source(time, alpha, b1, b2)
            # Broadcast values to h_ij matrix
            i0 = self._i0Segments[i]
            i1 = self._i1Segments[i]
            h_ij[i0:i1, i0:i1, 1:] = h

            # -----------------------------------------------------------------
            # Segment-to-segment thermal response factors for
            # borehole-to-borehole thermal interactions
            # -----------------------------------------------------------------
            if i+1 < nBoreholes:
                # Segments of the emitting borehole
                b1 = [seg
                      for (b, iBor) in zip(self.boreholes[i+1:], range(i+1, nBoreholes))
                      for seg in b.segments(self.nBoreSegments[iBor])]
                h = finite_line_source(time, alpha, b1, b2)
                # Broadcast values to h_ij matrix
                for j in range(i+1, nBoreholes):
                    j0 = self._i0Segments[j]
                    j1 = self._i1Segments[j]
                    h_ij[i0:i1, j0:j1, 1:] = h[:, j0-i1:j1-i1, :]
                    if j > i:
                        h_ij[j0:j1, i0:i1, 1:] = np.einsum(
                            'ijk,i,j->jik',
                            h[:, j0-i1:j1-i1, :],
                            segment_lengths[i0:i1],
                            1/segment_lengths[j0:j1])

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,1]

        # Interp1d object for thermal response factors
        h_ij = interp1d(np.hstack((0., time)), h_ij,
                        kind=kind, copy=True, axis=2)
        toc = tim.time()
        if self.disp: print(' {:.3f} sec'.format(toc - tic))

        return h_ij


class _Similarities(_BaseSolver):
    """
    Similarities solver for the evaluation of the g-function.

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

    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 12.
    segmentLengths : list, optional
        This is a 2D list that defines the segment lengths of borehole i. Each
        of the lists must add up to the total height of borehole i.
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
        Default is 'linear'.
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.
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
    .. [#Similarities-Cimmin2015] Cimmino, M. (2015). The effects of borehole
       thermal resistances and fluid flow rate on the g-functions of geothermal
       bore fields. International Journal of Heat and Mass Transfer, 91,
       1119-1127.
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
            Default is 'linear'.

        Returns
        -------
        h_ij : interp1d
            interp1d object (scipy.interpolate) of the matrix of
            segment-to-segment thermal response factors.

        """
        if self.disp:
            print('Calculating segment to segment response factors ...',
                  end='')
        # Number of time values
        nt = len(np.atleast_1d(time))
        # Initialize chrono
        tic = tim.time()
        # Initialize segment-to-segment response factors
        h_ij = np.zeros((self.nSources, self.nSources, nt+1), dtype=self.dtype)
        segment_lengths = self.segment_lengths()

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole thermal
        # interactions
        # ---------------------------------------------------------------------
        for group in self.borehole_to_self:
            # Index of first borehole in group
            i = group[0]
            # Find segment-to-segment similarities
            H1, D1, H2, D2, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs(i, i)
            # Locate thermal response factors in the h_ij matrix
            i_segment, j_segment, k_segment, l_segment = \
                self._map_segment_pairs(
                    i_pair, j_pair, k_pair, [(n, n) for n in group], [0])
            # Evaluate FLS at all time steps
            D1 = D1.reshape(1, -1)
            D2 = D2.reshape(1, -1)
            dis = self.boreholes[i].r_b
            h = finite_line_source_vectorized(time, alpha, dis, H1, D1, H2, D2)
            # Broadcast values to h_ij matrix
            h_ij[j_segment, i_segment, 1:] = h[0, k_segment, :]
        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for borehole-to-borehole
        # thermal interactions
        # ---------------------------------------------------------------------
        nGroups = len(self.borehole_to_borehole)
        for n in range(nGroups):
            # Index of first borehole pair in group
            i, j = self.borehole_to_borehole[n][0]
            # Find segment-to-segment similarities
            H1, D1, H2, D2, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs(i, j)
            # Locate thermal response factors in the h_ij matrix
            i_segment, j_segment, k_segment, l_segment = \
                self._map_segment_pairs(
                    i_pair, j_pair, k_pair, self.borehole_to_borehole[n],
                    self.borehole_to_borehole_indices[n])
            # Evaluate FLS at all time steps
            dis = np.reshape(self.borehole_to_borehole_distances[n], (-1, 1))
            D1 = D1.reshape(1, -1)
            D2 = D2.reshape(1, -1)
            h = finite_line_source_vectorized(time, alpha, dis, H1, D1, H2, D2)
            # Broadcast values to h_ij matrix
            h_ij[j_segment, i_segment, 1:] = h[l_segment, k_segment, :]
            if (self._compare_boreholes(self.boreholes[j], self.boreholes[i]) and
                self.nBoreSegments[i] == self.nBoreSegments[j]):
                h_ij[i_segment, j_segment, 1:] = h[l_segment, k_segment, :]
            else:
                h_ij[i_segment, j_segment, 1:] = (h[l_segment, k_segment, :].T \
                    * segment_lengths[j_segment]/segment_lengths[i_segment]).T

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,1]

        # Interp1d object for thermal response factors
        h_ij = interp1d(
            np.hstack((0., time)), h_ij,
            kind=kind, copy=True, assume_sorted=True, axis=2)
        toc = tim.time()
        if self.disp: print(' {:.3f} sec'.format(toc - tic))

        return h_ij

    def find_similarities(self):
        """
        Find similarities in the FLS solution for groups of boreholes.

        This function identifies pairs of boreholes for which the evaluation
        of the Finite Line Source (FLS) solution is equivalent.

        """
        if self.disp: print('Identifying similarities ...', end='')
        # Initialize chrono
        tic = tim.time()

        # Find similar pairs of boreholes
        # Boreholes can only be similar if their segments are similar
        self.borehole_to_self, self.borehole_to_borehole = \
            self._find_axial_borehole_pairs(self.boreholes)
        # Find distances for each similar pairs
        self.borehole_to_borehole_distances, self.borehole_to_borehole_indices = \
            self._find_distances(
                self.boreholes, self.borehole_to_borehole)

        # Stop chrono
        toc = tim.time()
        if self.disp: print(' {:.3f} sec'.format(toc - tic))

        return

    def _compare_boreholes(self, borehole1, borehole2):
        """
        Compare two boreholes and checks if they have the same dimensions :
        H, D, and r_b.

        Parameters
        ----------
        borehole1 : Borehole object
            First borehole.
        borehole2 : Borehole object
            Second borehole.

        Returns
        -------
        similarity : bool
            True if the two boreholes have the same dimensions.

        """
        # Compare lengths (H), buried depth (D) and radius (r_b)
        if (abs((borehole1.H - borehole2.H)/borehole1.H) < self.tol and
            abs((borehole1.r_b - borehole2.r_b)/borehole1.r_b) < self.tol and
            abs((borehole1.D - borehole2.D)/(borehole1.D + 1e-30)) < self.tol):
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_real_pairs(self, pair1, pair2):
        """
        Compare two pairs of boreholes or segments and return True if the two
        pairs have the same FLS solution for real sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        deltaD1 = pair1[1].D - pair1[0].D
        deltaD2 = pair2[1].D - pair2[0].D

        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of lengths in each pair
        equal_H = abs((pair1[0].H - pair1[1].H)/pair1[0].H) < self.tol
        # Equality of buried depths differences
        cond_deltaD = abs(deltaD1 - deltaD2)/abs(deltaD1 + 1e-30) < self.tol
        # Equality of buried depths differences if all boreholes have the same
        # length
        cond_deltaD_equal_H = abs((abs(deltaD1) - abs(deltaD2))/(abs(deltaD1) + 1e-30)) < self.tol
        if cond_H and (cond_deltaD or (equal_H and cond_deltaD_equal_H)):
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_image_pairs(self, pair1, pair2):
        """
        Compare two pairs of boreholes or segments and return True if the two
        pairs have the same FLS solution for mirror sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        sumD1 = pair1[1].D + pair1[0].D
        sumD2 = pair2[1].D + pair2[0].D

        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of buried depths sums
        cond_sumD = abs((sumD1 - sumD2)/(sumD1 + 1e-30)) < self.tol
        if cond_H and cond_sumD:
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_realandimage_pairs(self, pair1, pair2):
        """
        Compare two pairs of boreholes or segments and return True if the two
        pairs have the same FLS solution for both real and mirror sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        if (self._compare_real_pairs(pair1, pair2)
            and self._compare_image_pairs(pair1, pair2)):
            similarity = True
        else:
            similarity = False
        return similarity

    def _find_axial_borehole_pairs(self, boreholes):
        """
        Find axial (i.e. disregarding the radial distance) similarities between
        borehole pairs to simplify the evaluation of the FLS solution.

        Parameters
        ----------
        boreholes : list of Borehole objects
            Boreholes in the bore field.

        Returns
        -------
        borehole_to_self : list
            Lists of borehole indexes for each unique set of borehole
            dimensions (H, D, r_b) in the bore field.
        borehole_to_borehole : list
            Lists of tuples of borehole indexes for each unique pair of
            boreholes that share the same (pairwise) dimensions (H, D).

        """
        # Compare for the full (real + image) FLS solution
        compare_pairs = self._compare_realandimage_pairs

        nBoreholes = len(boreholes)
        borehole_to_self = []
        # Only check for similarities if there is more than one borehole
        if nBoreholes > 1:
            borehole_to_borehole = []
            for i in range(nBoreholes):
                # Compare the borehole to all known unique sets of dimensions
                for k in range(len(borehole_to_self)):
                    m = borehole_to_self[k][0]
                    # Add the borehole to the group if a similar borehole is
                    # found
                    if (self._compare_boreholes(boreholes[i], boreholes[m]) and
                        self.nBoreSegments[i] == self.nBoreSegments[m]):
                        borehole_to_self[k].append(i)
                        break
                else:
                    # If no similar boreholes are known, append the groups
                    borehole_to_self.append([i])
                for j in range(i + 1, nBoreholes):
                    pair0 = (boreholes[i], boreholes[j]) # pair
                    pair1 = (boreholes[j], boreholes[i]) # reciprocal pair
                    # Compare pairs of boreholes to known unique pairs
                    for k in range(len(borehole_to_borehole)):
                        m, n = borehole_to_borehole[k][0]
                        pair_ref = (boreholes[m], boreholes[n])
                        # Add the pair (or the reciprocal pair) to a group
                        # if a similar one is found
                        if (compare_pairs(pair0, pair_ref) and
                            self.nBoreSegments[i] == self.nBoreSegments[m] and
                            self.nBoreSegments[j] == self.nBoreSegments[n]):
                            borehole_to_borehole[k].append((i, j))
                            break
                        elif (compare_pairs(pair1, pair_ref) and
                              self.nBoreSegments[j] == self.nBoreSegments[m] and
                              self.nBoreSegments[i] == self.nBoreSegments[n]):
                            borehole_to_borehole[k].append((j, i))
                            break
                    # If no similar pairs are known, append the groups
                    else:
                        borehole_to_borehole.append([(i, j)])
        else:
            # Outputs for a single borehole
            borehole_to_self = [[0]]
            borehole_to_borehole = []
        return borehole_to_self, borehole_to_borehole

    def _find_distances(self, boreholes, borehole_to_borehole):
        """
        Find unique distances between pairs of boreholes for each unique pair
        of boreholes in the bore field.

        Parameters
        ----------
        boreholes : list of Borehole objects
            Boreholes in the bore field.
        borehole_to_borehole : list
            Lists of tuples of borehole indexes for each unique pair of
            boreholes that share the same (pairwise) dimensions (H, D).

        Returns
        -------
        borehole_to_borehole_distances : list
            Sorted lists of borehole-to-borehole radial distances for each
            unique pair of boreholes.
        borehole_to_borehole_indices : list
            Lists of indexes of distances associated with each borehole pair.

        """
        nGroups = len(borehole_to_borehole)
        borehole_to_borehole_distances = []
        borehole_to_borehole_indices = \
            [np.empty(len(group), dtype=np.uint) for group in borehole_to_borehole]
        # Find unique distances for each group
        for i in range(nGroups):
            borehole_to_borehole_distances.append([])
            pairs = borehole_to_borehole[i]
            nPairs = len(pairs)
            # Array of all borehole-to-borehole distances within the group
            all_distances = np.array(
                [boreholes[pair[0]].distance(boreholes[pair[1]]) for pair in pairs])
            # Indices to sort the distance array
            i_sort = all_distances.argsort()
            # Sort the distance array
            distances_sorted = all_distances[i_sort]
            j0 = 0
            j1 = 1
            nDis = 0
            # For each increasing distance in the sorted array :
            # 1 - find all distances that are within tolerance
            # 2 - add the average distance in the list of unique distances
            # 3 - associate the distance index to all pairs for the identified
            #     distances
            # 4 - re-start at the next distance index not yet accounted for.
            while j0 < nPairs and j1 > 0:
                # Find the first distance outside tolerance
                j1 = np.argmax(
                    distances_sorted >= (1+self.disTol)*distances_sorted[j0])
                if j1 > j0:
                    # Average distance between pairs of boreholes
                    borehole_to_borehole_distances[i].append(
                        np.mean(distances_sorted[j0:j1]))
                    # Apply distance index to borehole pairs
                    borehole_to_borehole_indices[i][i_sort[j0:j1]] = nDis
                else:
                    # Average distance between pairs of boreholes
                    borehole_to_borehole_distances[i].append(
                        np.mean(distances_sorted[j0:]))
                    # Apply distance index to borehole pairs
                    borehole_to_borehole_indices[i][i_sort[j0:]] = nDis
                j0 = j1
                nDis += 1
        return borehole_to_borehole_distances, borehole_to_borehole_indices

    def _map_axial_segment_pairs(self, i, j,
                                 reaSource=True, imgSource=True):
        """
        Find axial (i.e. disregarding the radial distance) similarities between
        segment pairs along two boreholes to simplify the evaluation of the
        FLS solution.

        The returned H1, D1, H2, and D2 can be used to evaluate the segment-to-
        segment response factors using scipy.integrate.quad_vec.

        Parameters
        ----------
        i : int
            Index of the first borehole.
        j : int
            Index of the second borehole.

        Returns
        -------
        H1 : float
            Length of the emitting segments.
        D1 : array
            Array of buried depths of the emitting segments.
        H2 : float
            Length of the receiving segments.
        D2 : array
            Array of buried depths of the receiving segments.
        i_pair : list
            Indices of the emitting segments along a borehole.
        j_pair : list
            Indices of the receiving segments along a borehole.
        k_pair : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair).

        """
        # Initialize local variables
        borehole1 = self.boreholes[i]
        borehole2 = self.boreholes[j]
        assert reaSource or imgSource, \
            "At least one of reaSource and imgSource must be True."
        if reaSource and imgSource:
            # Find segment pairs for the full (real + image) FLS solution
            compare_pairs = self._compare_realandimage_pairs
        elif reaSource:
            # Find segment pairs for the real FLS solution
            compare_pairs = self._compare_real_pairs
        elif imgSource:
            # Find segment pairs for the image FLS solution
            compare_pairs = self._compare_image_pairs
        # Dive both boreholes into segments
        segments1 = borehole1.segments(self.nBoreSegments[i])
        segments2 = borehole2.segments(self.nBoreSegments[j])
        # Segments have equal lengths
        H1 = segments1[0].H
        H2 = segments2[0].H
        # Prepare lists of segment buried depths
        D1 = []
        D2 = []
        # All possible pairs (i, j) of indices between segments
        i_pair = np.array(
            [ii
             for ii in range(self.nBoreSegments[i])
             for jj in range(self.nBoreSegments[j])],
            dtype=np.uint)
        j_pair = np.array(
            [jj
             for ii in range(self.nBoreSegments[i])
             for jj in range(self.nBoreSegments[j])],
            dtype=np.uint)
        # Empty list of indices for unique pairs
        k_pair = np.empty(self.nBoreSegments[i] * self.nBoreSegments[j],
                          dtype=np.uint)
        unique_pairs = []
        nPairs = 0

        p = 0
        for ii in range(self.nBoreSegments[i]):
            for jj in range(self.nBoreSegments[j]):
                pair = (segments1[ii], segments2[jj])
                # Compare the segment pairs to all known unique pairs
                for k in range(nPairs):
                    m, n = unique_pairs[k][0], unique_pairs[k][1]
                    pair_ref = (segments1[m], segments2[n])
                    # Stop if a similar pair is found and assign the index
                    if compare_pairs(pair, pair_ref):
                        k_pair[p] = k
                        break
                # If no similar pair is found : add a new pair, increment the
                # number of unique pairs, and extract the associated buried
                # depths
                else:
                    k_pair[p] = nPairs
                    D1.append(segments1[ii].D)
                    D2.append(segments2[jj].D)
                    unique_pairs.append((ii, jj))
                    nPairs += 1
                p += 1
        return H1, np.array(D1), H2, np.array(D2), i_pair, j_pair, k_pair

    def _map_segment_pairs(self, i_pair, j_pair, k_pair, borehole_to_borehole,
                           borehole_to_borehole_indices):
        """
        Return the maping of the unique segment-to-segment thermal response
        factors (h) to the complete h_ij array of the borefield, such that:

            h_ij[j_segment, i_segment, :nt] = h[:nt, l_segment, k_segment].T,

        where h is the array of unique segment-to-segment thermal response
        factors for a given unique pair of boreholes at all unique distances.

        Parameters
        ----------
        i_pair : list
            Indices of the emitting segments.
        j_pair : list
            Indices of the receiving segments.
        k_pair : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair).
        borehole_to_borehole : list
            Tuples of borehole indexes.
        borehole_to_borehole_indices : list
            Indexes of distances.

        Returns
        -------
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.
        l_segment : list
            Indices of unique distances for all pairs in (i_pair, j_pair)
            in the bore field.

        """
        i_segment = np.concatenate(
            [i_pair + self._i0Segments[i] for (i, j) in borehole_to_borehole])
        j_segment = np.concatenate(
            [j_pair + self._i0Segments[j] for (i, j) in borehole_to_borehole])
        k_segment = np.concatenate(
            [k_pair for (i, j) in borehole_to_borehole])
        l_segment = np.concatenate(
            [np.repeat(i, len(k_pair)) for i in borehole_to_borehole_indices])
        return i_segment, j_segment, k_segment, l_segment

    def _check_solver_specific_inputs(self):
        """
        This method ensures that solver specific inputs to the Solver object
        are what is expected.

        """
        assert isinstance(self.disTol, (np.floating, float)) and self.disTol > 0., \
            "The distance tolerance 'disTol' should be a positive float."
        assert isinstance(self.tol, (np.floating, float)) and self.tol > 0., \
            "The relative tolerance 'tol' should be a positive float."
        return
