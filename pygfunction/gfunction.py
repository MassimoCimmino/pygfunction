# -*- coding: utf-8 -*-
from time import perf_counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage
from scipy.constants import pi
from scipy.interpolate import interp1d as interp1d

from .boreholes import Borehole, _EquivalentBorehole, find_duplicates
from .heat_transfer import finite_line_source, finite_line_source_vectorized, \
    finite_line_source_equivalent_boreholes_vectorized, \
    finite_line_source_inclined_vectorized
from .networks import Network, _EquivalentNetwork, network_thermal_resistance
from .utilities import _initialize_figure, _format_axes
from . import utilities


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
            - 'equivalent' :
                The equivalent borehole method of Prieto and Cimmino (2021)
                [#gFunction-PriCim2021]_. Boreholes are assembled into groups
                of boreholes that share similar borehole wall temperatures and
                heat extraction rates. Each group is represented by an
                equivalent borehole and the group-to-group thermal interactions
                are calculated by the FLS solution. This is an approximation of
                the 'similarities' method.

        Default is 'equivalent'.
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
    m_flow_borehole : (nBoreholes,) array or (nBoreholes, nMassFlow) array, optional
        Fluid mass flow rate into each borehole. If a
        (nBoreholes, nMassFlow) array is supplied, the
        (nMassFlow, nMassFlow,) variable mass flow rate g-functions
        will be evaluated using the method of Cimmino (2024)
        [#gFunction-Cimmin2024]_. Only required for the 'MIFT' boundary
         condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
         provided.
        Default is None.
    m_flow_network : float or (nMassFlow,) array, optional
        Fluid mass flow rate into the network of boreholes. If an array
        is supplied, the (nMassFlow, nMassFlow,) variable mass flow
        rate g-functions will be evaluated using the method of Cimmino
        (2024) [#gFunction-Cimmin2024]_. Only required for the 'MIFT' boundary
         condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
         provided.
        Default is None.
    cp_f : float, optional
        Fluid specific isobaric heat capacity (in J/kg.degC). Only required
        for the 'MIFT' boundary condition.
        Default is None.
    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            nSegments : int or list, optional
                Number of line segments used per borehole, or list of number of
                line segments used for each borehole.
                Default is 8.
            segment_ratios : array, list of arrays, or callable, optional
                Ratio of the borehole length represented by each segment. The
                sum of ratios must be equal to 1. The shape of the array is of
                (nSegments,) or list of (nSegments[i],). If
                segment_ratios==None, segments of equal lengths are considered.
                If a callable is provided, it must return an array of size
                (nSegments,) when provided with nSegments (of type int) as an
                argument, or an array of size (nSegments[i],) when provided
                with an element of nSegments (of type list).
                Default is :func:`utilities.segment_ratios`.
            approximate_FLS : bool, optional
                Set to true to use the approximation of the FLS solution of
                Cimmino (2021) [#gFunction-Cimmin2021]_. This approximation
                does not require the numerical evaluation of any integral. When
                using the 'equivalent' solver, the approximation is only
                applied to the thermal response at the borehole radius. Thermal
                interaction between boreholes is evaluated using the FLS
                solution.
                Default is False.
            nFLS : int, optional
                Number of terms in the approximation of the FLS solution. This
                parameter is unused if `approximate_FLS` is set to False.
                Default is 10. Maximum is 25.
            mQuad : int, optional
                Number of Gauss-Legendre sample points for the integral over
                :math:`u` in the inclined FLS solution.
                Default is 11.
            linear_threshold : float, optional
                Threshold time (in seconds) under which the g-function is
                linearized. The g-function value is then interpolated between 0
                and its value at the threshold. If linear_threshold==None, the
                g-function is linearized for times
                `t < r_b**2 / (25 * self.alpha)`.
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

        The 'equivalent' solver accepts the following method-specific
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
            kClusters : int, optional
                Increment on the minimum number of equivalent boreholes
                determined by cutting the dendrogram of the bore field given
                by the hierarchical agglomerative clustering method. Increasing
                the value of this parameter increases the accuracy of the
                method.
                Default is 1.

    Notes
    -----
    - The 'equivalent' solver does not support the 'MIFT' boundary condition
      when boreholes are connected in series.
    - The 'equivalent' solver does not support inclined boreholes.
    - The g-function is linearized for times `t < r_b**2 / (25 * self.alpha)`.
      The g-function value is then interpolated between 0 and its value at the
      threshold.
    - If the 'MIFT' is used, only one of the 'm_flow_borehole' or
      'm_flow_network' can be supplied.

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
    .. [#gFunction-PriCim2021] Prieto, C., & Cimmino, M.
       (2021). Thermal interactions in large irregular fields of geothermal
       boreholes: the method of equivalent borehole. Journal of Building
       Performance Simulation, 14 (4), 446-460.
    .. [#gFunction-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.
    .. [#gFunction-Cimmin2024] Cimmino, M. (2024). g-Functions for fields of
       series- and parallel-connected boreholes with variable fluid mass flow
       rate and reversible flow direction. Renewable Energy, 228, 120661.

    """
    def __init__(self, boreholes_or_network, alpha, time=None,
                 method='equivalent', boundary_condition=None,
                 m_flow_borehole=None, m_flow_network=None,
                 cp_f=None, options={}):
        self.alpha = alpha
        self.time = time
        self.method = method
        self.boundary_condition = boundary_condition
        self.m_flow_borehole = m_flow_borehole
        self.m_flow_network = m_flow_network
        self.cp_f = cp_f
        self.options = options

        # Format inputs and assign default values where needed
        self._format_inputs(boreholes_or_network)
        # Check the validity of inputs
        self._check_inputs()

        # Load the chosen solver
        if self.method.lower()=='similarities':
            self.solver = _Similarities(
                self.boreholes, self.network, self.time,
                self.boundary_condition, self.m_flow_borehole,
                self.m_flow_network, self.cp_f, **self.options)
        elif self.method.lower()=='detailed':
            self.solver = _Detailed(
                self.boreholes, self.network, self.time,
                self.boundary_condition, self.m_flow_borehole,
                self.m_flow_network, self.cp_f, **self.options)
        elif self.method.lower()=='equivalent':
            self.solver = _Equivalent(
                self.boreholes, self.network, self.time,
                self.boundary_condition, self.m_flow_borehole,
                self.m_flow_network, self.cp_f, **self.options)
        else:
            raise ValueError(f"'{method}' is not a valid method.")

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
        time = np.maximum(np.atleast_1d(time), 0.)
        assert len(time) == 1 or np.all(time[:-1] <= time[1:]), \
            "Time values must be provided in increasing order."
        # Save time values
        self.time = time
        if self.solver.disp:
            print(60*'-')
            print(f"Calculating g-function for boundary condition : "
                  f"'{self.boundary_condition}'".center(60))
            print(60*'-')
        # Initialize chrono
        tic = perf_counter()

        # Evaluate g-function
        self.gFunc = self.solver.solve(time, self.alpha)
        toc = perf_counter()

        if self.solver.disp:
            print(f'Total time for g-function evaluation: '
                  f'{toc - tic:.3f} sec')
            print(60*'-')
        return self.gFunc

    def visualize_g_function(self, which=None):
        """
        Plot the g-function of the borefield.

        Parameters
        ----------
        which : list of tuple, optional
            Tuples (i, j) of the variable mass flow rate g-functions to plot.
            If None, all g-functions are plotted.
            Default is None.

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
        if self.solver.nMassFlow == 0:
            ax.plot(lntts, self.gFunc)
        elif which is None:
            for j in range(self.solver.nMassFlow):
                for i in range(self.solver.nMassFlow):
                    ax.plot(
                        lntts,
                        self.gFunc[i,j,:],
                        label=f'$g_{{{i}{j}}}$')
            plt.legend()
        else:
            for (i, j) in which:
                ax.plot(
                    lntts,
                    self.gFunc[i,j,:],
                    label=f'$g_{{{i}{j}}}$')
            plt.legend()

        # Adjust figure to window
        plt.tight_layout()
        return fig

    def visualize_heat_extraction_rates(self, iBoreholes=None, showTilt=True):
        """
        Plot the time-variation of the average heat extraction rates.

        Parameters
        ----------
        iBoreholes : list of int
            Borehole indices to plot heat extraction rates.
            If iBoreholes is None, heat extraction rates are plotted for all
            boreholes.
            Default is None.
        showTilt : bool
            Set to True to show borehole inclination.
            Default is True

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.solver.boreholes))
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
        ts = np.mean([b.H for b in self.solver.boreholes])**2/(9.*self.alpha)
        # Dimensionless time (log)
        lntts = np.log(self.time/ts)
        # Plot curves for requested boreholes
        for i, borehole in enumerate(self.solver.boreholes):
            if i in iBoreholes:
                # Draw heat extraction rate
                line = ax2.plot(lntts, Q_t[iBoreholes.index(i)])
                color = line[-1]._color
                # Draw colored marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color=color)
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color=color)
            else:
                # Draw black marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color='k')
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color='k')

        # Adjust figure to window
        plt.tight_layout()
        return fig

    def visualize_heat_extraction_rate_profiles(
            self, time=None, iBoreholes=None, showTilt=True):
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
        showTilt : bool
            Set to True to show borehole inclination.
            Default is True

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.solver.boreholes))
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
        for i, borehole in enumerate(self.solver.boreholes):
            if i in iBoreholes:
                # Draw heat extraction rate profile
                line = ax2.plot(
                    Q_b[iBoreholes.index(i)], z[iBoreholes.index(i)])
                color = line[-1]._color
                # Draw colored marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color=color)
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color=color)
            else:
                # Draw black marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color='k')
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color='k')

        plt.tight_layout()
        return fig

    def visualize_temperatures(self, iBoreholes=None, showTilt=True):
        """
        Plot the time-variation of the average borehole wall temperatures.

        Parameters
        ----------
        iBoreholes : list of int
            Borehole indices to plot temperatures.
            If iBoreholes is None, temperatures are plotted for all boreholes.
            Default is None.
        showTilt : bool
            Set to True to show borehole inclination.
            Default is True

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.solver.boreholes))
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
        ts = np.mean([b.H for b in self.solver.boreholes])**2/(9.*self.alpha)
        # Dimensionless time (log)
        lntts = np.log(self.time/ts)
        # Plot curves for requested boreholes
        for i, borehole in enumerate(self.solver.boreholes):
            if i in iBoreholes:
                # Draw borehole wall temperature
                line = ax2.plot(lntts, T_b[iBoreholes.index(i)])
                color = line[-1]._color
                # Draw colored marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color=color)
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color=color)
            else:
                # Draw black marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color='k')
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color='k')

        # Adjust figure to window
        plt.tight_layout()
        return fig

    def visualize_temperature_profiles(
            self, time=None, iBoreholes=None, showTilt=True):
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
        showTilt : bool
            Set to True to show borehole inclination.
            Default is True

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
        for i, borehole in enumerate(self.solver.boreholes):
            if i in iBoreholes:
                # Draw heat extraction rate profile
                line = ax2.plot(
                    T_b[iBoreholes.index(i)], z[iBoreholes.index(i)])
                color = line[-1]._color
                # Draw colored marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color=color)
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
                         marker='o',
                         color=color)
            else:
                # Draw black marker for borehole position
                if showTilt:
                    ax1.plot(
                        [borehole.x, borehole.x + borehole.H*np.sin(borehole.tilt)*np.cos(borehole.orientation)],
                        [borehole.y, borehole.y + borehole.H*np.sin(borehole.tilt)*np.sin(borehole.orientation)],
                         linestyle='--',
                         marker='None',
                         color='k')
                ax1.plot(borehole.x,
                         borehole.y,
                         linestyle='None',
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
                segment_ratios = self.solver.segment_ratios[i]
                Q_t.append(
                    np.sum(self.solver.Q_b[i0:i1,:]*segment_ratios[:,np.newaxis],
                           axis=0))
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
                    np.array([self.solver.boreholes[i].D,
                              self.solver.boreholes[i].D + self.solver.boreholes[i].H]))
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
                    segment_ratios = self.solver.segment_ratios[i]
                    z.append(
                        self.solver.boreholes[i].D \
                        + self.solver.boreholes[i]._segment_midpoints(
                            self.solver.nBoreSegments[i],
                            segment_ratios=segment_ratios))
                    Q_b.append(Q_bi)
                else:
                    # If there is only one segment, the heat extraction rate is
                    # duplicated to draw from z = D to z = D + H.
                    z.append(
                        np.array([self.solver.boreholes[i].D,
                                  self.solver.boreholes[i].D + self.solver.boreholes[i].H]))
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
                segment_ratios = self.solver.segment_ratios[i]
                T_b.append(
                    np.sum(self.solver.T_b[i0:i1,:]*segment_ratios[:,np.newaxis],
                           axis=0))
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
                    np.array([self.solver.boreholes[i].D,
                              self.solver.boreholes[i].D + self.solver.boreholes[i].H]))
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

                    segment_ratios = self.solver.segment_ratios[i]
                    z.append(
                        self.solver.boreholes[i].D \
                        + self.solver.boreholes[i]._segment_midpoints(
                            self.solver.nBoreSegments[i],
                            segment_ratios=segment_ratios))
                    T_b.append(T_bi)
                else:
                    # If there is only one segment, the temperature is
                    # duplicated to draw from z = D to z = D + H.
                    z.append(
                        np.array([self.solver.boreholes[i].D,
                                  self.solver.boreholes[i].D + self.solver.boreholes[i].H]))
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
            # Extract mass flow rate from Network object if provided in the object
            # and none of m_flow_borehole and m_flow_network are provided
            if self.m_flow_borehole is None and self.m_flow_network is None:
                if type(self.network.m_flow_network) is float:
                     self.m_flow_network = self.network.m_flow_network
                elif type(self.network.m_flow_network) is np.ndarray:
                     self.m_flow_borehole = self.network.m_flow_network
            if self.cp_f is None and type(self.network.cp_f) is float:
                self.cp_f = self.network.cp_f
        else:
            self.network = None
            self.boreholes = boreholes_or_network
            # If a borefield is provided and no boundary condition is provided,
            # use 'UBWT'
            if self.boundary_condition is None:
                self.boundary_condition = 'UBWT'
        # If the 'equivalent' solver is selected for the 'MIFT' condition,
        # switch to the 'similarities' solver if boreholes are in series
        if self.boundary_condition == 'MIFT' and  self.method.lower() == 'equivalent':
            if not np.all(np.array(self.network.c, dtype=int) == -1):
                warnings.warn(
                    "\nSolver 'equivalent' is only valid for "
                    "parallel-connected boreholes. Calculations will use the "
                    "'similarities' solver instead.")
                self.method = 'similarities'
            elif not (self.m_flow_borehole is None
                      or np.allclose(
                          self.m_flow_borehole,
                          self.m_flow_borehole[0])):
                warnings.warn(
                    "\nSolver 'equivalent' is only valid for equal mass flow "
                    "rates into the boreholes. Calculations will use the "
                    "'similarities' solver instead.")
                self.method = 'similarities'
            elif not np.all(
                    [np.allclose(self.network.p[0]._Rd, pipe._Rd)
                     for pipe in self.network.p]):
                warnings.warn(
                    "\nSolver 'equivalent' is only valid for boreholes with "
                    "the same piping  configuration. Calculations will use "
                    "the 'similarities' solver instead.")
                self.method = 'similarities'
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
        if self.boundary_condition == 'MIFT':
            assert not (self.m_flow_network is None and self.m_flow_borehole is None), \
                "The mass flow rate 'm_flow_borehole' or 'm_flow_network' must " \
                "be provided when using the 'MIFT' boundary condition."
            assert not (self.m_flow_network is not None and self.m_flow_borehole is not None), \
                "Only one of 'm_flow_borehole' or 'm_flow_network' can " \
                "be provided when using the 'MIFT' boundary condition."
            assert not self.cp_f is None, \
                "The heat capacity 'cp_f' must " \
                "be provided when using the 'MIFT' boundary condition."
            assert not (type(self.m_flow_borehole) is np.ndarray and self.m_flow_borehole.ndim and not np.size(self.m_flow_borehole, axis=1)==len(self.boreholes)), \
                "The number of mass flow rates in 'm_flow_borehole' must " \
                "correspond to the number of boreholes."
        assert type(self.time) is np.ndarray or isinstance(self.time, (np.floating, float)) or self.time is None, \
            "Time should be a float or an array."
        assert isinstance(self.alpha, (np.floating, float)), \
            "The thermal diffusivity 'alpha' should be a float or an array."
        acceptable_boundary_conditions = ['UHTR', 'UBWT', 'MIFT']
        assert type(self.boundary_condition) is str and self.boundary_condition in acceptable_boundary_conditions, \
            f"Boundary condition '{self.boundary_condition}' is not an acceptable boundary condition. \n" \
            f"Please provide one of the following inputs : {acceptable_boundary_conditions}"
        acceptable_methods = ['detailed', 'similarities', 'equivalent']
        assert type(self.method) is str and self.method in acceptable_methods, \
            f"Method '{self.method}' is not an acceptable method. \n" \
            f"Please provide one of the following inputs : {acceptable_methods}"
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
    # This function is deprecated as of v2.1. It will be removed in v3.0.
    warnings.warn("`pygfunction.gfunction.uniform_heat_extraction` is "
                  "deprecated as of v2.1. It will be removed in v3.0. "
                  "New features are not fully supported by the function. "
                  "Use the `pygfunction.gfunction.gFunction` class instead.",
                  DeprecationWarning)

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


def uniform_temperature(boreholes, time, alpha, nSegments=8,
                        segment_ratios=utilities.segment_ratios, kind='linear',
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
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered. If a callable is provided, it
        must return an array of size (nSegments,) when provided with nSegments
        (of type int) as an argument, or an array of size (nSegments[i],) when
        provided with an element of nSegments (of type list).
        Default is :func:`utilities.segment_ratios`.
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
    # This function is deprecated as of v2.1. It will be removed in v3.0.
    warnings.warn("`pygfunction.gfunction.uniform_temperature` is "
                  "deprecated as of v2.1. It will be removed in v3.0. "
                  "New features are not fully supported by the function. "
                  "Use the `pygfunction.gfunction.gFunction` class instead.",
                  DeprecationWarning)

    boundary_condition = 'UBWT'
    # Build options dict
    options = {'nSegments':nSegments,
               'segment_ratios':segment_ratios,
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
        kind='linear', nSegments=8, segment_ratios=utilities.segment_ratios,
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
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered. If a callable is provided, it
        must return an array of size (nSegments,) when provided with nSegments
        (of type int) as an argument, or an array of size (nSegments[i],) when
        provided with an element of nSegments (of type list).
        Default is :func:`utilities.segment_ratios`.
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
    # This function is deprecated as of v2.1. It will be removed in v3.0.
    warnings.warn("`pygfunction.gfunction.equal_inlet_temperature` is "
                  "deprecated as of v2.1. It will be removed in v3.0. "
                  "New features are not fully supported by the function. "
                  "Use the `pygfunction.gfunction.gFunction` class instead.",
                  DeprecationWarning)

    network = Network(
        boreholes, UTubes, m_flow_network=m_flow_borehole*len(boreholes),
        cp_f=cp_f, nSegments=nSegments)
    boundary_condition = 'MIFT'
    # Build options dict
    options = {'nSegments':nSegments,
               'segment_ratios':segment_ratios,
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


def mixed_inlet_temperature(
        network, m_flow_network, cp_f, time, alpha, kind='linear',
        nSegments=8, segment_ratios=utilities.segment_ratios,
        use_similarities=True, disTol=0.01, tol=1.0e-6, dtype=np.double,
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
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered.
        Default is None.
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
    # This function is deprecated as of v2.1. It will be removed in v3.0.
    warnings.warn("`pygfunction.gfunction.mixed_inlet_temperature` is "
                  "deprecated as of v2.1. It will be removed in v3.0. "
                  "New features are not fully supported by the function. "
                  "Use the `pygfunction.gfunction.gFunction` class instead.",
                  DeprecationWarning)

    boundary_condition = 'MIFT'
    # Build options dict
    options = {'nSegments':nSegments,
               'segment_ratios':segment_ratios,
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
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered. If a callable is provided, it
        must return an array of size (nSegments,) when provided with nSegments
        (of type int) as an argument, or an array of size (nSegments[i],) when
        provided with an element of nSegments (of type list).
        Default is :func:`utilities.segment_ratios`.
    m_flow_borehole : (nBoreholes,) array or (nBoreholes, nMassFlow) array, optional
        Fluid mass flow rate into each borehole. If a
        (nBoreholes, nMassFlow) array is supplied, the
        (nMassFlow, nMassFlow,) variable mass flow rate g-functions
        will be evaluated using the method of Cimmino (2024)
        [#gFunction-CimBer2024]_. Only required for the 'MIFT' boundary
         condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
         provided.
        Default is None.
    m_flow_network : float or (nMassFlow,) array, optional
        Fluid mass flow rate into the network of boreholes. If an array
        is supplied, the (nMassFlow, nMassFlow,) variable mass flow
        rate g-functions will be evaluated using the method of Cimmino
        (2024) [#gFunction-CimBer2024]_. Only required for the 'MIFT' boundary
         condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
         provided.
        Default is None.
    cp_f : float, optional
        Fluid specific isobaric heat capacity (in J/kg.degC). Only required
        for the 'MIFT' boundary condition.
        Default is None.
    approximate_FLS : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021). This approximation does not require the numerical evaluation of
        any integral. When using the 'equivalent' solver, the approximation is
        only applied to the thermal response at the borehole radius. Thermal
        interaction between boreholes is evaluated using the FLS solution.
        Default is False.
    nFLS : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximate_FLS` is set to False.
        Default is 10. Maximum is 25.
    mQuad : int, optional
        Number of Gauss-Legendre sample points for the integral over :math:`u`
        in the inclined FLS solution.
        Default is 11.
    linear_threshold : float, optional
        Threshold time (in seconds) under which the g-function is
        linearized. The g-function value is then interpolated between 0
        and its value at the threshold. If linear_threshold==None, the
        g-function is linearized for times
        `t < r_b**2 / (25 * self.alpha)`.
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
        Default is 'linear'.
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.

    """
    def __init__(self, boreholes, network, time, boundary_condition,
                 m_flow_borehole=None, m_flow_network=None, cp_f=None,
                 nSegments=8, segment_ratios=utilities.segment_ratios,
                 approximate_FLS=False, mQuad=11, nFLS=10,
                 linear_threshold=None, disp=False, profiles=False,
                 kind='linear', dtype=np.double, **other_options):
        self.boreholes = boreholes
        self.network = network
        # Convert time to a 1d array
        self.time = np.atleast_1d(time).flatten()
        self.linear_threshold = linear_threshold
        self.r_b_max = np.max([b.r_b for b in self.boreholes])
        self.boundary_condition = boundary_condition
        nBoreholes = len(self.boreholes)
        # Format number of segments and segment ratios
        if type(nSegments) is int:
            self.nBoreSegments = [nSegments] * nBoreholes
        else:
            self.nBoreSegments = nSegments
        if isinstance(segment_ratios, np.ndarray):
            segment_ratios = [segment_ratios] * nBoreholes
        elif segment_ratios is None:
            segment_ratios = [np.full(n, 1./n) for n in self.nBoreSegments]
        elif callable(segment_ratios):
            segment_ratios = [segment_ratios(n) for n in self.nBoreSegments]
        self.segment_ratios = segment_ratios
        # Shortcut for segment_ratios comparisons
        self._equal_segment_ratios = \
            (np.all(np.array(self.nBoreSegments, dtype=np.uint) == self.nBoreSegments[0])
             and np.all([np.allclose(segment_ratios, self.segment_ratios[0]) for segment_ratios in self.segment_ratios]))
        # Boreholes with a uniform discretization
        self._uniform_segment_ratios = [
            np.allclose(segment_ratios,
                        segment_ratios[0:1],
                        rtol=1e-6)
            for segment_ratios in self.segment_ratios]
        # Find indices of first and last segments along boreholes
        self._i0Segments = [sum(self.nBoreSegments[0:i])
                            for i in range(nBoreholes)]
        self._i1Segments = [sum(self.nBoreSegments[0:(i + 1)])
                            for i in range(nBoreholes)]
        self.nMassFlow = 0
        self.m_flow_borehole = m_flow_borehole
        if self.m_flow_borehole is not None:
            if not self.m_flow_borehole.ndim == 1:
                self.nMassFlow = np.size(self.m_flow_borehole, axis=1)
            self.m_flow_borehole = np.atleast_2d(self.m_flow_borehole)
            self.m_flow = self.m_flow_borehole
        self.m_flow_network = m_flow_network
        if self.m_flow_network is not None:
            if not isinstance(self.m_flow_network, (np.floating, float)):
                self.nMassFlow = len(self.m_flow_network)
            self.m_flow_network = np.atleast_1d(self.m_flow_network)
            self.m_flow = self.m_flow_network
        self.cp_f = cp_f
        self.approximate_FLS = approximate_FLS
        self.mQuad = mQuad
        self.nFLS = nFLS
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
        # Evaluate threshold time for g-function linearization
        if self.linear_threshold is None:
            time_threshold = self.r_b_max**2 / (25 * alpha)
        else:
            time_threshold = self.linear_threshold
        # Find the number of g-function values to be linearized
        p_long = np.searchsorted(self.time, time_threshold, side='right')
        if p_long > 0:
            time_long = np.concatenate([[time_threshold], self.time[p_long:]])
        else:
            time_long = self.time
        nt_long = len(time_long)
        # Calculate segment to segment thermal response factors
        h_ij = self.thermal_response_factors(time_long, alpha, kind=self.kind)
        # Segment lengths
        H_b = self.segment_lengths()
        if self.boundary_condition == 'MIFT':
            Hb_individual = np.array([b.H for b in self.boreSegments], dtype=self.dtype)
        H_tot = np.sum(H_b)
        if self.disp: print('Building and solving the system of equations ...',
                            end='')
        # Initialize chrono
        tic = perf_counter()

        if self.boundary_condition == 'UHTR':
            # Initialize g-function
            gFunc = np.zeros(nt)
            # Initialize segment heat extraction rates
            Q_b = 1
            # Initialize borehole wall temperatures
            T_b = np.zeros((self.nSources, nt), dtype=self.dtype)

            # Build and solve the system of equations at all times
            p0 = max(0, p_long-1)
            for p in range(nt_long):
                # Evaluate the g-function with uniform heat extraction along
                # boreholes

                # Thermal response factors evaluated at time t[p]
                h_dt = h_ij.y[:,:,p+1]
                # Borehole wall temperatures are calculated by the sum of
                # contributions of all segments
                T_b[:,p+p0] = np.sum(h_dt, axis=1)
                # The g-function is the average of all borehole wall
                # temperatures
                gFunc[p+p0] = np.sum(T_b[:,p+p0]*H_b)/H_tot

            # Linearize g-function for times under threshold
            if p_long > 0:
                gFunc[:p_long] = gFunc[p_long-1] * self.time[:p_long] / time_threshold
                T_b[:,:p_long] = T_b[:,p_long-1:p_long] * self.time[:p_long] / time_threshold

        elif self.boundary_condition == 'UBWT':
            # Initialize g-function
            gFunc = np.zeros(nt)
            # Initialize segment heat extraction rates
            Q_b = np.zeros((self.nSources, nt), dtype=self.dtype)
            T_b = np.zeros(nt, dtype=self.dtype)

            # Build and solve the system of equations at all times
            p0 = max(0, p_long-1)
            for p in range(nt_long):
                # Current thermal response factor matrix
                if p > 0:
                    dt = time_long[p] - time_long[p-1]
                else:
                    dt = time_long[p]
                # Thermal response factors evaluated at t=dt
                h_dt = h_ij(dt)
                # Reconstructed load history
                Q_reconstructed = self.load_history_reconstruction(
                    time_long[0:p+1], Q_b[:,p0:p+p0+1])
                # Borehole wall temperature for zero heat extraction at
                # current step
                T_b0 = self.temporal_superposition(
                    h_ij.y[:,:,1:], Q_reconstructed)

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
                Q_b[:,p+p0] = X[0:self.nSources]
                # The borehole wall temperatures are equal for all segments
                T_b[p+p0] = X[-1]
                gFunc[p+p0] = T_b[p+p0]

            # Linearize g-function for times under threshold
            if p_long > 0:
                gFunc[:p_long] = gFunc[p_long-1] * self.time[:p_long] / time_threshold
                Q_b[:,:p_long] = 1 + (Q_b[:,p_long-1:p_long] - 1) * self.time[:p_long] / time_threshold
                T_b[:p_long] = T_b[p_long-1] * self.time[:p_long] / time_threshold

        elif self.boundary_condition == 'MIFT':
            if self.nMassFlow == 0:
                # Initialize g-function
                gFunc = np.zeros((1, 1, nt))
                # Initialize segment heat extraction rates
                Q_b = np.zeros((1, self.nSources, nt), dtype=self.dtype)
                T_b = np.zeros((1, self.nSources, nt), dtype=self.dtype)
            else:
                # Initialize g-function
                gFunc = np.zeros((self.nMassFlow, self.nMassFlow, nt))
                # Initialize segment heat extraction rates
                Q_b = np.zeros(
                    (self.nMassFlow, self.nSources, nt), dtype=self.dtype)
                T_b = np.zeros(
                    (self.nMassFlow, self.nSources, nt), dtype=self.dtype)

            for j in range(np.maximum(self.nMassFlow, 1)):
                # Build and solve the system of equations at all times
                p0 = max(0, p_long-1)
                a_in_j, a_b_j = self.network.coefficients_borehole_heat_extraction_rate(
                        self.m_flow[j],
                        self.cp_f,
                        self.nBoreSegments,
                        segment_ratios=self.segment_ratios)
                k_s = self.network.p[0].k_s
                for p in range(nt_long):
                    # Current thermal response factor matrix
                    if p > 0:
                        dt = time_long[p] - time_long[p-1]
                    else:
                        dt = time_long[p]
                    # Thermal response factors evaluated at t=dt
                    h_dt = h_ij(dt)
                    # Reconstructed load history
                    Q_reconstructed = self.load_history_reconstruction(
                        time_long[0:p+1], Q_b[j,:,p0:p+p0+1])
                    # Borehole wall temperature for zero heat extraction at
                    # current step
                    T_b0 = self.temporal_superposition(
                        h_ij.y[:,:,1:], Q_reconstructed)

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
                    A = np.block(
                        [[h_dt,
                          -np.eye(self.nSources, dtype=self.dtype),
                          np.zeros((self.nSources, 1), dtype=self.dtype)],
                         [np.eye(self.nSources, dtype=self.dtype),
                          a_b_j/(2.0*pi*k_s*np.atleast_2d(Hb_individual).T),
                          a_in_j/(2.0*pi*k_s*np.atleast_2d(Hb_individual).T)],
                         [H_b, np.zeros(self.nSources + 1, dtype=self.dtype)]])
                    B = np.hstack(
                        (-T_b0,
                         np.zeros(self.nSources, dtype=self.dtype),
                         H_tot))
                    # Solve the system of equations
                    X = np.linalg.solve(A, B)
                    # Store calculated heat extraction rates
                    Q_b[j,:,p+p0] = X[0:self.nSources]
                    T_b[j,:,p+p0] = X[self.nSources:2*self.nSources]
                    # Inlet fluid temperature
                    T_f_in = X[-1]
                    # The gFunction is equal to the effective borehole wall
                    # temperature
                    # Outlet fluid temperature
                    T_f_out = T_f_in - 2 * pi * k_s * H_tot / (
                        np.sum(np.abs(self.m_flow[j]) * self.cp_f))
                    # Average fluid temperature
                    T_f = 0.5*(T_f_in + T_f_out)
                    # Borefield thermal resistance
                    R_field = network_thermal_resistance(
                        self.network, self.m_flow[j], self.cp_f)
                    # Effective borehole wall temperature
                    T_b_eff = T_f - 2 * pi * k_s * R_field
                    gFunc[j,j,p+p0] = T_b_eff

            for i in range(np.maximum(self.nMassFlow, 1)):
                for j in range(np.maximum(self.nMassFlow, 1)):
                    if not i == j:
                        # Inlet fluid temperature
                        a_in, a_b = self.network.coefficients_network_heat_extraction_rate(
                                self.m_flow[i],
                                self.cp_f,
                                self.nBoreSegments,
                                segment_ratios=self.segment_ratios)
                        T_f_in = (-2 * pi * k_s * H_tot - a_b @ T_b[j,:,p0:]) / a_in
                        # The gFunction is equal to the effective borehole wall
                        # temperature
                        # Outlet fluid temperature
                        T_f_out = T_f_in - 2 * pi * k_s * H_tot / (np.abs(self.m_flow[i]) * self.cp_f)
                        # Borefield thermal resistance
                        R_field = network_thermal_resistance(
                            self.network, self.m_flow[i], self.cp_f)
                        # Effective borehole wall temperature
                        T_b_eff = 0.5 * (T_f_in + T_f_out) - 2 * pi * k_s * R_field
                        gFunc[i,j,p0:] = T_b_eff

            # Linearize g-function for times under threshold
            if p_long > 0:
                gFunc[:,:,:p_long] = gFunc[:,:,p_long-1] * self.time[:p_long] / time_threshold
                Q_b[:,:,:p_long] = 1 + (Q_b[:,:,p_long-1:p_long] - 1) * self.time[:p_long] / time_threshold
                T_b[:,:,:p_long] = T_b[:,:,p_long-1:p_long] * self.time[:p_long] / time_threshold
            if self.nMassFlow == 0:
                gFunc = gFunc[0,0,:]
                Q_b = Q_b[0,:,:]
                T_b = T_b[0,:,:]

        # Store temperature and heat extraction rate profiles
        if self.profiles:
            self.Q_b = Q_b
            self.T_b = T_b
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')
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
        H_b = np.array([b.H for b in self.boreSegments], dtype=self.dtype)
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
        for b, nSegments, segment_ratios in zip(self.boreholes, self.nBoreSegments, self.segment_ratios):
            segments = b.segments(nSegments, segment_ratios=segment_ratios)
            boreSegments.extend(segments)

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
        assert self.network is None or isinstance(self.network, Network), \
            "The network is not a valid 'Network' object."
        if self.boundary_condition == 'MIFT':
            assert not (self.m_flow_network is None and self.m_flow_borehole is None), \
                "The mass flow rate 'm_flow_borehole' or 'm_flow_network' must " \
                "be provided when using the 'MIFT' boundary condition."
            assert not (self.m_flow_network is not None and self.m_flow_borehole is not None), \
                "Only one of 'm_flow_borehole' or 'm_flow_network' can " \
                "be provided when using the 'MIFT' boundary condition."
            assert not self.cp_f is None, \
                "The heat capacity 'cp_f' must " \
                "be provided when using the 'MIFT' boundary condition."
            assert not (type(self.m_flow_borehole) is np.ndarray and not np.size(self.m_flow_borehole, axis=1)==len(self.boreholes)), \
                "The number of mass flow rates in 'm_flow_borehole' must " \
                "correspond to the number of boreholes."
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
            f"Boundary condition '{self.boundary_condition}' is not an " \
            f"acceptable boundary condition. \n" \
            f"Please provide one of the following inputs : " \
            f"{acceptable_boundary_conditions}"
        assert type(self.approximate_FLS) is bool, \
            "The option 'approximate_FLS' should be set to True or False."
        assert type(self.nFLS) is int and 1 <= self.nFLS <= 25, \
            "The option 'nFLS' should be a positive int and lower or equal to " \
            "25."
        assert type(self.disp) is bool, \
            "The option 'disp' should be set to True or False."
        assert type(self.profiles) is bool, \
            "The option 'profiles' should be set to True or False."
        assert type(self.kind) is str, \
            "The option 'kind' should be set to a valid interpolation kind " \
            "in accordance with scipy.interpolate.interp1d options."
        acceptable_dtypes = (np.single, np.double)
        assert np.any([self.dtype is dtype for dtype in acceptable_dtypes]), \
            f"Data type '{self.dtype}' is not an acceptable data type. \n" \
            f"Please provide one of the following inputs : {acceptable_dtypes}"
        # Check segment ratios
        for j, (ratios, nSegments) in enumerate(
                zip(self.segment_ratios, self.nBoreSegments)):
            assert len(ratios) == nSegments, \
                f"The length of the segment ratios vectors must correspond to " \
                f"the number of segments, check borehole {j}."
            error = np.abs(1. - np.sum(ratios))
            assert(error < 1.0e-6), \
                f"Defined segment ratios must add up to 1. " \
                f", check borehole {j}."

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
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered. If a callable is provided, it
        must return an array of size (nSegments,) when provided with nSegments
        (of type int) as an argument, or an array of size (nSegments[i],) when
        provided with an element of nSegments (of type list).
        Default is :func:`utilities.segment_ratios`.
    approximate_FLS : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#Detailed-Cimmin2021]_. This approximation does not require the
        numerical evaluation of any integral.
        Default is False.
    nFLS : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximate_FLS` is set to False.
        Default is 10. Maximum is 25.
    mQuad : int, optional
        Number of Gauss-Legendre sample points for the integral over :math:`u`
        in the inclined FLS solution.
        Default is 11.
    linear_threshold : float, optional
        Threshold time (in seconds) under which the g-function is
        linearized. The g-function value is then interpolated between 0
        and its value at the threshold. If linear_threshold==None, the
        g-function is linearized for times
        `t < r_b**2 / (25 * self.alpha)`.
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
    .. [#Detailed-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

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
        obtain thermal response factors at any intermediate time by
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
        tic = perf_counter()
        # Initialize segment-to-segment response factors
        h_ij = np.zeros((self.nSources, self.nSources, nt+1), dtype=self.dtype)
        nBoreholes = len(self.boreholes)
        segment_lengths = self.segment_lengths()

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole
        # thermal interactions
        # ---------------------------------------------------------------------
        h, i_segment, j_segment = \
            self._thermal_response_factors_borehole_to_self(time, alpha)
        # Broadcast values to h_ij matrix
        h_ij[j_segment, i_segment, 1:] = h
        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for
        # borehole-to-borehole thermal interactions
        # ---------------------------------------------------------------------
        for i, (i0, i1) in enumerate(zip(self._i0Segments, self._i1Segments)):
            # Segments of the receiving borehole
            b2 = self.boreSegments[i0:i1]
            if i+1 < nBoreholes:
                # Segments of the emitting borehole
                b1 = self.boreSegments[i1:]
                h = finite_line_source(
                    time, alpha, b1, b2, approximation=self.approximate_FLS,
                    N=self.nFLS, M=self.mQuad)
                # Broadcast values to h_ij matrix
                h_ij[i0:i1, i1:, 1:] = h
                h_ij[i1:, i0:i1, 1:] = \
                    np.swapaxes(h, 0, 1) * np.divide.outer(
                        segment_lengths[i0:i1],
                        segment_lengths[i1:]).T[:,:,np.newaxis]

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,1]

        # Interp1d object for thermal response factors
        h_ij = interp1d(np.hstack((0., time)), h_ij,
                        kind=kind, copy=True, axis=2)
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')

        return h_ij

    def _thermal_response_factors_borehole_to_self(self, time, alpha):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments between each borehole and itself.

        Attributes
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).

        Returns
        -------
        h : array
            Finite line source solution.
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        """
        # Indices of the thermal response factors into h_ij
        i_segment = np.concatenate(
            [np.repeat(np.arange(i0, i1), nSegments)
             for i0, i1, nSegments in zip(
                     self._i0Segments, self._i1Segments, self.nBoreSegments)
            ])
        j_segment = np.concatenate(
            [np.tile(np.arange(i0, i1), nSegments)
             for i0, i1, nSegments in zip(
                     self._i0Segments, self._i1Segments, self.nBoreSegments)
            ])
        # Unpack parameters
        x = np.array([b.x for b in self.boreSegments])
        y = np.array([b.y for b in self.boreSegments])
        H = np.array([b.H for b in self.boreSegments])
        D = np.array([b.D for b in self.boreSegments])
        r_b = np.array([b.r_b for b in self.boreSegments])
        # Distances between boreholes
        dis = np.maximum(
            np.sqrt((x[i_segment] - x[j_segment])**2 + (y[i_segment] - y[j_segment])**2),
            r_b[i_segment])
        # FLS solution
        if np.all([b.is_vertical() for b in self.boreholes]):
            h = finite_line_source_vectorized(
                time, alpha,
                dis, H[i_segment], D[i_segment], H[j_segment], D[j_segment],
                approximation=self.approximate_FLS, N=self.nFLS)
        else:
            tilt = np.array([b.tilt for b in self.boreSegments])
            orientation = np.array([b.orientation for b in self.boreSegments])
            h = finite_line_source_inclined_vectorized(
                time, alpha,
                r_b[i_segment], x[i_segment], y[i_segment], H[i_segment],
                D[i_segment], tilt[i_segment], orientation[i_segment],
                x[j_segment], y[j_segment], H[j_segment], D[j_segment],
                tilt[j_segment], orientation[j_segment], M=self.mQuad,
                approximation=self.approximate_FLS, N=self.nFLS)
        return h, i_segment, j_segment



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
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered. If a callable is provided, it
        must return an array of size (nSegments,) when provided with nSegments
        (of type int) as an argument, or an array of size (nSegments[i],) when
        provided with an element of nSegments (of type list).
        Default is :func:`utilities.segment_ratios`.
    approximate_FLS : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#Similarities-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral.
        Default is False.
    nFLS : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximate_FLS` is set to False.
        Default is 10. Maximum is 25.
    mQuad : int, optional
        Number of Gauss-Legendre sample points for the integral over :math:`u`
        in the inclined FLS solution.
        Default is 11.
    linear_threshold : float, optional
        Threshold time (in seconds) under which the g-function is
        linearized. The g-function value is then interpolated between 0
        and its value at the threshold. If linear_threshold==None, the
        g-function is linearized for times
        `t < r_b**2 / (25 * self.alpha)`.
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
    .. [#Similarities-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

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
        tic = perf_counter()
        # Initialize segment-to-segment response factors
        h_ij = np.zeros((self.nSources, self.nSources, nt+1), dtype=self.dtype)

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole thermal
        # interactions (vertical boreholes)
        # ---------------------------------------------------------------------
        # Evaluate FLS at all time steps
        h, i_segment, j_segment, k_segment = \
            self._thermal_response_factors_borehole_to_self_vertical(
                time, alpha)
        # Broadcast values to h_ij matrix
        h_ij[j_segment, i_segment, 1:] = h[k_segment, :]
        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole thermal
        # interactions (inclined boreholes)
        # ---------------------------------------------------------------------
        # Evaluate FLS at all time steps
        h, i_segment, j_segment, k_segment = \
            self._thermal_response_factors_borehole_to_self_inclined(
                time, alpha)
        # Broadcast values to h_ij matrix
        h_ij[j_segment, i_segment, 1:] = h[k_segment, :]
        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for borehole-to-borehole
        # thermal interactions (vertical boreholes)
        # ---------------------------------------------------------------------
        for pairs, distances, distance_indices in zip(
                self.borehole_to_borehole_vertical,
                self.borehole_to_borehole_distances_vertical,
                self.borehole_to_borehole_indices_vertical):
            # Index of first borehole pair in group
            i, j = pairs[0]
            # Find segment-to-segment similarities
            H1, D1, H2, D2, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs_vertical(i, j)
            # Locate thermal response factors in the h_ij matrix
            i_segment, j_segment, k_segment, l_segment = \
                self._map_segment_pairs_vertical(
                    i_pair, j_pair, k_pair, pairs, distance_indices)
            # Evaluate FLS at all time steps
            dis = np.reshape(distances, (-1, 1))
            H1 = H1.reshape(1, -1)
            H2 = H2.reshape(1, -1)
            D1 = D1.reshape(1, -1)
            D2 = D2.reshape(1, -1)
            h = finite_line_source_vectorized(
                time, alpha, dis, H1, D1, H2, D2,
                approximation=self.approximate_FLS, N=self.nFLS)
            # Broadcast values to h_ij matrix
            h_ij[j_segment, i_segment, 1:] = h[l_segment, k_segment, :]
            if (self._compare_boreholes(self.boreholes[j], self.boreholes[i]) and
                self.nBoreSegments[i] == self.nBoreSegments[j] and
                self._uniform_segment_ratios[i] and
                self._uniform_segment_ratios[j]):
                h_ij[i_segment, j_segment, 1:] = h[l_segment, k_segment, :]
            else:
                h_ij[i_segment, j_segment, 1:] = (h * H2.T / H1.T)[l_segment, k_segment, :]
        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for borehole-to-borehole
        # thermal interactions (inclined boreholes)
        # ---------------------------------------------------------------------
        # Evaluate FLS at all time steps
        h, hT, i_segment, j_segment, k_segment = \
            self._thermal_response_factors_borehole_to_borehole_inclined(
                time, alpha)
        # Broadcast values to h_ij matrix
        h_ij[j_segment, i_segment, 1:] = h[k_segment, :]
        h_ij[i_segment, j_segment, 1:] = hT[k_segment, :]

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,1]

        # Interp1d object for thermal response factors
        h_ij = interp1d(
            np.hstack((0., time)), h_ij,
            kind=kind, copy=True, assume_sorted=True, axis=2)
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')

        return h_ij

    def _thermal_response_factors_borehole_to_borehole_inclined(
            self, time, alpha):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of inclined segments.

        Attributes
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).

        Returns
        -------
        h : array
            Finite line source solution.
        hT : array
            Reciprocal finite line source solution.
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.
        """
        rb1 = np.array([])
        x1 = np.array([])
        y1 = np.array([])
        H1 = np.array([])
        D1 = np.array([])
        tilt1 = np.array([])
        orientation1 = np.array([])
        x2 = np.array([])
        y2 = np.array([])
        H2 = np.array([])
        D2 = np.array([])
        tilt2 = np.array([])
        orientation2 = np.array([])
        i_segment = np.array([], dtype=np.uint)
        j_segment = np.array([], dtype=np.uint)
        k_segment = np.array([], dtype=np.uint)
        k0 = 0
        for pairs in self.borehole_to_borehole_inclined:
            # Index of first borehole pair in group
            i, j = pairs[0]
            # Find segment-to-segment similarities
            rb1_i, x1_i, y1_i, H1_i, D1_i, tilt1_i, orientation1_i, \
                x2_i, y2_i, H2_i, D2_i, tilt2_i, orientation2_i, \
                i_pair, j_pair, k_pair = \
                    self._map_axial_segment_pairs_inclined(i, j)
            # Locate thermal response factors in the h_ij matrix
            i_segment_i, j_segment_i, k_segment_i = \
                self._map_segment_pairs_inclined(i_pair, j_pair, k_pair, pairs)
            # Append lists
            rb1 = np.append(rb1, rb1_i)
            x1 = np.append(x1, x1_i)
            y1 = np.append(y1, y1_i)
            H1 = np.append(H1, H1_i)
            D1 = np.append(D1, D1_i)
            tilt1 = np.append(tilt1, tilt1_i)
            orientation1 = np.append(orientation1, orientation1_i)
            x2 = np.append(x2, x2_i)
            y2 = np.append(y2, y2_i)
            H2 = np.append(H2, H2_i)
            D2 = np.append(D2, D2_i)
            tilt2 = np.append(tilt2, tilt2_i)
            orientation2 = np.append(orientation2, orientation2_i)
            i_segment = np.append(i_segment, i_segment_i)
            j_segment = np.append(j_segment, j_segment_i)
            k_segment = np.append(k_segment, k_segment_i + k0)
            k0 += len(k_pair)
        # Evaluate FLS at all time steps
        h = finite_line_source_inclined_vectorized(
            time, alpha, rb1, x1, y1, H1, D1, tilt1, orientation1,
            x2, y2, H2, D2, tilt2, orientation2, M=self.mQuad,
            approximation=self.approximate_FLS, N=self.nFLS)
        hT = (h.T * H2 / H1).T
        return h, hT, i_segment, j_segment, k_segment

    def _thermal_response_factors_borehole_to_self_inclined(self, time, alpha):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments between each inclined borehole and itself.

        Attributes
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).

        Returns
        -------
        h : array
            Finite line source solution.
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.
        """
        rb1 = np.array([])
        x1 = np.array([])
        y1 = np.array([])
        H1 = np.array([])
        D1 = np.array([])
        tilt1 = np.array([])
        orientation1 = np.array([])
        x2 = np.array([])
        y2 = np.array([])
        H2 = np.array([])
        D2 = np.array([])
        tilt2 = np.array([])
        orientation2 = np.array([])
        i_segment = np.array([], dtype=np.uint)
        j_segment = np.array([], dtype=np.uint)
        k_segment = np.array([], dtype=np.uint)
        k0 = 0
        for group in self.borehole_to_self_inclined:
            # Index of first borehole in group
            i = group[0]
            # Find segment-to-segment similarities
            rb1_i, x1_i, y1_i, H1_i, D1_i, tilt1_i, orientation1_i, \
                x2_i, y2_i, H2_i, D2_i, tilt2_i, orientation2_i, \
                i_pair, j_pair, k_pair = \
                    self._map_axial_segment_pairs_inclined(i, i)
            # Locate thermal response factors in the h_ij matrix
            i_segment_i, j_segment_i, k_segment_i = \
                self._map_segment_pairs_inclined(
                    i_pair, j_pair, k_pair, [(n, n) for n in group])
            # Append lists
            rb1 = np.append(rb1, rb1_i)
            x1 = np.append(x1, x1_i)
            y1 = np.append(y1, y1_i)
            H1 = np.append(H1, H1_i)
            D1 = np.append(D1, D1_i)
            tilt1 = np.append(tilt1, tilt1_i)
            orientation1 = np.append(orientation1, orientation1_i)
            x2 = np.append(x2, x2_i)
            y2 = np.append(y2, y2_i)
            H2 = np.append(H2, H2_i)
            D2 = np.append(D2, D2_i)
            tilt2 = np.append(tilt2, tilt2_i)
            orientation2 = np.append(orientation2, orientation2_i)
            i_segment = np.append(i_segment, i_segment_i)
            j_segment = np.append(j_segment, j_segment_i)
            k_segment = np.append(k_segment, k_segment_i + k0)
            k0 += len(k_pair)
        # Evaluate FLS at all time steps
        h = finite_line_source_inclined_vectorized(
            time, alpha, rb1, x1, y1, H1, D1, tilt1, orientation1,
            x2, y2, H2, D2, tilt2, orientation2, M=self.mQuad,
            approximation=self.approximate_FLS, N=self.nFLS)
        return h, i_segment, j_segment, k_segment

    def _thermal_response_factors_borehole_to_self_vertical(self, time, alpha):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments between each vertical borehole and itself.

        Attributes
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).

        Returns
        -------
        h : array
            Finite line source solution.
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.
        """
        H1 = np.array([])
        D1 = np.array([])
        H2 = np.array([])
        D2 = np.array([])
        dis = np.array([])
        i_segment = np.array([], dtype=np.uint)
        j_segment = np.array([], dtype=np.uint)
        k_segment = np.array([], dtype=np.uint)
        k0 = 0
        for group in self.borehole_to_self_vertical:
            # Index of first borehole in group
            i = group[0]
            # Find segment-to-segment similarities
            H1_i, D1_i, H2_i, D2_i, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs_vertical(i, i)
            # Locate thermal response factors in the h_ij matrix
            i_segment_i, j_segment_i, k_segment_i, l_segment_i = \
                self._map_segment_pairs_vertical(
                    i_pair, j_pair, k_pair, [(n, n) for n in group], [0])
            # Append lists
            H1 = np.append(H1, H1_i)
            D1 = np.append(D1, D1_i)
            H2 = np.append(H2, H2_i)
            D2 = np.append(D2, D2_i)
            if len(self.borehole_to_self_vertical) > 1:
                dis = np.append(dis, np.full(len(H1_i), self.boreholes[i].r_b))
            else:
                dis = self.boreholes[i].r_b
            i_segment = np.append(i_segment, i_segment_i)
            j_segment = np.append(j_segment, j_segment_i)
            k_segment = np.append(k_segment, k_segment_i + k0)
            k0 += np.max(k_pair) + 1
        # Evaluate FLS at all time steps
        h = finite_line_source_vectorized(
            time, alpha, dis, H1, D1, H2, D2,
            approximation=self.approximate_FLS, N=self.nFLS)
        return h, i_segment, j_segment, k_segment

    def find_similarities(self):
        """
        Find similarities in the FLS solution for groups of boreholes.

        This function identifies pairs of boreholes for which the evaluation
        of the Finite Line Source (FLS) solution is equivalent.

        """
        if self.disp: print('Identifying similarities ...', end='')
        # Initialize chrono
        tic = perf_counter()

        # Find similar pairs of boreholes
        # Boreholes can only be similar if their segments are similar
        self.borehole_to_self_vertical, self.borehole_to_self_inclined, \
            self.borehole_to_borehole_vertical, self.borehole_to_borehole_inclined = \
                self._find_axial_borehole_pairs(self.boreholes)
        # Find distances for each similar pairs of vertical boreholes
        self.borehole_to_borehole_distances_vertical, self.borehole_to_borehole_indices_vertical = \
            self._find_distances(
                self.boreholes, self.borehole_to_borehole_vertical)

        # Stop chrono
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')

        return

    def _compare_boreholes(self, borehole1, borehole2):
        """
        Compare two boreholes and checks if they have the same dimensions :
        H, D, r_b, and tilt.

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
            abs((borehole1.D - borehole2.D)/(borehole1.D + 1e-30)) < self.tol and
            abs(abs(borehole1.tilt) - abs(borehole2.tilt))/(abs(borehole1.tilt) + 1e-30) < self.tol):
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_real_pairs_vertical(self, pair1, pair2):
        """
        Compare two pairs of vertical boreholes or segments and return True if
        the two pairs have the same FLS solution for real sources.

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

    def _compare_image_pairs_vertical(self, pair1, pair2):
        """
        Compare two pairs of vertical boreholes or segments and return True if
        the two pairs have the same FLS solution for mirror sources.

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

    def _compare_realandimage_pairs_vertical(self, pair1, pair2):
        """
        Compare two pairs of vertical boreholes or segments and return True if
        the two pairs have the same FLS solution for both real and mirror
        sources.

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
        if (self._compare_real_pairs_vertical(pair1, pair2)
            and self._compare_image_pairs_vertical(pair1, pair2)):
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_real_pairs_inclined(self, pair1, pair2):
        """
        Compare two pairs of inclined boreholes or segments and return True if
        the two pairs have the same FLS solution for real sources.

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
        dx1 = pair1[0].x - pair1[1].x; dx2 = pair2[0].x - pair2[1].x
        dy1 = pair1[0].y - pair1[1].y; dy2 = pair2[0].y - pair2[1].y
        dis1 = np.sqrt(dx1**2 + dy1**2); dis2 = np.sqrt(dx2**2 + dy2**2)
        theta_12_1 = np.arctan2(dy1, dx1); theta_12_2 = np.arctan2(dy2, dx2)
        deltaD1 = pair1[0].D - pair1[1].D; deltaD2 = pair2[0].D - pair2[1].D
        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of buried depths differences
        cond_deltaD = abs(deltaD1 - deltaD2)/(abs(deltaD1) + 1e-30) < self.tol
        # Equality of distances
        cond_dis = abs(dis1 - dis2)/(abs(dis1) + 1e-30) < self.disTol
        # Equality of tilts
        cond_beta = (
            abs(abs(pair1[0].tilt) - abs(pair2[0].tilt))/(abs(pair1[0].tilt) + 1e-30) < self.tol
            and abs(abs(pair1[1].tilt) - abs(pair2[1].tilt))/(abs(pair1[1].tilt) + 1e-30) < self.tol)
        # Equality of relative orientations
        sin_b1_cos_dt1_1 = np.sin(pair1[0].tilt) * np.cos(theta_12_1 - pair1[0].orientation)
        sin_b2_cos_dt2_1 = np.sin(pair1[1].tilt) * np.cos(theta_12_1 - pair1[1].orientation)
        sin_b1_cos_dt1_2 = np.sin(pair2[0].tilt) * np.cos(theta_12_2 - pair2[0].orientation)
        sin_b2_cos_dt2_2 = np.sin(pair2[1].tilt) * np.cos(theta_12_2 - pair2[1].orientation)
        cond_theta = (
            abs(sin_b1_cos_dt1_1 - sin_b1_cos_dt1_2) / (abs(sin_b1_cos_dt1_1) + 1e-30) > self.tol
            and abs(sin_b2_cos_dt2_1 - sin_b2_cos_dt2_2) / (abs(sin_b2_cos_dt2_1) + 1e-30) > self.tol)
        if cond_H and cond_deltaD and cond_dis and cond_beta and cond_theta:
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_image_pairs_inclined(self, pair1, pair2):
        """
        Compare two pairs of inclined boreholes or segments and return True if
        the two pairs have the same FLS solution for mirror sources.

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
        dx1 = pair1[0].x - pair1[1].x; dx2 = pair2[0].x - pair2[1].x
        dy1 = pair1[0].y - pair1[1].y; dy2 = pair2[0].y - pair2[1].y
        dis1 = np.sqrt(dx1**2 + dy1**2); dis2 = np.sqrt(dx2**2 + dy2**2)
        theta_12_1 = np.arctan2(dy1, dx1); theta_12_2 = np.arctan2(dy2, dx2)
        sumD1 = pair1[0].D + pair1[1].D; sumD2 = pair2[0].D + pair2[1].D
        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of buried depths sums
        cond_sumD = abs(sumD1 - sumD2)/(abs(sumD1) + 1e-30) < self.tol
        # Equality of distances
        cond_dis = abs(dis1 - dis2)/(abs(dis1) + 1e-30) < self.disTol
        # Equality of tilts
        cond_beta = (
            abs(abs(pair1[0].tilt) - abs(pair2[0].tilt))/(abs(pair1[0].tilt) + 1e-30) < self.tol
            and abs(abs(pair1[1].tilt) - abs(pair2[1].tilt))/(abs(pair1[1].tilt) + 1e-30) < self.tol)
        # Equality of relative orientations
        sin_b1_cos_dt1_1 = np.sin(pair1[0].tilt) * np.cos(theta_12_1 - pair1[0].orientation)
        sin_b2_cos_dt2_1 = np.sin(pair1[1].tilt) * np.cos(theta_12_1 - pair1[1].orientation)
        sin_b1_cos_dt1_2 = np.sin(pair2[0].tilt) * np.cos(theta_12_2 - pair2[0].orientation)
        sin_b2_cos_dt2_2 = np.sin(pair2[1].tilt) * np.cos(theta_12_2 - pair2[1].orientation)
        cond_theta = (
            abs(sin_b1_cos_dt1_1 - sin_b1_cos_dt1_2) / (abs(sin_b1_cos_dt1_1) + 1e-30) > self.tol
            and abs(sin_b2_cos_dt2_1 - sin_b2_cos_dt2_2) / (abs(sin_b2_cos_dt2_1) + 1e-30) > self.tol)
        if cond_H and cond_sumD and cond_dis and cond_beta and cond_theta:
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_realandimage_pairs_inclined(self, pair1, pair2):
        """
        Compare two pairs of inclined boreholes or segments and return True if
        the two pairs have the same FLS solution for both real and mirror
        sources.

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

        Notes
        -----
        For inclined boreholes the similarity condition is the same for real
        and image parts of the solution.

        """
        dx1 = pair1[0].x - pair1[1].x; dx2 = pair2[0].x - pair2[1].x
        dy1 = pair1[0].y - pair1[1].y; dy2 = pair2[0].y - pair2[1].y
        dis1 = np.sqrt(dx1**2 + dy1**2); dis2 = np.sqrt(dx2**2 + dy2**2)
        theta_12_1 = np.arctan2(dy1, dx1); theta_12_2 = np.arctan2(dy2, dx2)
        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of buried depths
        cond_D = (
            abs(pair1[0].D - pair2[0].D)/(abs(pair1[0].D) + 1e-30) < self.tol
            and abs(pair1[1].D - pair2[1].D)/(abs(pair1[1].D) + 1e-30) < self.tol)
        # Equality of distances
        cond_dis = abs(dis1 - dis2)/(abs(dis1) + 1e-30) < self.disTol
        # Equality of tilts
        cond_beta = (
            abs(abs(pair1[0].tilt) - abs(pair2[0].tilt))/(abs(pair1[0].tilt) + 1e-30) < self.tol
            and abs(abs(pair1[1].tilt) - abs(pair2[1].tilt))/(abs(pair1[1].tilt) + 1e-30) < self.tol)
        # Equality of relative orientations
        sin_b1_cos_dt1_1 = np.sin(pair1[0].tilt) * np.cos(theta_12_1 - pair1[0].orientation)
        sin_b2_cos_dt2_1 = np.sin(pair1[1].tilt) * np.cos(theta_12_1 - pair1[1].orientation)
        sin_b1_cos_dt1_2 = np.sin(pair2[0].tilt) * np.cos(theta_12_2 - pair2[0].orientation)
        sin_b2_cos_dt2_2 = np.sin(pair2[1].tilt) * np.cos(theta_12_2 - pair2[1].orientation)
        cond_theta = (
            abs(sin_b1_cos_dt1_1 - sin_b1_cos_dt1_2) / (abs(sin_b1_cos_dt1_1) + 1e-30) < self.tol
            and abs(sin_b2_cos_dt2_1 - sin_b2_cos_dt2_2) / (abs(sin_b2_cos_dt2_1) + 1e-30) < self.tol)
        if cond_H and cond_D and cond_dis and cond_beta and cond_theta:
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
        nBoreholes = len(boreholes)
        borehole_to_self_vertical = []
        borehole_to_self_inclined = []
        # Only check for similarities if there is more than one borehole
        if nBoreholes > 1:
            borehole_to_borehole_vertical = []
            borehole_to_borehole_inclined = []
            for i, (borehole_i, nSegments_i, ratios_i) in enumerate(
                    zip(boreholes, self.nBoreSegments, self.segment_ratios)):
                # Compare the borehole to all known unique sets of dimensions
                if borehole_i.is_vertical():
                    borehole_to_self = borehole_to_self_vertical
                    compare_pairs = self._compare_realandimage_pairs_vertical
                else:
                    borehole_to_self = borehole_to_self_inclined
                    compare_pairs = self._compare_realandimage_pairs_inclined
                for k, borehole_set in enumerate(borehole_to_self):
                    m = borehole_set[0]
                    # Add the borehole to the group if a similar borehole is
                    # found
                    if (self._compare_boreholes(borehole_i, boreholes[m]) and
                        (self._equal_segment_ratios or
                         (nSegments_i == self.nBoreSegments[m] and
                          np.allclose(ratios_i,
                                      self.segment_ratios[m],
                                      rtol=self.tol)))):
                        borehole_set.append(i)
                        break
                else:
                    # If no similar boreholes are known, append the groups
                    borehole_to_self.append([i])

                for j, (borehole_j, nSegments_j, ratios_j) in enumerate(
                        zip(boreholes[i+1:],
                            self.nBoreSegments[i+1:],
                            self.segment_ratios[i+1:]),
                        start=i+1):
                    pair0 = (borehole_i, borehole_j) # pair
                    pair1 = (borehole_j, borehole_i) # reciprocal pair
                    # Compare pairs of boreholes to known unique pairs
                    if borehole_i.is_vertical() and borehole_j.is_vertical():
                        borehole_to_borehole = borehole_to_borehole_vertical
                        compare_pairs = self._compare_realandimage_pairs_vertical
                    else:
                        borehole_to_borehole = borehole_to_borehole_inclined
                        compare_pairs = self._compare_realandimage_pairs_inclined
                    for pairs in borehole_to_borehole:
                        m, n = pairs[0]
                        pair_ref = (boreholes[m], boreholes[n])
                        # Add the pair (or the reciprocal pair) to a group
                        # if a similar one is found
                        if (compare_pairs(pair0, pair_ref) and
                            (self._equal_segment_ratios or
                             (nSegments_i == self.nBoreSegments[m] and
                              nSegments_j == self.nBoreSegments[n] and
                              np.allclose(ratios_i,
                                          self.segment_ratios[m],
                                          rtol=self.tol) and
                              np.allclose(ratios_j,
                                          self.segment_ratios[n],
                                          rtol=self.tol)))):
                            pairs.append((i, j))
                            break
                        elif (compare_pairs(pair1, pair_ref) and
                              (self._equal_segment_ratios or
                               (nSegments_j == self.nBoreSegments[m] and
                                nSegments_i == self.nBoreSegments[n] and
                                np.allclose(ratios_j,
                                            self.segment_ratios[m],
                                            rtol=self.tol) and
                                np.allclose(ratios_i,
                                            self.segment_ratios[n],
                                            rtol=self.tol)))):
                            pairs.append((j, i))
                            break
                    # If no similar pairs are known, append the groups
                    else:
                        borehole_to_borehole.append([(i, j)])

        else:
            # Outputs for a single borehole
            if boreholes[0].is_vertical:
                borehole_to_self_vertical = [[0]]
                borehole_to_self_inclined = []
            else:
                borehole_to_self_vertical = []
                borehole_to_self_inclined = [[0]]
            borehole_to_borehole_vertical = []
            borehole_to_borehole_inclined = []
        return borehole_to_self_vertical, borehole_to_self_inclined, \
            borehole_to_borehole_vertical, borehole_to_borehole_inclined

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
        borehole_to_borehole_distances = [[] for i in range(nGroups)]
        borehole_to_borehole_indices = \
            [np.empty(len(group), dtype=np.uint) for group in borehole_to_borehole]
        # Find unique distances for each group
        for i, (pairs, distances, distance_indices) in enumerate(
                zip(borehole_to_borehole,
                    borehole_to_borehole_distances,
                    borehole_to_borehole_indices)):
            nPairs = len(pairs)
            # Array of all borehole-to-borehole distances within the group
            all_distances = np.array(
                [boreholes[pair[0]].distance(boreholes[pair[1]])
                 for pair in pairs])
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
                    distances.append(np.mean(distances_sorted[j0:j1]))
                    # Apply distance index to borehole pairs
                    distance_indices[i_sort[j0:j1]] = nDis
                else:
                    # Average distance between pairs of boreholes
                    distances.append(np.mean(distances_sorted[j0:]))
                    # Apply distance index to borehole pairs
                    distance_indices[i_sort[j0:]] = nDis
                j0 = j1
                nDis += 1
        return borehole_to_borehole_distances, borehole_to_borehole_indices

    def _map_axial_segment_pairs_vertical(
            self, i, j, reaSource=True, imgSource=True):
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
        H1 : array
            Length of the emitting segments.
        D1 : array
            Array of buried depths of the emitting segments.
        H2 : array
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
            compare_pairs = self._compare_realandimage_pairs_vertical
        elif reaSource:
            # Find segment pairs for the real FLS solution
            compare_pairs = self._compare_real_pairs_vertical
        elif imgSource:
            # Find segment pairs for the image FLS solution
            compare_pairs = self._compare_image_pairs_vertical
        # Dive both boreholes into segments
        segments1 = borehole1.segments(
            self.nBoreSegments[i], segment_ratios=self.segment_ratios[i])
        segments2 = borehole2.segments(
            self.nBoreSegments[j], segment_ratios=self.segment_ratios[j])
        # Prepare lists of segment lengths
        H1 = []
        H2 = []
        # Prepare lists of segment buried depths
        D1 = []
        D2 = []
        # All possible pairs (i, j) of indices between segments
        i_pair = np.repeat(np.arange(self.nBoreSegments[i], dtype=np.uint),
                           self.nBoreSegments[j])
        j_pair = np.tile(np.arange(self.nBoreSegments[j], dtype=np.uint),
                         self.nBoreSegments[i])
        # Empty list of indices for unique pairs
        k_pair = np.empty(self.nBoreSegments[i] * self.nBoreSegments[j],
                          dtype=np.uint)
        unique_pairs = []
        nPairs = 0

        p = 0
        for ii, segment_i in enumerate(segments1):
            for jj, segment_j in enumerate(segments2):
                pair = (segment_i, segment_j)
                # Compare the segment pairs to all known unique pairs
                for k, pair_k in enumerate(unique_pairs):
                    m, n = pair_k[0], pair_k[1]
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
                    H1.append(segment_i.H)
                    H2.append(segment_j.H)
                    D1.append(segment_i.D)
                    D2.append(segment_j.D)
                    unique_pairs.append((ii, jj))
                    nPairs += 1
                p += 1
        return np.array(H1), np.array(D1), np.array(H2), np.array(D2), i_pair, j_pair, k_pair

    def _map_axial_segment_pairs_inclined(
            self, i, j, reaSource=True, imgSource=True):
        """
        Find axial similarities between segment pairs along two boreholes to
        simplify the evaluation of the FLS solution.

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
        rb1 : array
            Radii of the emitting heat sources.
        x1 : array
            x-Positions of the emitting heat sources.
        y1 : array
            y-Positions of the emitting heat sources.
        H1 : array
            Lengths of the emitting heat sources.
        D1 : array
            Buried depths of the emitting heat sources.
        tilt1 : array
            Angles (in radians) from vertical of the emitting heat sources.
        orientation1 : array
            Directions (in radians) of the tilt the emitting heat sources.
        x2 : array
            x-Positions of the receiving heat sources.
        y2 : array
            y-Positions of the receiving heat sources.
        H2 : array
            Lengths of the receiving heat sources.
        D2 : array
            Buried depths of the receiving heat sources.
        tilt2 : array
            Angles (in radians) from vertical of the receiving heat sources.
        orientation2 : array
            Directions (in radians) of the tilt the receiving heat sources.
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
            compare_pairs = self._compare_realandimage_pairs_inclined
        elif reaSource:
            # Find segment pairs for the real FLS solution
            compare_pairs = self._compare_real_pairs_inclined
        elif imgSource:
            # Find segment pairs for the image FLS solution
            compare_pairs = self._compare_image_pairs_inclined
        # Dive both boreholes into segments
        segments1 = borehole1.segments(
            self.nBoreSegments[i], segment_ratios=self.segment_ratios[i])
        segments2 = borehole2.segments(
            self.nBoreSegments[j], segment_ratios=self.segment_ratios[j])
        # Prepare lists of FLS-inclined arguments
        rb1 = []
        x1 = []
        y1 = []
        H1 = []
        D1 = []
        tilt1 = []
        orientation1 = []
        x2 = []
        y2 = []
        H2 = []
        D2 = []
        tilt2 = []
        orientation2 = []
        # All possible pairs (i, j) of indices between segments
        i_pair = np.repeat(np.arange(self.nBoreSegments[i], dtype=np.uint),
                           self.nBoreSegments[j])
        j_pair = np.tile(np.arange(self.nBoreSegments[j], dtype=np.uint),
                         self.nBoreSegments[i])
        # Empty list of indices for unique pairs
        k_pair = np.empty(self.nBoreSegments[i] * self.nBoreSegments[j],
                          dtype=np.uint)
        unique_pairs = []
        nPairs = 0

        p = 0
        for ii, segment_i in enumerate(segments1):
            for jj, segment_j in enumerate(segments2):
                pair = (segment_i, segment_j)
                # Compare the segment pairs to all known unique pairs
                for k, pair_k in enumerate(unique_pairs):
                    m, n = pair_k[0], pair_k[1]
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
                    rb1.append(segment_i.r_b)
                    x1.append(segment_i.x)
                    y1.append(segment_i.y)
                    H1.append(segment_i.H)
                    D1.append(segment_i.D)
                    tilt1.append(segment_i.tilt)
                    orientation1.append(segment_i.orientation)
                    x2.append(segment_j.x)
                    y2.append(segment_j.y)
                    H2.append(segment_j.H)
                    D2.append(segment_j.D)
                    tilt2.append(segment_j.tilt)
                    orientation2.append(segment_j.orientation)
                    unique_pairs.append((ii, jj))
                    nPairs += 1
                p += 1
        return np.array(rb1), np.array(x1), np.array(y1), np.array(H1), \
            np.array(D1), np.array(tilt1), np.array(orientation1), \
            np.array(x2), np.array(y2), np.array(H2), np.array(D2), \
            np.array(tilt2), np.array(orientation2), i_pair, j_pair, k_pair

    def _map_segment_pairs_vertical(
            self, i_pair, j_pair, k_pair, borehole_to_borehole,
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
        k_segment = np.tile(k_pair, len(borehole_to_borehole))
        l_segment = np.concatenate(
            [np.repeat(i, len(k_pair)) for i in borehole_to_borehole_indices])
        return i_segment, j_segment, k_segment, l_segment

    def _map_segment_pairs_inclined(
            self, i_pair, j_pair, k_pair, borehole_to_borehole):
        """
        Return the maping of the unique segment-to-segment thermal response
        factors (h) to the complete h_ij array of the borefield, such that:

            h_ij[j_segment, i_segment, :nt] = h[:nt, k_segment].T,

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

        Returns
        -------
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.

        """
        i_segment = np.concatenate(
            [i_pair + self._i0Segments[i] for (i, j) in borehole_to_borehole])
        j_segment = np.concatenate(
            [j_pair + self._i0Segments[j] for (i, j) in borehole_to_borehole])
        k_segment = np.tile(k_pair, len(borehole_to_borehole))
        return i_segment, j_segment, k_segment

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


class _Equivalent(_BaseSolver):
    """
    Equivalent solver for the evaluation of the g-function.

    This solver uses hierarchical agglomerative clustering to identify groups
    of boreholes that are expected to have similar borehole wall temperatures
    and heat extraction rates, as proposed by Prieto and Cimmino (2021)
    [#Equivalent-PriCim2021]_. Each group of boreholes is represented by a
    single equivalent borehole. The FLS solution is adapted to evaluate
    thermal interactions between groups of boreholes. This greatly reduces
    the number of evaluations of the FLS solution and the size of the system of
    equations to evaluate the g-function.

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
                [#Equivalent-CimBer2014]_.
            - 'UBWT' :
                **Uniform borehole wall temperature**. This is corresponds to
                boundary condition *BC-III* as defined by Cimmino and Bernier
                (2014) [#Equivalent-CimBer2014]_.
            - 'MIFT' :
                **Mixed inlet fluid temperatures**. This boundary condition was
                introduced by Cimmino (2015) [#Equivalent-Cimmin2015]_ for
                parallel-connected boreholes and extended to mixed
                configurations by Cimmino (2019) [#Equivalent-Cimmin2019]_.

    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered. If a callable is provided, it
        must return an array of size (nSegments,) when provided with nSegments
        (of type int) as an argument, or an array of size (nSegments[i],) when
        provided with an element of nSegments (of type list).
        Default is :func:`utilities.segment_ratios`.
    approximate_FLS : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#Equivalent-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral. When using the 'equivalent'
        solver, the approximation is only applied to the thermal response at
        the borehole radius. Thermal interaction between boreholes is evaluated
        using the FLS solution.
        Default is False.
    nFLS : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximate_FLS` is set to False.
        Default is 10. Maximum is 25.
    mQuad : int, optional
        Number of Gauss-Legendre sample points for the integral over :math:`u`
        in the inclined FLS solution.
        Default is 11.
    linear_threshold : float, optional
        Threshold time (in seconds) under which the g-function is
        linearized. The g-function value is then interpolated between 0
        and its value at the threshold. If linear_threshold==None, the
        g-function is linearized for times
        `t < r_b**2 / (25 * self.alpha)`.
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
    kClusters : int, optional
        Increment on the minimum number of equivalent boreholes determined by
        cutting the dendrogram of the bore field given by the hierarchical
        agglomerative clustering method. Increasing the value of this parameter
        increases the accuracy of the method.
        Default is 1.

    References
    ----------
    .. [#Equivalent-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#Equivalent-Cimmin2015] Cimmino, M. (2015). The effects of borehole
       thermal resistances and fluid flow rate on the g-functions of geothermal
       bore fields. International Journal of Heat and Mass Transfer, 91,
       1119-1127.
    .. [#Equivalent-Cimmin2018] Cimmino, M. (2018). Fast calculation of the
       g-functions of geothermal borehole fields using similarities in the
       evaluation of the finite line source solution. Journal of Building
       Performance Simulation, 11 (6), 655-668.
    .. [#Equivalent-PriCim2021] Prieto, C., & Cimmino, M.
       (2021). Thermal interactions in large irregular fields of geothermal
       boreholes: the method of equivalent borehole. Journal of Building
       Performance Simulation, 14 (4), 446-460.
    .. [#Equivalent-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

    """
    def initialize(self, disTol=0.01, tol=1.0e-6, kClusters=1, **kwargs):
        """
        Initialize paramteters. Identify groups for equivalent boreholes.

        Returns
        -------
        nSources : int
            Number of finite line heat sources in the borefield used to
            initialize the matrix of segment-to-segment thermal response
            factors (of size: nSources x nSources).

        """
        self.disTol = disTol
        self.tol = tol
        self.kClusters = kClusters
        # Check the validity of inputs
        self._check_solver_specific_inputs()
        # Initialize groups for equivalent boreholes
        nSources = self.find_groups()
        self.nBoreSegments = [self.nBoreSegments[0]] * self.nEqBoreholes
        self.segment_ratios = [self.segment_ratios[0]] * self.nEqBoreholes
        self.boreSegments = self.borehole_segments()
        self._i0Segments = [sum(self.nBoreSegments[0:i])
                            for i in range(self.nEqBoreholes)]
        self._i1Segments = [sum(self.nBoreSegments[0:(i + 1)])
                            for i in range(self.nEqBoreholes)]
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
        obtain thermal response factors at any intermediate time by
        h_ij(t)[:nSources,:nSources].

        Parameters
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
            print('Calculating segment to segment response factors ...',
                  end='')
        # Number of time values
        nt = len(np.atleast_1d(time))
        # Initialize chrono
        tic = perf_counter()
        # Initialize segment-to-segment response factors
        h_ij = np.zeros((self.nSources, self.nSources, nt+1), dtype=self.dtype)
        segment_lengths = self.segment_lengths()

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for borehole-to-borehole
        # thermal interactions
        # ---------------------------------------------------------------------
        # Groups correspond to unique pairs of borehole dimensions
        for pairs in self.borehole_to_borehole:
            i, j = pairs[0]
            # Prepare inputs to the FLS function
            dis, wDis = self._find_unique_distances(self.dis, pairs)
            H1, D1, H2, D2, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs(i, j)
            H1 = H1.reshape(1, -1)
            H2 = H2.reshape(1, -1)
            D1 = D1.reshape(1, -1)
            D2 = D2.reshape(1, -1)
            N2 = np.array(
                [[self.boreholes[j].nBoreholes for (i, j) in pairs]]).T
            # Evaluate FLS at all time steps
            h = finite_line_source_equivalent_boreholes_vectorized(
                time, alpha, dis, wDis, H1, D1, H2, D2, N2)
            # Broadcast values to h_ij matrix
            for k, (i, j) in enumerate(pairs):
                i_segment = self._i0Segments[i] + i_pair
                j_segment = self._i0Segments[j] + j_pair
                h_ij[j_segment, i_segment, 1:] = h[k, k_pair, :]
                if not i == j:
                    h_ij[i_segment, j_segment, 1:] = (h[k, k_pair, :].T \
                        * segment_lengths[j_segment]/segment_lengths[i_segment]).T

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole thermal
        # interactions
        # ---------------------------------------------------------------------
        # Groups correspond to unique borehole dimensions
        for group in self.borehole_to_self:
            # Index of first borehole in group
            i = group[0]
            # Find segment-to-segment similarities
            H1, D1, H2, D2, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs(i, i)
            # Evaluate FLS at all time steps
            dis = self.boreholes[i].r_b
            H1 = H1.reshape(1, -1)
            H2 = H2.reshape(1, -1)
            D1 = D1.reshape(1, -1)
            D2 = D2.reshape(1, -1)
            h = finite_line_source_vectorized(
                time, alpha, dis, H1, D1, H2, D2,
                approximation=self.approximate_FLS, N=self.nFLS)
            # Broadcast values to h_ij matrix
            for i in group:
                i_segment = self._i0Segments[i] + i_pair
                j_segment = self._i0Segments[i] + j_pair
                h_ij[j_segment, i_segment, 1:] = \
                    h_ij[j_segment, i_segment, 1:] + h[0, k_pair, :]

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,1]

        # Interp1d object for thermal response factors
        h_ij = interp1d(np.hstack((0., time)), h_ij,
                        kind=kind, copy=True, axis=2)
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')

        return h_ij

    def find_groups(self, tol=1e-6):
        """
        Identify groups of boreholes that can be represented by a single
        equivalent borehole for the calculation of the g-function.

        Hierarchical agglomerative clustering is applied to the superposed
        steady-state finite line source solution (i.e. the steady-state
        dimensionless borehole wall temperature due to a uniform heat
        extraction equal for all boreholes). The number of clusters is
        evaluated by cutting the dendrogram at the half-height of the longest
        branch and incrementing the number of intercepted branches by the value
        of the kClusters parameter.

        Parameters
        ----------
        tol : float
            Tolerance on the temperature to identify the maxiumum number of
            equivalent boreholes.
            Default is 1e-6.

        Returns
        -------
        nSources : int
            Number of heat sources in the bore field.

        """
        if self.disp: print('Identifying equivalent boreholes ...', end='')
        # Initialize chrono
        tic = perf_counter()

        # Temperature change of individual boreholes
        self.nBoreholes = len(self.boreholes)
        # Equivalent field formed by all boreholes
        eqField = _EquivalentBorehole(self.boreholes)
        if self.nBoreholes > 1:
            # Spatial superposition of the steady-state FLS solution
            data = np.sum(finite_line_source(np.inf, 1., self.boreholes, self.boreholes), axis=1).reshape(-1,1)
            # Split boreholes into groups of same dimensions
            unique_boreholes = self._find_unique_boreholes(self.boreholes)
            # Initialize empty list of clusters
            self.clusters = []
            self.nEqBoreholes = 0
            for group in unique_boreholes:
                if len(group) > 1:
                    # Maximum temperature
                    maxTemp = np.max(data[group])
                    # Hierarchical agglomerative clustering based on temperatures
                    clusterization = linkage(data[group], method='complete')
                    dcoord = np.array(
                        dendrogram(clusterization, no_plot=True)['dcoord'])
                    # Maximum number of clusters
                    # Height to cut each tree to obtain the minimum number of clusters
                    disLeft = dcoord[:,1] - dcoord[:,0]
                    disRight = dcoord[:,2] - dcoord[:,3]
                    if np.max(disLeft) >= np.max(disRight):
                        i = disLeft.argmax()
                        height = 0.5*(dcoord[i,1] + dcoord[i,0])
                    else:
                        i = disRight.argmax()
                        height = 0.5*(dcoord[i,2] + dcoord[i,3])
                    # Find the number of clusters and increment by kClusters
                    # Maximum number of clusters
                    nClustersMax = min(np.sum(dcoord[:,1] > tol*maxTemp) + 1,
                                       len(group))
                    # Optimal number of cluster
                    nClusters = np.max(
                        cut_tree(clusterization, height=height)) + 1
                    nClusters = min(nClusters + self.kClusters, nClustersMax)
                    # Cut the tree to find the borehole groups
                    clusters = cut_tree(
                        clusterization, n_clusters=nClusters)
                    self.clusters = self.clusters + \
                        [label + self.nEqBoreholes for label in clusters]
                else:
                    nClusters = 1
                    self.clusters.append(self.nEqBoreholes)
                self.nEqBoreholes += nClusters
        else:
            self.nEqBoreholes = self.nBoreholes
            self.clusters = range(self.nBoreholes)
        # Overwrite boreholes with equivalent boreholes
        self.boreholes = [_EquivalentBorehole(
            [borehole
             for borehole, cluster in zip(self.boreholes, self.clusters)
             if cluster==i])
            for i in range(self.nEqBoreholes)]
        self.wBoreholes = np.array([b.nBoreholes for b in self.boreholes])
        # Find similar pairs of boreholes
        self.borehole_to_self, self.borehole_to_borehole = \
            self._find_axial_borehole_pairs(self.boreholes)
        # Store unique distances in the bore field
        self.dis = eqField.unique_distance(eqField, self.disTol)[0][1:]

        if self.boundary_condition == 'MIFT':
            pipes = [self.network.p[self.clusters.index(i)]
                     for i in range(self.nEqBoreholes)]
            self.network = _EquivalentNetwork(
                self.boreholes,
                pipes,
                nSegments=self.nBoreSegments[0],
                segment_ratios=self.segment_ratios[0])

        # Stop chrono
        toc = perf_counter()
        if self.disp:
            print(f' {toc - tic:.3f} sec')
            print(f'Calculations will be done using {self.nEqBoreholes} '
                  f'equivalent boreholes')

        return self.nBoreSegments[0]*self.nEqBoreholes

    def segment_lengths(self):
        """
        Return the length of all segments in the bore field.

        The segments lengths are used for the energy balance in the calculation
        of the g-function. For equivalent boreholes, the length of segments
        is multiplied by the number of boreholes in the group.

        Returns
        -------
        H : array
            Array of segment lengths (in m).

        """
        # Borehole lengths
        H = np.array([seg.H*seg.nBoreholes
                      for (borehole, nSegments, ratios) in zip(
                              self.boreholes,
                              self.nBoreSegments,
                              self.segment_ratios)
                      for seg in borehole.segments(
                              nSegments, segment_ratios=ratios)],
                     dtype=self.dtype)
        return H

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
            for i, borehole_i in enumerate(boreholes):
                # Compare the borehole to all known unique sets of dimensions
                for k, borehole_set in enumerate(borehole_to_self):
                    m = borehole_set[0]
                    # Add the borehole to the group if a similar borehole is
                    # found
                    if self._compare_boreholes(borehole_i, boreholes[m]):
                        borehole_set.append(i)
                        break
                else:
                    # If no similar boreholes are known, append the groups
                    borehole_to_self.append([i])
                # Note : The range is different from similarities since
                # an equivalent borehole to itself includes borehole-to-
                # borehole thermal interactions
                for j, borehole_j in enumerate(boreholes[i:], start=i):
                    pair0 = (borehole_i, borehole_j) # pair
                    pair1 = (borehole_j, borehole_i) # reciprocal pair
                    # Compare pairs of boreholes to known unique pairs
                    for pairs in borehole_to_borehole:
                        m, n = pairs[0]
                        pair_ref = (boreholes[m], boreholes[n])
                        # Add the pair (or the reciprocal pair) to a group
                        # if a similar one is found
                        if compare_pairs(pair0, pair_ref):
                            pairs.append((i, j))
                            break
                        elif compare_pairs(pair1, pair_ref):
                            pairs.append((j, i))
                            break
                    # If no similar pairs are known, append the groups
                    else:
                        borehole_to_borehole.append([(i, j)])
        else:
            # Outputs for a single borehole
            borehole_to_self = [[0]]
            borehole_to_borehole = [[(0, 0)]]
        return borehole_to_self, borehole_to_borehole

    def _find_unique_boreholes(self, boreholes):
        """
        Find unique sets of dimensions (h, D, r_b) in the bore field.

        Parameters
        ----------
        boreholes : list of Borehole objects
            Boreholes in the bore field.

        Returns
        -------
        unique_boreholes : list
            List of list of borehole indices that correspond to unique
            borehole dimensions (H, D, r_b).

        """
        unique_boreholes = []
        for i, borehole_1 in enumerate(boreholes):
            for group in unique_boreholes:
                borehole_2 = boreholes[group[0]]
                # Add the borehole to a group if similar dimensions are found
                if self._compare_boreholes(borehole_1, borehole_2):
                    group.append(i)
                    break
            else:
                # If no similar boreholes are known, append the groups
                unique_boreholes.append([i])

        return unique_boreholes

    def _find_unique_distances(self, dis, indices):
        """
        Find the number of occurences of each unique distances between pairs
        of boreholes.

        Parameters
        ----------
        dis : array
            Array of unique distances (in meters) in the bore field.
        indices : list
            List of tuples of borehole indices.

        Returns
        -------
        dis : array
            Array of unique distances (in meters) in the bore field.
        wDis : array
            Array of number of occurences of each unique distance for each
            pair of equivalent boreholes in indices.

        """
        wDis = np.zeros((len(dis), len(indices)), dtype=np.uint)
        for k, pair in enumerate(indices):
            i, j = pair
            b1, b2 = self.boreholes[i], self.boreholes[j]
            # Generate a flattened array of distances between boreholes i and j
            if not i == j:
                dis_ij = b1.distance(b2).flatten()
            else:
                # Remove the borehole radius from the distances
                dis_ij = b1.distance(b2)[
                    ~np.eye(b1.nBoreholes, dtype=bool)].flatten()
            wDis_ij = np.zeros(len(dis), dtype=np.uint)
            # Get insert positions for the distances
            iDis = np.searchsorted(dis, dis_ij, side='left')
            # Find indexes where previous index is closer
            prev_iDis_is_less = ((iDis == len(dis))|(np.fabs(dis_ij - dis[np.maximum(iDis-1, 0)]) < np.fabs(dis_ij - dis[np.minimum(iDis, len(dis)-1)])))
            iDis[prev_iDis_is_less] -= 1
            np.add.at(wDis_ij, iDis, 1)
            wDis[:,k] = wDis_ij

        return dis.reshape((1, -1)), wDis

    def _map_axial_segment_pairs(self, iBor, jBor,
                                 reaSource=True, imgSource=True):
        """
        Find axial (i.e. disregarding the radial distance) similarities between
        segment pairs along two boreholes to simplify the evaluation of the
        FLS solution.

        The returned H1, D1, H2, and D2 can be used to evaluate the segment-to-
        segment response factors using scipy.integrate.quad_vec.

        Parameters
        ----------
        iBor : int
            Index of the first borehole.
        jBor : int
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
        borehole1 = self.boreholes[iBor]
        borehole2 = self.boreholes[jBor]
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
        segments1 = borehole1.segments(
            self.nBoreSegments[iBor], segment_ratios=self.segment_ratios[iBor])
        segments2 = borehole2.segments(
            self.nBoreSegments[jBor], segment_ratios=self.segment_ratios[jBor])
        # Prepare lists of segment lengths
        H1 = []
        H2 = []
        # Prepare lists of segment buried depths
        D1 = []
        D2 = []
        # All possible pairs (i, j) of indices between segments
        i_pair = np.repeat(np.arange(self.nBoreSegments[iBor], dtype=np.uint),
                           self.nBoreSegments[jBor])
        j_pair = np.tile(np.arange(self.nBoreSegments[jBor], dtype=np.uint),
                         self.nBoreSegments[iBor])
        # Empty list of indices for unique pairs
        k_pair = np.empty(self.nBoreSegments[iBor] * self.nBoreSegments[jBor],
                          dtype=np.uint)
        unique_pairs = []
        nPairs = 0

        p = 0
        for i, segment_i in enumerate(segments1):
            for j, segment_j in enumerate(segments2):
                pair = (segment_i, segment_j)
                # Compare the segment pairs to all known unique pairs
                for k, pair_k in enumerate(unique_pairs):
                    m, n = pair_k[0], pair_k[1]
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
                    H1.append(segment_i.H)
                    H2.append(segment_j.H)
                    D1.append(segment_i.D)
                    D2.append(segment_j.D)
                    unique_pairs.append((i, j))
                    nPairs += 1
                p += 1
        return np.array(H1), np.array(D1), np.array(H2), np.array(D2), i_pair, j_pair, k_pair

    def _check_solver_specific_inputs(self):
        """
        This method ensures that solver specific inputs to the Solver object
        are what is expected.

        """
        assert type(self.disTol) is float and self.disTol > 0., \
            "The distance tolerance 'disTol' should be a positive float."
        assert type(self.tol) is float and self.tol > 0., \
            "The relative tolerance 'tol' should be a positive float."
        assert type(self.kClusters) is int and self.kClusters >= 0, \
            "The precision increment 'kClusters' should be a positive int."
        assert np.all(np.array(self.nBoreSegments, dtype=np.uint) == self.nBoreSegments[0]), \
            "Solver 'equivalent' can only handle equal numbers of segments."
        assert np.all([np.allclose(segment_ratios, self.segment_ratios[0]) for segment_ratios in self.segment_ratios]), \
            "Solver 'equivalent' can only handle identical segment_ratios for all boreholes."
        assert not np.any([b.is_tilted() for b in self.boreholes]), \
            "Solver 'equivalent' can only handle vertical boreholes."
        if self.boundary_condition == 'MIFT':
            assert np.all(np.array(self.network.c, dtype=int) == -1), \
                "Solver 'equivalent' is only valid for parallel-connected " \
                "boreholes."
            assert (self.m_flow_borehole is None
                    or (self.m_flow_borehole.ndim==1 and np.allclose(self.m_flow_borehole, self.m_flow_borehole[0]))
                    or (self.m_flow_borehole.ndim==2 and np.all([np.allclose(self.m_flow_borehole[:, i], self.m_flow_borehole[0, i]) for i in range(self.nBoreholes)]))), \
                "Mass flow rates into the network must be equal for all " \
                "boreholes."
            # Use the total network mass flow rate.
            if (type(self.network.m_flow_network) is np.ndarray and \
                len(self.network.m_flow_network)==len(self.network.b)):
                self.network.m_flow_network = \
                    self.network.m_flow_network[0]*len(self.network.b)
            # Verify that all boreholes have the same piping configuration
            # This is best done by comparing the matrix of thermal resistances.
            assert np.all(
                [np.allclose(self.network.p[0]._Rd, pipe._Rd)
                 for pipe in self.network.p]), \
                "All boreholes must have the same piping configuration."
        return
