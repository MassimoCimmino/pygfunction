# -*- coding: utf-8 -*-
import warnings
from time import perf_counter
from typing import Union, List

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d as interp1d

from .borefield import Borefield
from .boreholes import Borehole, find_duplicates
from .media import Fluid
from .networks import Network
from .solvers import (
    Detailed,
    Equivalent,
    Similarities
)
from .utilities import (
    segment_ratios,
    _initialize_figure,
    _format_axes
)


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
                **Uniform heat transfer rate**. This corresponds to boundary
                condition *BC-I* as defined by Cimmino and Bernier (2014)
                [#gFunction-CimBer2014]_.
            - 'UBWT' :
                **Uniform borehole wall temperature**. This corresponds to
                boundary condition *BC-III* as defined by Cimmino and Bernier
                (2014) [#gFunction-CimBer2014]_.
            - 'MIFT' :
                **Mixed inlet fluid temperatures**. This boundary condition was
                introduced by Cimmino (2015) [#gFunction-Cimmin2015]_ for
                parallel-connected boreholes and extended to mixed
                configurations by Cimmino (2019) [#gFunction-Cimmin2019]_.

        If not given, chosen to be 'UBWT' if a list of boreholes is provided
        or 'MIFT' if a Network object is provided.
    m_flow_borehole : (nInlets,) array or (nMassFlow, nInlets,) array, optional
        Fluid mass flow rate into each circuit of the network. If a
        (nMassFlow, nInlets,) array is supplied, the
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
    - If the 'MIFT' boundary condition is used, only one of the
      'm_flow_borehole' or 'm_flow_network' can be supplied.

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
            self.solver = Similarities(
                self.boreholes, self.network, self.time,
                self.boundary_condition, self.m_flow_borehole,
                self.m_flow_network, self.cp_f, **self.options)
        elif self.method.lower()=='detailed':
            self.solver = Detailed(
                self.boreholes, self.network, self.time,
                self.boundary_condition, self.m_flow_borehole,
                self.m_flow_network, self.cp_f, **self.options)
        elif self.method.lower()=='equivalent':
            self.solver = Equivalent(
                self.boreholes, self.network, self.time,
                self.boundary_condition, self.m_flow_borehole,
                self.m_flow_network, self.cp_f, **self.options)
        else:
            raise ValueError(f"'{method}' is not a valid method.")

        # If a time vector is provided, evaluate the g-function
        if self.time is not None:
            self.gFunc = self.evaluate_g_function(self.time)

    @classmethod
    def from_static_params(cls,
                           H: npt.ArrayLike,
                           D: npt.ArrayLike,
                           r_b: npt.ArrayLike,
                           x: npt.ArrayLike,
                           y: npt.ArrayLike,
                           alpha: float,
                           time: npt.ArrayLike = None,
                           method: str = 'equivalent',
                           m_flow_network: float = None,
                           options={},
                           tilt: npt.ArrayLike = 0.,
                           orientation: npt.ArrayLike = 0.,
                           boundary_condition: str = 'MIFT',
                           pipe_type_str: str = None,
                           pos: List[tuple] = None,
                           r_in: Union[float, tuple, npt.ArrayLike] = None,
                           r_out: Union[float, tuple, npt.ArrayLike] = None,
                           k_s: float = None,
                           k_g: float = None,
                           k_p: Union[float, tuple, npt.ArrayLike] = None,
                           fluid_str: str = None,
                           fluid_concentration_pct: float = None,
                           fluid_temperature: float = 20,
                           epsilon: float = None,
                           reversible_flow: bool = True,
                           bore_connectivity: list = None,
                           J: int = 2,
                           ):

        """
        Constructs the 'gFunction' class from static parameters.

        Parameters
        ----------
        H : float or (nBoreholes,) array
            Borehole lengths (in meters).
        D : float or (nBoreholes,) array
            Borehole buried depths (in meters).
        r_b : float or (nBoreholes,) array
            Borehole radii (in meters).
        x : float or (nBoreholes,) array
            Position (in meters) of the head of the boreholes along the x-axis.
        y : float or (nBoreholes,) array
            Position (in meters) of the head of the boreholes along the y-axis.
        alpha : float
            Soil thermal diffusivity (in m2/s).
        time : float or array, optional
            Values of time (in seconds) for which the g-function is evaluated. The
            g-function is only evaluated at initialization if a value is provided.
            Default is None.
        method : str, optional
            Method for the evaluation of the g-function. Should be one of 'similarities', 'detailed', or 'equivalent'.
            Default is 'equivalent'. See 'gFunction' __init__ for more details.
        m_flow_network : float, optional
            Fluid mass flow rate into the network of boreholes (in kg/s).
            Default is None.
        options : dict, optional
            A dictionary of solver options. See 'gFunction' __init__ for more details.
        tilt : float or (nBoreholes,) array, optional
            Angle (in radians) from vertical of the axis of the boreholes.
            Default is 0.
        orientation : float or (nBoreholes,) array, optional
            Direction (in radians) of the tilt of the boreholes. Defaults to zero
            if the borehole is vertical.
            Default is 0.
        boundary_condition : str, optional
            Boundary condition for the evaluation of the g-function. Should be one of 'UHTR', 'UBWT', or 'MIFT'.
            Default is 'MIFT'.
        pipe_type_str : str, optional
            Pipe type used for 'MIFT' boundary condition. Should be one of 'COAXIAL_ANNULAR_IN', 'COAXIAL_ANNULAR_OUT',
            'DOUBLE_UTUBE_PARALLEL', 'DOUBLE_UTUBE_SERIES', or 'SINGLE_UTUBE'.
        pos : list of tuples, optional
            Position (x, y) (in meters) of the pipes inside the borehole.
        r_in : float, optional
            Inner radius (in meters) of the U-Tube pipes.
        r_out : float, optional
            Outer radius (in meters) of the U-Tube pipes.
        k_s : float, optional
            Soil thermal conductivity (in W/m-K).
        k_g : float, optional
            Grout thermal conductivity (in W/m-K).
        k_p : float, optional
            Pipe thermal conductivity (in W/m-K).
        fluid_str: str, optional
            The mixer for this application should be one of:

                - 'Water' - Complete water solution
                - 'MEG' - Ethylene glycol mixed with water
                - 'MPG' - Propylene glycol mixed with water
                - 'MEA' - Ethanol mixed with water
                - 'MMA' - Methanol mixed with water

        fluid_concentration_pct: float, optional
            Mass fraction of the mixing fluid added to water (in %).
            Lower bound = 0. Upper bound is dependent on the mixture.
        fluid_temperature: float, optional
            Temperature used for evaluating fluid properties (in degC).
            Default is 20.
        epsilon : float, optional
            Pipe roughness (in meters).
        reversible_flow : bool, optional
            True to treat a negative mass flow rate as the reversal of flow
            direction within the borehole. If False, the direction of flow is not
            reversed when the mass flow rate is negative, and the absolute value is
            used for calculations.
            Default True.
        bore_connectivity : list, optional
            Index of fluid inlet into each borehole. -1 corresponds to a borehole
            connected to the bore field inlet. If this parameter is not provided,
            parallel connections between boreholes is used.
            Default is None.
        J : int, optional
            Number of multipoles per pipe to evaluate the thermal resistances.
            J=1 or J=2 usually gives sufficient accuracy. J=0 corresponds to the
            line source approximation.
            Default is 2.

        Returns
        -------
        gFunction : 'gFunction' object
            The g-function.

        Notes
        -----
        - When using the 'MIFT' boundary condition, the parameters `pipe_type_str`,
          `fluid_str`, `fluid_concentration_pct`, `fluid_temperature`, and
          `epsilon` are required.

        """

        if boundary_condition.upper() not in ['UBWT', 'UHTR', 'MIFT']:
            raise ValueError(f"'{boundary_condition}' is not a valid boundary condition.")

        # construct all required pieces
        borefield = Borefield(H, D, r_b, x, y, tilt, orientation)

        if boundary_condition.upper() == 'MIFT':
            boreholes = borefield.to_boreholes()
            cp_f = Fluid(fluid_str, fluid_concentration_pct).cp
            boreholes_or_network= Network.from_static_params(
                boreholes,
                pipe_type_str,
                pos,
                r_in,
                r_out,
                k_s,
                k_g,
                k_p,
                m_flow_network,
                epsilon,
                fluid_str,
                fluid_concentration_pct,
                fluid_temperature,
                reversible_flow,
                bore_connectivity,
                J,
            )
        else:
            boreholes_or_network = borefield
            cp_f = None

        return cls(boreholes_or_network, alpha, time=time, method=method, boundary_condition=boundary_condition,
                   m_flow_network=m_flow_network, cp_f=cp_f, options=options)

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
        from ._mpl import plt

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
            if which is None:
                which = [
                    (i, j) for j in range(self.solver.nMassFlow)
                    for i in range(self.solver.nMassFlow)]
            for (i, j) in which:
                ax.plot(
                    lntts,
                    self.gFunc[i,j,:],
                    label=f'$g_{{{i}{j}}}$')
            plt.legend()

        # Adjust figure to window
        plt.tight_layout()
        return fig

    def visualize_heat_extraction_rates(
            self, iBoreholes=None, showTilt=True, which=None):
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
        which : list of int, optional
            Indices i of the diagonal variable mass flow rate g-functions for
            which to plot heat extraction rates.
            If None, all diagonal g-functions are plotted.
            Default is None.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        from ._mpl import plt

        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.solver.boreholes))
        # Import heat extraction rates
        Q_t = self._heat_extraction_rates(iBoreholes)
        # Borefield characteristic time
        ts = np.mean([b.H for b in self.solver.boreholes])**2/(9.*self.alpha)
        # Dimensionless time (log)
        lntts = np.log(self.time/ts)

        if self.solver.nMassFlow == 0:
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
        else:
            m_flow = self.solver.m_flow
            if which is None:
                which = [n for n in range(self.solver.nMassFlow)]
            for n in which:
                # Configure figure and axes
                fig = _initialize_figure()
                fig.suptitle(
                    f'Heat extraction rates for m_flow={m_flow[n]} kg/s')
                ax1 = fig.add_subplot(121)
                ax1.set_xlabel(r'$x$ [m]')
                ax1.set_ylabel(r'$y$ [m]')
                ax1.axis('equal')
                _format_axes(ax1)
                ax2 = fig.add_subplot(122)
                ax2.set_xlabel(r'ln$(t/t_s)$')
                ax2.set_ylabel(r'$\bar{Q}_b$')
                _format_axes(ax2)

                # Plot curves for requested boreholes
                for i, borehole in enumerate(self.solver.boreholes):
                    if i in iBoreholes:
                        # Draw heat extraction rate
                        line = ax2.plot(lntts, Q_t[iBoreholes.index(i)][n])
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
            self, time=None, iBoreholes=None, showTilt=True, which=None):
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
        which : list of int, optional
            Indices i of the diagonal variable mass flow rate g-functions for
            which to plot heat extraction rates.
            If None, all diagonal g-functions are plotted.
            Default is None.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        from ._mpl import plt

        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.solver.boreholes))
        # Import heat extraction rate profiles
        z, Q_b = self._heat_extraction_rate_profiles(time, iBoreholes)

        if self.solver.nMassFlow == 0:
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

            # Adjust figure to window
            plt.tight_layout()
        else:
            m_flow = self.solver.m_flow
            if which is None:
                which = [n for n in range(self.solver.nMassFlow)]
            for n in which:
                # Configure figure and axes
                fig = _initialize_figure()
                fig.suptitle(
                    f'Heat extraction rate profiles for m_flow={m_flow[n]} kg/s')
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
                            Q_b[iBoreholes.index(i)][n],
                            z[iBoreholes.index(i)])
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

    def visualize_temperatures(
            self, iBoreholes=None, showTilt=True, which=None):
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
        which : list of int, optional
            Indices i of the diagonal variable mass flow rate g-functions for
            which to plot borehole wall temperatures.
            If None, all diagonal g-functions are plotted.
            Default is None.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        from ._mpl import plt

        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.solver.boreholes))
        # Import temperatures
        T_b = self._temperatures(iBoreholes)
        # Borefield characteristic time
        ts = np.mean([b.H for b in self.solver.boreholes])**2/(9.*self.alpha)
        # Dimensionless time (log)
        lntts = np.log(self.time/ts)


        if self.solver.nMassFlow == 0:
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
        else:
            m_flow = self.solver.m_flow
            if which is None:
                which = [n for n in range(self.solver.nMassFlow)]
            for n in which:
                # Configure figure and axes
                fig = _initialize_figure()
                fig.suptitle(
                    f'Borehole wall temperatures for m_flow={m_flow[n]} kg/s')
                ax1 = fig.add_subplot(121)
                ax1.set_xlabel(r'$x$ [m]')
                ax1.set_ylabel(r'$y$ [m]')
                ax1.axis('equal')
                _format_axes(ax1)
                ax2 = fig.add_subplot(122)
                ax2.set_xlabel(r'ln$(t/t_s)$')
                ax2.set_ylabel(r'$\bar{T}_b$')
                _format_axes(ax2)
                # Plot curves for requested boreholes
                for i, borehole in enumerate(self.solver.boreholes):
                    if i in iBoreholes:
                        # Draw borehole wall temperature
                        line = ax2.plot(lntts, T_b[iBoreholes.index(i)][n])
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
            self, time=None, iBoreholes=None, showTilt=True, which=None):
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
        which : list of int, optional
            Indices i of the diagonal variable mass flow rate g-functions for
            which to plot borehole wall temperatures.
            If None, all diagonal g-functions are plotted.
            Default is None.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        from ._mpl import plt

        # If iBoreholes is None, then plot all boreholes
        if iBoreholes is None:
            iBoreholes = range(len(self.boreholes))
        # Import temperature profiles
        z, T_b = self._temperature_profiles(time, iBoreholes)

        if self.solver.nMassFlow == 0:
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
                    # Draw borehole wall temperature profile
                    line = ax2.plot(
                        T_b[iBoreholes.index(i)],
                        z[iBoreholes.index(i)])
                    color = line[-1]._color
                    # Draw colored marker for borehole position
                    if showTilt:
                        ax1.plot(
                            [borehole.x, borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                            [borehole.y, borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
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
                            [borehole.x, borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                            [borehole.y, borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
                             linestyle='--',
                             marker='None',
                             color='k')
                    ax1.plot(borehole.x,
                             borehole.y,
                             linestyle='None',
                             marker='o',
                             color='k')

            plt.tight_layout()
        else:
            m_flow = self.solver.m_flow
            if which is None:
                which = [n for n in range(self.solver.nMassFlow)]
            for n in which:
                # Configure figure and axes
                fig = _initialize_figure()
                fig.suptitle(
                f'Borehole wall temperature profiles for m_flow={m_flow[n]} kg/s')
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
                        # Draw borehole wall temperature profile
                        line = ax2.plot(
                            T_b[iBoreholes.index(i)][n],
                            z[iBoreholes.index(i)])
                        color = line[-1]._color
                        # Draw colored marker for borehole position
                        if showTilt:
                            ax1.plot(
                                [borehole.x, borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                                [borehole.y, borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
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
                                [borehole.x, borehole.x + borehole.H * np.sin(borehole.tilt) * np.cos(borehole.orientation)],
                                [borehole.y, borehole.y + borehole.H * np.sin(borehole.tilt) * np.sin(borehole.orientation)],
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
            Borehole indices to extract heat extraction rates.

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
                if self.solver.nMassFlow == 0:
                    Q_t.append(
                        np.sum(
                            self.solver.Q_b[i0:i1, :]
                            * segment_ratios[:, np.newaxis],
                            axis=0))
                else:
                    Q_t.append(
                        np.sum(
                            self.solver.Q_b[:, i0:i1, :]
                            * segment_ratios[:, np.newaxis],
                            axis=1))
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
                    if self.solver.nMassFlow == 0:
                        Q_bi = self.solver.Q_b[i0:i1, -1]
                    else:
                        Q_bi = self.solver.Q_b[:, i0:i1, -1]
                else:
                    # Otherwise, heat extraction rates are interpolated.
                    if self.solver.nMassFlow == 0:
                        Q_bi = interp1d(
                            self.time,
                            self.solver.Q_b[i0:i1,:],
                            kind='linear',
                            copy=False,
                            axis=1)(time)
                    else:
                        Q_bi = interp1d(
                            self.time,
                            self.solver.Q_b[i0:i1,:],
                            kind='linear',
                            copy=False,
                            axis=2)(time)
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
                    if self.solver.nMassFlow == 0:
                        Q_b.append(np.repeat(Q_bi, 2, axis=0))
                    else:
                        Q_b.append(np.repeat(Q_bi, 2, axis=1))
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
                if self.solver.nMassFlow == 0:
                    T_b.append(
                        np.sum(
                            self.solver.T_b[i0:i1, :]
                            * segment_ratios[:, np.newaxis],
                            axis=0))
                else:
                    T_b.append(
                        np.sum(
                            self.solver.T_b[:, i0:i1, :]
                            * segment_ratios[:, np.newaxis],
                            axis=1))
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
                    if self.solver.nMassFlow == 0:
                        T_bi = self.solver.T_b[i0:i1, -1]
                    else:
                        T_bi = self.solver.T_b[:, i0:i1, -1]
                else:
                    # Otherwise, temperatures are interpolated.
                    T_bi = interp1d(self.time,
                                    self.solver.T_b[i0:i1, :],
                                    kind='linear',
                                    copy=False,
                                    axis=1)(time)
                    if self.solver.nMassFlow == 0:
                        T_bi = interp1d(
                            self.time,
                            self.solver.T_b[i0:i1,:],
                            kind='linear',
                            copy=False,
                            axis=1)(time)
                    else:
                        T_bi = interp1d(
                            self.time,
                            self.solver.T_b[i0:i1,:],
                            kind='linear',
                            copy=False,
                            axis=2)(time)
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
                    if self.solver.nMassFlow == 0:
                        T_b.append(np.repeat(T_bi, 2, axis=0))
                    else:
                        T_b.append(np.repeat(T_bi, 2, axis=1))
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
        assert isinstance(self.boreholes, (list, Borefield)), \
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
            assert not (type(self.m_flow_borehole) is np.ndarray and self.m_flow_borehole.ndim == 2 and not np.size(self.m_flow_borehole, axis=1)==self.network.nInlets), \
                "The number of mass flow rates in 'm_flow_borehole' must " \
                "correspond to the number of circuits in the network."
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
                        segment_ratios=segment_ratios, kind='linear',
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
        kind='linear', nSegments=8, segment_ratios=segment_ratios,
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
        nSegments=8, segment_ratios=segment_ratios,
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
