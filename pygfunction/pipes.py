# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi
from scipy.special import binom
import warnings

from .utilities import _initialize_figure, _format_axes


class _BasePipe(object):
    """
    Template for pipe classes.

    Pipe classes inherit from this class.

    Attributes
    ----------
    borehole : Borehole object
        Borehole class object of the borehole containing the U-Tube.
    nPipes : int
        Number of U-Tubes, equals to 1.
    nInlets : int
        Total number of pipe inlets, equals to 1.
    nOutlets : int
        Total number of pipe outlets, equals to 1.

    Notes
    -----
    The expected array shapes of input parameters and outputs are documented
    for each class method. `nInlets` and `nOutlets` are the number of inlets
    and outlets to the borehole, and both correspond to the number of
    independent parallel pipes. `nSegments` is the number of discretized
    segments along the borehole. `nPipes` is the number of pipes (i.e. the
    number of U-tubes) in the borehole. `nDepths` is the number of depths at
    which temperatures are evaluated.

    """
    def __init__(self, borehole):
        self.b = borehole
        self.nPipes = 1
        self.nInlets = 1
        self.nOutlets = 1

    def get_temperature(
            self, z, T_f_in, T_b, m_flow_borehole, cp_f, segment_ratios=None):
        """
        Returns the fluid temperatures of the borehole at a depth (z).

        Parameters
        ----------
        z : float or (nDepths,) array
            Depths (in meters) to evaluate the fluid temperatures.
        T_f_in : float or (nInlets,) array
            Inlet fluid temperatures (in Celsius).
        T_b : float or (nSegments,) array
            Borehole wall temperatures (in Celsius).
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rates (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        T_f : (2*nPipes,) or (nDepths, 2*nPipes,) array
            Fluid temperature (in Celsius) in each pipe. The returned shape
            depends on the type of the parameter `z`.

        """
        T_b = np.atleast_1d(T_b)
        nSegments = len(T_b)
        # Build coefficient matrices
        z_all = np.atleast_1d(z).flatten()
        a_in, a_b = self.coefficients_temperature(
            z_all, m_flow_borehole, cp_f, nSegments,
            segment_ratios=segment_ratios)
        # Evaluate fluid temperatures
        T_f = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b

        # Return 1d array if z was supplied as scalar
        if np.isscalar(z):
            T_f = T_f.flatten()
        return T_f

    def get_inlet_temperature(
            self, Q_f, T_b, m_flow_borehole, cp_f, segment_ratios=None):
        """
        Returns the outlet fluid temperatures of the borehole.

        Parameters
        ----------
        Q_f : float or (nInlets,) array
            Heat extraction from the fluid circuits (in Watts).
        T_b : float or (nSegments,) array
            Borehole wall temperatures (in Celsius).
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rates (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        T_in : float or (nOutlets,) array
            Inlet fluid temperatures (in Celsius) into each inlet pipe. The
            returned type corresponds to the type of the parameter `Q_f`.

        """
        T_b = np.atleast_1d(T_b)
        nSegments = len(T_b)
        # Build coefficient matrices
        a_qf, a_b = self.coefficients_inlet_temperature(
            m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)
        # Evaluate outlet temperatures
        T_f_in = a_qf @ np.atleast_1d(Q_f) + a_b @ T_b
        # Return float if Qf was supplied as scalar
        if np.isscalar(Q_f) and not np.isscalar(T_f_in):
            T_f_in = T_f_in.item()
        return T_f_in

    def get_outlet_temperature(
            self, T_f_in, T_b, m_flow_borehole, cp_f, segment_ratios=None):
        """
        Returns the outlet fluid temperatures of the borehole.

        Parameters
        ----------
        T_f_in : float or (nInlets,) array
            Inlet fluid temperatures (in Celsius).
        T_b : float or (nSegments,) array
            Borehole wall temperatures (in Celsius).
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rates (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        T_f_out : float or (nOutlets,) array
            Outlet fluid temperatures (in Celsius) from each outlet pipe. The
            returned type corresponds to the type of the parameter `T_f_in`.

        """
        T_b = np.atleast_1d(T_b)
        nSegments = len(T_b)
        # Build coefficient matrices
        a_in, a_b = self.coefficients_outlet_temperature(
            m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)
        # Evaluate outlet temperatures
        T_f_out = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b
        # Return float if Tin was supplied as scalar
        if np.isscalar(T_f_in) and not np.isscalar(T_f_out):
            T_f_out = T_f_out.item()
        return T_f_out

    def get_borehole_heat_extraction_rate(
            self, T_f_in, T_b, m_flow_borehole, cp_f, segment_ratios=None):
        """
        Returns the heat extraction rates of the borehole.

        Parameters
        ----------
        T_f_in : float or (nInlets,) array
            Inlet fluid temperatures (in Celsius).
        T_b : float or (nSegments,) array
            Borehole wall temperatures (in Celsius).
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rates (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        Q_b : float or (nSegments,) array
            Heat extraction rates along each borehole segment (in Watts). The
            returned type corresponds to the type of the parameter `T_b`.

        """
        T_b = np.atleast_1d(T_b)
        nSegments = len(T_b)
        a_in, a_b = self.coefficients_borehole_heat_extraction_rate(
            m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)
        Q_b = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b
        # Return float if Tb was supplied as scalar
        if np.isscalar(T_b) and not np.isscalar(Q_b):
            Q_b = Q_b.item()
        return Q_b

    def get_fluid_heat_extraction_rate(
            self, T_f_in, T_b, m_flow_borehole, cp_f, segment_ratios=None):
        """
        Returns the heat extraction rates of the borehole.

        Parameters
        ----------
        T_f_in : float or (nInlets,) array
            Inlet fluid temperatures (in Celsius).
        T_b : float or (nSegments,) array
            Borehole wall temperatures (in Celsius).
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rates (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        Q_f : float or (nOutlets,) array
            Heat extraction rates from each fluid circuit (in Watts). The
            returned type corresponds to the type of the parameter `T_f_in`.

        """
        T_b = np.atleast_1d(T_b)
        nSegments = len(T_b)
        a_in, a_b = self.coefficients_fluid_heat_extraction_rate(
            m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)
        Q_f = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b
        # Return float if Tb was supplied as scalar
        if np.isscalar(T_f_in) and not np.isscalar(Q_f):
            Q_f = Q_f.item()
        return Q_f

    def get_total_heat_extraction_rate(
            self, T_f_in, T_b, m_flow_borehole, cp_f, segment_ratios=None):
        """
        Returns the total heat extraction rate of the borehole.

        Parameters
        ----------
        T_f_in : float or (nInlets,) array
            Inlet fluid temperatures (in Celsius).
        T_b : float or (nSegments,) array
            Borehole wall temperatures (in Celsius).
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rates (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        Q_t : float
            Total net heat extraction rate of the borehole (in Watts).

        """
        Q_f = self.get_fluid_heat_extraction_rate(
            T_f_in, T_b, m_flow_borehole, cp_f, segment_ratios=segment_ratios)
        Q_t = np.sum(Q_f)
        return Q_t

    def coefficients_inlet_temperature(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate outlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,in}} = \\mathbf{a_{q,f}} \\mathbf{Q_{f}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rates (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_qf : (nOutlets, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (nOutlets, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_inlet_temperature is 3
        method_id = 3
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios, method_id):
            a_qf, a_b = self._get_stored_coefficients(method_id)
        else:
            # Coefficient matrices for fluid heat extraction rates:
            # [Q_{f}] = [b_in]*[T_{f,in}] + [b_b]*[T_{b}]
            b_in, b_b = self.coefficients_fluid_heat_extraction_rate(
                m_flow_borehole, cp_f, nSegments,
                segment_ratios=segment_ratios)
            b_in_m1 = np.linalg.inv(b_in)

            # Matrices for fluid heat extraction rates:
            # [T_{f,in}] = [a_qf]*[Q_{f}] + [a_b]*[T_{b}]
            a_qf = b_in_m1
            a_b = -b_in_m1 @ b_b

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios, (a_qf, a_b),
                method_id)

        return a_qf, a_b

    def coefficients_outlet_temperature(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate outlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,out}} = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rates (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nOutlets, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (nOutlets, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_outlet_temperature is 4
        method_id = 4
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Check if _continuity_condition_base need to be called
            # method_id for _continuity_condition_base is 0
            if self._check_coefficients(
                    m_flow_borehole, cp_f, nSegments, segment_ratios, 0):
                b_in, b_out, b_b = self._get_stored_coefficients(0)
            else:
                # Coefficient matrices from continuity condition:
                # [b_out]*[T_{f,out}] = [b_in]*[T_{f,in}] + [b_b]*[T_b]
                b_in, b_out, b_b = self._continuity_condition_base(
                    m_flow_borehole, cp_f, nSegments,
                    segment_ratios=segment_ratios)

                # Store coefficients
                self._set_stored_coefficients(
                    m_flow_borehole, cp_f, nSegments, segment_ratios,
                    (b_in, b_out, b_b), 0)

            # Final coefficient matrices for outlet temperatures:
            # [T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]
            b_out_m1 = np.linalg.inv(b_out)
            a_in = b_out_m1 @ b_in
            a_b = b_out_m1 @ b_b

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios, (a_in, a_b),
                method_id)

        return a_in, a_b

    def coefficients_temperature(
            self, z, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate fluid temperatures at a depth
        (z).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        z : float or (nDepths,) array
            Depths (in meters) to evaluate the fluid temperature coefficients.
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in :
        (2*nPipes, nInlets,) array, or (nDepths, 2*nPipes, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_b :
        (2*nPipes, nSegments,) array, or (nDepths, 2*nPipes, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_temperature is 5
        method_id = 5

        # Coefficient matrices for outlet temperatures:
        # [T_{f,out}] = [b_in]*[T_{f,in}] + [b_b]*[T_b]
        b_in, b_b = self.coefficients_outlet_temperature(
            m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)

        # Check if _continuity_condition_head need to be called
        # method_id for _continuity_condition_head is 1
        if self._check_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios, 1):
            c_in, c_out, c_b = self._get_stored_coefficients(1)
        else:
            # Coefficient matrices for temperatures at depth (z = 0):
            # [T_f](0) = [c_in]*[T_{f,in}] + [c_out]*[T_{f,out}]
            #                              + [c_b]*[T_b]
            c_in, c_out, c_b = self._continuity_condition_head(
                m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios,
                (c_in, c_out, c_b), 1)

        # Coefficient matrices from general solution:
        # [T_f](z) = [d_f0]*[T_f](0) + [d_b]*[T_b]
        d_f0, d_b = self._general_solution(
            z, m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)

        # Final coefficient matrices for temperatures at depth (z):
        # [T_f](z) = [a_in]*[T_{f,in}] + [a_b]*[T_b]
        a_in = d_f0 @ (c_in + c_out @ b_in)
        a_b = d_f0 @ (c_b + c_out @ b_b) + d_b

        return a_in, a_b

    def coefficients_borehole_heat_extraction_rate(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_b} = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nSegments, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (nSegments, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_borehole_heat_extraction_rate is 6
        method_id = 6

        nPipes = self.nPipes
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(
                m_flow_borehole, cp_f, nSegments, segment_ratios)
            m_flow_pipe = self._m_flow_pipe
            cp_pipe = self._cp_pipe
            mcp = np.hstack((-m_flow_pipe[0:nPipes],
                             m_flow_pipe[-nPipes:]))*cp_pipe

            # Initialize coefficient matrices
            a_in = np.zeros((nSegments, self.nInlets))
            a_b = np.zeros((nSegments, nSegments))
            # Heat extraction rates are calculated from an energy balance on a
            # borehole segment.
            z = self.b._segment_edges(nSegments, segment_ratios=segment_ratios)
            aTf, bTf = self.coefficients_temperature(
                z, m_flow_borehole, cp_f, nSegments,
                segment_ratios=segment_ratios)
            a_in = mcp @ (aTf[:-1,:,:] - aTf[1:,:,:])
            a_b = mcp @ (bTf[:-1,:,:] - bTf[1:,:,:])

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios,
                (a_in, a_b), method_id)

        return a_in, a_b

    def coefficients_fluid_heat_extraction_rate(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_f} = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nOutlets, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (nOutlets, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_fluid_heat_extraction_rate is 7
        method_id = 7
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(
                m_flow_borehole, cp_f, nSegments, segment_ratios)

            # Coefficient matrices for outlet temperatures:
            # [T_{f,out}] = [b_in]*[T_{f,in}] + [b_b]*[T_b]
            b_in, b_b = self.coefficients_outlet_temperature(
                m_flow_borehole, cp_f, nSegments,
                segment_ratios=segment_ratios)

            # Intermediate matrices for fluid heat extraction rates:
            # [Q_{f}] = [c_in]*[T_{f,in}] + [c_out]*[T_{f,out}]
            MCP = self._m_flow_in * self._cp_in
            c_in = -np.diag(MCP)
            c_out = np.diag(MCP)

            # Matrices for fluid heat extraction rates:
            # [Q_{f}] = [a_in]*[T_{f,in}] + [a_b]*[T_{b}]
            a_in = c_in + c_out @ b_in
            a_b = c_out @ b_b

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_borehole, cp_f, nSegments, segment_ratios, (a_in, a_b),
                method_id)

        return a_in, a_b

    def effective_borehole_thermal_resistance(self, m_flow_borehole, cp_f):
        """
        Evaluate the effective borehole thermal resistance, defined by:

            .. math::

                \\frac{Q_b}{H} = \\frac{T^*_b - \\bar{T}_f}{R^*_b}

                \\bar{T}_f = \\frac{1}{2}(T_{f,in} + T_{f,out})

        where :math:`Q_b` is the borehole heat extraction rate (in Watts),
        :math:`H` is the borehole length, :math:`T^*_b` is the effective
        borehole wall temperature, :math:`R^*_b` is the effective borehole
        thermal resistance, :math:`T_{f,in}` is the inlet fluid temperature,
        and :math:`T_{f,out}` is the outlet fluid temperature.

        Parameters
        ----------
        m_flow_borehole : float
            Fluid mass flow rate (in kg/s) into the borehole.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.K)

        Returns
        -------
        R_b : float
            Effective borehole thermal resistance (in m.K/W).

        """
        # Coefficient for T_{f,out} = a_out*T_{f,in} + [b_out]*[T_b]
        a_out = self.coefficients_outlet_temperature(
            m_flow_borehole, cp_f, nSegments=1)[0].item()
        # Coefficient for Q_b = [a_Q]*T{f,in} + [b_Q]*[T_b]
        a_Q = self.coefficients_borehole_heat_extraction_rate(
                m_flow_borehole, cp_f, nSegments=1)[0].item()
        # Borehole length
        H = self.b.H
        # Effective borehole thermal resistance
        R_b = -0.5*H*(1. + a_out)/a_Q
        return R_b

    def local_borehole_thermal_resistance(self):
        """
        Evaluate the local (cross-sectional) borehole thermal resistance,
        defined by:

            .. math::

                Q'_b(z) = \\frac{T_b(z) - \\bar{T}_f(z)}{R_b}

        where :math:`Q'_b(z)` is the borehole heat extraction rate per unit
        depth at a depth :math:`(z)`, :math:`T_b(z)` is the borehole wall
        temperature, :math:`\\bar{T}_f(z)` is the arithmetic mean fluid
        temperature and :math:`R_b` is the local borehole thermal resistance.

        Returns
        -------
        R_b : float
            Local borehole thermal resistance (in m.K/W).

        """
        # Evaluate borehole thermal resistance
        R_b = 1 / np.trace(1 / self._Rd)
        return R_b

    def update_thermal_resistances(self):
        """ Update the delta-circuit of thermal resistances.
        """
        raise NotImplementedError(
            'update_thermal_resistances class method not implemented, '
            'this method should update the array of delta-circuit thermal '
            'resistances.')
        return

    def visualize_pipes(self):
        """
        Plot the cross-section view of the borehole.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # Configure figure and axes
        fig = _initialize_figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.axis('equal')
        _format_axes(ax)

        # Color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        lw = plt.rcParams['lines.linewidth']

        # Borehole wall outline
        ax.plot([-self.b.r_b, 0., self.b.r_b, 0.],
                [0., self.b.r_b, 0., -self.b.r_b],
                'k.', alpha=0.)
        borewall = plt.Circle(
            (0., 0.), radius=self.b.r_b, fill=False,
            color='k', linestyle='--', lw=lw)
        ax.add_patch(borewall)

        # Pipes
        for i in range(self.nPipes):
            # Coordinates of pipes
            (x_in, y_in) = self.pos[i]
            (x_out, y_out) = self.pos[i + self.nPipes]

            # Pipe outline (inlet)
            pipe_in_in = plt.Circle(
                (x_in, y_in), radius=self.r_in,
                fill=False, linestyle='-', color=colors[i], lw=lw)
            pipe_in_out = plt.Circle(
                (x_in, y_in), radius=self.r_out,
                fill=False, linestyle='-', color=colors[i], lw=lw)
            ax.text(x_in, y_in, i, ha="center", va="center")

            # Pipe outline (outlet)
            pipe_out_in = plt.Circle(
                (x_out, y_out), radius=self.r_in,
                fill=False, linestyle='-', color=colors[i], lw=lw)
            pipe_out_out = plt.Circle(
                (x_out, y_out), radius=self.r_out,
                fill=False, linestyle='-', color=colors[i], lw=lw)
            ax.text(x_out, y_out, i + self.nPipes,
                    ha="center", va="center")

            ax.add_patch(pipe_in_in)
            ax.add_patch(pipe_in_out)
            ax.add_patch(pipe_out_in)
            ax.add_patch(pipe_out_out)

        plt.tight_layout()

        return fig

    def _initialize_stored_coefficients(self):
        nMethods = 8    # Number of class methods
        self._stored_coefficients = [() for i in range(nMethods)]
        self._stored_m_flow_cp = [np.empty(self.nInlets)*np.nan
                                  for i in range(nMethods)]
        self._stored_nSegments = [np.nan for i in range(nMethods)]
        self._stored_segment_ratios = [np.nan for i in range(nMethods)]
        self._m_flow_cp_model_variables = np.empty(self.nInlets)*np.nan
        self._nSegments_model_variables = np.nan
        self._segment_ratios_model_variables = np.nan


    def _set_stored_coefficients(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios, coefficients, method_id):
        self._stored_coefficients[method_id] = coefficients
        self._stored_m_flow_cp[method_id] = m_flow_borehole*cp_f
        self._stored_nSegments[method_id] = nSegments
        self._stored_segment_ratios[method_id] = segment_ratios

    def _get_stored_coefficients(self, method_id):
        coefficients = self._stored_coefficients[method_id]
        return coefficients

    def _check_model_variables(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios, tol=1e-6):
        stored_m_flow_cp = self._m_flow_cp_model_variables
        stored_nSegments = self._nSegments_model_variables
        stored_segment_ratios = self._segment_ratios_model_variables
        if stored_segment_ratios is None and segment_ratios is None:
            check_ratios = True
        elif isinstance(stored_segment_ratios, np.ndarray) and isinstance(segment_ratios, np.ndarray):
            check_ratios = (len(segment_ratios) == len(stored_segment_ratios) and
                            np.all(np.abs(segment_ratios - stored_segment_ratios) < np.abs(stored_segment_ratios)*tol))
        else:
            check_ratios = False
        if (np.all(np.abs(m_flow_borehole*cp_f - stored_m_flow_cp) < np.abs(stored_m_flow_cp)*tol)
            and nSegments == stored_nSegments
            and check_ratios):
            check = True
        else:
            self._update_model_variables(
                m_flow_borehole, cp_f, nSegments, segment_ratios)
            self._m_flow_cp_model_variables = m_flow_borehole*cp_f
            self._nSegments_model_variables = nSegments
            self._segment_ratios_model_variables = segment_ratios
            check = False

        return check

    def _check_coefficients(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios, method_id,
            tol=1e-6):
        stored_m_flow_cp = self._stored_m_flow_cp[method_id]
        stored_nSegments = self._stored_nSegments[method_id]
        stored_segment_ratios = self._stored_segment_ratios[method_id]
        if nSegments == stored_nSegments:
            if stored_segment_ratios is None and segment_ratios is None:
                check_ratios = True
            elif isinstance(stored_segment_ratios, np.ndarray) and isinstance(segment_ratios, np.ndarray):
                check_ratios = np.all(np.abs(segment_ratios - stored_segment_ratios) < np.abs(stored_segment_ratios)*tol)
            else:
                check_ratios = False
        else:
            check_ratios = False
        if (check_ratios
            and np.all(np.abs(m_flow_borehole*cp_f - stored_m_flow_cp) < np.abs(stored_m_flow_cp)*tol)):
            check = True
        else:
            check = False

        return check
    
    def _check_geometry(self):
        """ Verifies the inputs to the pipe object and raises an error if
            the geometry is not valid.
        """
        # Verify that thermal properties are greater than 0.
        if not self.k_s > 0.:
            raise ValueError(
                f'The ground thermal conductivity must be greater than zero. '
                f'A value of {self.k_s} was provided.')
        if not self.k_g > 0.:
            raise ValueError(
                f'The grout thermal conductivity must be greater than zero. '
                f'A value of {self.k_g} was provided.')
        if not self.R_fp > 0.:
            raise ValueError(
                f'The fluid to outer pipe wall thermal resistance must be'
                f'greater than zero. '
                f'A value of {self.R_fp} was provided.')

        # Verify that the pipe radius is greater than zero.
        if not self.r_in > 0.:
            raise ValueError(
                f'The pipe inner radius must be greater than zero. '
                f'A value of {self.r_in} was provided.')

        # Verify that the outer pipe radius is greater than the inner pipe
        # radius.
        if not self.r_out > self.r_in:
            raise ValueError(
                f'The pipe outer radius must be greater than the pipe inner '
                f'radius. A value of {self.r_out} was provided.')

        # Verify that the number of multipoles is zero or greater.
        if not self.J >= 0:
            raise ValueError(
                f'The number of terms in the multipole expansion must be zero '
                f'or greater. A value of {self.J} was provided.')

        # Verify that the pipes are contained within the borehole.
        for i in range(2*self.nPipes):
            r_pipe = np.sqrt(self.pos[i][0]**2 + self.pos[i][1]**2)
            if not r_pipe + self.r_out <= self.b.r_b:
                raise ValueError(
                    f'Pipes must be entirely contained within the borehole. '
                    f'Pipe {i} is partly or entirely outside the borehole.')

        # Verify that the pipes do not collide to one another.
        for i in range(2*self.nPipes):
            for j in range(i+1, 2*self.nPipes):
                dx = self.pos[i][0] - self.pos[j][0]
                dy = self.pos[i][1] - self.pos[j][1]
                dis = np.sqrt(dx**2 + dy**2)
                if not dis >= 2*self.r_out:
                    raise ValueError(
                        f'Pipes {i} and {j} are overlapping.')

        return True

    def _continuity_condition_base(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """ Returns coefficients for the relation
            [a_out]*[T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]
        """
        raise NotImplementedError(
            '_continuity_condition_base class method not implemented, '
            'this method should return matrices for the relation: '
            '[a_out]*[T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]')

    def _continuity_condition_head(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """ Returns coefficients for the relation
            [T_f](z=0) = [a_in]*[T_{f,in}] + [a_out]*[T_{f,out}] + [a_b]*[T_b]
        """
        raise NotImplementedError(
            '_continuity_condition_head class method not implemented, '
            'this method should return matrices for the relation: '
            '[T_f](z=0) = [a_in]*[T_{f,in}] + [a_out]*[T_{f,out}] '
            '+ [a_b]*[T_b]')

    def _general_solution(
            self, z, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """ Returns coefficients for the relation
            [T_f](z) = [a_f0]*[T_f](0) + [a_b]*[T_b]
        """
        raise NotImplementedError(
            '_general_solution class method not implemented, '
            'this method should return matrices for the relation: '
            '[T_f](z) = [a_f0]*[T_f](0) + [a_b]*[T_b]')

    def _update_model_variables(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios):
        """
        Evaluate common coefficients needed in other class methods.
        """
        raise NotImplementedError(
            '_update_coefficients class method not implemented, '
            'this method should evaluate common coefficients needed in other '
            'class methods.')

    def _format_inputs(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios):
        """
        Format arrays of mass flow rates and heat capacity.
        """
        raise NotImplementedError(
            '_format_inputs class method not implemented, '
            'this method should format 1d arrays for the inlet mass flow '
            'rates (_m_flow_in), mass flow rates in each pipe (_m_flow_pipe), '
            'heat capacity at each inlet (_cp_in) and heat capacity in each '
            'pipe (_cp_pipe).')


class SingleUTube(_BasePipe):
    """
    Class for single U-Tube boreholes.

    Contains information regarding the physical dimensions and thermal
    characteristics of the pipes and the grout material, as well as methods to
    evaluate fluid temperatures and heat extraction rates based on the work of
    Hellstrom [#Single-Hellstrom1991]_.

    Attributes
    ----------
    pos : list of tuples
        Position (x, y) (in meters) of the pipes inside the borehole.
    r_in : float
        Inner radius (in meters) of the U-Tube pipes.
    r_out : float
        Outer radius (in meters) of the U-Tube pipes.
    borehole : Borehole object
        Borehole class object of the borehole containing the U-Tube.
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    R_fp : float
        Fluid to outer pipe wall thermal resistance (m-K/W).
    J : int, optional
        Number of multipoles per pipe to evaluate the thermal resistances.
        Default is 2.
    nPipes : int
        Number of U-Tubes, equals to 1.
    nInlets : int
        Total number of pipe inlets, equals to 1.
    nOutlets : int
        Total number of pipe outlets, equals to 1.

    Notes
    -----
    The expected array shapes of input parameters and outputs are documented
    for each class method. `nInlets` and `nOutlets` are the number of inlets
    and outlets to the borehole, and both are equal to 1 for a single U-tube
    borehole. `nSegments` is the number of discretized segments along the
    borehole. `nPipes` is the number of pipes (i.e. the number of U-tubes) in
    the borehole, equal to 1. `nDepths` is the number of depths at which
    temperatures are evaluated.

    References
    ----------
    .. [#Single-Hellstrom1991] Hellstrom, G. (1991). Ground heat storage.
       Thermal Analyses of Duct Storage Systems I: Theory. PhD Thesis.
       University of Lund, Department of Mathematical Physics. Lund, Sweden.

    """
    def __init__(self, pos, r_in, r_out, borehole, k_s, k_g, R_fp, J=2):
        self.pos = pos
        self.r_in = r_in
        self.r_out = r_out
        self.b = borehole
        self.k_s = k_s
        self.k_g = k_g
        self.R_fp = R_fp
        self.J = J
        self.nPipes = 1
        self.nInlets = 1
        self.nOutlets = 1
        self._check_geometry()

        # Delta-circuit thermal resistances
        self.update_thermal_resistances(self.R_fp)
        return

    def update_thermal_resistances(self, R_fp):
        """
        Update the delta-circuit of thermal resistances.

        This methods updates the values of the delta-circuit thermal
        resistances based on the provided fluid to outer pipe wall thermal
        resistance.

        Parameters
        ----------
        R_fp : float
            Fluid to outer pipe wall thermal resistance (m-K/W).

        """
        self.R_fp = R_fp
        # Delta-circuit thermal resistances
        self._Rd = thermal_resistances(
            self.pos, self.r_out, self.b.r_b, self.k_s, self.k_g, R_fp,
            J=self.J)[1]
        # Initialize stored_coefficients
        self._initialize_stored_coefficients()
        return

    def _continuity_condition_base(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Equation that satisfies equal fluid temperatures in both legs of
        each U-tube pipe at depth (z = H).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{a_{out}} T_{f,out} =
                \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nOutlets, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_out : (nOutlets, nOutlets,) array
            Array of coefficients for outlet fluid temperature.
        a_b : (nOutlets, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # Check if model variables need to be updated
        self._check_model_variables(
            m_flow_borehole, cp_f, nSegments, segment_ratios)

        # Evaluate coefficient matrices from Hellstrom (1991):
        a_in = ((self._f1(self.b.H) + self._f2(self.b.H))
                / (self._f3(self.b.H) - self._f2(self.b.H)))
        a_in = np.array([[a_in]])

        a_out = np.array([[1.0]])

        a_b = np.zeros((self.nOutlets, nSegments))

        z = self.b._segment_edges(nSegments, segment_ratios=segment_ratios)[::-1]
        F4 = self._F4(z)
        dF4 = F4[:-1] - F4[1:]
        F5 = self._F5(z)
        dF5 = F5[:-1] - F5[1:]
        a_b[0, :] = (dF4 + dF5) / (self._f3(self.b.H) - self._f2(self.b.H))

        return a_in, a_out, a_b

    def _continuity_condition_head(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate fluid temperatures at depth
        (z = 0). These coefficients take into account connections between
        U-tube pipes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z=0) = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{out}} \\mathbf{T_{f,out}}
                + \\mathbf{a_{b}} \\mathbf{T_{b}}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (2*nPipes, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_out : (2*nPipes, nOutlets,) array
            Array of coefficients for outlet fluid temperature.
        a_b : (2*nPipes, nSegments,) array
            Array of coefficients for borehole wall temperature.

        """
        # Check if model variables need to be updated
        self._check_model_variables(
            m_flow_borehole, cp_f, nSegments, segment_ratios)

        # There is only one pipe
        a_in = np.array([[1.0], [0.0]])
        a_out = np.array([[0.0], [1.0]])
        a_b = np.zeros((2, nSegments))

        return a_in, a_out, a_b

    def _general_solution(
            self, z, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        General solution for fluid temperatures at a depth (z).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{a_{f0}} \\mathbf{T_{f}}(z=0)
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        z : float or (nDepths,) array
            Depths (in meters) to evaluate the fluid temperature coefficients.
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_f0 :
        (2*nPipes, 2*nPipes,) array, or (nDepths, 2*nPipes, 2*nPipes,) array
            Array of coefficients for inlet fluid temperature.
        a_b :
        (2*nPipes, nSegments,) array, or (nDepths, 2*nPipes, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        z_array = np.atleast_1d(z)
        # Check if model variables need to be updated
        self._check_model_variables(
            m_flow_borehole, cp_f, nSegments, segment_ratios)

        a_f0 = np.block([[[self._f1(z)], [self._f2(z)]],
                         [[-self._f2(z)], [self._f3(z)]]]).transpose(2, 0, 1)

        a_b = np.zeros((len(z_array), 2*self.nPipes, nSegments))
        z_edges = self.b._segment_edges(
            nSegments, segment_ratios=segment_ratios)
        dz = np.maximum(np.subtract.outer(z_array, z_edges), 0.)
        dF4 = self._F4(dz)
        dF5 = self._F5(dz)
        a_b[:, 0, :] = (dF4[:, :-1] - dF4[:, 1:])
        a_b[:, 1, :] = -(dF5[:, :-1] - dF5[:, 1:])
        # Remove first dimension if z is a scalar
        if np.isscalar(z):
            a_f0 = a_f0[0, :, :]
            a_b = a_b[0, :, :]

        return a_f0, a_b

    def _update_model_variables(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios):
        """
        Evaluate dimensionless resistances for Hellstrom (1991) solution.

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
        """

        # Format mass flow rate and heat capacity inputs
        self._format_inputs(m_flow_borehole, cp_f, nSegments, segment_ratios)
        m_flow_in = self._m_flow_in
        cp_in = self._cp_in

        # Dimensionless delta-circuit conductances
        self._beta1 = 1./(self._Rd[0][0]*m_flow_in[0]*cp_in[0])
        self._beta2 = 1./(self._Rd[1][1]*m_flow_in[0]*cp_in[0])
        self._beta12 = 1./(self._Rd[0][1]*m_flow_in[0]*cp_in[0])
        self._beta = 0.5*(self._beta2 - self._beta1)
        # Eigenvalues
        self._gamma = np.sqrt(0.25*(self._beta1+self._beta2)**2
                              + self._beta12*(self._beta1+self._beta2))
        self._delta = 1./self._gamma \
            * (self._beta12 + 0.5*(self._beta1+self._beta2))

    def _format_inputs(self, m_flow_borehole, cp_f, nSegments, segment_ratios):
        """
        Format mass flow rate and heat capacity inputs.

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
        """

        # Format mass flow rate inputs
        if np.isscalar(m_flow_borehole):
            # Mass flow rate in each fluid circuit
            m_flow_in = m_flow_borehole*np.ones(self.nInlets)
        else:
            # Mass flow rate in each fluid circuit
            m_flow_in = m_flow_borehole
        self._m_flow_in = m_flow_in
        # Mass flow rate in pipes
        m_flow_pipe = np.tile(m_flow_in, 2*self.nPipes)
        self._m_flow_pipe = m_flow_pipe

        # Format heat capacity inputs
        if np.isscalar(cp_f):
            # Heat capacity in each fluid circuit
            cp_in = cp_f*np.ones(self.nInlets)
        else:
            # Heat capacity in each fluid circuit
            cp_in = cp_f
        self._cp_in = cp_in
        # Heat capacity in pipes
        cp_pipe = np.tile(cp_in, 2*self.nPipes)
        self._cp_pipe = cp_pipe

    def _f1(self, z):
        """
        Calculate function f1 from Hellstrom (1991)

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperature coefficients.
        """

        f1 = np.exp(self._beta*z)*(np.cosh(self._gamma*z)
                                   - self._delta*np.sinh(self._gamma*z))
        return f1

    def _f2(self, z):
        """
        Calculate function f2 from Hellstrom (1991)

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperature coefficients.
        """

        f2 = np.exp(self._beta*z)*self._beta12/self._gamma \
            * np.sinh(self._gamma*z)
        return f2

    def _f3(self, z):
        """
        Calculate function f3 from Hellstrom (1991)

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperature coefficients.
        """

        f3 = np.exp(self._beta*z)*(np.cosh(self._gamma*z)
                                   + self._delta*np.sinh(self._gamma*z))
        return f3

    def _f4(self, z):
        """
        Calculate function f4 from Hellstrom (1991)

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperature coefficients.
        """

        A = self._delta*self._beta1 + self._beta2*self._beta12/self._gamma
        f4 = np.exp(self._beta*z) \
            * (self._beta1*np.cosh(self._gamma*z) - A*np.sinh(self._gamma*z))
        return f4

    def _f5(self, z):
        """
        Calculate function f5 from Hellstrom (1991)

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperature coefficients.
        """

        B = self._delta*self._beta2 + self._beta1*self._beta12/self._gamma
        f5 = np.exp(self._beta*z) \
            * (self._beta2*np.cosh(self._gamma*z) + B*np.sinh(self._gamma*z))
        return f5

    def _F4(self, z):
        """
        Calculate integral of function f4 from Hellstrom (1991)

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperature coefficients.
        """

        A = self._delta*self._beta1 + self._beta2*self._beta12/self._gamma
        C = self._beta1*self._beta + A*self._gamma
        S = - (self._beta1*self._gamma + self._beta*A)
        denom = (self._beta**2 - self._gamma**2)
        F4 = np.exp(self._beta*z) / denom \
            * (C*np.cosh(self._gamma*z) + S*np.sinh(self._gamma*z))
        return F4

    def _F5(self, z):
        """
        Calculate integral of function f5 from Hellstrom (1991)

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperature coefficients.
        """

        B = self._delta*self._beta2 + self._beta1*self._beta12/self._gamma
        C = self._beta2*self._beta - B*self._gamma
        S = - (self._beta2*self._gamma - self._beta*B)
        denom = (self._beta**2 - self._gamma**2)
        F5 = np.exp(self._beta*z) / denom \
            * (C*np.cosh(self._gamma*z) + S*np.sinh(self._gamma*z))
        return F5


class MultipleUTube(_BasePipe):
    """
    Class for multiple U-Tube boreholes.

    Contains information regarding the physical dimensions and thermal
    characteristics of the pipes and the grout material, as well as methods to
    evaluate fluid temperatures and heat extraction rates based on the work of
    Cimmino [#Cimmino2016]_ for boreholes with any number of U-tubes.

    Attributes
    ----------
    pos : list of tuples
        Position (x, y) (in meters) of the pipes inside the borehole.
    r_in : float
        Inner radius (in meters) of the U-Tube pipes.
    r_out : float
        Outer radius (in meters) of the U-Tube pipes.
    borehole : Borehole object
        Borehole class object of the borehole containing the U-Tube.
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    R_fp : float
        Fluid to outer pipe wall thermal resistance (m-K/W).
    J : int, optional
        Number of multipoles per pipe to evaluate the thermal resistances.
        Default is 2.
    nPipes : int
        Number of U-Tubes.
    config : str, defaults to 'parallel'
        Configuration of the U-Tube pipes:
            'parallel' : U-tubes are connected in parallel.
            'series' : U-tubes are connected in series.
    nInlets : int
        Total number of pipe inlets, equals to 1.
    nOutlets : int
        Total number of pipe outlets, equals to 1.

    Notes
    -----
    The expected array shapes of input parameters and outputs are documented
    for each class method. `nInlets` and `nOutlets` are the number of inlets
    and outlets to the borehole, and both are equal to 1 for a multiple U-tube
    borehole. `nSegments` is the number of discretized segments along the
    borehole. `nPipes` is the number of pipes (i.e. the number of U-tubes) in
    the borehole. `nDepths` is the number of depths at which temperatures are
    evaluated.

    References
    ----------
    .. [#Cimmino2016] Cimmino, M. (2016). Fluid and borehole wall temperature
       profiles in vertical geothermal boreholes with multiple U-tubes.
       Renewable Energy, 96, 137-147.

    """
    def __init__(self, pos, r_in, r_out, borehole, k_s,
                 k_g, R_fp, nPipes, config='parallel', J=2):
        self.pos = pos
        self.r_in = r_in
        self.r_out = r_out
        self.b = borehole
        self.k_s = k_s
        self.k_g = k_g
        self.R_fp = R_fp
        self.J = J
        self.nPipes = nPipes
        self.nInlets = 1
        self.nOutlets = 1
        self.config = config.lower()
        self._check_geometry()

        # Delta-circuit thermal resistances
        self.update_thermal_resistances(self.R_fp)
        return

    def update_thermal_resistances(self, R_fp):
        """
        Update the delta-circuit of thermal resistances.

        This methods updates the values of the delta-circuit thermal
        resistances based on the provided fluid to outer pipe wall thermal
        resistance.

        Parameters
        ----------
        R_fp : float
            Fluid to outer pipe wall thermal resistance (m-K/W).

        """
        self.R_fp = R_fp
        # Delta-circuit thermal resistances
        self._Rd = thermal_resistances(
            self.pos, self.r_out, self.b.r_b, self.k_s, self.k_g, R_fp,
            J=self.J)[1]
        # Initialize stored_coefficients
        self._initialize_stored_coefficients()
        return

    def _continuity_condition_base(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Equation that satisfies equal fluid temperatures in both legs of
        each U-tube pipe at depth (z = H).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{a_{out}} T_{f,out} = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nOutlets, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_out : (nOutlets, nOutlets,) array
            Array of coefficients for outlet fluid temperature.
        a_b : (nOutlets, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # Check if model variables need to be updated
        self._check_model_variables(
            m_flow_borehole, cp_f, nSegments, segment_ratios)

        # Coefficient matrices from continuity condition:
        # [b_u]*[T_{f,u}](z=0) = [b_d]*[T_{f,d}](z=0) + [b_b]*[T_b]
        b_d, b_u, b_b = self._continuity_condition(
            m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)
        b_u_m1 = np.linalg.inv(b_u)

        if self.config == 'parallel':
            # Intermediate coefficient matrices:
            # [T_{f,d}](z=0) = [c_in]*[T_{f,in}]
            c_in = np.ones((self.nPipes, 1))

            # Intermediate coefficient matrices:
            # [T_{f,out}] = d_u*[T_{f,u}](z=0)
            mcp = self._m_flow_pipe[-self.nPipes:]*self._cp_pipe[-self.nPipes:]
            d_u = np.reshape(mcp/np.sum(mcp), (1, -1))

            # Final coefficient matrices for continuity at depth (z = H):
            # [a_out][T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]
            a_in = d_u @ b_u_m1 @ b_d @ c_in
            a_out = np.array([[1.0]])
            a_b = d_u @ b_u_m1 @ b_b

        elif self.config == 'series':
            # Intermediate coefficient matrices:
            # [T_{f,d}](z=0) = [c_in]*[T_{f,in}] + [c_u]*[T_{f,u}](z=0)
            c_in = np.eye(self.nPipes, M=1)
            c_u = np.eye(self.nPipes, k=-1)

            # Intermediate coefficient matrices:
            # [d_u]*[T_{f,u}](z=0) = [d_in]*[T_{f,in}] + [d_b]*[T_b]
            d_u = b_u - b_d @ c_u
            d_in = b_d @ c_in
            d_b = b_b
            d_u_m1 = np.linalg.inv(d_u)

            # Intermediate coefficient matrices:
            # [T_{f,out}] = e_u*[T_{f,u}](z=0)
            e_u = np.eye(self.nPipes, M=1, k=-self.nPipes+1).T

            # Final coefficient matrices for continuity at depth (z = H):
            # [a_out][T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]
            a_in = e_u @ d_u_m1 @ d_in
            a_out = np.array([[1.0]])
            a_b = e_u @ d_u_m1 @ d_b
        else:
            raise NotImplementedError(f"Configuration '{self.config}' "
                                      f"not implemented.")

        return a_in, a_out, a_b

    def _continuity_condition_head(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate fluid temperatures at depth
        (z = 0). These coefficients take into account connections between
        U-tube pipes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z=0) = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{out}} \\mathbf{T_{f,out}}
                + \\mathbf{a_{b}} \\mathbf{T_{b}}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (2*nPipes, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_out : (2*nPipes, nOutlets,) array
            Array of coefficients for outlet fluid temperature.
        a_b : (2*nPipes, nSegments,) array
            Array of coefficients for borehole wall temperature.

        """
        # Check if model variables need to be updated
        self._check_model_variables(
            m_flow_borehole, cp_f, nSegments, segment_ratios)

        if self.config == 'parallel':
            a_in = np.vstack((np.ones((self.nPipes, self.nInlets)),
                              np.zeros((self.nPipes, self.nInlets))))
            a_out = np.vstack((np.zeros((self.nPipes, self.nOutlets)),
                               np.ones((self.nPipes, self.nOutlets))))
            a_b = np.zeros((2*self.nPipes, nSegments))

        elif self.config == 'series':
            # Coefficient matrices from continuity condition:
            # [b_u]*[T_{f,u}](z=0) = [b_d]*[T_{f,d}](z=0) + [b_b]*[T_b]
            b_d, b_u, b_b = self._continuity_condition(
                m_flow_borehole, cp_f, nSegments,
                segment_ratios=segment_ratios)

            # Intermediate coefficient matrices:
            # [T_{f,d}](z=0) = [c_in]*[T_{f,in}] + [c_u]*[T_{f,u}](z=0)
            c_in = np.eye(self.nPipes, M=1)
            c_u = np.eye(self.nPipes, k=-1)

            # Intermediate coefficient matrices:
            # [d_u]*[T_{f,u}](z=0) = [d_in]*[T_{f,in}] + [d_b]*[T_b]
            d_u = b_u - b_d @ c_u
            d_in = b_d @ c_in
            d_b = b_b
            d_u_m1 = np.linalg.inv(d_u)

            # Intermediate coefficient matrices:
            # [T_f](z=0) = [e_d]*[T_{f,d}](z=0) + [e_u]*[T_{f,u}](z=0)
            e_d = np.eye(2*self.nPipes, M=self.nPipes)
            e_u = np.eye(2*self.nPipes, M=self.nPipes, k=-self.nPipes)

            # Final coefficient matrices for temperatures at depth (z = 0):
            # [T_f](z=0) = [a_in]*[T_{f,in}]+[a_out]*[T_{f,out}]+[a_b]*[T_b]
            a_in = e_d @ (c_in + c_u @ d_u_m1 @ d_in) + e_u @ d_u_m1 @ d_in
            a_out = np.zeros((2*self.nPipes, self.nOutlets))
            a_b = e_d @ c_u @ d_u_m1 @ d_b + e_u @ d_u_m1 @ d_b
        else:
            raise NotImplementedError(f"Configuration '{self.config}' not "
                                      "implemented.")

        return a_in, a_out, a_b

    def _continuity_condition(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate fluid temperatures in downward
        and upward flowing pipes at depth (z = 0).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{a_{u}} \\mathbf{T_{f,u}}(z=0) =
                + \\mathbf{a_{d}} \\mathbf{T_{f,d}}(z=0)
                + \\mathbf{a_{b}} \\mathbf{T_{b}}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_d : (nPipes, nPipes,) array
            Array of coefficients for fluid temperature in downward flowing
            pipes.
        a_u : (nPipes, nPipes,) array
            Array of coefficients for fluid temperature in upward flowing
            pipes.
        a_b : (nPipes, nSegments,) array
            Array of coefficients for borehole wall temperature.

        """
        # Load coefficients
        sumA = self._sumA
        V = self._V
        Vm1 = self._Vm1
        L = self._L
        Dm1 = self._Dm1

        # Matrix exponential at depth (z = H)
        H = self.b.H
        E = np.real(V @ np.diag(np.exp(L*H)) @ Vm1)

        # Coefficient matrix for borehole wall temperatures
        IIm1 = np.hstack((np.eye(self.nPipes), -np.eye(self.nPipes)))
        a_b = np.zeros((self.nPipes, nSegments))
        z = self.b._segment_edges(nSegments, segment_ratios=segment_ratios)[::-1]
        exp_Lz = np.exp(np.multiply.outer(L, z))
        dexp_Lz = exp_Lz[:,:-1] - exp_Lz[:,1:]
        a_b = np.real(((IIm1 @ V @ Dm1) * (Vm1 @ sumA)) @ dexp_Lz)
            

        # Configuration-specific inlet and outlet coefficient matrices
        IZER = np.vstack((np.eye(self.nPipes),
                          np.zeros((self.nPipes, self.nPipes))))
        ZERI = np.vstack((np.zeros((self.nPipes, self.nPipes)),
                          np.eye(self.nPipes)))
        a_u = IIm1 @ E @ ZERI
        a_d = -IIm1 @ E @ IZER

        return a_d, a_u, a_b

    def _general_solution(
            self, z, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        General solution for fluid temperatures at a depth (z).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{a_{f0}} \\mathbf{T_{f}}(z=0)
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        z : float or (nDepths,) array
            Depths (in meters) to evaluate the fluid temperature coefficients.
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_f0 :
        (2*nPipes, 2*nPipes,) array, or (nDepths, 2*nPipes, 2*nPipes,) array
            Array of coefficients for inlet fluid temperature.
        a_b :
        (2*nPipes, nSegments,) array, or (nDepths, 2*nPipes, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        z_array = np.atleast_1d(z)
        # Check if model variables need to be updated
        self._check_model_variables(
            m_flow_borehole, cp_f, nSegments, segment_ratios)

        # Load coefficients
        sumA = self._sumA
        V = self._V
        Vm1 = self._Vm1
        L = self._L
        Dm1 = self._Dm1

        # Coefficient matrix for fluid temperatures at z=0
        a_f0 = np.real(
            (V * np.exp(np.multiply.outer(z_array, L))[:,np.newaxis,:]) @ Vm1)

        # Coefficient matrix for borehole wall temperatures
        a_b = np.zeros((len(z_array), 2*self.nPipes, nSegments))
        z_edges = self.b._segment_edges(
            nSegments, segment_ratios=segment_ratios)
        dz = np.maximum(np.subtract.outer(z_array, z_edges), 0.)
        exp_Lz = np.exp(np.multiply.outer(L, dz)).transpose(1, 0, 2)
        dexp_Lz = exp_Lz[:,:,1:] - exp_Lz[:,:,:-1]
        a_b = np.real(((V @ Dm1) * (Vm1 @ sumA)) @ dexp_Lz)
        # Remove first dimension if z is a scalar
        if np.isscalar(z):
            a_f0 = a_f0[0, :, :]
            a_b = a_b[0, :, :]

        return a_f0, a_b

    def _update_model_variables(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios):
        """
        Evaluate eigenvalues and eigenvectors for the system of differential
        equations.

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
        """

        nPipes = self.nPipes
        # Format mass flow rate and heat capacity inputs
        self._format_inputs(m_flow_borehole, cp_f, nSegments, segment_ratios)
        m_flow_pipe = self._m_flow_pipe
        cp_pipe = self._cp_pipe

        # Coefficient matrix for differential equations
        self._A = 1.0 / (self._Rd.T * m_flow_pipe * cp_pipe).T
        for i in range(2*nPipes):
            self._A[i, i] = -self._A[i, i] - sum(
                [self._A[i, j] for j in range(2*nPipes) if not i == j])
        for i in range(nPipes, 2*nPipes):
            self._A[i, :] = - self._A[i, :]
        self._sumA = np.sum(self._A, axis=1)
        # Eigenvalues and eigenvectors of A
        self._L, self._V = np.linalg.eig(self._A)
        # Inverse of eigenvector matrix
        self._Vm1 = np.linalg.inv(self._V)
        # Diagonal matrix of eigenvalues and inverse
        self._D = np.diag(self._L)
        self._Dm1 = np.diag(1./self._L)

    def _format_inputs(self, m_flow_borehole, cp_f, nSegments, segment_ratios):
        """
        Format mass flow rate and heat capacity inputs.

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
        """

        nPipes = self.nPipes
        # Format mass flow rate inputs
        # Mass flow rate in pipes
        if self.config.lower() == 'parallel':
            m_flow_pipe = np.tile(m_flow_borehole/nPipes, 2*self.nPipes)
        elif self.config.lower() == 'series':
            m_flow_pipe = np.tile(m_flow_borehole, 2*self.nPipes)
        self._m_flow_pipe = m_flow_pipe
        # Mass flow rate in each fluid circuit
        m_flow_in = np.atleast_1d(m_flow_borehole)
        self._m_flow_in = m_flow_in

        # Format heat capacity inputs
        # Heat capacity in each fluid circuit
        cp_in = np.atleast_1d(cp_f)
        self._cp_in = cp_in
        # Heat capacity in pipes
        cp_pipe = np.tile(cp_in, 2*self.nPipes)
        self._cp_pipe = cp_pipe


class IndependentMultipleUTube(MultipleUTube):
    """
    Class for multiple U-Tube boreholes with independent U-tubes.

    Contains information regarding the physical dimensions and thermal
    characteristics of the pipes and the grout material, as well as methods to
    evaluate fluid temperatures and heat extraction rates based on the work of
    Cimmino [#Cimmino2016b]_ for boreholes with any number of U-tubes.

    Attributes
    ----------
    pos : list of tuples
        Position (x, y) (in meters) of the pipes inside the borehole.
    r_in : float
        Inner radius (in meters) of the U-Tube pipes.
    r_out : float
        Outer radius (in meters) of the U-Tube pipes.
    borehole : Borehole object
        Borehole class object of the borehole containing the U-Tube.
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    R_fp : float
        Fluid to outer pipe wall thermal resistance (m-K/W).
    J : int, optional
        Number of multipoles per pipe to evaluate the thermal resistances.
        Default is 2.
    nPipes : int
        Number of U-Tubes.
    nInlets : int
        Total number of pipe inlets, equals to nPipes.
    nOutlets : int
        Total number of pipe outlets, equals to nPipes.

    Notes
    -----
    The expected array shapes of input parameters and outputs are documented
    for each class method. `nInlets` and `nOutlets` are the number of inlets
    and outlets to the borehole, and both are equal to the number of pipes.
    `nSegments` is the number of discretized segments along the borehole.
    `nPipes` is the number of pipes (i.e. the number of U-tubes) in the
    borehole. `nDepths` is the number of depths at which temperatures are
    evaluated.

    References
    ----------
    .. [#Cimmino2016b] Cimmino, M. (2016). Fluid and borehole wall temperature
       profiles in vertical geothermal boreholes with multiple U-tubes.
       Renewable Energy, 96, 137-147.

    """
    def __init__(self, pos, r_in, r_out, borehole, k_s,
                 k_g, R_fp, nPipes, J=2):
        self.pos = pos
        self.r_in = r_in
        self.r_out = r_out
        self.b = borehole
        self.k_s = k_s
        self.k_g = k_g
        self.R_fp = R_fp
        self.J = J
        self.nPipes = nPipes
        self.nInlets = nPipes
        self.nOutlets = nPipes
        self._check_geometry()

        # Delta-circuit thermal resistances
        self.update_thermal_resistances(self.R_fp)
        return

    def update_thermal_resistances(self, R_fp):
        """
        Update the delta-circuit of thermal resistances.

        This methods updates the values of the delta-circuit thermal
        resistances based on the provided fluid to outer pipe wall thermal
        resistance.

        Parameters
        ----------
        R_fp : float
            Fluid to outer pipe wall thermal resistance (m-K/W).

        """
        self.R_fp = R_fp
        # Delta-circuit thermal resistances
        self._Rd = thermal_resistances(
            self.pos, self.r_out, self.b.r_b, self.k_s, self.k_g, R_fp,
            J=self.J)[1]
        # Initialize stored_coefficients
        self._initialize_stored_coefficients()
        return

    def _continuity_condition_base(
            self, m_flow_borehole, cp_f, nSegments, segment_ratios=None):
        """
        Equation that satisfies equal fluid temperatures in both legs of
        each U-tube pipe at depth (z = H).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{a_{out}} T_{f,out} = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nOutlets, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_out : (nOutlets, nOutlets,) array
            Array of coefficients for outlet fluid temperature.
        a_b : (nOutlets, nSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # Check if model variables need to be updated
        self._check_model_variables(
            m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)

        # Coefficient matrices from continuity condition:
        # [b_u]*[T_{f,u}](z=0) = [b_d]*[T_{f,d}](z=0) + [b_b]*[T_b]
        a_in, a_out, a_b = self._continuity_condition(
            m_flow_borehole, cp_f, nSegments, segment_ratios=segment_ratios)

        return a_in, a_out, a_b

    def _continuity_condition_head(
            self, m_flow_borehole, cp, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate fluid temperatures at depth
        (z = 0). These coefficients take into account connections between
        U-tube pipes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z=0) = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{out}} \\mathbf{T_{f,out}}
                + \\mathbf{a_{b}} \\mathbf{T_{b}}

        Parameters
        ----------
        m_flow_borehole : float or (nInlets,) array
            Inlet mass flow rate (in kg/s) into the borehole.
        cp_f : float or (nInlets,) array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.
        segment_ratios : (nSegments,) array, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments of
            equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (2*nPipes, nInlets,) array
            Array of coefficients for inlet fluid temperature.
        a_out : (2*nPipes, nOutlets,) array
            Array of coefficients for outlet fluid temperature.
        a_b : (2*nPipes, nSegments,) array
            Array of coefficients for borehole wall temperature.

        """
        # Check if model variables need to be updated
        self._check_model_variables(
            m_flow_borehole, cp, nSegments, segment_ratios)

        a_in = np.eye(2*self.nPipes, M=self.nPipes, k=0)
        a_out = np.eye(2*self.nPipes, M=self.nPipes, k=-self.nPipes)
        a_b = np.zeros((2*self.nPipes, nSegments))

        return a_in, a_out, a_b

    def _format_inputs(self, m_flow_borehole, cp_f, nSegments, segment_ratios):
        """
        Format mass flow rate and heat capacity inputs.
        """
        # Format mass flow rate inputs
        # Mass flow rate in each fluid circuit
        m_flow_in = np.atleast_1d(m_flow_borehole)
        if not len(m_flow_in) == self.nInlets:
            raise ValueError(
                'Incorrect length of mass flow vector.')
        self._m_flow_in = m_flow_in
        # Mass flow rate in pipes
        m_flow_pipe = np.tile(m_flow_in, 2)
        self._m_flow_pipe = m_flow_pipe

        # Format heat capacity inputs
        # Heat capacity in each fluid circuit
        cp_in = np.atleast_1d(cp_f)
        if len(cp_in) == 1:
            cp_in = np.tile(cp_f, self.nInlets)
        elif not len(cp_in) == self.nInlets:
            raise ValueError(
                'Incorrect length of heat capacity vector.')
        self._cp_in = cp_in
        # Heat capacity in pipes
        cp_pipe = np.tile(cp_in, 2)
        self._cp_pipe = cp_pipe


class Coaxial(SingleUTube):
    """
    Class for coaxial boreholes.

    Contains information regarding the physical dimensions and thermal
    characteristics of the pipes and the grout material, as well as methods to
    evaluate fluid temperatures and heat extraction rates based on the work of
    Hellstrom [#Coaxial-Hellstrom1991]_.

    Attributes
    ----------
    pos : tuple
        Position (x, y) (in meters) of the pipes inside the borehole.
    r_in : (2,) array
        Inner radii (in meters) of the coaxial pipes. The first element of the
        array corresponds to the inlet pipe.
    r_out : (2,) array
        Outer radii (in meters) of the coaxial pipes. The first element of the
        array corresponds to the inlet pipe.
    borehole : Borehole object
        Borehole class object of the borehole containing the U-Tube.
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    R_ff : float
        Fluid to fluid thermal resistance of the inner pipe to the outer pipe
        (in m-K/W).
    R_fp : float
        Fluid to outer pipe wall thermal resistance of the outer pipe in
        contact with the grout (in m-K/W).
    J : int, optional
        Number of multipoles per pipe to evaluate the thermal resistances.
        Default is 2.
    nPipes : int
        Number of U-Tubes, equals to 1.
    nInlets : int
        Total number of pipe inlets, equals to 1.
    nOutlets : int
        Total number of pipe outlets, equals to 1.

    Notes
    -----
    The expected array shapes of input parameters and outputs are documented
    for each class method. `nInlets` and `nOutlets` are the number of inlets
    and outlets to the borehole, and both are equal to 1 for a coaxial
    borehole. `nSegments` is the number of discretized segments along the
    borehole. `nPipes` is the number of pipes (i.e. the number of U-tubes) in
    the borehole, equal to 1. `nDepths` is the number of depths at which
    temperatures are evaluated.

    References
    ----------
    .. [#Coaxial-Hellstrom1991] Hellstrom, G. (1991). Ground heat storage.
       Thermal Analyses of Duct Storage Systems I: Theory. PhD Thesis.
       University of Lund, Department of Mathematical Physics. Lund, Sweden.

    """
    def __init__(self, pos, r_in, r_out, borehole, k_s, k_g, R_ff, R_fp, J=2):
        if isinstance(pos, tuple):
            pos = [pos]
        self.pos = pos
        self.r_in = r_in
        self.r_out = r_out
        self.b = borehole
        self.k_s = k_s
        self.k_g = k_g
        self.R_ff = R_ff
        self.R_fp = R_fp
        self.J = J
        self.nPipes = 1
        self.nInlets = 1
        self.nOutlets = 1

        # Determine the indexes of the inner and outer pipes
        self._iInner = r_out.argmin()
        self._iOuter = r_out.argmax()

        self._check_geometry()

        # Delta-circuit thermal resistances
        self.update_thermal_resistances(self.R_ff, self.R_fp)
        return

    def update_thermal_resistances(self, R_ff, R_fp):
        """
        Update the delta-circuit of thermal resistances.

        This methods updates the values of the delta-circuit thermal
        resistances based on the provided fluid to fluid and fluid to outer
        pipe wall thermal resistances.

        Parameters
        ----------
        R_ff : float
            Fluid to fluid thermal resistance of the inner pipe to the outer
            pipe (in m-K/W).
        R_fp : float
            Fluid to outer pipe wall thermal resistance of the outer pipe in
            contact with the grout (in m-K/W).

        """
        self.R_ff = R_ff
        self.R_fp = R_fp
        # Outer pipe to borehole wall thermal resistance
        R_fg = thermal_resistances(
            self.pos, self.r_out[self._iOuter], self.b.r_b, self.k_s, self.k_g,
            R_fp, J=self.J)[1][0]
        # Delta-circuit thermal resistances
        self._Rd = np.zeros((2*self.nPipes, 2*self.nPipes))
        self._Rd[self._iInner, self._iInner] = np.inf
        self._Rd[self._iInner, self._iOuter] = R_ff
        self._Rd[self._iOuter, self._iInner] = R_ff
        self._Rd[self._iOuter, self._iOuter] = R_fg
        # Initialize stored_coefficients
        self._initialize_stored_coefficients()
        return

    def visualize_pipes(self):
        """
        Plot the cross-section view of the borehole.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).

        """
        # Configure figure and axes
        fig = _initialize_figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.axis('equal')
        _format_axes(ax)

        # Color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        lw = plt.rcParams['lines.linewidth']

        # Borehole wall outline
        ax.plot([-self.b.r_b, 0., self.b.r_b, 0.],
                [0., self.b.r_b, 0., -self.b.r_b],
                'k.', alpha=0.)
        borewall = plt.Circle(
            (0., 0.), radius=self.b.r_b, fill=False,
            color='k', linestyle='--', lw=lw)
        ax.add_patch(borewall)

        # Pipes
        for i, (pos, color) in enumerate(zip(self.pos, colors)):
            # Coordinates of pipes
            (x_in, y_in) = pos
            (x_out, y_out) = pos

            # Pipe outline (inlet)
            pipe_in_in = plt.Circle(
                (x_in, y_in), radius=self.r_in[0],
                fill=False, linestyle='-', color=color, lw=lw)
            pipe_in_out = plt.Circle(
                (x_in, y_in), radius=self.r_out[0],
                fill=False, linestyle='-', color=color, lw=lw)
            if self._iInner == 0:
                ax.text(x_in, y_in, i, ha="center", va="center")
            else:
                ax.text(x_in + 0.5 * (self.r_out[0] + self.r_in[1]), y_in, i,
                        ha="center", va="center")

            # Pipe outline (outlet)
            pipe_out_in = plt.Circle(
                (x_out, y_out), radius=self.r_in[1],
                fill=False, linestyle='-', color=color, lw=lw)
            pipe_out_out = plt.Circle(
                (x_out, y_out), radius=self.r_out[1],
                fill=False, linestyle='-', color=color, lw=lw)
            if self._iInner == 1:
                ax.text(x_out, y_out, i + self.nPipes, ha="center", va="center")
            else:
                ax.text(x_out + 0.5 * (self.r_out[0] + self.r_in[1]), y_out,
                        i + self.nPipes, ha="center", va="center")

            ax.add_patch(pipe_in_in)
            ax.add_patch(pipe_in_out)
            ax.add_patch(pipe_out_in)
            ax.add_patch(pipe_out_out)

        plt.tight_layout()

        return fig
    
    def _check_geometry(self):
        """ Verifies the inputs to the pipe object and raises an error if
            the geometry is not valid.
        """
        # Verify that thermal properties are greater than 0.
        if not self.k_s > 0.:
            raise ValueError(
                f'The ground thermal conductivity must be greater than zero. '
                f'A value of {self.k_s} was provided.')
        if not self.k_g > 0.:
            raise ValueError(
                f'The grout thermal conductivity must be greater than zero. '
                f'A value of {self.k_g} was provided.')
        if not np.all(self.R_ff) >= 0.:
            raise ValueError(
                f'The fluid to fluid thermal resistance must be'
                f'greater or equal to zero. '
                f'A value of {self.R_ff} was provided.')
        if not np.all(self.R_fp) > 0.:
            raise ValueError(
                f'The fluid to outer pipe wall thermal resistance must be'
                f'greater than zero. A value of {self.R_fp} was provided.')

        # Verify that the pipe radius is greater than zero.
        if not np.all(self.r_in) > 0.:
            raise ValueError(
                f'The pipe inner radius must be greater than zero. '
                f'A value of {self.r_in} was provided.')

        # Verify that the outer pipe radius is greater than the inner pipe
        # radius.
        if not np.all(np.greater(self.r_out, self.r_in)):
            raise ValueError(
                f'The pipe outer radius must be greater than the pipe inner '
                f'radius. A value of {self.r_out} was provided.')

        # Verify that the inner radius of the outer pipe is greater than the
        # outer radius of the inner pipe.
        if not np.greater(self.r_in[self._iOuter], self.r_out[self._iInner]):
            raise ValueError(
                'The inner radius of the outer pipe must be greater than the'
                ' outer radius of the inner pipe.')

        # Verify that the number of multipoles is zero or greater.
        if not self.J >= 0:
            raise ValueError(
                f'The number of terms in the multipole expansion must be zero '
                f'or greater. A value of {self.J} was provided.')

        # Verify that the pipes are contained within the borehole.
        for i in range(len(self.pos)):
            r_pipe = np.sqrt(self.pos[i][0]**2 + self.pos[i][1]**2)
            radii = r_pipe + self.r_out
            if not np.any(np.greater_equal(self.b.r_b, radii)):
                raise ValueError(
                    f'Pipes must be entirely contained within the borehole. '
                    f'Pipe {i} is partly or entirely outside the '
                    f'borehole.')

        return True


thermal_resistances_dict = {
    'pos': None, 'r_out': None, 'r_b': None, 'k_s': None, 'k_g': None,
    'R_fp': None, 'J': None, 'R': None, 'Rd': None}
def _compare_thermal_resistances_inputs(pos, r_out, r_b, k_s, k_g, R_fp, J, tol=1e-6):
    for arg in ('pos', 'r_out', 'r_b', 'k_s', 'k_g', 'R_fp', 'J'):
        if thermal_resistances_dict[arg] is None:
            return False
    if not (len(pos) == len(thermal_resistances_dict['pos']) and
            len(r_out) == len(thermal_resistances_dict['r_out']) and
            len(R_fp) == len(thermal_resistances_dict['R_fp'])):
        return False
    if (np.allclose(r_out, thermal_resistances_dict['r_out'], rtol=tol) and
        np.abs(r_b - thermal_resistances_dict['r_b']) / r_b < tol and
        np.abs(k_s - thermal_resistances_dict['k_s']) / k_s < tol and
        np.abs(k_g - thermal_resistances_dict['k_g']) / k_g < tol and
        np.allclose(R_fp, thermal_resistances_dict['R_fp']) and
        J == thermal_resistances_dict['J']):
        for (x, y), (x_ref, y_ref) in zip(
                pos, thermal_resistances_dict['pos']):
            if (np.abs(x - x_ref) > np.abs(x*tol) or
                np.abs(y - y_ref) > np.abs(y*tol)):
                return False
        return True
    return False
        
def thermal_resistances(pos, r_out, r_b, k_s, k_g, R_fp, J=2):
    """
    Evaluate thermal resistances and delta-circuit thermal resistances.

    This function evaluates the thermal resistances and delta-circuit thermal
    resistances between pipes in a borehole using the multipole method
    [#Claesson2011]_. Thermal resistances are defined by:

    .. math:: \\mathbf{T_f} - T_b = \\mathbf{R} \\cdot \\mathbf{Q_{p}}

    Delta-circuit thermal resistances are defined by:

    .. math::

        q_{p,i,j} = \\frac{T_{f,i} - T_{f,j}}{R^\\Delta_{i,j}}

        q_{p,i,i} = \\frac{T_{f,i} - T_b}{R^\\Delta_{i,i}}

    Parameters
    ----------
    pos : list
        List of positions (x,y) (in meters) of pipes around the center
        of the borehole.
    r_out : float or array
        Outer radius of the pipes (in meters).
    r_b : float
        Borehole radius (in meters).
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    R_fp : float or array
        Fluid-to-outer-pipe-wall thermal resistance (in m-K/W).
    J : int, optional
        Number of multipoles per pipe to evaluate the thermal resistances.
        J=1 or J=2 usually gives sufficient accuracy. J=0 corresponds to the
        line source approximation [#Hellstrom1991b]_.
        Default is 2.

    Returns
    -------
    R : array
        Thermal resistances (in m-K/W).
    Rd : array
        Delta-circuit thermal resistances (in m-K/W).

    Examples
    --------
    >>> pos = [(-0.06, 0.), (0.06, 0.)]
    >>> R, Rd = gt.pipes.thermal_resistances(pos, 0.01, 0.075, 2., 1., 0.1,
                                             J=0)
    R = [[ 0.36648149, -0.04855895],
         [-0.04855895,  0.36648149]]
    Rd = [[ 0.31792254, -2.71733044],
          [-2.71733044,  0.31792254]]

    References
    ----------
    .. [#Hellstrom1991b] Hellstrom, G. (1991). Ground heat storage. Thermal
       Analyses of Duct Storage Systems I: Theory. PhD Thesis. University of
       Lund, Department of Mathematical Physics. Lund, Sweden.
    .. [#Claesson2011] Claesson, J., & Hellstrom, G. (2011).
       Multipole method to calculate borehole thermal resistances in a borehole
       heat exchanger. HVAC&R Research, 17(6), 895-911.

    """
    # Number of pipes
    n_p = len(pos)
    # If r_out and/or Rfp are supplied as float, build arrays of size n_p
    if np.isscalar(r_out):
        r_out = np.ones(n_p)*r_out
    if np.isscalar(R_fp):
        R_fp = np.ones(n_p)*R_fp
    if _compare_thermal_resistances_inputs(pos, r_out, r_b, k_s, k_g, R_fp, J):
        return thermal_resistances_dict['R'], thermal_resistances_dict['Rd']

    R = np.zeros((n_p, n_p))
    if J == 0:
        # Line source approximation
        sigma = (k_g - k_s)/(k_g + k_s)
        for i in range(n_p):
            xi = pos[i][0]
            yi = pos[i][1]
            for j in range(n_p):
                xj = pos[j][0]
                yj = pos[j][1]
                if i == j:
                    # Same-pipe thermal resistance
                    r = np.sqrt(xi**2 + yi**2)
                    R[i, j] = R_fp[i] + 1./(2.*pi*k_g) \
                        *(np.log(r_b/r_out[i]) - sigma*np.log(1 - r**2/r_b**2))
                else:
                    # Pipe to pipe thermal resistance
                    r = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    ri = np.sqrt(xi**2 + yi**2)
                    rj = np.sqrt(xj**2 + yj**2)
                    dij = np.sqrt((1. - ri**2/r_b**2)*(1.-rj**2/r_b**2) +
                                  r**2/r_b**2)
                    R[i, j] = -1./(2.*pi*k_g) \
                        *(np.log(r/r_b) + sigma*np.log(dij))
    else:
        # Resistances from multipole method are evaluated from the solution of
        # n_p problems
        for m in range(n_p):
            Q_p = np.zeros(n_p)
            Q_p[m] = 1.0
            (T_f, T, it, eps_max) = multipole(pos, r_out, r_b, k_s, k_g,
                                              R_fp, 0., Q_p, J)
            R[:,m] = T_f

    # Delta-circuit thermal resistances
    K = -np.linalg.inv(R)
    for i in range(n_p):
        K[i, i] = -(K[i, i] +
                    sum([K[i, j] for j in range(n_p) if not i == j]))
    Rd = 1.0/K

    thermal_resistances_dict['pos'] = pos
    thermal_resistances_dict['r_out'] = r_out
    thermal_resistances_dict['r_b'] = r_b
    thermal_resistances_dict['k_s'] = k_s
    thermal_resistances_dict['k_g'] = k_g
    thermal_resistances_dict['R_fp'] = R_fp
    thermal_resistances_dict['J'] = J
    thermal_resistances_dict['R'] = R
    thermal_resistances_dict['Rd'] = Rd
    return R, Rd


def borehole_thermal_resistance(pipe, m_flow_borehole, cp_f):
    """
    Evaluate the effective borehole thermal resistance.

    Parameters
    ----------
    pipe : pipe object
        Model for pipes inside the borehole.
    m_flow_borehole : float
        Fluid mass flow rate (in kg/s) into the borehole.
    cp_f : float
        Fluid specific isobaric heat capacity (in J/kg.K)

    Returns
    -------
    R_b : float
        Effective borehole thermal resistance (m.K/W).

    """
    # This function is deprecated as of v2.2. It will be removed in v3.0.
    warnings.warn("`pygfunction.pipes.borehole_thermal_resistance` is "
                  "deprecated as of v2.2. It will be removed in v3.0. "
                  "Use the `_BasePipe.effective_borehole_thermal_resistance` "
                  "class method instead.",
                  DeprecationWarning)
    # Coefficient for T_{f,out} = a_out*T_{f,in} + [b_out]*[T_b]
    a_out = pipe.coefficients_outlet_temperature(
        m_flow_borehole, cp_f, nSegments=1)[0].item()
    # Coefficient for Q_b = [a_Q]*T{f,in} + [b_Q]*[T_b]
    a_Q = pipe.coefficients_borehole_heat_extraction_rate(
            m_flow_borehole, cp_f, nSegments=1)[0].item()
    # Borehole length
    H = pipe.b.H
    # Effective borehole thermal resistance
    R_b = -0.5*H*(1. + a_out)/a_Q

    return R_b


def fluid_friction_factor_circular_pipe(
        m_flow_pipe, r_in, mu_f, rho_f, epsilon, tol=1.0e-6):
    """
    Evaluate the Darcy-Weisbach friction factor.

    Parameters
    ----------
    m_flow_pipe : float
        Fluid mass flow rate (in kg/s) into the pipe.
    r_in : float
        Inner radius of the pipes (in meters).
    mu_f : float
        Fluid dynamic viscosity (in kg/m-s).
    rho_f : float
        Fluid density (in kg/m3).
    epsilon : float
        Pipe roughness (in meters).
    tol : float
        Relative convergence tolerance on Darcy friction factor.
        Default is 1.0e-6.

    Returns
    -------
    fDarcy : float
        Darcy friction factor.

    Examples
    --------

    """
    # Hydraulic diameter
    D = 2.*r_in
    # Relative roughness
    E = epsilon / D
    # Fluid velocity
    V_flow = m_flow_pipe / rho_f
    A_cs = pi * r_in**2
    V = V_flow / A_cs
    # Reynolds number
    Re = rho_f * V * D / mu_f

    if Re < 2.3e3:
        # Darcy friction factor for laminar flow
        fDarcy = 64.0 / Re
    else:
        # Colebrook-White equation for rough pipes
        fDarcy = 0.02
        df = 1.0e99
        while abs(df/fDarcy) > tol:
            one_over_sqrt_f = -2.0 * np.log10(E / 3.7
                                              + 2.51/(Re*np.sqrt(fDarcy)))
            fDarcy_new = 1.0 / one_over_sqrt_f**2
            df = fDarcy_new - fDarcy
            fDarcy = fDarcy_new

    return fDarcy


def convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon):
    """
    Evaluate the convective heat transfer coefficient for circular pipes.

    The Nusselt number must first be determined to find the convection
    coefficient. Determination of the Nusselt number in turbulent flow is done
    by calling :func:`_Nusselt_number_turbulent_flow`. An analytical solution
    for constant pipe wall surface temperature is used for laminar flow.

    Since :func:`_Nusselt_number_turbulent_flow` is only valid for Re > 3000.
    and to avoid dicontinuities in the values of the convective heat transfer
    coefficient near the onset of the turbulence region (approximately
    Re = 2300.), linear interpolation is used over the range 2300 < Re < 4000
    for the evaluation of the Nusselt number.

    This approach was verified by Gnielinski (2013)
    [#Gnielinksi2013]_.

    Parameters
    ----------
    m_flow_pipe : float
        Fluid mass flow rate (in kg/s) into the pipe.
    r_in : float
        Inner radius of the pipe (in meters).
    mu_f : float
        Fluid dynamic viscosity (in kg/m-s).
    rho_f : float
        Fluid density (in kg/m3).
    k_f : float
        Fluid thermal conductivity (in W/m-K).
    cp_f : float
        Fluid specific heat capacity (in J/kg-K).
    epsilon : float
        Pipe roughness (in meters).

    Returns
    -------
    h_fluid : float
        Convective heat transfer coefficient (in W/m2-K).

    Examples
    --------

    References
    -----------
    .. [#Gnielinksi2013] Gnielinski, V. (2013). On heat transfer in tubes.
        International Journal of Heat and Mass Transfer, 63, 134–140.
        https://doi.org/10.1016/j.ijheatmasstransfer.2013.04.015

    """
    # Hydraulic diameter
    D = 2.*r_in
    # Fluid velocity
    V_flow = m_flow_pipe / rho_f
    A_cs = pi * r_in**2
    V = V_flow / A_cs
    # Reynolds number
    Re = rho_f * V * D / mu_f
    # Prandtl number
    Pr = cp_f * mu_f / k_f
    # Darcy friction factor
    fDarcy = fluid_friction_factor_circular_pipe(
        m_flow_pipe, r_in, mu_f, rho_f, epsilon)

    # To ensure there are no dramatic jumps in the equation, an interpolation
    # in a transition region of 2300 <= Re <= 4000 will be used.
    # Cengel and Ghajar (2015, pg. 476) state that Re> 4000 is a conservative
    # value to consider the flow to be turbulent in piping networks.

    Re_crit_lower = 2300.
    Re_crit_upper = 4000.

    if Re >= Re_crit_upper:
        # Nusselt number from Gnielinski
        Nu = _Nusselt_number_turbulent_flow(Re, Pr, fDarcy)
    elif Re_crit_lower < Re:
        Nu_lam = 3.66  # constant surface temperature laminar Nusselt number
        # Nusselt number at the upper bound of the "transition" region between
        # laminar value and Gnielinski correlation (Re = 4000.)
        Nu_turb = _Nusselt_number_turbulent_flow(Re_crit_upper, Pr, fDarcy)
        # Interpolate between the laminar (Re = 2300.) and turbulent
        # (Re = 4000.) values. Equations (16)-(17) from Gnielinski (2013).
        gamma = (Re - Re_crit_lower) / (Re_crit_upper - Re_crit_lower)
        Nu = (1 - gamma) * Nu_lam + gamma * Nu_turb
    else:
        Nu = 3.66

    h_fluid = k_f * Nu / D

    return h_fluid


def convective_heat_transfer_coefficient_concentric_annulus(
        m_flow_pipe, r_a_in, r_a_out, mu_f, rho_f, k_f, cp_f, epsilon):
    """
    Evaluate the inner and outer convective heat transfer coefficient for the
    annulus region of a concentric pipe.

    Grundman (2007) referenced Hellström (1991) [#Hellstrom1991b]_ in the
    discussion about inner and outer convection coefficients in an annulus
    region of a concentric pipe arrangement.

    The following is valid for :math:`Re < 2300` and
    :math:`0.1 \leq Pr \leq 1000` :

        .. math::
            \\text{Nu}_{a,in} = 3.66 + 1.2(r^*)^{-0.8}

        .. math::
            \\text{Nu}_{a,out} = 3.66 + 1.2(r^*)^{0.5}

    where :math:`r^* = r_{a,in} / r_{a,out}` is the ratio of the inner over
    the outer annulus radius (Çengel and Ghajar 2015, pg. 476).

    Cengel and Ghajar (2015) [#ConvCoeff-CengelGhajar2015]_ state that inner
    and outer Nusselt numbers are approximately equivalent for turbulent flow.
    They additionally state that Gnielinski
    :func:`_Nusselt_number_turbulent_flow` can be used for turbulent flow.
    Linear interpolation is used over the range 2300 < Re < 4000 for the
    evaluation of the Nusselt number, as proposed by Gnielinski (2013)
    [#Gnielinksi2013]_.

    Parameters
    ----------
    m_flow_pipe: float
        Fluid mass flow rate (in kg/s) into the pipe.
    r_a_in: float
        Pipe annulus inner radius (in meters).
    r_a_out: float
        Pipe annulus outer radius (in meters).
    mu_f : float
        Fluid dynamic viscosity (in kg/m-s).
    rho_f : float
        Fluid density (in kg/m3).
    k_f : float
        Fluid thermal conductivity (in W/m-K).
    cp_f : float
        Fluid specific heat capacity (in J/kg-K).
    epsilon : float
        Pipe roughness (in meters).

    Returns
    -------
    h_fluid_a_in: float
        Convective heat transfer coefficient of the inner pipe annulus
        region (in W/m2-K).
    h_fluid_a_out: float
        Convective heat transfer coefficient of the outer pipe annulus
        region (in W/m2-K).

    References
    -----------
    .. [#Grundman2007] Grundman, R. (2007) Improved design methods for ground
        heat exchangers. Oklahoma State University, M.S. Thesis.
    .. [#ConvCoeff-CengelGhajar2015] Çengel, Y.A., & Ghajar, A.J. (2015). Heat
        and mass transfer: fundamentals & applications (Fifth edition.).
        McGraw-Hill.

    """
    # Hydraulic diameter and radius for concentric tube annulus region
    D_h = 2 * (r_a_out - r_a_in)
    r_h = D_h / 2
    # Cross-sectional area of the annulus region
    A_c = pi * ((r_a_out ** 2) - (r_a_in ** 2))
    # Volume flow rate
    V_dot = m_flow_pipe / rho_f
    # Average velocity
    V = V_dot / A_c
    # Reynolds number
    Re = rho_f * V * D_h / mu_f
    # Prandtl number
    Pr = cp_f * mu_f / k_f
    # Ratio of radii (Grundman, 2007)
    r_star = r_a_in / r_a_out
    # Darcy-Wiesbach friction factor
    fDarcy = fluid_friction_factor_circular_pipe(
        m_flow_pipe, r_h, mu_f, rho_f, epsilon)

    # To ensure there are no dramatic jumps in the equation, an interpolation
    # in a transition region of 2300 <= Re <= 4000 will be used.

    Re_crit_lower = 2300.
    Re_crit_upper = 4000.

    if Re >= Re_crit_upper:
        # Nusselt number from Gnielinski, applied to both surfaces if the
        # flow is turbulent
        Nu = _Nusselt_number_turbulent_flow(Re, Pr, fDarcy)
        Nu_a_in = Nu
        Nu_a_out = Nu
    elif Re_crit_lower < Re:
        # Inner and outer surfaces Nusselt numbers in the laminar region
        Nu_a_in_lam = 3.66 + 1.2 * r_star**(-0.8)
        Nu_a_out_lam = 3.66 + 1.2 * r_star**0.5
        # Nusselt number at the upper bound of the "transition" region between
        # laminar value and Gnielinski correlation (Re = 4000.)
        Nu_turb = _Nusselt_number_turbulent_flow(Re_crit_upper, Pr, fDarcy)
        # Interpolate between the laminar (Re = 2300.) and turbulent
        # (Re = 4000.) values. Equations (16)-(17) from Gnielinski (2013).
        gamma = (Re - Re_crit_lower) / (Re_crit_upper - Re_crit_lower)
        Nu_a_in = (1 - gamma) * Nu_a_in_lam + gamma * Nu_turb
        Nu_a_out = (1 - gamma) * Nu_a_out_lam + gamma * Nu_turb
    else:
        # Inner and outer surfaces Nusselt numbers in the laminar region
        Nu_a_in = 3.66 + 1.2 * r_star**(-0.8)
        Nu_a_out = 3.66 + 1.2 * r_star**0.5

    h_fluid_a_in = k_f * Nu_a_in / D_h
    h_fluid_a_out = k_f * Nu_a_out / D_h

    return h_fluid_a_in, h_fluid_a_out


def conduction_thermal_resistance_circular_pipe(r_in, r_out, k_p):
    """
    Evaluate the conduction thermal resistance for circular pipes.

    Parameters
    ----------
    r_in : float
        Inner radius of the pipes (in meters).
    r_out : float
        Outer radius of the pipes (in meters).
    k_p : float
        Pipe thermal conductivity (in W/m-K).

    Returns
    -------
    R_p : float
        Conduction thermal resistance (in m-K/W).

    Examples
    --------

    """
    R_p = np.log(r_out/r_in)/(2*pi*k_p)

    return R_p


def multipole(pos, r_out, r_b, k_s, k_g, R_fp, T_b, q_p, J,
              x_T=np.empty(0), y_T=np.empty(0),
              eps=1e-5, it_max=100):
    """
    Multipole method to calculate borehole thermal resistances in a borehole
    heat exchanger.

    Adapted from the work of Claesson and Hellstrom [#Claesson2011b]_.

    Parameters
    ----------
    pos : list
        List of positions (x,y) (in meters) of pipes around the center
        of the borehole.
    r_out : float or array
        Outer radius of the pipes (in meters).
    r_b : float
        Borehole radius (in meters).
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    R_fp : float or array
        Fluid-to-outer-pipe-wall thermal resistance (in m-K/W).
    J : int
        Number of multipoles per pipe to evaluate the thermal resistances.
        J=1 or J=2 usually gives sufficient accuracy. J=0 corresponds to the
        line source approximation.
    q_p : array
        Thermal energy flows (in W/m) from pipes.
    T_b : float
        Average borehole wall temperature (in degC).
    eps : float, optional
        Iteration relative accuracy.
        Default is 1e-5.
    it_max : int, optional
        Maximum number of iterations.
        Default is 100.
    x_T : array, optional
        x-coordinates (in meters) to calculate temperatures.
        Default is np.empty(0).
    y_T : array, optional
        y-coordinates (in meters) to calculate temperatures.
        Default is np.empty(0).

    Returns
    -------
    T_f : array
        Fluid temperatures (in degC) in the pipes.
    T : array
        Requested temperatures (in degC).
    it : int
        Total number of iterations
    eps_max : float
        Maximum error.

    References
    ----------
    .. [#Claesson2011b] Claesson, J., & Hellstrom, G. (2011).
       Multipole method to calculate borehole thermal resistances in a borehole
       heat exchanger. HVAC&R Research, 17(6), 895-911.

    """
    # Pipe coordinates in complex form
    n_p = len(pos)
    z_p = np.array([x + 1.j*y for (x, y) in pos])
    # If r_out and/or Rfp are supplied as float, build arrays of size n_p
    if np.isscalar(r_out):
        r_out = np.full(n_p, r_out)
    if np.isscalar(R_fp):
        R_fp = np.full(n_p, R_fp)

    # -------------------------------------
    # Thermal resistance matrix R0 (EQ. 33)
    # -------------------------------------
    pikg = 1.0 / (2.0*pi*k_g)
    sigma = (k_g - k_s)/(k_g + k_s)
    beta_p = 2*pi*k_g*R_fp
    rbm = r_b**2/(r_b**2 - np.abs(z_p)**2)
    R0 = np.diag(pikg * (np.log(r_b/r_out) + beta_p + sigma*np.log(rbm)))
    rbmn = r_b**2/np.abs(r_b**2 - np.multiply.outer(np.conj(z_p), z_p))
    dz = np.abs(np.subtract.outer(z_p + 1e-12, z_p))
    R0 = R0 + (1 - np.eye(n_p)) * pikg * (np.log(r_b / dz) + sigma * np.log(rbmn))

    # Initialize maximum error and iteration counter
    eps_max = 1.0e99
    it = 0
    # -------------------
    # Multipoles (EQ. 38)
    # -------------------
    if J > 0:
        P = np.zeros((n_p, J), dtype=np.cfloat)
        coeff = -np.array([[(1 - (k+1)*beta_m)/(1 + (k+1)*beta_m)
                           for k in range(J)] for beta_m in beta_p])
        while eps_max > eps and it < it_max:
            it += 1
            eps_max = 0.
            F = _F_mk(q_p, P, n_p, J, r_b, r_out, z_p, pikg, sigma)
            P_new = coeff*np.conj(F)
            if it == 1:
                diff0 = np.max(np.abs(P_new-P)) - np.min(np.abs(P_new-P))
            diff = np.max(np.abs(P_new-P)) - np.min(np.abs(P_new-P))
            eps_max = diff / (diff0 + 1e-30)
            P = P_new
    else:
        P = np.zeros((n_p, 0))

    # --------------------------
    # Fluid temperatures(EQ. 32)
    # --------------------------
    T_f = T_b + R0 @ q_p
    if J > 0:
        for j in range(J):
            dz = np.subtract.outer(z_p + 1e-12, z_p)
            T_f = T_f + np.real((1 - np.eye(n_p)) / dz**(j+1) @ (P[:,j] * r_out**(j+1)))
            zz = np.multiply.outer(np.conj(z_p), z_p)
            T_f = T_f + sigma * np.real(np.conj(z_p)**(j+1) * (1 / (r_b**2 - zz)**(j+1) @ (P[:,j] * r_out**(j+1))))

    # -------------------------------
    # Requested temperatures (EQ. 28)
    # -------------------------------
    n_T = len(x_T)
    T = np.zeros(n_T)
    for i, (x, y) in enumerate(zip(x_T, y_T)):
        z_T = x + 1.j*y
        dT0 = 0. + 0.j
        dTJ = 0. + 0.j
        for n, (z_n, r_out_n, q_n, T_n, P_n) in enumerate(
                zip(z_p, r_out, q_p, T_f, P)):
            if np.abs(z_T - z_n)/r_out_n < 1.0:
                # Coordinate inside pipe
                T[i] = T_n
                break
            # Order 0
            if np.abs(z_T) <= r_b:
                # Coordinate inside borehole
                W0 = np.log(r_b/(z_T - z_n)) \
                        + sigma*np.log(r_b**2/(r_b**2 - z_n*np.conj(z_T)))
            else:
                # Coordinate outside borehole
                W0 = (1. + sigma)*np.log(r_b/(z_T - z_n)) \
                        + sigma*(1. + sigma)/(1. - sigma)*np.log(r_b/z_T)
            dT0 += q_n*pikg*W0
            # Multipoles
            for j in range(J):
                if np.abs(z_T) <= r_b:
                    # Coordinate inside borehole
                    WJ = (r_out_n/(z_T - z_n))**(j+1) \
                            + sigma*((r_out_n*np.conj(z_T))
                                     /(r_b**2 - z_n*np.conj(z_T)))**(j+1)
                else:
                    # Coordinate outside borehole
                    WJ = (1. + sigma)*(r_out_n/(z_T - z_n))**(j+1)
                dTJ += P_n[j]*WJ
        else:
            T[i] += T_b + np.real(dT0 + dTJ)

    return T_f, T, it, eps_max


def _F_mk(q_p, P, n_p, J, r_b, r_out, z, pikg, sigma):
    """
    Complex matrix F_mk from Claesson and Hellstrom (2011), EQ. 34.

    Parameters
    ----------
    q_p : array
        Thermal energy flows (in W/m) from pipes.
    P : array
        Multipoles.
    n_p : int
        Total numper of pipes.
    J : int
        Number of multipoles per pipe to evaluate the thermal resistances.
        J=1 or J=2 usually gives sufficient accuracy. J=0 corresponds to the
        line source approximation.
    r_b : float
        Borehole radius (in meters).
    r_out : float or array
        Outer radius of the pipes (in meters).
    z : array
        Array of pipe coordinates in complex notation (x + 1.j*y). 
    pikg : float
        Inverse of 2*pi times the grout thermal conductivity, 1.0/(2.0*pi*k_g).
    sigma : array
        Dimensionless parameter for the ground and grout thermal
        conductivities, (k_g - k_s)/(k_g + k_s).

    Returns
    -------
    F : array
        Matrix F_mk from Claesson and Hellstrom (2011), EQ. 34.

    """
    F = np.zeros((n_p, J), dtype=np.cfloat)
    for k in range(J):
        # First term
        F[:,k] = F[:,k] + r_out**(k+1) * pikg / (k + 1) * ((1 - np.eye(n_p)) / np.add.outer(-z, z + 1e-12)**(k+1) @ q_p)
        # Second term
        F[:,k] = F[:,k] + sigma * r_out**(k+1) * pikg / (k + 1) * (1 / (r_b**2 - np.multiply.outer(z, np.conj(z)))**(k+1) @ (q_p * np.conj(z)**(k+1)))
        for j in range(J):
            # Third term
            F[:,k] = F[:,k] + binom(j+k+1, j) * (-r_out)**(k+1) * ((1 - np.eye(n_p)) / np.add.outer(z + 1e-12, -z)**(j+k+2) @ (P[:,j] * r_out**(j+1)))
            j_pend = min(k, j) + 2
            for jp in range(j_pend):
                F[:,k] = F[:,k] + sigma * binom(j+1, jp) * binom(j+k-jp+1, j) * r_out**(k+1) * z**(j+1-jp) * (1 / (r_b**2 - np.multiply.outer(z, np.conj(z)))**(k+j+2-jp) @ (np.conj(P[:,j]) * r_out**(j+1) * np.conj(z)**(k+1-jp)))

    return F


def _Nusselt_number_turbulent_flow(Re, Pr, fDarcy):
    """
    An empirical equation developed by Volker Gnielinski (1975)
    [#Gnielinski1975]_ based on experimental data for turbulent flow in pipes.
    Cengel and Ghajar (2015, pg. 497) [#Nusselt-CengelGhajar2015]_ say that the
    Gnielinski equation should be preferred for determining the Nusselt number
    in the transition and turbulent region.

    .. math::
        	\\text{Nu} = \\dfrac{(f/8)(\\text{Re}-1000)\\text{Pr}}
        	{1 + 12.7(f/8)^{0.5} (\\text{Pr}^{2/3}-1)} \\;\\;\\;
        	\\bigg(
            \\begin{array}{c}
                0.5 \leq \\text{Pr} \leq 2000 \\\\
                3 \\times 10^5 <  \\text{Re} < 5 \\times 10^6
            \\end{array}
            \\bigg)

    .. note::
        This equation does not apply to Re < 3000.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Pr : float
        Prandlt Number.
    fDarcy : float
        Darcy friction factor.

    Returns
    -------
    Nu : float
        The Nusselt number

    References
    ------------
    .. [#Gnielinski1975] Gnielinski, V. (1975). Neue Gleichungen für
        den Wärme- und den Stoffübergang in turbulent durchströmten Rohren und
        Kanälen. Forschung im Ingenieurwesen, 41(1), 8–16.
        https://doi.org/10.1007/BF02559682
    .. [#Nusselt-CengelGhajar2015] Çengel, Y.A., & Ghajar, A.J. (2015). Heat
        and mass transfer: fundamentals & applications (Fifth edition.).
        McGraw-Hill.

    """

    # Warn the user if the Reynolds number is out of bounds, but don't break
    if not 3.0E03 < Re < 5.0E06:
        warnings.warn(f'This Nusselt calculation is only valid for Reynolds '
                      f'number in the range of 3.0E03 < Re < 5.0E06, your '
                      f'value falls outside of the range at Re={Re:.4f}')

    # Warn the user if the Prandlt number is out of bounds
    if not 0.5 <= Pr <= 2000.:
        warnings.warn(f'This Nusselt calculation is only valid for Prandlt '
                      f'numbers in the range of 0.5 <= Pr <= 2000, your value '
                      f'falls outside of the range at Pr={Pr:.4f}')

    Nu = 0.125 * fDarcy * (Re - 1.0e3) * Pr / \
        (1.0 + 12.7 * np.sqrt(0.125*fDarcy) * (Pr**(2.0/3.0) - 1.0))
    return Nu
