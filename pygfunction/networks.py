# -*- coding: utf-8 -*-
from typing import Union, List

import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

from .borefield import Borefield
from .boreholes import Borehole
from .enums import PipeType
from .media import Fluid
from .pipes import SingleUTube, MultipleUTube, Coaxial
from .pipes import fluid_to_pipe_thermal_resistance, fluid_to_fluid_thermal_resistance


class Network(object):
    """
    Class for networks of boreholes with series, parallel, and mixed
    connections between the boreholes.

    Contains information regarding the physical dimensions and thermal
    characteristics of the pipes and the grout material in each borehole, the
    topology of the connections between boreholes, as well as methods to
    evaluate fluid temperatures and heat extraction rates based on the work of
    Cimmino (2018, 2019, 2024) [#Network-Cimmin2018]_, [#Network-Cimmin2019]_,
    [#Network-Cimmino2024]_.

    Attributes
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    pipes : list of pipe objects
        List of pipes included in the bore field.
    bore_connectivity : list, optional
        Index of fluid inlet into each borehole. -1 corresponds to a borehole
        connected to the bore field inlet. If this parameter is not provided,
        parallel connections between boreholes is used.
        Default is None.
    m_flow_network : float or array, optional
        Total mass flow rate into the network or inlet mass flow rates
        into each circuit of the network (in kg/s). If a float is supplied,
        the total mass flow rate is split equally into all circuits. This
        parameter is used to initialize the coefficients if it is provided.
        Default is None.
    cp_f : float, optional
        Fluid specific isobaric heat capacity (in J/kg.degC). This parameter is
        used to initialize the coefficients if it is provided.
        Default is None.
    nSegments : int, optional
        Number of line segments used per borehole. This parameter is used to
        initialize the coefficients if it is provided.
        Default is None.
    segment_ratios :
    (nSegments,) array or list of (nSegments[i],) arrays, optional
        Ratio of the borehole length represented by each segment. The sum of
        ratios must be equal to 1. If segment_ratios==None, segments of equal
        lengths are considered.
        Default is None.

    Notes
    -----
    The expected array shapes of input parameters and outputs are documented
    for each class method. `nInlets` and `nOutlets` are the number of inlets
    and outlets to the network, and both correspond to the number of parallel
    circuits. `nTotalSegments` is the sum of the number of discretized segments
    along every borehole. `nBoreholes` is the total number of boreholes in the
    network.

    References
    ----------
    .. [#Network-Cimmin2018] Cimmino, M. (2018). g-Functions for bore fields with
       mixed parallel and series connections considering the axial fluid
       temperature variations. Proceedings of the IGSHPA Sweden Research Track
       2018. Stockholm, Sweden. pp. 262-270.
    .. [#Network-Cimmin2019] Cimmino, M. (2019). Semi-analytical method for
       g-function calculation of bore fields with series- and
       parallel-connected boreholes. Science and Technology for the Built
       Environment, 25 (8), 1007-1022.
    .. [#Network-Cimmino2024] Cimmino, M. (2024). g-Functions for fields of
       series- and parallel-connected boreholes with variable fluid mass flow
       rate and reversible flow direction. Renewable Energy, 228, 120661.

    """
    def __init__(self, boreholes, pipes, bore_connectivity=None,
                 m_flow_network=None, cp_f=None, nSegments=None,
                 segment_ratios=None):
        self.b = boreholes
        self.H_tot = sum([b.H for b in self.b])
        self.nBoreholes = len(boreholes)
        self.p = pipes
        if bore_connectivity is None:
            bore_connectivity = [-1]*self.nBoreholes
        self.c = bore_connectivity
        self.m_flow_network = m_flow_network
        self.cp_f = cp_f

        # Verify that borehole connectivity is valid
        self._verify_bore_connectivity()
        self._find_inlets_outlets()

        # Initialize stored_coefficients
        self._initialize_coefficients_connectivity()
        self._initialize_stored_coefficients(
            m_flow_network, cp_f, nSegments, segment_ratios)

    @classmethod
    def from_static_params(cls,
                           boreholes: Union[List[Borehole], Borefield],
                           pipe_type_str: str,
                           pos: List[tuple],
                           r_in: Union[float, tuple, npt.ArrayLike],
                           r_out: Union[float, tuple, npt.ArrayLike],
                           k_s: float,
                           k_g: float,
                           k_p: Union[float, tuple, npt.ArrayLike],
                           m_flow_network: float,
                           epsilon: float,
                           fluid_str: str,
                           fluid_concentraton_percent: float,
                           fluid_temperature: float,
                           reversible_flow: bool = True,
                           bore_connectivity: list = None,
                           J: int = 2):
        """
        Constructs the 'Network' class from static parameters.

        Parameters
        ----------
        boreholes : list of Borehole objects
            List of boreholes included in the bore field.
        pipe_type_str : str
            Should be one of 'COAXIAL_ANNULAR_IN', 'COAXIAL_ANNULAR_OUT',
            'DOUBLE_UTUBE_PARALLEL', 'DOUBLE_UTUBE_SERIES', or 'SINGLE_UTUBE'.
        pos : list of tuples
            Position (x, y) (in meters) of the pipes inside the borehole.
        r_in : float
            Inner radius (in meters) of the U-Tube pipes.
        r_out : float
            Outer radius (in meters) of the U-Tube pipes.
        k_s : float
            Soil thermal conductivity (in W/m-K).
        k_g : float
            Grout thermal conductivity (in W/m-K).
        k_p : float, tuple, or (2,) array
            Pipe thermal conductivity (in W/m-K).
        m_flow_network : float
            Fluid mass flow rate into the network of boreholes (in kg/s).
        epsilon : float
            Pipe roughness (in meters).
        fluid_str: str
            The mixer for this application should be one of:

                - 'Water' - Complete water solution
                - 'MEG' - Ethylene glycol mixed with water
                - 'MPG' - Propylene glycol mixed with water
                - 'MEA' - Ethanol mixed with water
                - 'MMA' - Methanol mixed with water

        fluid_concentration_pct: float
            Mass fraction of the mixing fluid added to water (in %).
            Lower bound = 0. Upper bound is dependent on the mixture.
        fluid_temperature: float, optional
            Temperature used for evaluating fluid properties (in degC).
            Default is 20.
        reversible_flow : bool, optional
            True to treat a negative mass flow rate as the reversal of flow
            direction within the borehole. If False, the direction of flow is not
            reversed when the mass flow rate is negative, and the absolute value is
            used for calculations.
            Default is True.
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
        Network : 'Network' object.
            The network.

        """
        # Convert borefield to list
        if isinstance(boreholes, Borefield):
            boreholes = boreholes.to_boreholes()

        # The total fluid mass flow rate is divided equally amongst inlets
        if bore_connectivity is None:
            m_flow_borehole = abs(m_flow_network / len(boreholes))
        else:
            m_flow_borehole = abs(m_flow_network / bore_connectivity.count(-1))

        # Pipe and fluid types
        pipe_type = PipeType[pipe_type_str.upper()]
        fluid = Fluid(fluid_str, fluid_concentraton_percent, fluid_temperature)

        if pipe_type == PipeType.SINGLE_UTUBE:
            # Single U-tube borehole
            R_fp = fluid_to_pipe_thermal_resistance(
                pipe_type, m_flow_borehole, r_in, r_out, k_p, epsilon, fluid)
            pipes = [
                SingleUTube(
                    pos, r_in, r_out, borehole, k_s, k_g, R_fp, J, reversible_flow)
                for borehole in boreholes
                ]

        elif pipe_type == PipeType.DOUBLE_UTUBE_PARALLEL:
            # Double U-tube borehole (parallel)
            R_fp = fluid_to_pipe_thermal_resistance(
                pipe_type, m_flow_borehole, r_in, r_out, k_p, epsilon, fluid)
            pipes = [
                MultipleUTube(
                    pos, r_in, r_out, borehole, k_s, k_g, R_fp, 2, 'parallel', J, reversible_flow)
                for borehole in boreholes
                ]

        elif pipe_type == PipeType.DOUBLE_UTUBE_SERIES:
            # Double U-tube borehole (series)
            R_fp = fluid_to_pipe_thermal_resistance(
                pipe_type, m_flow_borehole, r_in, r_out, k_p, epsilon, fluid)
            pipes = [
                MultipleUTube(
                    pos, r_in, r_out, borehole, k_s, k_g, R_fp, 2, 'series', J, reversible_flow)
                for borehole in boreholes
                ]

        elif pipe_type in [PipeType.COAXIAL_ANNULAR_IN, PipeType.COAXIAL_ANNULAR_OUT]:
            # Coaxial borehole
            R_fp = fluid_to_pipe_thermal_resistance(
                pipe_type, m_flow_borehole, r_in, r_out, k_p, epsilon, fluid)
            R_ff = fluid_to_fluid_thermal_resistance(
                pipe_type, m_flow_borehole, r_in, r_out, k_p, epsilon, fluid)
            pipes = [
                Coaxial(
                    pos, np.array(r_in), np.array(r_out), borehole, k_s, k_g, R_ff, R_fp, J, reversible_flow)
                for borehole in boreholes
                ]

        else:
            raise ValueError(f"Unsupported pipe_type: '{pipe_type_str}'")

        return cls(boreholes=boreholes, pipes=pipes, m_flow_network=m_flow_network, bore_connectivity=bore_connectivity,
                   cp_f=fluid.cp)

    def get_inlet_temperature(
            self, T_f_in, T_b, m_flow_network, cp_f, nSegments,
            segment_ratios=None):
        """
        Returns the inlet fluid temperatures of all boreholes.

        Parameters
        ----------
        T_f_in : float or (1,) array
            Inlet fluid temperatures into network (in Celsius).
        T_b : float or (nTotalSegments,) array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        T_f_in : (nBoreholes,) array
            Inlet fluid temperature (in Celsius) into each borehole.

        """
        # Build coefficient matrices
        a_in, a_b = self.coefficients_inlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
        # Evaluate outlet temperatures
        if np.isscalar(T_b):
            T_b = np.tile(T_b, np.size(a_b, axis=1))
        T_f_in_borehole = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b
        return T_f_in_borehole

    def get_outlet_temperature(
            self, T_f_in, T_b, m_flow_network, cp_f, nSegments,
            segment_ratios=None):
        """
        Returns the outlet fluid temperatures of all boreholes.

        Parameters
        ----------
        T_f_in : float or (1,) array
            Inlet fluid temperatures into network (in Celsius).
        T_b : float or (nTotalSegments,) array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        T_f_out : (nBoreholes,) array
            Outlet fluid temperatures (in Celsius) from each borehole.

        """
        # Build coefficient matrices
        a_in, a_b = self.coefficients_outlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
        # Evaluate outlet temperatures
        if np.isscalar(T_b):
            T_b = np.tile(T_b, np.size(a_b, axis=1))
        T_f_out = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b
        return T_f_out

    def get_borehole_heat_extraction_rate(
            self, T_f_in, T_b, m_flow_network, cp_f, nSegments,
            segment_ratios=None):
        """
        Returns the heat extraction rates of all boreholes.

        Parameters
        ----------
        T_f_in : float or (1,) array
            Inlet fluid temperatures into network (in Celsius).
        T_b : float or (nTotalSegments,) array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        Q_b : (nTotalSegments,) array
            Heat extraction rates along each borehole segment (in Watts).

        """
        a_in, a_b = self.coefficients_borehole_heat_extraction_rate(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
        if np.isscalar(T_b):
            T_b = np.tile(T_b, np.size(a_b, axis=1))
        Q_b = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b

        return Q_b

    def get_fluid_heat_extraction_rate(
            self, T_f_in, T_b, m_flow_network, cp_f, nSegments,
            segment_ratios=None):
        """
        Returns the total heat extraction rates of all boreholes.

        Parameters
        ----------
        T_f_in : float or (1,) array
            Inlet fluid temperatures into network (in Celsius).
        T_b : float or (nTotalSegments,) array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        Q_f : (nBoreholes,) array
            Total heat extraction rates from each borehole (in Watts).

        """
        a_in, a_b = self.coefficients_fluid_heat_extraction_rate(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
        if np.isscalar(T_b):
            T_b = np.tile(T_b, np.size(a_b, axis=1))
        Q_f = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b

        return Q_f

    def get_network_inlet_temperature(
            self, Q_t, T_b, m_flow_network, cp_f, nSegments,
            segment_ratios=None):
        """
        Returns the inlet fluid temperature of the network.

        Parameters
        ----------
        Q_t : float or (1,) array
            Total heat extraction rate from the network (in Watts).
        T_b : float or (nTotalSegments,) array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        T_f_in : float or (1,) array
            Inlet fluid temperature (in Celsius) into the network. The returned
            type corresponds to the type of the parameter `Qt`.

        """
        # Build coefficient matrices
        a_q, a_b = self.coefficients_network_inlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
        # Evaluate outlet temperatures
        if np.isscalar(T_b):
            T_b = np.tile(T_b, np.size(a_b, axis=1))
        T_f_in = a_q @ np.atleast_1d(Q_t) + a_b @ T_b
        if np.isscalar(Q_t):
            T_f_in = T_f_in.item()
        return T_f_in

    def get_network_outlet_temperature(
            self, T_f_in, T_b, m_flow_network, cp_f, nSegments,
            segment_ratios=None):
        """
        Returns the outlet fluid temperature of the network.

        Parameters
        ----------
        T_f_in : float or (1,) array
            Inlet fluid temperatures into network (in Celsius).
        T_b : float or (nTotalSegments,) array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        T_f_out : float or (1,) array
            Outlet fluid temperature (in Celsius) from the network. The
            returned type corresponds to the type of the parameter `Tin`.

        """
        # Build coefficient matrices
        a_in, a_b = self.coefficients_network_outlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
        # Evaluate outlet temperatures
        if np.isscalar(T_b):
            T_b = np.tile(T_b, np.size(a_b, axis=1))
        T_f_out = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b
        if np.isscalar(T_f_in):
            T_f_out = T_f_out.item()
        return T_f_out

    def get_network_heat_extraction_rate(
            self, T_f_in, T_b, m_flow_network, cp_f, nSegments,
            segment_ratios=None):
        """
        Returns the total heat extraction rate of the network.

        Parameters
        ----------
        T_f_in : float or (1,) array
            Inlet fluid temperatures into network (in Celsius).
        T_b : float or (nTotalSegments,) array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        Q_t : float or (1,) array
            Heat extraction rate of the network (in Watts). The returned type
            corresponds to the type of the parameter `Tin`.

        """
        a_in, a_b = self.coefficients_network_heat_extraction_rate(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
        if np.isscalar(T_b):
            T_b = np.tile(T_b, np.size(a_b, axis=1))
        Q_t = a_in @ np.atleast_1d(T_f_in) + a_b @ T_b
        if np.isscalar(T_f_in):
            Q_t = Q_t.item()

        return Q_t

    def coefficients_inlet_temperature(
            self, m_flow_network, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate intlet fluid temperatures of all
        boreholes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,borehole,in}} =
                \\mathbf{a_{in}} T_{f,network,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nBoreholes, 1,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (nBoreholes, nTotalSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_inlet_temperature is 0
        method_id = 0
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(
                m_flow_network, cp_f, nSegments, segment_ratios)
            # Coefficient matrices for borehole inlet temperatures:
            # [T_{f,b,in}] = [c_in]*[T_{f,n,in}] + [c_out]*[T_{f,b,out}]
            c_in = self._c_in
            c_out = self._c_out
            # Coefficient matrices for borehole outlet temperatures:
            # [T_{f,b,out}] = [A]*[T_{f,b,in}] + [B]*[T_{b}]
            AB = list(zip(*[pipe.coefficients_outlet_temperature(
                m_flow, cp, nSegments, segment_ratios=ratios)
                for pipe, m_flow, cp, nSegments, ratios in zip(
                        self.p,
                        self._m_flow_borehole,
                        self._cp_borehole,
                        self.nSegments,
                        self._segment_ratios)]))
            A = block_diag(*AB[0])
            B = block_diag(*AB[1])
            # Coefficient matrices for borehole inlet temperatures:
            # [T_{f,b,in}] = [a_in]*[T_{f,n,in}] + [a_b]*[T_{b}]
            ICA = np.eye(self.nBoreholes) - c_out @ A
            a_in = np.linalg.solve(ICA, c_in)
            a_b = np.linalg.solve(ICA, c_out @ B)

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, (a_in, a_b),
                method_id)

        return a_in, a_b

    def coefficients_outlet_temperature(
            self, m_flow_network, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate outlet fluid temperatures of all
        boreholes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,borehole,out}} =
                \\mathbf{a_{in}} T_{f,network,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nBoreholes, 1,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (nBoreholes, nTotalSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_outlet_temperature is 1
        method_id = 1
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(
                m_flow_network, cp_f, nSegments, segment_ratios)
            # Coefficient matrices for borehole inlet temperatures:
            # [T_{f,b,in}] = [c_in]*[T_{f,n,in}] + [c_out]*[T_{f,b,out}]
            c_in = self._c_in
            c_out = self._c_out
            # Coefficient matrices for borehole outlet temperatures:
            # [T_{f,b,out}] = [A]*[T_{f,b,in}] + [B]*[T_{b}]
            AB = list(zip(*[pipe.coefficients_outlet_temperature(
                m_flow, cp, nSegments, segment_ratios=ratios)
                for pipe, m_flow, cp, nSegments, ratios in zip(
                        self.p,
                        self._m_flow_borehole,
                        self._cp_borehole,
                        self.nSegments,
                        self._segment_ratios)]))
            A = block_diag(*AB[0])
            B = block_diag(*AB[1])
            # Coefficient matrices for borehole outlet temperatures:
            # [T_{f,b,out}] = [a_in]*[T_{f,n,in}] + [a_b]*[T_{b}]
            IAC = np.eye(self.nBoreholes) - A @ c_out
            a_in = np.linalg.solve(IAC, A @ c_in)
            a_b = np.linalg.solve(IAC, B)

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, (a_in, a_b),
                method_id)

        return a_in, a_b

    def coefficients_network_inlet_temperature(
            self, m_flow_network, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate inlet fluid temperature of the
        network.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,network,in}} =
                \\mathbf{a_{q,f}} Q_{f}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        a_qf : (1, 1,) array
            Array of coefficients for total heat extraction rate.
        a_b : (1, nTotalSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_network_inlet_temperature is 2
        method_id = 2
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, method_id):
            a_qf, a_b = self._get_stored_coefficients(method_id)
        else:
            # Coefficient matrices for network heat extraction rates:
            # [Q_{tot}] = [b_in]*[T_{f,n,in}] + [b_b]*[T_{b}]
            b_in, b_b = self.coefficients_network_heat_extraction_rate(
                    m_flow_network, cp_f, nSegments,
                    segment_ratios=segment_ratios)
            # Coefficient matrices for network inlet temperature:
            # [T_{f,n,in}] = [a_qf]*[Q_{tot}] + [a_b]*[T_{b}]
            b_in_inv = np.linalg.inv(b_in)
            a_qf = b_in_inv
            a_b = -b_in_inv.dot(b_b)

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, (a_qf, a_b),
                method_id)

        return a_qf, a_b

    def coefficients_network_outlet_temperature(
            self, m_flow_network, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate outlet fluid temperature of the
        network.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,network,out}} =
                \\mathbf{a_{in}} T_{f,network,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (1, 1,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (1, nTotalSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_network_outlet_temperature is 3
        method_id = 3
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Coefficient matrices for borehole outlet temperatures:
            # [T_{f,b,out}] = [b_in]*[T_{f,n,in}] + [b_b]*[T_{b}]
            b_in, b_b = self.coefficients_outlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
            # Coefficient matrices for network outlet temperature:
            # [T_{f,n,out}] = [a_in]*[T_{f,n,in}] + [a_b]*[T_{b}]
            mix_out = self._coefficients_mixing(m_flow_network)
            a_in = mix_out @ b_in
            a_b = mix_out @ b_b

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, (a_in, a_b),
                method_id)

        return a_in, a_b

    def coefficients_borehole_heat_extraction_rate(
            self, m_flow_network, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate heat extraction rates of all
        boreholes segments.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_b} =
                \\mathbf{a_{in}} T_{f,network,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nTotalSegments, 1,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (nTotalSegments, nTotalSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_borehole_heat_extraction_rate is 4
        method_id = 4
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(
                m_flow_network, cp_f, nSegments, segment_ratios)
            # Coefficient matrices for borehole inlet temperatures:
            # [T_{f,b,in}] = [b_in]*[T_{f,n,in}] + [b_b]*[T_{b}]
            b_in, b_b = self.coefficients_inlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
            # Coefficient matrices for borehole heat extraction rates:
            # [Q_{b}] = [A]*[T_{f,b,in}] + [B]*[T_{b}]
            AB = list(zip(*[pipe.coefficients_borehole_heat_extraction_rate(
                m_flow, cp, nSegments, segment_ratios=ratios)
                for pipe, m_flow, cp, nSegments, ratios in zip(
                        self.p,
                        self._m_flow_borehole,
                        self._cp_borehole,
                        self.nSegments,
                        self._segment_ratios)]))
            A = block_diag(*AB[0])
            B = block_diag(*AB[1])
            # Coefficient matrices for borehole heat extraction rates:
            # [Q_{b}] = [a_in]*[T_{f,n,in}] + [a_b]*[T_{b}]
            a_in = A @ b_in
            a_b = A @ b_b + B

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, (a_in, a_b),
                method_id)

        return a_in, a_b

    def coefficients_fluid_heat_extraction_rate(
            self, m_flow_network, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate heat extraction rates of all
        boreholes.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_f} =
                \\mathbf{a_{in}} T_{f,network,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (nBoreholes, 1,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (nBoreholes, nTotalSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_fluid_heat_extraction_rate is 5
        method_id = 5
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(
                m_flow_network, cp_f, nSegments, segment_ratios)
            # Coefficient matrices for borehole inlet temperatures:
            # [T_{f,b,in}] = [b_in]*[T_{f,n,in}] + [b_b]*[T_{b}]
            b_in, b_b = self.coefficients_inlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
            # Coefficient matrices for fluid heat extraction rates:
            # [Q_{f}] = [A]*[T_{f,b,in}] + [B]*[T_{b}]
            AB = list(zip(*[pipe.coefficients_fluid_heat_extraction_rate(
                m_flow, cp, nSegments, segment_ratios=ratios)
                for pipe, m_flow, cp, nSegments, ratios in zip(
                        self.p,
                        self._m_flow_borehole,
                        self._cp_borehole,
                        self.nSegments,
                        self._segment_ratios)]))
            A = block_diag(*AB[0])
            B = block_diag(*AB[1])
            # Coefficient matrices for fluid heat extraction rates:
            # [Q_{f}] = [a_in]*[T_{f,n,in}] + [a_b]*[T_{b}]
            a_in = A @ b_in
            a_b = A @ b_b + B

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, (a_in, a_b),
                method_id)

        return a_in, a_b

    def coefficients_network_heat_extraction_rate(
            self, m_flow_network, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate total heat extraction rate of
        the network.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_network} =
                \\mathbf{a_{in}} T_{f,network,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (1, 1,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (1, nTotalSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_network_heat_extraction_rate is 6
        method_id = 6
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Coefficient matrices for fluid heat extraction rates:
            # [Q_{f}] = [b_in]*[T_{f,n,in}] + [b_b]*[T_{b}]
            b_in, b_b = self.coefficients_fluid_heat_extraction_rate(
                    m_flow_network, cp_f, nSegments,
                    segment_ratios=segment_ratios)
            # The total network heat extraction rate is the sum of heat
            # extraction rates from all boreholes:
            # [Q_{tot}] = [a_in]*[T_{f,n,in}] + [a_b]*[T_{b}]
            a_in = np.reshape(np.sum(b_in, axis=0), (1,-1))
            a_b = np.reshape(np.sum(b_b, axis=0), (1,-1))

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, (a_in, a_b),
                method_id)

        return a_in, a_b

    def _coefficients_mixing(self, m_flow_network):
        """
        Returns coefficients for the relation:

            .. math::

                T_{f,network,out} =
                \\mathbf{a_{out}} \\mathbf{T_{f,borehole,out}}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.

        Returns
        -------
        mix_out : (1, nOutlets,) array
            Array of coefficients for outlet fluid temperatures of all
            boreholes.

        """
        if not self._check_mixing_coefficients(m_flow_network):
            self._mix_out = np.zeros((1, self.nBoreholes))
            self._mix_out[0, self.iOutlets] = \
                np.abs(self._m_flow_in) / np.sum(np.abs(self._m_flow_in))
            self._mixing_m_flow = m_flow_network
        return self._mix_out

    def _initialize_coefficients_connectivity(self):
        """
        Initializes coefficients for the relation:

            .. math::

                \\mathbf{T_{f,borehole,in}} =
                \\mathbf{c_{in}} T_{f,network,in}
                + \\mathbf{c_{out}} \\mathbf{T_{f,borehole,out}}

        """
        # Flow in positive direction
        self._c_in_pos = np.zeros((self.nBoreholes, 1))
        self._c_out_pos = np.zeros((self.nBoreholes, self.nBoreholes))
        for i in range(self.nInlets):
            self._c_in_pos[self._iInlets_pos[i], 0] = 1.
        for i in range(self.nBoreholes):
            if not self.c[i] == -1:
                self._c_out_pos[i, self.c[i]] = 1.
        # Flow in negative direction
        self._c_in_neg = np.zeros((self.nBoreholes, 1))
        self._c_out_neg = np.zeros((self.nBoreholes, self.nBoreholes))
        for i in range(self.nInlets):
            self._c_in_neg[self._iInlets_neg[i], 0] = 1.
        for i in range(self.nBoreholes):
            if not self.c[i] == -1:
                self._c_out_neg[self.c[i], i] = 1.

        return

    def _initialize_stored_coefficients(
            self, m_flow_network, cp_f, nSegments, segment_ratios):
        nMethods = 7    # Number of class methods
        self._stored_coefficients = [() for _ in range(nMethods)]
        self._stored_m_flow_cp = [np.empty(self.nInlets)*np.nan
                                  for _ in range(nMethods)]
        self._stored_nSegments = [np.nan for _ in range(nMethods)]
        self._stored_segment_ratios = [np.nan for _ in range(nMethods)]
        self._m_flow_cp_model_variables = np.empty(self.nInlets)*np.nan
        self._nSegments_model_variables = np.nan
        self._segment_ratios_model_variables = np.nan
        self._mixing_m_flow = np.empty(self.nInlets)*np.nan*np.nan
        self._mixing_m_flow[:] = np.nan
        self._mix_out = np.empty((1, self.nBoreholes))*np.nan

        # If m_flow, cp_f, and nSegments are specified, evaluate and store all
        # matrix coefficients.
        if m_flow_network is not None and cp_f is not None and nSegments is not None:
            self.coefficients_inlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
            self.coefficients_outlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
            self.coefficients_network_inlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
            self.coefficients_network_outlet_temperature(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
            self.coefficients_borehole_heat_extraction_rate(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
            self.coefficients_fluid_heat_extraction_rate(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)
            self.coefficients_network_heat_extraction_rate(
                m_flow_network, cp_f, nSegments, segment_ratios=segment_ratios)

        return

    def _set_stored_coefficients(
            self, m_flow_network, cp_f, nSegments, segment_ratios,
            coefficients, method_id):
        self._stored_coefficients[method_id] = coefficients
        self._stored_m_flow_cp[method_id] = m_flow_network*cp_f
        self._stored_nSegments[method_id] = nSegments
        self._stored_segment_ratios[method_id] = segment_ratios

        return

    def _get_stored_coefficients(self, method_id):
        coefficients = self._stored_coefficients[method_id]

        return coefficients

    def _check_mixing_coefficients(self, m_flow_network, tol=1e-6):
        mixing_m_flow = self._mixing_m_flow
        if np.all(np.abs(m_flow_network - mixing_m_flow) < np.abs(mixing_m_flow)*tol):
            check = True
        else:
            check = False

        return check

    def _check_coefficients(
            self, m_flow_network, cp_f, nSegments, segment_ratios, method_id,
            tol=1e-6):
        stored_m_flow_cp = self._stored_m_flow_cp[method_id]
        stored_nSegments = self._stored_nSegments[method_id]
        stored_segment_ratios = self._stored_segment_ratios[method_id]
        if stored_segment_ratios is None and segment_ratios is None:
            check_ratios = True
        elif isinstance(stored_segment_ratios, list) and isinstance(segment_ratios, list):
            check_ratios = (np.all([len(segment_ratios[i]) == len(stored_segment_ratios[i]) for i in range(len(segment_ratios))]) and
                            np.all([np.all(np.abs(segment_ratios[i] - stored_segment_ratios[i]) < np.abs(stored_segment_ratios[i])*tol) for i in range(len(segment_ratios))]))
        else:
            check_ratios = False
        if (np.all(np.abs(m_flow_network*cp_f - stored_m_flow_cp) < np.abs(stored_m_flow_cp)*tol)
            and nSegments == stored_nSegments and check_ratios):
            check = True
        else:
            check = False

        return check

    def _format_inputs(self, m_flow_network, cp_f, nSegments, segment_ratios):
        """
        Format mass flow rate and heat capacity inputs.
        """
        # Format mass flow rate inputs
        # Mass flow rate in each fluid circuit
        m_flow_in = np.atleast_1d(m_flow_network)
        if len(m_flow_in) == 1:
            m_flow_in = np.tile(m_flow_network/self.nInlets, self.nInlets)
        elif not len(m_flow_in) == self.nInlets:
            raise ValueError(
                'Incorrect length of mass flow vector.')
        self._m_flow_in = m_flow_in
        # Flow direction
        if np.all(m_flow_network >= 0.):
            self._is_reversed = False
            self._c_in = self._c_in_pos
            self._c_out = self._c_out_pos
            self.iInlets = self._iInlets_pos
            self.iOutlets = self._iOutlets_pos
        elif np.all(m_flow_network <= 0.):
            self._is_reversed = True
            self._c_in = self._c_in_neg
            self._c_out = self._c_out_neg
            self.iInlets = self._iInlets_neg
            self.iOutlets = self._iOutlets_neg
        else:
            raise ValueError(
                'All elements of m_flow_network should be of the same sign.')

        # Format heat capacity inputs
        # Heat capacity in each fluid circuit
        cp_in = np.atleast_1d(cp_f)
        if len(cp_in) == 1:
            cp_in = np.tile(cp_f, self.nInlets)
        elif not len(cp_in) == self.nInlets:
            raise ValueError(
                'Incorrect length of heat capacity vector.')
        elif not np.all(cp_in == cp_in[0]):
            raise ValueError(
                'The heat capacity should be the same in all circuits.')
        self._cp_in = cp_in

        # Mass flow rate in boreholes
        m_flow_borehole = np.array([m_flow_in[i] for i in self.iCircuit])
        self._m_flow_borehole = m_flow_borehole
        # Heat capacity in boreholes
        cp_borehole = np.array([cp_in[i] for i in self.iCircuit])
        self._cp_borehole = cp_borehole

        # Format number of segments for each borehole
        nSeg = np.atleast_1d(nSegments)
        if len(nSeg) == 1:
            self.nSegments = [nSeg[0]] * self.nBoreholes
        elif not len(nSeg) == self.nBoreholes:
            raise ValueError(
                'Incorrect length of number of segments list.')
        else:
            self.nSegments = nSegments

        # Format segment ratios
        if segment_ratios is None:
            self._segment_ratios = [None] * self.nBoreholes
        elif isinstance(segment_ratios, np.ndarray):
            self._segment_ratios = [segment_ratios] * self.nBoreholes
        elif isinstance(segment_ratios, list):
            self._segment_ratios = segment_ratios
        else:
            raise ValueError(
                'Incorrect format of the segment ratios list.')


    def _find_inlets_outlets(self):
        """
        Finds the numbers of boreholes connected to the inlet and outlet of the
        network and the indices of the boreholes.

        This function raises an error if the supplied borehole connectivity is
        invalid.

        Parameters
        ----------
        bore_connectivity : list
            Index of fluid inlet into each borehole. -1 corresponds to a borehole
            connected to the bore field inlet.
        nBoreholes : int
            Number of boreholes in the bore field.

        """
        # Number and indices of inlets
        nInlets = self.c.count(-1)
        iInlets = [i for i in range(self.nBoreholes) if self.c[i] == -1]
        # Number and indices of outlets
        iOutlets = [i for i in range(self.nBoreholes) if i not in self.c]
        nOutlets = len(iOutlets)
        iCircuit = [iInlets.index(self._path_to_inlet(i)[-1])
                    for i in range(self.nBoreholes)]
        if not nInlets == nOutlets:
            raise ValueError(
                'The network should have as many inlets as outlets.')

        # Number of inlets and outlets in network
        self.nInlets = nInlets
        self.nOutlets = nOutlets
        # Indices of inlets and outlets in network
        self._iInlets_pos = iInlets
        self._iOutlets_pos = iOutlets
        self._iInlets_neg = iOutlets
        self._iOutlets_neg = iInlets
        # Indices of circuit of each borehole in network
        self.iCircuit = iCircuit

        return


    def _path_to_inlet(self, bore_index):
        """
        Returns the path from a borehole to the bore field inlet.

        Parameters
        ----------
        bore_index : int
            Index of borehole to evaluate path.

        Returns
        -------
        path : list
            List of boreholes leading to the bore field inlet, starting from
            borehole bore_index

        """
        # Initialize path
        path = [bore_index]
        # Index of borehole feeding into borehole (bore_index)
        index_in = self.c[bore_index]
        # Stop when bore field inlet is reached (index_in == -1)
        while not index_in == -1:
            # Add index of upstream borehole to path
            path.append(index_in)
            # Get index of next upstream borehole
            index_in = self.c[index_in]

        return path


    def _verify_bore_connectivity(self):
        """
        Verifies that borehole connectivity is valid.

        This function raises an error if the supplied borehole connectivity is
        invalid.

        """
        if not len(self.c) == self.nBoreholes:
            raise ValueError(
                'The length of the borehole connectivity list does not correspond '
                'to the number of boreholes in the bore field.')
        if max(self.c) >= self.nBoreholes:
            raise ValueError(
                'The borehole connectivity list contains borehole indices that '
                'are not part of the network.')
        # Cycle through each borehole and verify that connections lead to -1
        # (-1 is the bore field inlet) and that no two boreholes have the same
        # index of fluid inlet (except for -1).
        for index_in in self.c:
            n = 0 # Initialize step counter
            if index_in != -1 and self.c.count(index_in) > 1:
                raise ValueError(
                    'Two boreholes cannot have the same inlet, except fort the '
                    'network inlet (index of -1).')
            # Stop when bore field inlet is reached (index_in == -1)
            while not index_in == -1:
                index_in = self.c[index_in]
                n += 1 # Increment step counter
                # Raise error if n exceeds the number of boreholes
                if n > self.nBoreholes:
                    raise ValueError(
                        'The borehole connectivity list is invalid.')
        return


class _EquivalentNetwork(Network):
    """
    Class for networks of equivalent boreholes with parallel connections
    between the equivalent boreholes.

    Contains information regarding the physical dimensions and thermal
    characteristics of the pipes and the grout material in each boreholes, the
    topology of the connections between boreholes, as well as methods to
    evaluate fluid temperatures and heat extraction rates based on the work of
    Cimmino (2018, 2019, 2024) [#Network-Cimmin2018]_, [#Network-Cimmin2019]_,
    [#Network-Cimmin2024]_.

    Attributes
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    pipes : list of pipe objects
        List of pipes included in the bore field.
    m_flow_network : float or array, optional
        Total mass flow rate into the network or inlet mass flow rates
        into each circuit of the network (in kg/s). If a float is supplied,
        the total mass flow rate is split equally into all circuits. This
        parameter is used to initialize the coefficients if it is provided.
        Default is None.
    cp_f : float, optional
        Fluid specific isobaric heat capacity (in J/kg.degC). This parameter is
        used to initialize the coefficients if it is provided.
        Default is None.
    nSegments : int, optional
        Number of line segments used per borehole. This parameter is used to
        initialize the coefficients if it is provided.
        Default is None.
    segment_ratios :
    (nSegments,) array or list of (nSegments[i],) arrays, optional
        Ratio of the borehole length represented by each segment. The sum of
        ratios must be equal to 1. If segment_ratios==None, segments of equal
        lengths are considered.
        Default is None.

    Notes
    -----
    The expected array shapes of input parameters and outputs are documented
    for each class method. `nInlets` and `nOutlets` are the number of inlets
    and outlets to the network, and both correspond to the number of parallel
    circuits. `nTotalSegments` is the sum of the number of discretized segments
    along every borehole. `nBoreholes` is the total number of boreholes in the
    network.

    References
    ----------
    .. [#Network-Cimmin2018] Cimmino, M. (2018). g-Functions for bore fields with
       mixed parallel and series connections considering the axial fluid
       temperature variations. Proceedings of the IGSHPA Sweden Research Track
       2018. Stockholm, Sweden. pp. 262-270.
    .. [#Network-Cimmin2019] Cimmino, M. (2019). Semi-analytical method for
       g-function calculation of bore fields with series- and
       parallel-connected boreholes. Science and Technology for the Built
       Environment, 25 (8), 1007-1022.
    .. [#Network-Cimmino2024] Cimmino, M. (2024). g-Functions for fields of
       series- and parallel-connected boreholes with variable fluid mass flow
       rate and reversible flow direction. Renewable Energy, 228, 120661.

    """
    def __init__(self, equivalentBoreholes, pipes, m_flow_network=None,
                 cp_f=None, nSegments=None, segment_ratios=None):
        self.b = equivalentBoreholes
        self.H_tot = sum([b.H*b.nBoreholes for b in self.b])
        self.nBoreholes = len(equivalentBoreholes)
        self.wBoreholes = np.array(
            [[b.nBoreholes for b in equivalentBoreholes]]).T
        self.nBoreholes_total = np.sum(self.wBoreholes)
        self.p = pipes
        self.c = [-1]*self.nBoreholes
        self.m_flow_network = m_flow_network
        self.cp_f = cp_f

        # Verify that borehole connectivity is valid
        self._find_inlets_outlets()

        # Initialize stored_coefficients
        self._initialize_coefficients_connectivity()
        self._initialize_stored_coefficients(
            m_flow_network, cp_f, nSegments, segment_ratios)
        return

    def coefficients_network_heat_extraction_rate(
            self, m_flow_network, cp_f, nSegments, segment_ratios=None):
        """
        Build coefficient matrices to evaluate total heat extraction rate of
        the network.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_network} =
                \\mathbf{a_{in}} T_{f,network,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.
        segment_ratios :
        (nSegments,) array or list of (nSegments[i],) arrays, optional
            Ratio of the borehole length represented by each segment. The sum
            of ratios must be equal to 1. If segment_ratios==None, segments
            of equal lengths are considered.
            Default is None.

        Returns
        -------
        a_in : (1, 1,) array
            Array of coefficients for inlet fluid temperature.
        a_b : (1, nTotalSegments,) array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_network_heat_extraction_rate is 6
        method_id = 6
        # Check if stored coefficients are available
        if self._check_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Coefficient matrices for fluid heat extraction rates:
            # [Q_{f}] = [b_in]*[T_{f,n,in}] + [b_b]*[T_{b}]
            b_in, b_b = self.coefficients_fluid_heat_extraction_rate(
                    m_flow_network, cp_f, nSegments,
                    segment_ratios=segment_ratios)
            # The total network heat extraction rate is the sum of heat
            # extraction rates from all boreholes:
            # [Q_{tot}] = [a_in]*[T_{f,n,in}] + [a_b]*[T_{b}]
            a_in = np.reshape(self.wBoreholes.T @ b_in, (1,-1))
            a_b = np.reshape(self.wBoreholes.T @ b_b, (1,-1))

            # Store coefficients
            self._set_stored_coefficients(
                m_flow_network, cp_f, nSegments, segment_ratios, (a_in, a_b),
                method_id)

        return a_in, a_b

    def _coefficients_mixing(self, m_flow_network):
        """
        Returns coefficients for the relation:

            .. math::

                T_{f,network,out} =
                \\mathbf{a_{out}} \\mathbf{T_{f,borehole,out}}

        Parameters
        ----------
        m_flow_network : float or (nInlets,) array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.

        Returns
        -------
        mix_out : (1, nOutlets,) array
            Array of coefficients for outlet fluid temperatures of all
            boreholes.

        """
        if not self._check_mixing_coefficients(m_flow_network):
            self._mix_out = np.zeros((1, self.nBoreholes))
            m_flow_in = np.abs(self._m_flow_in)
            wBoreholes = self.wBoreholes.flatten()
            self._mix_out[0, self.iOutlets] = \
                m_flow_in * wBoreholes / np.sum(m_flow_in * wBoreholes)
            self._mixing_m_flow = m_flow_network
        return self._mix_out

    def _format_inputs(self, m_flow_network, cp_f, nSegments, segment_ratios):
        """
        Format mass flow rate and heat capacity inputs.
        """
        # Format mass flow rate inputs
        # Mass flow rate in each fluid circuit
        m_flow_in = np.atleast_1d(m_flow_network)
        if len(m_flow_in) == 1:
            m_flow_in = np.array(
                [m_flow_network/self.nBoreholes_total for b in self.b])
        elif not len(m_flow_in) == self.nInlets:
            raise ValueError(
                'Incorrect length of mass flow vector.')
        self._m_flow_in = m_flow_in
        # Flow direction
        self._is_reversed = m_flow_network < 0.
        if self._is_reversed:
            self._c_in = self._c_in_neg
            self._c_out = self._c_out_neg
            self.iInlets = self._iOutlets_neg
            self.iOutlets = self._iInlets_neg
        else:
            self._c_in = self._c_in_pos
            self._c_out = self._c_out_pos
            self.iInlets = self._iOutlets_pos
            self.iOutlets = self._iInlets_pos

        # Format heat capacity inputs
        # Heat capacity in each fluid circuit
        cp_in = np.atleast_1d(cp_f)
        if len(cp_in) == 1:
            cp_in = np.tile(cp_f, self.nInlets)
        elif not len(cp_in) == self.nInlets:
            raise ValueError(
                'Incorrect length of heat capacity vector.')
        elif not np.all(cp_in == cp_in[0]):
            raise ValueError(
                'The heat capacity should be the same in all circuits.')
        self._cp_in = cp_in

        # Mass flow rate in boreholes
        m_flow_borehole = np.array([m_flow_in[i] for i in self.iCircuit])
        self._m_flow_borehole = m_flow_borehole
        # Heat capacity in boreholes
        cp_borehole = np.array([cp_in[i] for i in self.iCircuit])
        self._cp_borehole = cp_borehole

        # Format number of segments for each borehole
        nSeg = np.atleast_1d(nSegments)
        if len(nSeg) == 1 and not isinstance(nSegments, list):
            self.nSegments = [nSegments] * self.nBoreholes
        elif not len(nSeg) == self.nBoreholes:
            raise ValueError(
                'Incorrect length of number of segments list.')
        else:
            self.nSegments = nSegments

        # Format segment ratios
        if segment_ratios is None:
            self._segment_ratios = [None] * self.nBoreholes
        elif isinstance(segment_ratios, np.ndarray):
            self._segment_ratios = [segment_ratios] * self.nBoreholes
        elif isinstance(segment_ratios, list):
            self._segment_ratios = segment_ratios
        else:
            raise ValueError(
                'Incorrect format of the segment ratios list.')


def network_thermal_resistance(network, m_flow_network, cp_f):
    """
    Evaluate the effective bore field thermal resistance.

    As proposed in Cimmino (2018, 2019) [#Network-Cimmin2018]_,
    [#Network-Cimmin2019]_.

    Parameters
    ----------
    network : network object
        Model of the network.
    m_flow_network : float or (nInlets, ) array
        Total mass flow rate into the network or inlet mass flow rates
        into each circuit of the network (in kg/s). If a float is supplied,
        the total mass flow rate is split equally into all circuits.
    cp_f : float
        Fluid specific isobaric heat capacity (in J/kg.degC).

    Returns
    -------
    R_field : float
        Effective bore field thermal resistance (m.K/W).

    """
    # Number of boreholes
    nBoreholes = len(network.b)

    # Total borehole length
    H_tot = network.H_tot


    # Coefficients for T_{f,out} = A_out*T_{f,in} + [B_out]*[T_b], and
    # Q_b = [A_Q]*T{f,in} + [B_Q]*[T_b]
    A_out, B_out = network.coefficients_network_outlet_temperature(
            m_flow_network, cp_f, 1)
    A_Q, B_Q = network.coefficients_network_heat_extraction_rate(
            m_flow_network, cp_f, 1)

    # Effective bore field thermal resistance
    R_field = -0.5*H_tot*(1. + A_out)/A_Q
    if not np.isscalar(R_field):
        R_field = R_field.item()

    return R_field
