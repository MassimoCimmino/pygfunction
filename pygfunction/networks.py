from __future__ import absolute_import, division, print_function

import numpy as np


class Network(object):
    """
    Class for networks of boreholes with series, parallel, and mixed
    connections between the boreholes.

    Contains information regarding the physical dimensions and thermal
    characteristics of the pipes and the grout material in each boreholes, the
    topology of the connections between boreholes, as well as methods to
    evaluate fluid temperatures and heat extraction rates based on the work of
    Cimmino [#Cimmino2018]_.

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
    m_flow : float or array, optional
        Total mass flow rate into the network or inlet mass flow rates
        into each circuit of the network (in kg/s). If a float is supplied,
        the total mass flow rate is split equally into all circuits. This
        parameter is used to initialize the coefficients if it is provided.
        Default is None.
    cp : float or array, optional
        Fluid specific isobaric heat capacity (in J/kg.degC). ust be the same
        for all circuits (a single float can be supplied). This parameter is
        used to initialize the coefficients if it is provided.
        Default is None.
    nSegments : int, optional
        Number of line segments used per borehole. This parameter is used to
        initialize the coefficients if it is provided.
        Default is None.
    

    References
    ----------
    .. [#Cimmino2018] Cimmino, M. (2018). g-Functions for bore fields with
       mixed parallel and series connections considering the axial fluid
       temperature variations. Proceedings of the IGSHPA Sweden Research Track
       2018. Stockholm, Sweden. pp. 262-270.

    """
    def __init__(self, boreholes, pipes, bore_connectivity=None, m_flow=None,
                 cp=None, nSegments=None):
        self.b = boreholes
        self.nBoreholes = len(boreholes)
        self.p = pipes
        if bore_connectivity is None:
            bore_connectivity = [-1]*self.nBoreholes
        self.c = bore_connectivity
        self.m_flow = m_flow
        self.cp = cp

        # Verify that borehole connectivity is valid
        _verify_bore_connectivity(bore_connectivity, self.nBoreholes)
        iInlets, nInlets, iOutlets, nOutlets, iCircuit = _find_inlets_outlets(
                bore_connectivity, self.nBoreholes)

        # Number of inlets and outlets in network
        self.nInlets = nInlets
        self.nOutlets = nOutlets
        # Indices of inlets and outlets in network
        self.iInlets = iInlets
        self.iOutlets = iOutlets
        # Indices of circuit of each borehole in network
        self.iCircuit = iCircuit

        # Initialize stored_coefficients
        self._initialize_stored_coefficients(m_flow, cp, nSegments)

    def get_inlet_temperature(self, Tin, Tb, m_flow, cp, nSegments):
        """
        Returns the inlet fluid temperatures of all boreholes.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures into network (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        Tin : array
            Inlet fluid temperature (in Celsius) into each borehole.

        """
        # Build coefficient matrices
        a_in, a_b = self.coefficients_inlet_temperature(
                m_flow, cp, nSegments)
        # Evaluate outlet temperatures
        if np.isscalar(Tb):
            Tb = np.tile(Tb, sum(self.nSegments))
        Tin_borehole = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()
        return Tin_borehole

    def get_outlet_temperature(self, Tin, Tb, m_flow, cp, nSegments):
        """
        Returns the outlet fluid temperatures of all boreholes.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures into network (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        Tout : array
            Outlet fluid temperatures (in Celsius) from each borehole.

        """
        # Build coefficient matrices
        a_in, a_b = self.coefficients_outlet_temperature(
                m_flow, cp, nSegments)
        # Evaluate outlet temperatures
        if np.isscalar(Tb):
            Tb = np.tile(Tb, sum(self.nSegments))
        Tout = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()
        return Tout

    def get_borehole_heat_extraction_rate(self, Tin, Tb, m_flow, cp, nSegments):
        """
        Returns the heat extraction rates of all boreholes.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures into network (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        Qb : float or array
            Heat extraction rates along each borehole segment (in Watts).

        """
        a_in, a_b = self.coefficients_borehole_heat_extraction_rate(
                m_flow, cp, nSegments)
        if np.isscalar(Tb):
            Tb = np.tile(Tb, sum(self.nSegments))
        Qb = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()

        return Qb

    def get_fluid_heat_extraction_rate(self, Tin, Tb, m_flow, cp, nSegments):
        """
        Returns the total heat extraction rates of all boreholes.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures into network (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        Qf : float or array
            Total heat extraction rates from each borehole (in Watts).

        """
        a_in, a_b = self.coefficients_fluid_heat_extraction_rate(
                m_flow, cp, nSegments)
        if np.isscalar(Tb):
            Tb = np.tile(Tb, sum(self.nSegments))
        Qf = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()

        return Qf

    def get_network_inlet_temperature(self, Qt, Tb, m_flow, cp, nSegments):
        """
        Returns the inlet fluid temperature of the network.

        Parameters
        ----------
        Qt : float or array
            Total heat extraction rate from the network (in Watts).
        Tb : float or array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        Tin : float or array
            Inlet fluid temperature (in Celsius) into the network.

        """
        # Build coefficient matrices
        a_in, a_b = self.coefficients_network_inlet_temperature(
                m_flow, cp, nSegments)
        # Evaluate outlet temperatures
        if np.isscalar(Tb):
            Tb = np.tile(Tb, sum(self.nSegments))
        Tin = a_in.dot(Qt).flatten() + a_b.dot(Tb).flatten()
        if np.isscalar(Qt):
            Tin = np.asscalar(Tin)
        return Tin

    def get_network_outlet_temperature(self, Tin, Tb, m_flow, cp, nSegments):
        """
        Returns the outlet fluid temperature of the network.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures into network (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        Tout : float or array
            Outlet fluid temperature (in Celsius) from the network.

        """
        # Build coefficient matrices
        a_in, a_b = self.coefficients_network_outlet_temperature(
                m_flow, cp, nSegments)
        # Evaluate outlet temperatures
        if np.isscalar(Tb):
            Tb = np.tile(Tb, sum(self.nSegments))
        Tout = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()
        if np.isscalar(Tin):
            Tout = np.asscalar(Tout)
        return Tout

    def get_network_heat_extraction_rate(self, Tin, Tb, m_flow, cp, nSegments):
        """
        Returns the total heat extraction rate of the network.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures into network (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius). If a float is supplied,
            the same temperature is applied to all segments of all boreholes.
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        Qt : float or array
            Heat extraction rate of the network (in Watts).

        """
        a_in, a_b = self.coefficients_network_heat_extraction_rate(
                m_flow, cp, nSegments)
        if np.isscalar(Tb):
            Tb = np.tile(Tb, sum(self.nSegments))
        Qt = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()

        return Qt

    def coefficients_inlet_temperature(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_inlet_temperature is 0
        method_id = 0
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(m_flow, cp, nSegments)
            B = [self.p[i].coefficients_outlet_temperature(
                    self._m_flow_borehole[i],
                    self._cp_borehole[i],
                    self.nSegments[i])
                 for i in range(self.nBoreholes)]
            C = [(np.eye(1), np.zeros((1, self.nSegments[i])))
                 for i in range(self.nBoreholes)]
            a_in, a_b = self._network_coefficients_from_pipe_coefficients(B, C)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_in, a_b),
                                          method_id)

        return a_in, a_b

    def coefficients_outlet_temperature(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_outlet_temperature is 1
        method_id = 1
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(m_flow, cp, nSegments)
            B = [self.p[i].coefficients_outlet_temperature(
                    self._m_flow_borehole[i],
                    self._cp_borehole[i],
                    self.nSegments[i])
                 for i in range(self.nBoreholes)]
            C = B
            a_in, a_b = self._network_coefficients_from_pipe_coefficients(B, C)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_in, a_b),
                                          method_id)

        return a_in, a_b

    def coefficients_network_inlet_temperature(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate intlet fluid temperature of the
        network.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,network,in}} =
                \\mathbf{a_{q,f}} Q_{f}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        a_qf : array
            Array of coefficients for total heat extraction rate.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_network_inlet_temperature is 2
        method_id = 2
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_qf, a_b = self._get_stored_coefficients(method_id)
        else:
            b_in, b_b = self.coefficients_network_heat_extraction_rate(
                    m_flow, cp, nSegments)
            b_in_inv = np.linalg.inv(b_in)
            a_qf = b_in_inv
            a_b = -b_in_inv.dot(b_b)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_qf, a_b),
                                          method_id)

        return a_qf, a_b

    def coefficients_network_outlet_temperature(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_network_outlet_temperature is 3
        method_id = 3
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            b_in, b_b = self.coefficients_outlet_temperature(m_flow, cp, nSegments)
            iOutlets = self.iOutlets
            m_flow = np.zeros((1,self.nBoreholes))
            m_flow[0,iOutlets] = self._m_flow_in/np.sum(self._m_flow_in)
            a_in = m_flow.dot(b_in)
            a_b = m_flow.dot(b_b)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_in, a_b),
                                          method_id)

        return a_in, a_b

    def coefficients_borehole_heat_extraction_rate(self,
                                                   m_flow, cp, nSegments):
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
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_borehole_heat_extraction_rate is 4
        method_id = 4
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(m_flow, cp, nSegments)
            B = [self.p[i].coefficients_outlet_temperature(
                    self._m_flow_borehole[i],
                    self._cp_borehole[i],
                    self.nSegments[i])
                 for i in range(self.nBoreholes)]
            C = [self.p[i].coefficients_borehole_heat_extraction_rate(
                    self._m_flow_borehole[i],
                    self._cp_borehole[i],
                    self.nSegments[i])
                 for i in range(self.nBoreholes)]
            a_in, a_b = self._network_coefficients_from_pipe_coefficients(B, C)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_in, a_b),
                                          method_id)

        return a_in, a_b

    def coefficients_fluid_heat_extraction_rate(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_fluid_heat_extraction_rate is 5
        method_id = 5
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(m_flow, cp, nSegments)
            B = [self.p[i].coefficients_outlet_temperature(
                    self._m_flow_borehole[i],
                    self._cp_borehole[i],
                    self.nSegments[i])
                 for i in range(self.nBoreholes)]
            C = [self.p[i].coefficients_fluid_heat_extraction_rate(
                    self._m_flow_borehole[i],
                    self._cp_borehole[i],
                    self.nSegments[i])
                 for i in range(self.nBoreholes)]
            a_in, a_b = self._network_coefficients_from_pipe_coefficients(B, C)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_in, a_b),
                                          method_id)

        return a_in, a_b

    def coefficients_network_heat_extraction_rate(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a single float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_network_heat_extraction_rate is 6
        method_id = 6
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # The total network heat extraction rate is the some of heat
            # extraction rates from all boreholes
            b_in, b_b = self.coefficients_fluid_heat_extraction_rate(
                    m_flow, cp, nSegments)
            a_in = np.reshape(np.sum(b_in, axis=0), (1,-1))
            a_b = np.reshape(np.sum(b_b, axis=0), (1,-1))

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_in, a_b),
                                          method_id)

        return a_in, a_b

    def _network_coefficients_from_pipe_coefficients(self,
            coefficients_outlet_temperature,
            coefficients_output_variable):
        """
        Build network coefficient matrices from list of pipe coefficients
        for the outlet temperatures and for the desired variable.

        This class method builds the coefficient matrices (a_in, a_b) of the
        system:

            [Desired_variable(network)] = [a_in]*T_{in,network} + [a_b]*[T_b]

        from the coefficients (b_in, b_b) and (c_in, c_b) of the systems:

            T_{out,borehole} = [b_in]*T_{in,borehole} + [b_b]*[T_b]

            [Desired_variable(borehole)] = [c_in]*T_{in,borehole} + [c_b]*[T_b]

        Parameters
        ----------
        coefficients_outlet_temperature : list of tuples of arrays
            List of tuples of coefficients (a_in, a_b) for outlet temperature
            of the boreholes from the pipe objects.
        coefficients_output_variable : list of tuples of arrays
            List of tuples of coefficients (a_in, a_b) for the desired variable
            from the pipe objects.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # Initialize lists of coefficients (A_in, A_b) for the desired
        # variable
        A_in = [np.zeros((np.shape(coefficients_output_variable[i][0])[0], 1)) for i in range(self.nBoreholes)]
        A_b = [[np.zeros((np.shape(coefficients_output_variable[i][1])[0], self.nSegments[j])) for j in range(self.nBoreholes)] for i in range(self.nBoreholes)]

        # Solve each circuit independently
        for iOutlet in self.iOutlets:
            # Identify path from inlet to outlet of current circuit
            outlet_to_inlet = _path_to_inlet(self.c, iOutlet)
            inlet_to_outlet = outlet_to_inlet[::-1]
            iInlet = inlet_to_outlet[0]
            # For boreholes connected to the network inlet, the coefficient
            # sub-matrices of (a_in, a_b) are equal to (c_in, c_c)
            A_in[iInlet] = coefficients_output_variable[iInlet][0]
            A_b[iInlet][iInlet] = coefficients_output_variable[iInlet][1]
            b_in_previous = coefficients_outlet_temperature[iInlet][0]

            # Progress towards outlet:
            # The value of the desired variable for any given borehole is
            # dependent on the inlet fluid temperature of the network and the
            # borehole wall temperatures of all boreholes in the path from the
            # given borehole to the inlet.
            for i in inlet_to_outlet[1:]:
                b_in_current = coefficients_outlet_temperature[i][0]
                c_in = coefficients_output_variable[i][0]
                A_in[i] = c_in.dot(b_in_previous)
                b_in_previous = b_in_current.dot(b_in_previous)

                A_b[i][i] = coefficients_output_variable[i][1]

                path_to_inlet = _path_to_inlet(self.c, i)
                d_in_previous = np.ones((1,1))
                for j in path_to_inlet[1:]:
                    b_in_current = coefficients_outlet_temperature[j][0]
                    b_b_current = coefficients_outlet_temperature[j][1]
                    A_b[i][j] = np.linalg.multi_dot([c_in, d_in_previous, b_b_current])
                    d_in_previous = b_in_current.dot(d_in_previous)
        a_in = np.vstack(A_in)
        a_b = np.block(A_b)

        return a_in, a_b

    def _initialize_stored_coefficients(self, m_flow, cp, nSegments):
        nMethods = 7    # Number of class methods
        self._stored_coefficients = [() for i in range(nMethods)]
        self._stored_m_flow_cp = [np.empty(self.nInlets)
                                  for i in range(nMethods)]
        self._stored_nSegments = [np.nan for i in range(nMethods)]
        self._m_flow_cp_model_variables = np.empty(self.nInlets)
        self._nSegments_model_variables = np.nan

        # If m_flow, cp, and nSegments are specified, evaluate and store all
        # matrix coefficients.
        if m_flow is not None and cp is not None and nSegments is not None:
            self.coefficients_inlet_temperature(m_flow, cp, nSegments)
            self.coefficients_outlet_temperature(m_flow, cp, nSegments)
            self.coefficients_network_inlet_temperature(m_flow, cp, nSegments)
            self.coefficients_network_outlet_temperature(m_flow, cp, nSegments)
            self.coefficients_borehole_heat_extraction_rate(m_flow, cp, nSegments)
            self.coefficients_fluid_heat_extraction_rate(m_flow, cp, nSegments)
            self.coefficients_network_heat_extraction_rate(m_flow, cp, nSegments)

        return

    def _set_stored_coefficients(self, m_flow, cp, nSegments, coefficients,
                                 method_id):
        self._stored_coefficients[method_id] = coefficients
        self._stored_m_flow_cp[method_id] = m_flow*cp
        self._stored_nSegments[method_id] = nSegments

        return

    def _get_stored_coefficients(self, method_id):
        coefficients = self._stored_coefficients[method_id]

        return coefficients

    def _check_coefficients(self, m_flow, cp, nSegments, method_id, tol=1e-6):
        stored_m_flow_cp = self._stored_m_flow_cp[method_id]
        stored_nSegments = self._stored_nSegments[method_id]
        if (np.allclose(m_flow*cp, stored_m_flow_cp, rtol=tol)
                and nSegments == stored_nSegments):
            check = True
        else:
            check = False

        return check

    def _format_inputs(self, m_flow, cp, nSegments):
        """
        Format mass flow rate and heat capacity inputs.
        """
        # Format mass flow rate inputs
        # Mass flow rate in each fluid circuit
        m_flow_in = np.atleast_1d(m_flow)
        if len(m_flow_in) == 1:
            m_flow_in = np.tile(m_flow/self.nInlets, self.nInlets)
        elif not len(m_flow_in) == self.nInlets:
            raise ValueError(
                'Incorrect length of mass flow vector.')
        self._m_flow_in = m_flow_in

        # Format heat capacity inputs
        # Heat capacity in each fluid circuit
        cp_in = np.atleast_1d(cp)
        if len(cp_in) == 1:
            cp_in = np.tile(cp, self.nInlets)
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
            self.nSegments = [nSegments for i in range(self.nBoreholes)]
        elif not len(nSeg) == self.nBoreholes:
            raise ValueError(
                'Incorrect length of number of segments list.')
        else:
            self.nSegments = nSegments


def network_thermal_resistance(network, m_flow, cp):
    """
    Evaluate the effective bore field thermal resistance.

    As proposed in [#Cimmino2018]_.

    Parameters
    ----------
    network : network object
        Model of the network.
    m_flow : float or array
        Total mass flow rate into the network or inlet mass flow rates
        into each circuit of the network (in kg/s). If a float is supplied,
        the total mass flow rate is split equally into all circuits.
    cp : float or array
        Fluid specific isobaric heat capacity (in J/kg.degC).
        Must be the same for all circuits (a single float can be supplied).

    Returns
    -------
    Rfield : float
        Effective bore field thermal resistance (m.K/W).

    """
    # Number of boreholes
    nBoreholes = len(network.b)

    # Total borehole length
    H_tot = sum([network.b[i].H for i in range(nBoreholes)])


    # Coefficients for T_{f,out} = A_out*T_{f,in} + [B_out]*[T_b], and
    # Q_b = [A_Q]*T{f,in} + [B_Q]*[T_b]
    A_out, B_out = network.coefficients_network_outlet_temperature(
            m_flow, cp, 1)
    A_Q, B_Q = network.coefficients_network_heat_extraction_rate(
            m_flow, cp, 1)

    # Effective bore field thermal resistance
    Rfield = -0.5*H_tot*(1. + A_out)/A_Q
    if not np.isscalar(Rfield):
        Rfield = np.asscalar(Rfield)

    return Rfield


def _find_inlets_outlets(bore_connectivity, nBoreholes):
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
    nInlets = bore_connectivity.count(-1)
    iInlets = [i for i in range(nBoreholes) if bore_connectivity[i]==-1]
    # Number and indices of outlets
    iOutlets = [i for i in range(nBoreholes) if i not in bore_connectivity]
    nOutlets = len(iOutlets)
    iCircuit = [iInlets.index(_path_to_inlet(bore_connectivity, i)[-1])
                for i in range(nBoreholes)]
    if not nInlets == nOutlets:
        raise ValueError(
            'The network should have as many inlets as outlets.')

    return iInlets, nInlets, iOutlets, nOutlets, iCircuit


def _path_to_inlet(bore_connectivity, bore_index):
    """
    Returns the path from a borehole to the bore field inlet.

    Parameters
    ----------
    bore_connectivity : list
        Index of fluid inlet into each borehole. -1 corresponds to a borehole
        connected to the bore field inlet.
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
    index_in = bore_connectivity[bore_index]
    # Stop when bore field inlet is reached (index_in == -1)
    while not index_in == -1:
        # Add index of upstream borehole to path
        path.append(index_in)
        # Get index of next upstream borehole
        index_in = bore_connectivity[index_in]

    return path


def _verify_bore_connectivity(bore_connectivity, nBoreholes):
    """
    Verifies that borehole connectivity is valid.

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
    if not len(bore_connectivity) == nBoreholes:
        raise ValueError(
            'The length of the borehole connectivity list does not correspond '
            'to the number of boreholes in the bore field.')
    if max(bore_connectivity) >= nBoreholes:
        raise ValueError(
            'The borehole connectivity list contains borehole indices that '
            'are not part of the network.')
    # Cycle through each borehole and verify that connections lead to -1
    # (-1 is the bore field inlet) and that no two boreholes have the same
    # index of fluid inlet (except for -1).
    for i in range(nBoreholes):
        n = 0 # Initialize step counter
        # Index of borehole feeding into borehole i
        index_in = bore_connectivity[i]
        if index_in != -1 and bore_connectivity.count(index_in) > 1:
            raise ValueError(
                'Two boreholes cannot have the same inlet, except fort the '
                'network inlet (index of -1).')
        # Stop when bore field inlet is reached (index_in == -1)
        while not index_in == -1:
            index_in = bore_connectivity[index_in]
            n += 1 # Increment step counter
            # Raise error if n exceeds the number of boreholes
            if n > nBoreholes:
                raise ValueError(
                    'The borehole connectivity list is invalid.')
    return
