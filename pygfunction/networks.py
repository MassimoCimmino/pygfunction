from __future__ import division, print_function, absolute_import

import numpy as np


class Network(object):
    """
    Network class.

    #TODO.

    Attributes
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    pipes : list of pipe objects
        #TODO.
    bore_connectivity : list
        Index of fluid inlet into each borehole. -1 corresponds to a borehole
        connected to the bore field inlet.

    """
    def __init__(self, boreholes, pipes, bore_connectivity):
        self.b = boreholes
        self.nBoreholes = len(boreholes)
        self.p = pipes
        self.c = bore_connectivity

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

    def coefficients_inlet_temperature(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate intlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,borehole,in}} = \\mathbf{a_{q,f}} \\mathbf{Q_{f}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a signle float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        a_qf : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """

        return a_qf, a_b

    def coefficients_outlet_temperature(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate outlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,borehole,out}} =
                \\mathbf{a_{in}} \\mathbf{T_{f,network,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a signle float can be supplied).
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

        return a_in, a_b

    def coefficients_network_inlet_temperature(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate intlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,network,in}} = \\mathbf{a_{q,f}} \\mathbf{Q_{f}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a signle float can be supplied).
        nSegments : int or list
            Number of borehole segments for each borehole. If an int is
            supplied, all boreholes are considered to have the same number of
            segments.

        Returns
        -------
        a_qf : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """

        return a_qf, a_b
#
    def coefficients_network_outlet_temperature(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate outlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,network,out}} =
                \\mathbf{a_{in}} \\mathbf{T_{f,network,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a signle float can be supplied).
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

        return a_in, a_b

    def coefficients_borehole_heat_extraction_rate(self,
                                                   m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_b} = \\mathbf{a_{in}} \\mathbf{T_{f,network,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a signle float can be supplied).
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

        return a_in, a_b

    def coefficients_fluid_heat_extraction_rate(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_f} = \\mathbf{a_{in}} \\mathbf{T_{f,network,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a signle float can be supplied).
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

        return a_in, a_b

    def coefficients_network_heat_extraction_rate(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_network} =
                \\mathbf{a_{in}} \\mathbf{T_{f,network,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Total mass flow rate into the network or inlet mass flow rates
            into each circuit of the network (in kg/s). If a float is supplied,
            the total mass flow rate is split equally into all circuits.
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
            Must be the same for all circuits (a signle float can be supplied).
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

        return a_in, a_b

    def _initialize_stored_coefficients(self):
        nMethods = 8    # Number of class methods
        self._stored_coefficients = [() for i in range(nMethods)]
        self._stored_m_flow_cp = [np.empty(self.nInlets)
                                  for i in range(nMethods)]
        self._stored_nSegments = [np.nan for i in range(nMethods)]
        self._m_flow_cp_model_variables = np.empty(self.nInlets)
        self._nSegments_model_variables = np.nan

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
            nSeg = [nSegments for i in range(self.nBoreholes)]
        elif not len(nSeg) == self.nBoreholes:
            raise ValueError(
                'Incorrect length of number of segments list.')
        self.nSegments = nSeg


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