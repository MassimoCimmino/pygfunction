from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.constants import pi
from scipy.special import binom

from .boreholes import _path_to_inlet, _verify_bore_connectivity


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

    """
    def __init__(self, borehole):
        self.b = borehole
        self.nPipes = 1
        self.nInlets = 1
        self.nOutlets = 1

    def get_temperature(self, z, Tin, Tb, m_flow, cp):
        """
        Returns the fluid temperatures of the borehole at a depth (z).

        Parameters
        ----------
        z : float or array
            Depths (in meters) to evaluate the fluid temperatures.
        Tin : float or array
            Inlet fluid temperatures (in Celsius).
        Tb : array
            Borehole wall temperatures (in Celsius).
        m_flow : float or array
            Inlet mass flow rates (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).

        Returns
        -------
        Tf : array
            Fluid temperature (in Celsius) in each pipe.

        """
        nSegments = len(np.atleast_1d(Tb))
        z_all = np.atleast_1d(z).flatten()
        Tf = np.zeros((len(z_all), 2*self.nPipes))
        for i in range(len(z_all)):
            zi = z_all[i]
            # Build coefficient matrices
            a_in, a_b = self.coefficients_temperature(zi,
                                                      m_flow,
                                                      cp,
                                                      nSegments)
            # Evaluate fluid temperatures
            Tf[i,:] = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()

        # Return 1d array if z was supplied as scalar
        if np.isscalar(z):
            Tf = Tf.flatten()
        return Tf

    def get_inlet_temperature(self, Qf, Tb, m_flow, cp):
        """
        Returns the outlet fluid temperatures of the borehole.

        Parameters
        ----------
        Qf : float or array
            Heat extraction from the fluid circuits (in Watts).
        Tb : float or array
            Borehole wall temperatures (in Celsius).
        m_flow : float or array
            Inlet mass flow rates (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).

        Returns
        -------
        Tin : float or array
            Inlet fluid temperatures (in Celsius) into each inlet pipe.

        """
        nSegments = len(np.atleast_1d(Tb))
        # Build coefficient matrices
        a_qf, a_b = self.coefficients_inlet_temperature(m_flow,
                                                        cp,
                                                        nSegments)
        # Evaluate outlet temperatures
        Tin = a_qf.dot(Qf).flatten() + a_b.dot(Tb).flatten()
        # Return float if Tin was supplied as scalar
        if np.isscalar(Qf) and not np.isscalar(Tin):
            Tin = np.asscalar(Tin)
        return Tin

    def get_outlet_temperature(self, Tin, Tb, m_flow, cp):
        """
        Returns the outlet fluid temperatures of the borehole.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius).
        m_flow : float or array
            Inlet mass flow rates (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).

        Returns
        -------
        Tout : float or array
            Outlet fluid temperatures (in Celsius) from each outlet pipe.

        """
        nSegments = len(np.atleast_1d(Tb))
        # Build coefficient matrices
        a_in, a_b = self.coefficients_outlet_temperature(m_flow,
                                                         cp,
                                                         nSegments)
        # Evaluate outlet temperatures
        Tout = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()
        # Return float if Tin was supplied as scalar
        if np.isscalar(Tin) and not np.isscalar(Tout):
            Tout = np.asscalar(Tout)
        return Tout

    def get_borehole_heat_extraction_rate(self, Tin, Tb, m_flow, cp):
        """
        Returns the heat extraction rates of the borehole.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius).
        m_flow : float or array
            Inlet mass flow rates (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).

        Returns
        -------
        Qb : float or array
            Heat extraction rates along each borehole segment (in Watts).

        """
        nSegments = len(np.atleast_1d(Tb))
        a_in, a_b = self.coefficients_borehole_heat_extraction_rate(m_flow,
                                                                    cp,
                                                                    nSegments)
        Qb = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()
        # Return float if Tb was supplied as scalar
        if np.isscalar(Tb) and not np.isscalar(Qb):
            Qb = np.asscalar(Qb)
        return Qb

    def get_fluid_heat_extraction_rate(self, Tin, Tb, m_flow, cp):
        """
        Returns the heat extraction rates of the borehole.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius).
        m_flow : float or array
            Inlet mass flow rates (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).

        Returns
        -------
        Qf : float or array
            Heat extraction rates from each fluid circuit (in Watts).

        """
        nSegments = len(np.atleast_1d(Tb))
        a_in, a_b = self.coefficients_fluid_heat_extraction_rate(m_flow,
                                                                 cp,
                                                                 nSegments)
        Qf = a_in.dot(Tin).flatten() + a_b.dot(Tb).flatten()
        # Return float if Tb was supplied as scalar
        if np.isscalar(Tin) and not np.isscalar(Qf):
            Qf = np.asscalar(Qf)
        return Qf

    def get_total_heat_extraction_rate(self, Tin, Tb, m_flow, cp):
        """
        Returns the total heat extraction rate of the borehole.

        Parameters
        ----------
        Tin : float or array
            Inlet fluid temperatures (in Celsius).
        Tb : float or array
            Borehole wall temperatures (in Celsius).
        m_flow : float or array
            Inlet mass flow rates (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).

        Returns
        -------
        Q : float
            Total net heat extraction rate of the borehole (in Watts).

        """
        Qf = self.get_fluid_heat_extraction_rate(Tin, Tb, m_flow, cp)
        Q = np.sum(Qf)
        return Q

    def coefficients_inlet_temperature(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate outlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,in}} = \\mathbf{a_{q,f}} \\mathbf{Q_{f}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Inlet mass flow rates (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_qf : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_inlet_temperature is 3
        method_id = 3
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_qf, a_b = self._get_stored_coefficients(method_id)
        else:
            # Coefficient matrices for fluid heat extraction rates:
            # [Q_{f}] = [b_in]*[T_{f,in}] + [b_b]*[T_{b}]
            b_in, b_b = self.coefficients_fluid_heat_extraction_rate(m_flow,
                                                                     cp,
                                                                     nSegments)
            b_in_m1 = np.linalg.inv(b_in)

            # Matrices for fluid heat extraction rates:
            # [T_{f,in}] = [a_qf]*[Q_{f}] + [a_b]*[T_{b}]
            a_qf = b_in_m1
            a_b = -b_in_m1.dot(b_b)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_qf, a_b),
                                          method_id)

        return a_qf, a_b

    def coefficients_outlet_temperature(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate outlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_{f,out}} = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Inlet mass flow rates (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_outlet_temperature is 4
        method_id = 4
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Check if _continuity_condition_base need to be called
            # method_id for _continuity_condition_base is 0
            if self._check_coefficients(m_flow, cp, nSegments, 0):
                b_in, b_out, b_b = self._get_stored_coefficients(0)
            else:
                # Coefficient matrices from continuity condition:
                # [b_out]*[T_{f,out}] = [b_in]*[T_{f,in}] + [b_b]*[T_b]
                b_in, b_out, b_b = self._continuity_condition_base(m_flow,
                                                                   cp,
                                                                   nSegments)

                # Store coefficients
                self._set_stored_coefficients(m_flow, cp, nSegments,
                                              (b_in, b_out, b_b), 0)

            # Final coefficient matrices for outlet temperatures:
            # [T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]
            b_out_m1 = np.linalg.inv(b_out)
            a_in = b_out_m1.dot(b_in)
            a_b = b_out_m1.dot(b_b)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_in, a_b),
                                          method_id)

        return a_in, a_b

    def coefficients_temperature(self, z, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate fluid temperatures at a depth
        (z).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperature coefficients.
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_temperature is 5
        method_id = 5

        # Coefficient matrices for outlet temperatures:
        # [T_{f,out}] = [b_in]*[T_{f,in}] + [b_b]*[T_b]
        b_in, b_b = self.coefficients_outlet_temperature(m_flow, cp, nSegments)

        # Check if _continuity_condition_head need to be called
        # method_id for _continuity_condition_head is 1
        if self._check_coefficients(m_flow, cp, nSegments, 1):
            c_in, c_out, c_b = self._get_stored_coefficients(1)
        else:
            # Coefficient matrices for temperatures at depth (z = 0):
            # [T_f](0) = [c_in]*[T_{f,in}] + [c_out]*[T_{f,out}]
            #                              + [c_b]*[T_b]
            c_in, c_out, c_b = self._continuity_condition_head(m_flow,
                                                               cp,
                                                               nSegments)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments,
                                          (c_in, c_out, c_b), 1)

        # Coefficient matrices from general solution:
        # [T_f](z) = [d_f0]*[T_f](0) + [d_b]*[T_b]
        d_f0, d_b = self._general_solution(z, m_flow, cp, nSegments)

        # Final coefficient matrices for temperatures at depth (z):
        # [T_f](z) = [a_in]*[T_{f,in}] + [a_b]*[T_b]
        a_in = d_f0.dot(c_in + c_out.dot(b_in))
        a_b = d_f0.dot(c_b + c_out.dot(b_b)) + d_b

        return a_in, a_b

    def coefficients_borehole_heat_extraction_rate(self,
                                                   m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_b} = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_borehole_heat_extraction_rate is 6
        method_id = 6

        nPipes = self.nPipes
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(m_flow, cp, nSegments)
            m_flow_pipe = self._m_flow_pipe
            cp_pipe = self._cp_pipe
            mcp = np.hstack((-m_flow_pipe[0:nPipes],
                             m_flow_pipe[-nPipes:]))*cp_pipe

            # Initialize coefficient matrices
            a_in = np.zeros((nSegments, self.nInlets))
            a_b = np.zeros((nSegments, nSegments))
            # Heat extraction rates are calculated from an energy balance on a
            # borehole segment.
            z1 = 0.
            aTf1, bTf1 = self.coefficients_temperature(z1,
                                                       m_flow,
                                                       cp,
                                                       nSegments)
            for i in range(nSegments):
                z2 = (i + 1) * self.b.H / nSegments
                aTf2, bTf2 = self.coefficients_temperature(z2,
                                                           m_flow,
                                                           cp,
                                                           nSegments)
                a_in[i, :] = mcp.dot(aTf1 - aTf2)
                a_b[i, :] = mcp.dot(bTf1 - bTf2)
                aTf1, bTf1 = aTf2, bTf2

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_in, a_b),
                                          method_id)

        return a_in, a_b

    def coefficients_fluid_heat_extraction_rate(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_f} = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # method_id for coefficients_fluid_heat_extraction_rate is 7
        method_id = 7
        # Check if stored coefficients are available
        if self._check_coefficients(m_flow, cp, nSegments, method_id):
            a_in, a_b = self._get_stored_coefficients(method_id)
        else:
            # Update input variables
            self._format_inputs(m_flow, cp, nSegments)

            # Coefficient matrices for outlet temperatures:
            # [T_{f,out}] = [b_in]*[T_{f,in}] + [b_b]*[T_b]
            b_in, b_b = self.coefficients_outlet_temperature(m_flow, cp,
                                                             nSegments)

            # Intermediate matrices for fluid heat extraction rates:
            # [Q_{f}] = [c_in]*[T_{f,in}] + [c_out]*[T_{f,out}]
            MCP = self._m_flow_in * self._cp_in
            c_in = -np.diag(MCP)
            c_out = np.diag(MCP)

            # Matrices for fluid heat extraction rates:
            # [Q_{f}] = [a_in]*[T_{f,in}] + [a_b]*[T_{b}]
            a_in = c_in + c_out.dot(b_in)
            a_b = c_out.dot(b_b)

            # Store coefficients
            self._set_stored_coefficients(m_flow, cp, nSegments, (a_in, a_b),
                                          method_id)

        return a_in, a_b

    def visualize_pipes(self):
        """
        Plot the cross-section view of the borehole.

        Returns
        -------
        fig : figure
            Figure object (matplotlib).
    
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator

        # Initialize figure
        LW = 1.5    # Line width
        FS = 12.    # Font size

        plt.rc('figure', figsize=(80.0/25.4, 80.0/25.4))
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Borehole wall outline
        borewall = plt.Circle((0., 0.), radius=self.b.r_b,
                              fill=False, linestyle='--', linewidth=LW)
        ax.add_patch(borewall)

        # Pipes
        for i in range(self.nPipes):
            # Coordinates of pipes
            (x_in, y_in) = self.pos[i]
            (x_out, y_out) = self.pos[i + self.nPipes]

            # Pipe outline (inlet)
            pipe_in_in = plt.Circle((x_in, y_in), radius=self.r_in,
                                    fill=False, linestyle='-', linewidth=LW)
            pipe_in_out = plt.Circle((x_in, y_in), radius=self.r_out,
                                     fill=False, linestyle='-', linewidth=LW)
            ax.text(x_in, y_in, i + 1,
                    ha="center", va="center", size=FS)

            # Pipe outline (outlet)
            pipe_out_in = plt.Circle((x_out, y_out), radius=self.r_in,
                                     fill=False, linestyle='-', linewidth=LW)
            pipe_out_out = plt.Circle((x_out, y_out), radius=self.r_out,
                                      fill=False, linestyle='-', linewidth=LW)
            ax.text(x_out, y_out, i + self.nPipes + 1,
                    ha="center", va="center", size=FS)

            ax.add_patch(pipe_in_in)
            ax.add_patch(pipe_in_out)
            ax.add_patch(pipe_out_in)
            ax.add_patch(pipe_out_out)

        # Configure figure axes
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.axis('equal')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
    
        return fig

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

    def _check_model_variables(self, m_flow, cp, nSegments, tol=1e-6):
        stored_m_flow_cp = self._m_flow_cp_model_variables
        stored_nSegments = self._nSegments_model_variables
        if (np.allclose(m_flow*cp, stored_m_flow_cp, rtol=tol)
                and nSegments == stored_nSegments):
            check = True
        else:
            self._update_model_variables(m_flow, cp, nSegments)
            self._m_flow_cp_model_variables = m_flow*cp
            self._nSegments_model_variables = nSegments
            check = False

        return check

    def _check_coefficients(self, m_flow, cp, nSegments, method_id, tol=1e-6):
        stored_m_flow_cp = self._stored_m_flow_cp[method_id]
        stored_nSegments = self._stored_nSegments[method_id]
        if (np.allclose(m_flow*cp, stored_m_flow_cp, rtol=tol)
                and nSegments == stored_nSegments):
            check = True
        else:
            check = False

        return check

    def _continuity_condition_base(self, m_flow, cp, nSegments):
        """ Returns coefficients for the relation
            [a_out]*[T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]
        """
        raise NotImplementedError(
            '_continuity_condition_base class method not implemented, '
            'this method should return matrices for the relation: '
            '[a_out]*[T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]')

    def _continuity_condition_head(self, m_flow, cp, nSegments):
        """ Returns coefficients for the relation
            [T_f](z=0) = [a_in]*[T_{f,in}] + [a_out]*[T_{f,out}] + [a_b]*[T_b]
        """
        raise NotImplementedError(
            '_continuity_condition_head class method not implemented, '
            'this method should return matrices for the relation: '
            '[T_f](z=0) = [a_in]*[T_{f,in}] + [a_out]*[T_{f,out}] '
            '+ [a_b]*[T_b]')

    def _general_solution(self, z, m_flow, cp, nSegments):
        """ Returns coefficients for the relation
            [T_f](z) = [a_f0]*[T_f](0) + [a_b]*[T_b]
        """
        raise NotImplementedError(
            '_general_solution class method not implemented, '
            'this method should return matrices for the relation: '
            '[T_f](z) = [a_f0]*[T_f](0) + [a_b]*[T_b]')

    def _update_model_variables(self, m_flow, cp, nSegments):
        """
        Evaluate common coefficients needed in other class methods.
        """
        raise NotImplementedError(
            '_update_coefficients class method not implemented, '
            'this method should evaluate common coefficients needed in other '
            'class methods.')

    def _format_inputs(self, m_flow, cp, nSegments):
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
    Hellstrom [#Hellstrom1991]_.

    Attributes
    ----------
    pos : list of tuples
        Position (x, y) (in meters) of the pipes inside the borehole.
    r_in : float
        Inner radius (in meters) of the U-Tube pipes.
    r_out : float
        Outter radius (in meters) of the U-Tube pipes.
    borehole : Borehole object
        Borehole class object of the borehole containing the U-Tube.
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    R_fp : float
        Fluid to outter pipe wall thermal resistance (m-K/W).
    J : int, optional
        Number of multipoles per pipe to evaluate the thermal resistances.
        Default is 2.
    nPipes : int
        Number of U-Tubes, equals to 1.
    nInlets : int
        Total number of pipe inlets, equals to 1.
    nOutlets : int
        Total number of pipe outlets, equals to 1.

    References
    ----------
    .. [#Hellstrom1991] Hellstrom, G. (1991). Ground heat storage. Thermal
       Analyses of Duct Storage Systems I: Theory. PhD Thesis. University of
       Lund, Department of Mathematical Physics. Lund, Sweden.

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

        # Delta-circuit thermal resistances
        self._Rd = thermal_resistances(pos, r_out, borehole.r_b,
                                       k_s, k_g, self.R_fp, J=self.J)[1]

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

    def _continuity_condition_base(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_out : array
            Array of coefficients for outlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # Check if model variables need to be updated
        self._check_model_variables(m_flow, cp, nSegments)

        # Evaluate coefficient matrices from Hellstrom (1991):
        a_in = ((self._f1(self.b.H) + self._f2(self.b.H))
                / (self._f3(self.b.H) - self._f2(self.b.H)))
        a_in = np.array([[a_in]])

        a_out = np.array([[1.0]])

        a_b = np.zeros((self.nOutlets, nSegments))
        for i in range(nSegments):
            z1 = (nSegments - i - 1) * self.b.H / nSegments
            z2 = (nSegments - i) * self.b.H / nSegments
            dF4 = self._F4(z2) - self._F4(z1)
            dF5 = self._F5(z2) - self._F5(z1)
            a_b[0, i] = (dF4 + dF5) / (self._f3(self.b.H) - self._f2(self.b.H))

        return a_in, a_out, a_b

    def _continuity_condition_head(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_out : array
            Array of coefficients for outlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperature.

        """
        # Check if model variables need to be updated
        self._check_model_variables(m_flow, cp, nSegments)

        # There is only one pipe
        a_in = np.array([[1.0], [0.0]])
        a_out = np.array([[0.0], [1.0]])
        a_b = np.zeros((2, nSegments))

        return a_in, a_out, a_b

    def _general_solution(self, z, m_flow, cp, nSegments):
        """
        General solution for fluid temperatures at a depth (z).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{a_{f0}} \\mathbf{T_{f}}(z=0)
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_f0 : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # Check if model variables need to be updated
        self._check_model_variables(m_flow, cp, nSegments)

        a_f0 = np.array([[self._f1(z), self._f2(z)],
                        [-self._f2(z), self._f3(z)]])

        a_b = np.zeros((2*self.nPipes, nSegments))
        N = int(np.ceil(z/self.b.H*nSegments))
        for i in range(N):
            z1 = z - min((i+1)*self.b.H/nSegments, z)
            z2 = z - i * self.b.H / nSegments
            dF4 = self._F4(z2) - self._F4(z1)
            dF5 = self._F5(z2) - self._F5(z1)
            a_b[0, i] = dF4
            a_b[1, i] = -dF5

        return a_f0, a_b

    def _update_model_variables(self, m_flow, cp, nSegments):
        """
        Evaluate dimensionless resistances for Hellstrom (1991) solution.
        """
        # Format mass flow rate and heat capacity inputs
        self._format_inputs(m_flow, cp, nSegments)
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

    def _format_inputs(self, m_flow, cp, nSegments):
        """
        Format mass flow rate and heat capacity inputs.
        """
        # Format mass flow rate inputs
        if np.isscalar(m_flow):
            # Mass flow rate in each fluid circuit
            m_flow_in = m_flow*np.ones(self.nInlets)
        else:
            # Mass flow rate in each fluid circuit
            m_flow_in = m_flow
        self._m_flow_in = m_flow_in
        # Mass flow rate in pipes
        m_flow_pipe = np.tile(m_flow_in, 2*self.nPipes)
        self._m_flow_pipe = m_flow_pipe

        # Format heat capacity inputs
        if np.isscalar(cp):
            # Heat capacity in each fluid circuit
            cp_in = cp*np.ones(self.nInlets)
        else:
            # Heat capacity in each fluid circuit
            cp_in = cp
        self._cp_in = cp_in
        # Heat capacity in pipes
        cp_pipe = np.tile(cp_in, 2*self.nPipes)
        self._cp_pipe = cp_pipe

    def _f1(self, z):
        """
        Calculate function f1 from Hellstrom (1991)
        """
        f1 = np.exp(self._beta*z)*(np.cosh(self._gamma*z)
                                   - self._delta*np.sinh(self._gamma*z))
        return f1

    def _f2(self, z):
        """
        Calculate function f2 from Hellstrom (1991)
        """
        f2 = np.exp(self._beta*z)*self._beta12/self._gamma \
            * np.sinh(self._gamma*z)
        return f2

    def _f3(self, z):
        """
        Calculate function f3 from Hellstrom (1991)
        """
        f3 = np.exp(self._beta*z)*(np.cosh(self._gamma*z)
                                   + self._delta*np.sinh(self._gamma*z))
        return f3

    def _f4(self, z):
        """
        Calculate function f4 from Hellstrom (1991)
        """
        A = self._delta*self._beta1 + self._beta2*self._beta12/self._gamma
        f4 = np.exp(self._beta*z) \
            * (self._beta1*np.cosh(self._gamma*z) - A*np.sinh(self._gamma*z))
        return f4

    def _f5(self, z):
        """
        Calculate function f5 from Hellstrom (1991)
        """
        B = self._delta*self._beta2 + self._beta1*self._beta12/self._gamma
        f5 = np.exp(self._beta*z) \
            * (self._beta2*np.cosh(self._gamma*z) + B*np.sinh(self._gamma*z))
        return f5

    def _F4(self, z):
        """
        Calculate integral of function f4 from Hellstrom (1991)
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
        Outter radius (in meters) of the U-Tube pipes.
    borehole : Borehole object
        Borehole class object of the borehole containing the U-Tube.
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    R_fp : float
        Fluid to outter pipe wall thermal resistance (m-K/W).
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

        # Delta-circuit thermal resistances
        self._Rd = thermal_resistances(pos, r_out, borehole.r_b,
                                       k_s, k_g, self.R_fp, J=self.J)[1]

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

    def _continuity_condition_base(self, m_flow, cp, nSegments):
        """
        Equation that satisfies equal fluid temperatures in both legs of
        each U-tube pipe at depth (z = H).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{a_{out}} T_{f,out} = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_out : array
            Array of coefficients for outlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # Check if model variables need to be updated
        self._check_model_variables(m_flow, cp, nSegments)

        # Coefficient matrices from continuity condition:
        # [b_u]*[T_{f,u}](z=0) = [b_d]*[T_{f,d}](z=0) + [b_b]*[T_b]
        b_d, b_u, b_b = self._continuity_condition(m_flow, cp, nSegments)
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
            a_in = np.linalg.multi_dot([d_u, b_u_m1, b_d, c_in])
            a_out = np.array([[1.0]])
            a_b = np.linalg.multi_dot([d_u, b_u_m1, b_b])

        elif self.config == 'series':
            # Intermediate coefficient matrices:
            # [T_{f,d}](z=0) = [c_in]*[T_{f,in}] + [c_u]*[T_{f,u}](z=0)
            c_in = np.eye(self.nPipes, M=1)
            c_u = np.eye(self.nPipes, k=-1)

            # Intermediate coefficient matrices:
            # [d_u]*[T_{f,u}](z=0) = [d_in]*[T_{f,in}] + [d_b]*[T_b]
            d_u = b_u - b_d.dot(c_u)
            d_in = b_d.dot(c_in)
            d_b = b_b
            d_u_m1 = np.linalg.inv(d_u)

            # Intermediate coefficient matrices:
            # [T_{f,out}] = e_u*[T_{f,u}](z=0)
            e_u = np.eye(self.nPipes, M=1, k=-self.nPipes+1).T

            # Final coefficient matrices for continuity at depth (z = H):
            # [a_out][T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]
            a_in = np.linalg.multi_dot([e_u, d_u_m1, d_in])
            a_out = np.array([[1.0]])
            a_b = np.linalg.multi_dot([e_u, d_u_m1, d_b])

        return a_in, a_out, a_b

    def _continuity_condition_head(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_out : array
            Array of coefficients for outlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperature.

        """
        # Check if model variables need to be updated
        self._check_model_variables(m_flow, cp, nSegments)

        if self.config == 'parallel':
            a_in = np.vstack((np.ones((self.nPipes, self.nInlets)),
                              np.zeros((self.nPipes, self.nInlets))))
            a_out = np.vstack((np.zeros((self.nPipes, self.nOutlets)),
                               np.ones((self.nPipes, self.nOutlets))))
            a_b = np.zeros((2*self.nPipes, nSegments))

        elif self.config == 'series':
            # Coefficient matrices from continuity condition:
            # [b_u]*[T_{f,u}](z=0) = [b_d]*[T_{f,d}](z=0) + [b_b]*[T_b]
            b_d, b_u, b_b = self._continuity_condition(m_flow, cp, nSegments)

            # Intermediate coefficient matrices:
            # [T_{f,d}](z=0) = [c_in]*[T_{f,in}] + [c_u]*[T_{f,u}](z=0)
            c_in = np.eye(self.nPipes, M=1)
            c_u = np.eye(self.nPipes, k=-1)

            # Intermediate coefficient matrices:
            # [d_u]*[T_{f,u}](z=0) = [d_in]*[T_{f,in}] + [d_b]*[T_b]
            d_u = b_u - b_d.dot(c_u)
            d_in = b_d.dot(c_in)
            d_b = b_b
            d_u_m1 = np.linalg.inv(d_u)

            # Intermediate coefficient matrices:
            # [T_f](z=0) = [e_d]*[T_{f,d}](z=0) + [e_u]*[T_{f,u}](z=0)
            e_d = np.eye(2*self.nPipes, M=self.nPipes)
            e_u = np.eye(2*self.nPipes, M=self.nPipes, k=-self.nPipes)

            # Final coefficient matrices for temperatures at depth (z = 0):
            # [T_f](z=0) = [a_in]*[T_{f,in}]+[a_out]*[T_{f,out}]+[a_b]*[T_b]
            a_in = e_d.dot(c_in + np.linalg.multi_dot([c_u, d_u_m1, d_in])) \
                + np.linalg.multi_dot([e_u, d_u_m1, d_in])
            a_out = np.zeros((2*self.nPipes, self.nOutlets))
            a_b = np.linalg.multi_dot([e_d, c_u, d_u_m1, d_b]) \
                + np.linalg.multi_dot([e_u, d_u_m1, d_b])

        return a_in, a_out, a_b

    def _continuity_condition(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_d : array
            Array of coefficients for fluid temperature in downward flowing
            pipes.
        a_u : array
            Array of coefficients for fluid temperature in upward flowing
            pipes.
        a_b : array
            Array of coefficients for borehole wall temperature.

        """
        # Load coefficients
        A = self._A
        V = self._V
        Vm1 = self._Vm1
        L = self._L
        Dm1 = self._Dm1

        # Matrix exponential at depth (z = H)
        H = self.b.H
        E = (V.dot(np.diag(np.exp(L*H)))).dot(Vm1)

        # Coefficient matrix for borehole wall temperatures
        IIm1 = np.hstack((np.eye(self.nPipes), -np.eye(self.nPipes)))
        Ones = np.ones((2*self.nPipes, 1))
        a_b = np.zeros((self.nPipes, nSegments))
        for v in range(nSegments):
            z1 = H - v*H/nSegments
            z2 = H - (v + 1)*H/nSegments
            dE = np.diag(np.exp(L*z1) - np.exp(L*z2))
            a_b[:, v:v+1] = np.linalg.multi_dot([IIm1,
                                                 V,
                                                 Dm1,
                                                 dE,
                                                 Vm1,
                                                 A,
                                                 Ones])

        # Configuration-specific inlet and outlet coefficient matrices
        IZER = np.vstack((np.eye(self.nPipes),
                          np.zeros((self.nPipes, self.nPipes))))
        ZERI = np.vstack((np.zeros((self.nPipes, self.nPipes)),
                          np.eye(self.nPipes)))
        a_u = np.linalg.multi_dot([IIm1, E, ZERI])
        a_d = np.linalg.multi_dot([-IIm1, E, IZER])

        return a_d, a_u, a_b

    def _general_solution(self, z, m_flow, cp, nSegments):
        """
        General solution for fluid temperatures at a depth (z).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{a_{f0}} \\mathbf{T_{f}}(z=0)
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_f0 : array
            Array of coefficients for inlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # Check if model variables need to be updated
        self._check_model_variables(m_flow, cp, nSegments)

        # Load coefficients
        A = self._A
        V = self._V
        Vm1 = self._Vm1
        L = self._L
        Dm1 = self._Dm1

        # Matrix exponential at depth (z)
        a_f0 = (V.dot(np.diag(np.exp(L*z)))).dot(Vm1)

        # Coefficient matrix for borehole wall temperatures
        a_b = np.zeros((2*self.nPipes, nSegments))
        Ones = np.ones((2*self.nPipes, 1))
        for v in range(nSegments):
            dz1 = z - min(z, v*self.b.H/nSegments)
            dz2 = z - min(z, (v + 1)*self.b.H/nSegments)
            E1 = np.diag(np.exp(L*dz1))
            E2 = np.diag(np.exp(L*dz2))
            a_b[:,v:v+1] = np.linalg.multi_dot([V,
                                                Dm1,
                                                E2 - E1,
                                                Vm1,
                                                A,
                                                Ones])

        return a_f0, a_b

    def _update_model_variables(self, m_flow, cp, nSegments):
        """
        Evaluate eigenvalues and eigenvectors for the system of differential
        equations.
        """
        nPipes = self.nPipes
        # Format mass flow rate and heat capacity inputs
        self._format_inputs(m_flow, cp, nSegments)
        m_flow_pipe = self._m_flow_pipe
        cp_pipe = self._cp_pipe

        # Coefficient matrix for differential equations
        self._A = 1.0 / (self._Rd.T * m_flow_pipe * cp_pipe).T
        for i in range(2*nPipes):
            self._A[i, i] = -self._A[i, i] - sum(
                [self._A[i, j] for j in range(2*nPipes) if not i == j])
        for i in range(nPipes, 2*nPipes):
            self._A[i, :] = - self._A[i, :]
        # Eigenvalues and eigenvectors of A
        self._L, self._V = np.linalg.eig(self._A)
        # Inverse of eigenvector matrix
        self._Vm1 = np.linalg.inv(self._V)
        # Diagonal matrix of eigenvalues and inverse
        self._D = np.diag(self._L)
        self._Dm1 = np.diag(1./self._L)

    def _format_inputs(self, m_flow, cp, nSegments):
        """
        Format mass flow rate and heat capacity inputs.
        """
        nPipes = self.nPipes
        # Format mass flow rate inputs
        # Mass flow rate in pipes
        if self.config.lower() == 'parallel':
            m_flow_pipe = np.tile(m_flow/nPipes, 2*self.nPipes)
        elif self.config.lower() == 'series':
            m_flow_pipe = np.tile(m_flow, 2*self.nPipes)
        self._m_flow_pipe = m_flow_pipe
        # Mass flow rate in each fluid circuit
        m_flow_in = np.atleast_1d(m_flow)
        self._m_flow_in = m_flow_in

        # Format heat capacity inputs
        # Heat capacity in each fluid circuit
        cp_in = np.atleast_1d(cp)
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
        Outter radius (in meters) of the U-Tube pipes.
    borehole : Borehole object
        Borehole class object of the borehole containing the U-Tube.
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    R_fp : float
        Fluid to outter pipe wall thermal resistance (m-K/W).
    J : int, optional
        Number of multipoles per pipe to evaluate the thermal resistances.
        Default is 2.
    nPipes : int
        Number of U-Tubes.
    nInlets : int
        Total number of pipe inlets, equals to nPipes.
    nOutlets : int
        Total number of pipe outlets, equals to nPipes.

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

        # Delta-circuit thermal resistances
        self._Rd = thermal_resistances(pos, r_out, borehole.r_b,
                                       k_s, k_g, self.R_fp, J=self.J)[1]

        # Initialize stored_coefficients
        self._initialize_stored_coefficients()

    def _continuity_condition_base(self, m_flow, cp, nSegments):
        """
        Equation that satisfies equal fluid temperatures in both legs of
        each U-tube pipe at depth (z = H).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{a_{out}} T_{f,out} = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_out : array
            Array of coefficients for outlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperatures.

        """
        # Check if model variables need to be updated
        self._check_model_variables(m_flow, cp, nSegments)

        # Coefficient matrices from continuity condition:
        # [b_u]*[T_{f,u}](z=0) = [b_d]*[T_{f,d}](z=0) + [b_b]*[T_b]
        a_in, a_out, a_b = self._continuity_condition(m_flow, cp, nSegments)

        return a_in, a_out, a_b

    def _continuity_condition_head(self, m_flow, cp, nSegments):
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
        m_flow : float or array
            Inlet mass flow rate (in kg/s).
        cp : float or array
            Fluid specific isobaric heat capacity (in J/kg.degC).
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array of coefficients for inlet fluid temperature.
        a_out : array
            Array of coefficients for outlet fluid temperature.
        a_b : array
            Array of coefficients for borehole wall temperature.

        """
        # Check if model variables need to be updated
        self._check_model_variables(m_flow, cp, nSegments)

        a_in = np.eye(2*self.nPipes, M=self.nPipes, k=0)
        a_out = np.eye(2*self.nPipes, M=self.nPipes, k=-self.nPipes)
        a_b = np.zeros((2*self.nPipes, nSegments))

        return a_in, a_out, a_b

    def _format_inputs(self, m_flow, cp, nSegments):
        """
        Format mass flow rate and heat capacity inputs.
        """
        # Format mass flow rate inputs
        # Mass flow rate in each fluid circuit
        m_flow_in = np.atleast_1d(m_flow)
        self._m_flow_in = m_flow_in
        # Mass flow rate in pipes
        m_flow_pipe = np.tile(m_flow_in, 2)
        self._m_flow_pipe = m_flow_pipe

        # Format heat capacity inputs
        # Heat capacity in each fluid circuit
        cp_in = np.atleast_1d(cp)
        self._cp_in = cp_in
        # Heat capacity in pipes
        cp_pipe = np.tile(cp_in, 2)
        self._cp_pipe = cp_pipe


def thermal_resistances(pos, r_out, r_b, k_s, k_g, Rfp, J=2):
    """
    Evaluate thermal resistances and delta-circuit thermal resistances.

    This function evaluates the thermal resistances and delta-circuit thermal
    resistances between pipes in a borehole using the multipole method
    [#Claesson2011]_. Thermal resistances are defined by:

    .. math:: \\mathbf{T_f} - T_b = \\mathbf{R} \\cdot \\mathbf{Q_{pipes}}

    Delta-circuit thermal resistances are defined by:

    .. math::

        Q_{i,j} = \\frac{T_{f,i} - T_{f,j}}{R^\\Delta_{i,j}}

        Q_{i,i} = \\frac{T_{f,i} - T_b}{R^\\Delta_{i,i}}

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
    Rfp : float or array
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
    if np.isscalar(Rfp):
        Rfp = np.ones(n_p)*Rfp

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
                    R[i, j] = Rfp[i] + 1./(2.*pi*k_g) \
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
                                              Rfp, 0., Q_p, J)
            R[:,m] = T_f

    # Delta-circuit thermal resistances
    K = -np.linalg.inv(R)
    for i in range(n_p):
        K[i, i] = -(K[i, i] +
                    sum([K[i, j] for j in range(n_p) if not i == j]))
    Rd = 1.0/K

    return R, Rd


def borehole_thermal_resistance(pipe, m_flow, cp):
    """
    Evaluate the effective borehole thermal resistance.

    Parameters
    ----------
    pipe : pipe object
        Model for pipes inside the borehole.
    m_flow : float
        Fluid mass flow rate (in kg/s).
    cp : float
        Fluid specific isobaric heat capacity (in J/kg.K)

    Returns
    -------
    Rb : float
        Effective borehole thermal resistance (m.K/W).

    """
    # Coefficient for T_{f,out} = a_out*T_{f,in} + [b_out]*[T_b]
    a_out = np.asscalar(
            pipe.coefficients_outlet_temperature(m_flow, cp, nSegments=1)[0])
    # Coefficient for Q_b = [a_Q]*T{f,in} + [b_Q]*[T_b]
    a_Q = np.asscalar(pipe.coefficients_borehole_heat_extraction_rate(
            m_flow, cp, nSegments=1)[0])
    # Borehole length
    H = pipe.b.H
    # Effective borehole thermal resistance
    Rb = -0.5*H*(1. + a_out)/a_Q

    return Rb


def field_thermal_resistance(pipes, bore_connectivity, m_flow, cp):
    """
    Evaluate the effective bore field thermal resistance.

    As proposed in [#Cimmino2018]_.

    Parameters
    ----------
    pipes : list of pipe objects
        Models for pipes inside each borehole.
    bore_connectivity : list
        Index of fluid inlet into each borehole. -1 corresponds to a borehole
        connected to the bore field inlet.
    m_flow : float or array
        Fluid mass flow rate in each borehole (in kg/s).
    cp : float
        Fluid specific isobaric heat capacity (in J/kg.K)

    Returns
    -------
    Rfield : float
        Effective bore field thermal resistance (m.K/W).

    References
    ----------
    .. [#Cimmino2018] Cimmino, M. (2018). g-Functions for bore fields with
       mixed parallel and series connections considering the axial fluid
       temperature variations. IGSHPA Research Track, Stockholm. In review.

    """
    # Number of boreholes
    nBoreholes = len(pipes)
    # If m_flow is supplied as float, apply m_flow to all boreholes
    if np.isscalar(m_flow):
        m_flow = np.tile(m_flow, nBoreholes)
    # Total mass flow rate in the bore field
    m_flow_tot = sum([m_flow[i] for i in range(nBoreholes)
                      if bore_connectivity[i] == -1])

    # Total borehole length
    H_tot = sum([pipes[i].b.H for i in range(nBoreholes)])

    # Verify that borehole connectivity is valid
    _verify_bore_connectivity(bore_connectivity, nBoreholes)


    # Coefficients for T_{f,out} = A_out*T_{f,in} + [B_out]*[T_b], and
    # Q_b = [A_Q]*T{f,in} + [B_Q]*[T_b]
    A_Q = np.array([np.asscalar(
            pipes[i].coefficients_borehole_heat_extraction_rate(
                    m_flow[i], cp, nSegments=1)[0])
            for i in range(len(pipes))])
    for i in range(len(pipes)):
        path = _path_to_inlet(bore_connectivity, i)
        for j in path[1:]:
            a_out = np.asscalar(pipes[j].coefficients_outlet_temperature(
                    m_flow[j], cp, nSegments=1)[0])
            A_Q[i] = A_Q[i]*a_out
    A_out = 1. + np.sum(A_Q)/(m_flow_tot*cp)

    # Effective bore field thermal resistance
    Rfield = -0.5*H_tot*(1. + A_out)/np.sum(A_Q)

    return Rfield


def fluid_friction_factor_circular_pipe(m_flow, r_in, visc, den, epsilon,
                                        tol=1.0e-6):
    """
    Evaluate the Darcy-Weisbach friction factor.

    Parameters
    ----------
    m_flow : float
        Fluid mass flow rate (in kg/s).
    r_in : float
        Inner radius of the pipes (in meters).
    visc : float
        Fluid dynamic viscosity (in kg/m-s).
    den : float
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
    V_flow = m_flow / den
    A_cs = pi * r_in**2
    V = V_flow / A_cs
    # Reynolds number
    Re = den * V * D / visc

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


def convective_heat_transfer_coefficient_circular_pipe(m_flow, r_in, visc, den,
                                                       k, cp, epsilon):
    """
    Evaluate the convective heat transfer coefficient for circular pipes.

    Parameters
    ----------
    m_flow : float
        Fluid mass flow rate (in kg/s).
    r_in : float
        Inner radius of the pipes (in meters).
    visc : float
        Fluid dynamic viscosity (in kg/m-s).
    den : float
        Fluid density (in kg/m3).
    k : float
        Fluid thermal conductivity (in W/m-K).
    cp : float
        Fluid specific heat capacity (in J/kg-K).
    epsilon : float
        Pipe roughness (in meters).

    Returns
    -------
    h_fluid : float
        Convective heat transfer coefficient (in W/m2-K).

    Examples
    --------

    """
    # Hydraulic diameter
    D = 2.*r_in
    # Fluid velocity
    V_flow = m_flow / den
    A_cs = pi * r_in**2
    V = V_flow / A_cs
    # Reynolds number
    Re = den * V * D / visc
    # Prandtl number
    Pr = cp * visc / k
    # Darcy friction factor
    fDarcy = fluid_friction_factor_circular_pipe(m_flow, r_in, visc, den,
                                                 epsilon)
    if Re > 2300.:
        # Nusselt number from Gnielinski
        Nu = 0.125*fDarcy * (Re - 1.0e3) * Pr / \
            (1.0 + 12.7 * np.sqrt(0.125*fDarcy) * (Pr**(2.0/3.0) - 1.0))
    else:
        Nu = 3.66
    h_fluid = k * Nu / D

    return h_fluid


def conduction_thermal_resistance_circular_pipe(r_in, r_out, k):
    """
    Evaluate the conduction thermal resistance for circular pipes.

    Parameters
    ----------
    r_in : float
        Inner radius of the pipes (in meters).
    r_out : float
        Outer radius of the pipes (in meters).
    k : float
        Pipe thermal conductivity (in W/m-K).

    Returns
    -------
    R_pipe : float
        Conduction thermal resistance (in m-K/W).

    Examples
    --------

    """
    R_pipe = np.log(r_out/r_in)/(2*pi*k)

    return R_pipe


def multipole(pos, r_p, r_b, k_s, k_g, Rfp, T_b, Q_p, J,
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
    r_p : float or array
        Outer radius of the pipes (in meters).
    r_b : float
        Borehole radius (in meters).
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    Rfp : float or array
        Fluid-to-outer-pipe-wall thermal resistance (in m-K/W).
    J : int
        Number of multipoles per pipe to evaluate the thermal resistances.
        J=1 or J=2 usually gives sufficient accuracy. J=0 corresponds to the
        line source approximation.
    Q_p : array
        Thermal energy flows (in W/m) from pipes.
    T_b_av : float
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
    z_p = np.array([pos[i][0] + 1.j*pos[i][1] for i in range(n_p)])
    # If r_out and/or Rfp are supplied as float, build arrays of size n_p
    if np.isscalar(r_p):
        r_p = np.ones(n_p)*r_p
    if np.isscalar(Rfp):
        Rfp = np.ones(n_p)*Rfp

    # -------------------------------------
    # Thermal resistance matrix R0 (EQ. 33)
    # -------------------------------------
    pikg = 1.0 / (2.0*pi*k_g)
    sigma = (k_g - k_s)/(k_g + k_s)
    beta_p = 2*pi*k_g*Rfp
    R0 = np.zeros((n_p, n_p))
    for i in range(n_p):
        rbm = r_b**2/(r_b**2 - np.abs(z_p[i])**2)
        R0[i, i] = pikg*(np.log(r_b/r_p[i]) + beta_p[i] + sigma*np.log(rbm))
        for j in range(n_p):
            if i != j:
                dz = np.abs(z_p[i] - z_p[j])
                rbm = r_b**2/np.abs(r_b**2 - z_p[j]*np.conj(z_p[i]))
                R0[i, j] = pikg*(np.log(r_b/dz) + sigma*np.log(rbm))

    # Initialize maximum error and iteration counter
    eps_max = 1.0e99
    it = 0
    # -------------------
    # Multipoles (EQ. 38)
    # -------------------
    if J > 0:
        P = np.zeros((n_p, J), dtype=np.cfloat)
        coeff = -np.array([[(1 - (k+1)*beta_p[m])/(1 + (k+1)*beta_p[m])
                           for k in range(J)] for m in range(n_p)])
        while eps_max > eps and it < it_max:
            it += 1
            eps_max = 0.
            F = _F_mk(Q_p, P, n_p, J, r_b, r_p, z_p, pikg, sigma)
            P_new = coeff*np.conj(F)
            if it == 1:
                diff0 = np.max(np.abs(P_new-P)) - np.min(np.abs(P_new-P))
            diff = np.max(np.abs(P_new-P)) - np.min(np.abs(P_new-P))
            eps_max = diff / diff0
            P = P_new

    # --------------------------
    # Fluid temperatures(EQ. 32)
    # --------------------------
    T_f = T_b + R0.dot(Q_p)
    if J > 0:
        for m in range(n_p):
            dTfm = 0. + 0.j
            for n in range(n_p):
                for j in range(J):
                    # Second term
                    if n != m:
                        dTfm += P[n,j]*(r_p[n]/(z_p[m]-z_p[n]))**(j+1)
                    # Third term
                    dTfm += sigma*P[n,j]*(r_p[n]*np.conj(z_p[m]) \
                                   /(r_b**2 - z_p[n]*np.conj(z_p[m])))**(j+1)
            T_f[m] += np.real(dTfm)

    # -------------------------------
    # Requested temperatures (EQ. 28)
    # -------------------------------
    n_T = len(x_T)
    T = np.zeros(n_T)
    for i in range(n_T):
        z_T = x_T[i] + 1.j*y_T[i]
        dT0 = 0. + 0.j
        dTJ = 0. + 0.j
        for n in range(n_p):
            if np.abs(z_T - z_p[n])/r_p[n] < 1.0:
                # Coordinate inside pipe
                T[i] = T_f[n]
                break
            # Order 0
            if np.abs(z_T) <= r_b:
                # Coordinate inside borehole
                W0 = np.log(r_b/(z_T - z_p[n])) \
                        + sigma*np.log(r_b**2/(r_b**2 - z_p[n]*np.conj(z_T)))
            else:
                # Coordinate outside borehole
                W0 = (1. + sigma)*np.log(r_b/(z_T - z_p[n])) \
                        + sigma*(1. + sigma)/(1. - sigma)*np.log(r_b/z_T)
            dT0 += Q_p[n]*pikg*W0
            # Multipoles
            for j in range(J):
                if np.abs(z_T) <= r_b:
                    # Coordinate inside borehole
                    WJ = (r_p[n]/(z_T - z_p[n]))**(j+1) \
                            + sigma*((r_p[n]*np.conj(z_T))
                                     /(r_b**2 - z_p[n]*np.conj(z_T)))**(j+1)
                else:
                    # Coordinate outside borehole
                    WJ = (1. + sigma)*(r_p[n]/(z_T - z_p[n]))**(j+1)
                dTJ += P[n,j]*WJ
        else:
            T[i] += T_b + np.real(dT0 + dTJ)

    return T_f, T, it, eps_max


def _F_mk(Q_p, P, n_p, J, r_b, r_p, z, pikg, sigma):
    """
    Complex matrix F_mk from Claesson and Hellstrom (2011), EQ. 34
    """
    F = np.zeros((n_p, J), dtype=np.cfloat)
    for m in range(n_p):
        for k in range(J):
            fmk = 0. + 0.j
            for n in range(n_p):
                # First term
                if m != n:
                    fmk += Q_p[n]*pikg/(k+1)*(r_p[m]/(z[n] - z[m]))**(k+1)
                # Second term
                fmk += sigma*Q_p[n]*pikg/(k+1)*(r_p[m]*np.conj(z[n])/(
                        r_b**2 - z[m]*np.conj(z[n])))**(k+1)
                for j in range(J):
                    # Third term
                    if m != n:
                        fmk += P[n,j]*binom(j+k+1, j) \
                                *r_p[n]**(j+1)*(-r_p[m])**(k+1) \
                                /(z[m] - z[n])**(j+k+2)
                    # Fourth term
                    j_pend = np.min((k, j)) + 1
                    for jp in range(j_pend):
                        fmk += sigma*np.conj(P[n,j])*binom(j+1, jp) \
                                *binom(j+k-jp+1, j)*r_p[n]**(j+1) \
                                *r_p[m]**(k+1)*z[m]**(j+1-jp) \
                                *np.conj(z[n])**(k+1-jp) \
                                /(r_b**2 - z[m]*np.conj(z[n]))**(k+j+2-jp)
            F[m,k] = fmk

    return F
