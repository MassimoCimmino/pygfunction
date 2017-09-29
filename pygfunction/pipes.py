from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.constants import pi


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
        self.borehole = borehole
        self.nPipes = 1
        self.nInlets = 1
        self.nOutlets = 1

    def get_temperature(self, z, Tin, Tb, m_flow, cp):
        """
        Returns the fluid temperatures of the borehole at a depth (z).

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperatures.
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
        # Build coefficient matrices
        a_in, a_b = self.coefficients_temperature(z,
                                                  m_flow,
                                                  cp,
                                                  nSegments)
        # Evaluate fluid temperatures
        Tf = (a_in.dot(Tin) + a_b.dot(Tb)).flatten()
        return Tf

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
        Tout = (a_in.dot(Tin) + a_b.dot(Tb)).flatten()
        # Return float if Tin was supplied as scalar
        if np.isscalar(Tin) and not np.isscalar(Tout):
            Tout = np.asscalar(Tout)
        return Tout

    def get_heat_extraction_rate(self, Tin, Tb, m_flow, cp):
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
            Heat extraction rates along each borehole segment.

        """
        nSegments = len(np.atleast_1d(Tb))
        a_in, a_b = self.coefficients_heat_extraction_rate(m_flow,
                                                           cp,
                                                           nSegments)
        Qb = (a_in.dot(Tin) + a_b.dot(Tb)).flatten()
        # Return float if Tb was supplied as scalar
        if np.isscalar(Tb) and not np.isscalar(Qb):
            Qb = np.asscalar(Qb)
        return Qb

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
        Qb : float
            Total net heat extraction rate of the borehole.

        """
        Tout = self.get_outlet_temperature(Tin, Tb, m_flow, cp)
        Q = np.sum(m_flow * cp * (Tout - Tin))
        return Q

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
        # Update model variables
        self._update_coefficients(m_flow, cp, nSegments)

        # Coefficient matrices from continuity condition:
        # [b_out]*[T_{f,out}] = [b_in]*[T_{f,in}] + [b_b]*[T_b]
        b_in, b_out, b_b = self._continuity_condition_base(m_flow, cp, nSegments)

        # Final coefficient matrices for outlet temperatures:
        # [T_{f,out}] = [a_in]*[T_{f,in}] + [a_b]*[T_b]
        b_out_m1 = np.inv(b_out)
        a_in = b_out_m1.dot(b_in)
        a_b = b_out_m1.dot(b_b)

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
        # Update model variables
        self._update_coefficients(m_flow, cp, nSegments)

        # Coefficient matrices for outlet temperatures:
        # [T_{f,out}] = [b_in]*[T_{f,in}] + [b_b]*[T_b]
        b_in, b_b = self.coefficients_outlet_temperature(m_flow, cp, nSegments)

        # Coefficient matrices for temperatures at depth (z = 0):
        # [T_f](0) = [c_in]*[T_{f,in}] + [c_out]*[T_{f,out}] + [c_b]*[T_b]
        c_in, c_out, c_b = self._continuity_condition_head(m_flow,
                                                           cp,
                                                           nSegments)

        # Coefficient matrices from general solution:
        # [T_f](z) = [d_f0]*[T_f](0) + [d_b]*[T_b]
        d_f0, d_b = self._general_solution(z, m_flow, cp, nSegments)

        # Final coefficient matrices for temperatures at depth (z):
        # [T_f](z) = [a_in]*[T_{f,in}] + [a_b]*[T_b]
        a_in = d_f0.dot(c_in + c_out.dot(b_in))
        a_b = d_f0.dot(c_b + c_out.dot(b_b)) + d_b

        return a_in, a_b

    def coefficients_heat_extraction_rate(self, m_flow, cp, nSegments):
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
        # Update model variables
        self._update_coefficients(m_flow, cp, nSegments)
        M = np.hstack((-self._m_flow_pipe*cp, self._m_flow_pipe*cp))
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
            a_in[i, :] = M.dot(aTf1 - aTf2)
            a_b[i, :] = M.dot(bTf1 - bTf2)
            aTf1, bTf1 = aTf2, bTf2

        return a_in, a_b

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

    def _update_coefficients(self, m_flow, cp, nSegments):
        """
        Evaluate common coefficients needed in other class methods.
        """
        raise NotImplementedError(
            '_update_coefficients class method not implemented, '
            'this method should Evaluate common coefficients needed in other '
            'class methods.')


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
    def __init__(self, pos, r_in, r_out, borehole, k_s, k_g, R_fp):
        self.pos = pos
        self.r_in = r_in
        self.r_out = r_out
        self.b = borehole
        self.k_s = k_s
        self.k_g = k_g
        self.R_fp = R_fp
        self.nPipes = 1
        self.nInlets = 1
        self.nOutlets = 1

        # Delta-circuit thermal resistances
        self._Rd = thermal_resistances(pos, r_out, borehole.r_b,
                                       k_s, k_g, self.R_fp)[1]

    def _continuity_condition_base(self, m_flow, cp, nSegments):
        """
        Equation that satisfies equal fluid temperatures in both legs of
        each U-tube pipe at depth (z = H).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{a_{out}} T_{f,out} = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
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
        # There is only one pipe
        a_in = np.array([[1.0], [0.0]])
        a_out = np.array([[0.0], [1.0]])
        a_b = np.zeros((2.0, nSegments))

        return a_in, a_out

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

    def _update_coefficients(self, m_flow, cp, nSegments):
        """
        Evaluate dimensionless resistances for Hellstrom solution.
        """
        # Mass flow rate in pipes
        self._m_flow_pipe = m_flow
        # Dimensionless delta-circuit conductances
        self._beta1 = 1./(self._Rd[0][0]*m_flow*cp)
        self._beta2 = 1./(self._Rd[1][1]*m_flow*cp)
        self._beta12 = 1./(self._Rd[0][1]*m_flow*cp)
        self._beta = 0.5*(self._beta2 - self._beta1)
        # Eigenvalues
        self._gamma = np.sqrt(0.25*(self._beta1+self._beta2)**2
                              + self._beta12*(self._beta1+self._beta2))
        self._delta = 1./self._gamma \
            * (self._beta12 + 0.5*(self._beta1+self._beta2))

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
                 k_g, R_fp, nPipes, config='parallel'):
        self.pos = pos
        self.r_in = r_in
        self.r_out = r_out
        self.b = borehole
        self.k_s = k_s
        self.k_g = k_g
        self.R_fp = R_fp
        self.nPipes = nPipes
        self.nInlets = 1
        self.nOutlets = 1
        self.config = config.lower()

        # Delta-circuit thermal resistances
        self._Rd = thermal_resistances(pos, r_out, borehole.r_b,
                                       k_s, k_g, self.R_fp)[1]

    def coefficients_heat_extraction_rate(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_b} = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float
            Inlet mass flow rate (in kg/s).
        cp : float
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
        if self.config.lower() == 'parallel':
            m_flow_pipe = m_flow / self.nPipes
        else:
            m_flow_pipe = m_flow
        M = m_flow_pipe*cp*np.concatenate([-np.ones((1, self.nPipes)),
                                           np.ones((1, self.nPipes))], axis=1)

        aQ = np.zeros((nSegments, self.nInlets))
        bQ = np.zeros((nSegments, nSegments))

        for i in range(nSegments):
            z1 = i * self.b.H / nSegments
            z2 = (i + 1) * self.b.H / nSegments
            aTf1, bTf1 = self.coefficients_temperature(z1,
                                                       m_flow,
                                                       cp,
                                                       nSegments)
            aTf2, bTf2 = self.coefficients_temperature(z2,
                                                       m_flow,
                                                       cp,
                                                       nSegments)
            aQ[i, :] = M.dot(aTf1 - aTf2)
            bQ[i, :] = M.dot(bTf1 - bTf2)
        return aQ, bQ

    def coefficients_temperature(self, z, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate fluid temperatures at a depth
        (z).

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float
            Inlet mass flow rate (in kg/s).
        cp : float
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
        self._matrixExponentialsInOut(m_flow, cp, nSegments)
        Eoutm1 = np.linalg.inv(self._Eout)

        # Intermediate matrices for [Tf](z=0) = [b_in] * Tin + [b_b] * [Tb]
        if self.config == 'parallel':
            b_in = np.concatenate((np.ones((self.nPipes, 1)),
                                   Eoutm1.dot(self._Ein)), axis=0)
            b_b = np.concatenate((np.zeros((self.nPipes, nSegments)),
                                  Eoutm1.dot(self._Eb)), axis=0)
        elif self.config == 'series':
            b_in = np.concatenate((np.eye(self.nPipes, k=-1),
                                   np.eye(self.nPipes)), axis=0)
            b_in = np.linalg.multi_dot([b_in,
                                        Eoutm1,
                                        self._Ein])
            b_in = np.reshape(b_in, (2*self.nPipes, 1))
            b_in[0, 0] += 1.0
            b_b = np.concatenate((np.eye(self.nPipes, k=-1),
                                  np.eye(self.nPipes)), axis=0)
            b_b = np.linalg.multi_dot([b_b, Eoutm1, self._Eb])
            b_b = np.reshape(b_b, (2*self.nPipes, nSegments))

        # Final matrices for [Tf](z) = [a_in] * Tin + [a_b] * [Tb]
        Vm1 = np.linalg.inv(self._V)
        D = np.diag(self._L)
        Dm1 = np.linalg.inv(D)
        E = np.linalg.multi_dot([self._V,
                                 np.diag(np.exp(self._L*z)),
                                 Vm1])
        Ones = np.ones((2*self.nPipes, 1))

        a_in = E.dot(b_in)
        a_b = E.dot(b_b)
        da_bv = []
        N = int(np.ceil(z/self.b.H*nSegments))
        # Build the coefficient matrix for borehole wall temperature
        # from energy balances on each segment
        for v in range(N):
            z1 = z - v*self.b.H/nSegments
            z2 = z - min((v+1)*self.b.H/nSegments, z)
            dE = np.diag(np.exp(self._L*z1)) - np.diag(np.exp(self._L*z2))
            da_bv.append(np.linalg.multi_dot([self._V,
                                              Dm1,
                                              dE,
                                              Vm1,
                                              self._A,
                                              Ones]))
        for v in range(N, nSegments):
            da_bv.append(np.zeros((2*self.nPipes, 1)))
        a_b = E.dot(b_b) - np.concatenate(da_bv, axis=1)

        return a_in, a_b

    def coefficients_outlet_temperature(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate outlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                T_{f,out} = \\mathbf{a_{in}} T_{f,in}
                + \\mathbf{a_{b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : float
            Inlet mass flow rates (in kg/s).
        cp : float
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
        self._matrixExponentialsInOut(m_flow, cp, nSegments)

        Eoutm1 = np.linalg.inv(self._Eout)
        if self.config == 'parallel':
            nm1 = 1.0/self.nPipes * np.ones((1, self.nPipes))
            a_in = np.linalg.multi_dot([nm1,
                                        Eoutm1,
                                        self._Ein])
            a_b = np.linalg.multi_dot([Eoutm1,
                                       self._Eb])
        elif self.config == 'series':
            le1 = np.zeros((1, self.nPipes))
            le1[0, -1] = 1.0
            a_in = np.linalg.multi_dot([le1,
                                        Eoutm1,
                                        self._Ein])
            a_b = np.linalg.multi_dot([le1,
                                       Eoutm1,
                                       self._Eb])
        return a_in, a_b

    def _matrixExponentialsInOut(self, m_flow, cp, nSegments):
        """ Evaluate configuration-specific coefficient matrices
        """
        nPipes = self.nPipes
        if self.config.lower() == 'parallel':
            m_flow_pipe = m_flow / nPipes
        else:
            m_flow_pipe = m_flow

        # Coefficient matrix for differential equations
        self._A = 1.0 / (self._Rd * m_flow_pipe * cp)
        for i in range(2*nPipes):
            self._A[i, i] = -self._A[i, i] - sum(
                [self._A[i, j] for j in range(2*nPipes) if not i == j])
        for i in range(nPipes, 2*nPipes):
            self._A[i, :] = - self._A[i, :]
        # Eigenvalues and eigenvectors of A
        self._L, self._V = np.linalg.eig(self._A)

        z = self.b.H
        Vm1 = np.linalg.inv(self._V)
        D = np.diag(self._L)
        Dm1 = np.linalg.inv(D)
        E = (self._V.dot(np.diag(np.exp(self._L*z)))).dot(Vm1)
        IIm1 = np.concatenate((np.eye(nPipes), -np.eye(nPipes)), 1)
        Ones = np.ones((2*nPipes, 1))

        # Coefficient matrix for borehole wall temperatures
        self._Eb = np.zeros((nPipes, nSegments))
        for v in range(nSegments):
            z1 = self.b.H - v * self.b.H/nSegments
            z2 = self.b.H - (v + 1) * self.b.H/nSegments
            dE = np.diag(np.exp(self._L*z1)) - np.diag(np.exp(self._L*z2))
            self._Eb[:, v:v+1] = np.linalg.multi_dot([IIm1,
                                                     self._V,
                                                     Dm1,
                                                     dE,
                                                     Vm1,
                                                     self._A,
                                                     Ones])

        # Configuration-specific inlet and outlet coefficient matrices
        IZER = np.concatenate((np.eye(nPipes), np.zeros((nPipes, nPipes))), 0)
        ZERI = np.concatenate((np.zeros((nPipes, nPipes)), np.eye(nPipes)), 0)
        if self.config == 'parallel':
            self._Eout = IIm1.dot(E).dot(ZERI)
            self._Ein = -IIm1.dot(E).dot(IZER).dot(np.ones((nPipes, 1)))
        elif self.config == 'series':
            self._Eout = IIm1.dot(E).dot(ZERI)
            self._Ein = -IIm1.dot(E).dot(IZER)
            self._Eout[:, 0:nPipes-1] += - self._Ein[:, 1:nPipes]
            self._Ein = self._Ein[:, 0]


class IndependentMultipleUTube(_BasePipe):
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
                 k_g, R_fp, nPipes):
        self.pos = pos
        self.r_in = r_in
        self.r_out = r_out
        self.b = borehole
        self.k_s = k_s
        self.k_g = k_g
        self.R_fp = R_fp
        self.nPipes = nPipes
        self.nInlets = nPipes
        self.nOutlets = nPipes

        # Delta-circuit thermal resistances
        self._Rd = thermal_resistances(pos, r_out, borehole.r_b,
                                       k_s, k_g, self.R_fp)[1]

        # Stored conditions
        self._m_flow = np.nan
        self._cp = np.nan
        self._nSegments = np.nan
        self._m_flow_Q = np.nan
        self._cp_Q = np.nan
        self._nSegments_Q = np.nan
        self._m_flow_T0 = np.nan
        self._cp_T0 = np.nan
        self._nSegments_T0 = np.nan

    def coefficients_heat_extraction_rate(self, m_flow, cp, nSegments):
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

        if not self._is_same_heat_extraction_rate(m_flow, cp, nSegments):
            m_flow_pipe = np.reshape(m_flow, (1, self.nPipes))
            M = cp*np.concatenate([-m_flow_pipe,
                                   m_flow_pipe], axis=1)

            aQ = np.zeros((nSegments, self.nInlets))
            bQ = np.zeros((nSegments, nSegments))

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
                aQ[i, :] = M.dot(aTf1 - aTf2)
                bQ[i, :] = M.dot(bTf1 - bTf2)
                aTf1, bTf1 = aTf2, bTf2
            self._a_in_Q = aQ
            self._a_b_Q = bQ
        else:
            aQ = self._a_in_Q
            bQ = self._a_b_Q
            self._m_flow_Q = m_flow
            self._cp_Q = cp
            self._nSegments_Q = nSegments
        return aQ, bQ

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
        # Intermediate matrices for [Tf](z=0) = [b_in] * Tin + [b_b] * [Tb]
        b_in0, b_b0 = self.coefficients_outlet_temperature(m_flow,
                                                           cp,
                                                           nSegments)
        b_in = np.concatenate((np.eye(self.nPipes),
                               b_in0), axis=0)
        b_b = np.concatenate((np.zeros((self.nPipes, nSegments)),
                              b_b0), axis=0)

        # Final matrices for [Tf](z) = [a_in] * Tin + [a_b] * [Tb]
        E, F = self._matrix_exponentials(z,
                                         m_flow,
                                         cp,
                                         nSegments)[-2:]
        a_in = E.dot(b_in)
        a_b = E.dot(b_b) - F

        return a_in, a_b

    def coefficients_outlet_temperature(self, m_flow, cp, nSegments):
        """
        Build coefficient matrices to evaluate outlet fluid temperatures.

        Returns coefficients for the relation:

            .. math::

                T_{f,out} = \\mathbf{a_{in}} \\mathbf{T_{f,in}}
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

        if not self._is_same_outlet_temperature(m_flow, cp, nSegments):
            E_in, E_out, E_b, E, F = self._matrix_exponentials(self.b.H,
                                                               m_flow,
                                                               cp,
                                                               nSegments)
            E_out_inv = np.linalg.inv(E_out)
            a_in = np.linalg.multi_dot([E_out_inv,
                                        E_in])
            a_b = np.linalg.multi_dot([E_out_inv,
                                       E_b])
            self._a_in_T0 = a_in
            self._a_b_T0 = a_b
            self._m_flow_T0 = m_flow
            self._cp_T0 = cp
            self._nSegments_T0 = nSegments
        else:
            a_in = self._a_in_T0
            a_b = self._a_b_T0
        return a_in, a_b

    def _matrix_exponentials(self, z, m_flow, cp, nSegments):
        """ Evaluate configuration-specific coefficient matrices
        """
        nPipes = self.nPipes

        if not self._is_SameMatrix(m_flow, cp, nSegments):
            # Coefficient matrix for differential equations
            self._A = 1.0 / (self._Rd * cp)
            for i in range(nPipes):
                self._A[i, :] = self._A[i, :]/m_flow[i]
                self._A[i+nPipes, :] = self._A[i+nPipes, :]/m_flow[i]
            for i in range(2*nPipes):
                self._A[i, i] = -self._A[i, i] - sum(
                    [self._A[i, j] for j in range(2*nPipes) if not i == j])
            for i in range(nPipes, 2*nPipes):
                self._A[i, :] = - self._A[i, :]

            # Eigenvalues and eigenvectors of A
            self._L, self._V = np.linalg.eig(self._A)
            self._Vm1 = np.linalg.inv(self._V)
            self._D = np.diag(self._L)
            self._Dm1 = np.linalg.inv(self._D)

            # Update conditions
            self._m_flow = m_flow
            self._cp = cp
            self._nSegments = nSegments

        Vm1 = self._Vm1
        D = self._D
        Dm1 = self._Dm1
        E = (self._V.dot(np.diag(np.exp(self._L*z)))).dot(Vm1)
        IIm1 = np.concatenate((np.eye(nPipes),
                               -np.eye(nPipes)), 1)
        ZOI = np.concatenate((np.zeros((nPipes, nPipes)),
                              np.eye(nPipes)), 0)
        IOZ = np.concatenate((np.eye(nPipes),
                              np.zeros((nPipes, nPipes))), 0)
        Ones = np.ones((2*nPipes, 1))

        # Coefficient matrix for borehole wall temperatures
        F = np.zeros((2*nPipes, nSegments))
        for v in range(nSegments):
            dz1 = z - min(z, v*self.b.H/nSegments)
            dz2 = z - min(z, (v + 1)*self.b.H/nSegments)
            E1 = np.diag(np.exp(self._L*dz1))
            E2 = np.diag(np.exp(self._L*dz2))
            F[:,v:v+1] = np.linalg.multi_dot([self._V,
                                              Dm1,
                                              E1 - E2,
                                              Vm1,
                                              self._A,
                                              Ones])

        E_in = np.linalg.multi_dot([-IIm1, E, IOZ])
        E_out = np.linalg.multi_dot([IIm1, E, ZOI])

        E_b = np.linalg.multi_dot([IIm1, F])

        return E_in, E_out, E_b, E, F

    def _is_SameMatrix(self, m_flow, cp, nSegments, tol=1.0e-6):
        dm_flow = np.max(np.abs((m_flow - self._m_flow) \
                                / (self._m_flow + 1.0e-30)))
        dcp = abs((cp - self._cp) / self._cp)
        if nSegments == self._nSegments_Q and dm_flow < tol and dcp < tol:
            is_SameMatrix = True
        else:
            is_SameMatrix = False
        return is_SameMatrix

    def _is_same_outlet_temperature(self, m_flow, cp, nSegments, tol=1.0e-6):
        dm_flow = np.max(np.abs((m_flow - self._m_flow_T0) \
                                / (self._m_flow_T0 + 1.0e-30)))
        dcp = abs((cp - self._cp_T0) / self._cp_T0)
        if nSegments == self._nSegments_T0 and dm_flow < tol and dcp < tol:
            is_same_outlet_temperature = True
        else:
            is_same_outlet_temperature = False
        return is_same_outlet_temperature

    def _is_same_heat_extraction_rate(self, m_flow, cp, nSegments, tol=1.0e-6):
        dm_flow = np.max(np.abs((m_flow - self._m_flow_Q) \
                                / (self._m_flow_Q + 1.0e-30)))
        dcp = abs((cp - self._cp_Q) / self._cp_Q)
        if nSegments == self._nSegments_T0 and dm_flow < tol and dcp < tol:
            is_same_heat_extraction_rate = True
        else:
            is_same_heat_extraction_rate = False
        return is_same_heat_extraction_rate


def thermal_resistances(pos, r_out, r_b, k_s, k_g, Rfp, method='LineSource'):
    """
    Evaluate thermal resistances and delta-circuit thermal resistances.

    This function evaluates the thermal resistances and delta-circuit thermal
    resistances between pipes in a borehole. Thermal resistances are defined
    by:

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
    r_out : float
        Outer radius of the pipes (in meters).
    r_b : float
        Borehole radius (in meters).
    k_s : float
        Soil thermal conductivity (in W/m-K).
    k_g : float
        Grout thermal conductivity (in W/m-K).
    Rfp : float
        Fluid-to-outer-pipe-wall thermal resistance (in m-K/W).
    method : str, defaults to 'LineSource'
        Method used to evaluate the thermal resistances:
            'LineSource' : Line source approximation, from [#Hellstrom1991b]_.

    Returns
    -------
    R : array
        Thermal resistances.
    Rd : array
        Delta-circuit thermal resistances.

    Examples
    --------
    >>> pos = [(-0.06, 0.), (0.06, 0.)]
    >>> R, Rd = gt.heat_transfer.thermal_resistances(pos, 0.01, 0.075, 2., 1., 0.1)
    R = [[ 0.36648149, -0.04855895],
         [-0.04855895,  0.36648149]]
    Rd = [[ 0.31792254, -2.71733044],
          [-2.71733044,  0.31792254]]

    References
    ----------
    .. [#Hellstrom1991b] Hellstrom, G. (1991). Ground heat storage. Thermal
       Analyses of Duct Storage Systems I: Theory. PhD Thesis. University of
       Lund, Department of Mathematical Physics. Lund, Sweden.

    """
    if method.lower() == 'linesource':
        n = len(pos)

        R = np.zeros((n, n))
        sigma = (k_g - k_s)/(k_g + k_s)
        for i in range(n):
            xi = pos[i][0]
            yi = pos[i][1]
            for j in range(n):
                xj = pos[j][0]
                yj = pos[j][1]
                if i == j:
                    # Same-pipe thermal resistance
                    r = np.sqrt(xi**2 + yi**2)
                    R[i, j] = Rfp + 1./(2.*pi*k_g) \
                        * (np.log(r_b/r_out) - sigma*np.log(1. - r**2/r_b**2))
                else:
                    # Pipe to pipe thermal resistance
                    r = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    ri = np.sqrt(xi**2 + yi**2)
                    rj = np.sqrt(xj**2 + yj**2)
                    dij = np.sqrt((1. - ri**2/r_b**2)*(1.-rj**2/r_b**2) +
                                  r**2/r_b**2)
                    R[i, j] = -1./(2.*pi*k_g) \
                        * (np.log(r/r_b) + sigma*np.log(dij))

        S = -np.linalg.inv(R)
        for i in range(n):
            S[i, i] = -(S[i, i] +
                        sum([S[i, j] for j in range(n) if not i == j]))
        Rd = 1.0/S

    return R, Rd


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
        if Re * E > 65:
            # Colebrook-White equation for rough pipes
            fDarcy = 0.02
            df = 1.0e99
            while abs(df/fDarcy) > tol:
                one_over_sqrt_f = -2.0 * np.log10(E / 3.7
                                                  + 2.51 / (Re * np.sqrt(fDarcy)))
                fDarcy_new = 1.0 / one_over_sqrt_f**2
                df = fDarcy_new - fDarcy
                fDarcy = fDarcy_new
        else:
            # Blasius equation for smooth pipes
            fDarcy = 0.3164 * Re**(-0.25)

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
    # Nusselt number from Gnielinski
    Nu = 0.125*fDarcy * (Re - 1.0e3) * Pr / \
        (1.0 + 12.7 * np.sqrt(0.125*fDarcy) * (Pr**(2.0/3.0) - 1.0))
    h_fluid = k * Nu / D

    return h_fluid
