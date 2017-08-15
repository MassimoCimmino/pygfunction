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

    def get_temperature(self, z, Tin, Tb, m_flow, fluid):
        """
        Returns the fluid temperatures of the borehole.

        Parameters
        ----------
        z : float
            Depth (in meters) to evaluate the fluid temperatures.
        Tin : array
            Array of inlet fluid temperatures (in Celsius).
        Tb : array
            Array of borehole wall temperatures (in Celsius).
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties

        Returns
        -------
        Tf : array
            Fluid temperature (in Celsius) in each pipe.

        """
        Tb = np.reshape(Tb, (-1, 1))
        nSegments = len(Tb)
        Tin = np.reshape(Tin, (-1, 1))
        # Build coefficient matrices
        a_in, a_b = self.coefficients_temperature(z,
                                                  m_flow,
                                                  fluid,
                                                  nSegments)
        # Evaluate fluid temperatures
        Tf = (a_in.dot(Tin) + a_b.dot(Tb)).flatten()
        return Tf

    def get_outlet_temperature(self, Tin, Tb, m_flow, fluid):
        """
        Returns the outlet fluid temperatures of the borehole.

        Parameters
        ----------
        Tin : array
            Array of inlet fluid temperatures (in Celsius).
        Tb : array
            Array of borehole wall temperatures (in Celsius).
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties

        Returns
        -------
        Tout : array
            Outlet fluid temperatures (in Celsius) from each outlet pipe.

        """
        Tb = np.reshape(Tb, (-1, 1))
        nSegments = len(Tb)
        Tin = np.reshape(Tin, (-1, 1))
        # Build coefficient matrices
        a_in, a_b = self.coefficients_outlet_temperature(m_flow,
                                                         fluid,
                                                         nSegments)
        # Evaluate outlet temperatures
        Tout = (a_in.dot(Tin) + a_b.dot(Tb)).flatten()
        return Tout

    def get_heat_extraction_rate(self, Tin, Tb, m_flow, fluid):
        """
        Returns the heat extraction rates of the borehole.

        Parameters
        ----------
        Tin : array
            Array of inlet fluid temperatures (in Celsius).
        Tb : array
            Array of borehole wall temperatures (in Celsius).
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties

        Returns
        -------
        Qb : array
            Array of heat extraction rates along each borehole segment.

        """
        Tb = np.reshape(Tb, (-1, 1))
        nSegments = len(Tb)
        Tin = np.reshape(Tin, (-1, 1))
        a_in, a_b = self.coefficients_heat_extraction_rate(m_flow,
                                                           fluid,
                                                           nSegments)
        Qb = (a_in.dot(Tin) + a_b.dot(Tb)).flatten()
        return Qb

    def get_total_heat_extraction_rate(self, Tin, Tb, m_flow, fluid):
        """
        Returns the total heat extraction rate of the borehole.

        Parameters
        ----------
        Tin : array
            Array of inlet fluid temperatures (in Celsius).
        Tb : array
            Array of borehole wall temperatures (in Celsius).
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties

        Returns
        -------
        Q : float
            Total net heat extraction rate of the borehole.

        """
        cp = fluid.get_SpecificIsobaricHeatCapacity()
        Tout = self.get_outlet_temperature(Tin, Tb, m_flow, fluid)
        Q = m_flow * cp * (Tout - Tin).flatten()
        return Q

    def coefficients_heat_extraction_rate(self, m_flow, fluid, nSegments):
        """ Returns coefficients for the relation
            [Q_Segments] = [a_in] * [Tin] + [a_b] * [Tb]
        """
        raise NotImplementedError(
            'coefficients_heat_extraction_rate class method not implemented, '
            'this method should return matrices for the relation: '
            '[Q_Segments] = [a_in] * [Tin] + [a_b] * [Tb]')

    def coefficients_outlet_temperature(self, m_flow, fluid, nSegments):
        """ Returns coefficients for the relation
            [Tout] = [a_in] * [Tin] + [a_b] * [Tb]
        """
        raise NotImplementedError(
            'coefficients_outlet_temperature class method not implemented, '
            'this method should return matrices for the relation: '
            '[Tout] = [a_in].[Tin] + [a_b].[Tb]')

    def coefficients_temperature(self, z, m_flow, fluid, nSegments):
        """ Returns coefficients for the relation
            [Tf](z) = [a_in] * [Tin] + [a_b] * [Tb]
        """
        raise NotImplementedError(
            'coefficients_temperature class method not implemented, '
            'this method should return matrices for the relation: '
            '[Tf](z) = [a_in].[Tin] + [a_b].[Tb]')


class SingleUTube(_BasePipe):
    """
    Class for single U-Tube boreholes.

    Contains information regarding the physical dimensions and thermal
    characteristics of the pipes and the grout material, as well as methods to
    evaluate fluid temperatures and heat extraction rates based on the work of
    Hellstrom [#Hellstrom1991]_.

    The outlet fluid temperature is given by:

        .. math::

            T_{f,out} = \\frac{f_1(H)+f_2(H)}{f_3(H)-f_2(H)} T_{f,in}
            + \\int_0^H \\frac{T_b(z)[f_4(H-z)+f_5(H-z)]}{f_3(H)-f_2(H)}dz

    with:

        .. math::

            f_1(z) &= e^{\\beta z}[cosh(\\gamma z) - \\delta sinh(\\gamma z)]

            f_2(z) &= e^{\\beta z}
            [\\frac{\\beta_{12}}{\\gamma} sinh(\\gamma z)]

            f_3(z) &= e^{\\beta z}[cosh(\\gamma z) + \\delta sinh(\\gamma z)]

            f_4(z) &= e^{\\beta z}[\\beta_1 cosh(\\gamma z) -
            (\\delta\\beta_1+\\frac{\\beta_2\\beta_{12}}{\\gamma})
            sinh(\\gamma z)]

            f_5(z) &= e^{\\beta z}[\\beta_2 cosh(\\gamma z) +
            (\\delta\\beta_2+\\frac{\\beta_1\\beta_{12}}{\\gamma})
            sinh(\\gamma z)]

    and:

        .. math::

            \\beta_1 &= \\frac{1}{R_{11}^\\Delta \\dot{m} c_p}

            \\beta_2 &= \\frac{1}{R_{22}^\\Delta \\dot{m} c_p}

            \\beta_{12} &= \\frac{1}{R_{12}^\\Delta \\dot{m} c_p}

            \\beta &= \\frac{\\beta_2 - \\beta_1}{2}

            \\gamma &= \\sqrt{\\frac{(\\beta_1 + \\beta_2)^2}{4} +
            \\beta_{12} (\\beta_1 + \\beta_2)}

            \\delta &= \\frac{1}{\\gamma} (\\beta_{12}
            + \\frac{\\beta_1 + \\beta_2}{2})

    The fluid temperature in each pipe are then calculated from:

        .. math::

            T_{f,1}(z) &= T_{f,in} f_1(z) + T_{f,out} f_2(z)
            + \\int_0^z T_b(\\zeta) f_4(z-\\zeta) d\\zeta

            T_{f,2}(z) &= - T_{f,in} f_2(z) + T_{f,out} f_3(z)
            - \\int_0^z T_b(\\zeta) f_5(z-\\zeta) d\\zeta

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

    def coefficients_heat_extraction_rate(self, m_flow, fluid, nSegments):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_b} = \\mathbf{A_{T_f}} T_{f,in}
                + \\mathbf{A_{T_b}} \\mathbf{T_b}

        with:

            .. math::

                \\mathbf{A_{T_f}} = \\left(\\mathbf{M}
                (\\mathbf{B_{i,T_f}} - \\mathbf{B_{i+1,T_f}})\\right)^T

                \\mathbf{A_{T_b}} = \\left(\\mathbf{M}
                (\\mathbf{B_{i,T_b}} - \\mathbf{B_{i+1,T_b}})\\right)^T

                \\mathbf{M} = \\left[ \\begin{matrix}
                -\\dot{m} c_p & \\dot{m} c_p
                \\end{matrix} \\right]

        and with :math:`\\mathbf{B_{i,T_f}}` and :math:`\\mathbf{B_{i,T_b}}`
        the coefficient matrices returned by the
        coefficients_temperature evaluated at :math:`z=i\\frac{H}{N}`.

        Parameters
        ----------
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties.
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array for inlet fluid temperature.
        a_b : array
            Array for borehole wall temperatures.

        """
        self._hellstrom_coefficients(m_flow, fluid)
        cp = fluid.get_SpecificIsobaricHeatCapacity()
        M = m_flow*cp*np.array([[-1.0, 1.0]])
        # Initialize coefficient matrices
        a_in = np.zeros((nSegments, 1))
        a_b = np.zeros((nSegments, nSegments))
        # Heat extraction rates are calculated from an energy balance on a
        # borehole segment.
        for i in range(nSegments):
            z1 = i * self.b.H / nSegments
            z2 = (i + 1) * self.b.H / nSegments
            aTf1, bTf1 = self.coefficients_temperature(z1, m_flow,
                                                       fluid, nSegments)
            aTf2, bTf2 = self.coefficients_temperature(z2, m_flow,
                                                       fluid, nSegments)
            a_in[i, :] = M.dot(aTf1 - aTf2)
            a_b[i, :] = M.dot(bTf1 - bTf2)
        return a_in, a_b

    def coefficients_outlet_temperature(self, m_flow, fluid, nSegments):
        """
        Build coefficient matrices to evaluate outlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                T_{f,out} = \\mathbf{A_{T_f}} T_{f,in}
                + \\mathbf{A_{T_b}} \\mathbf{T_b}

        with:

            .. math::

                \\mathbf{A_{T_f}} &= \\left[ \\begin{matrix}
                \\frac{f_1(H)+f2(H)}{f_3(H)-f_2(H)}
                \\end{matrix} \\right]

                \\mathbf{A_{T_b}} &= \\left[ \\begin{matrix}
                a_{0,T_b} & \\cdots & a_{N-1,T_b}
                \\end{matrix} \\right]

                a_{i,T_b} &=
                \\int_{(N-i-1)\\frac{H}{N}}^{(N-i)\\frac{H}{N}}
                \\frac{f_4(z) + f_5(z)}{f_3(H)-f_2(H)} dz

        Parameters
        ----------
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties.
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array for inlet fluid temperature.
        a_b : array
            Array for borehole wall temperatures.

        """
        self._hellstrom_coefficients(m_flow, fluid)
        a_in = ((self._f1(self.b.H) + self._f2(self.b.H))
                / (self._f3(self.b.H) - self._f2(self.b.H)))
        a_in = np.array([[a_in]])
        a_b = np.zeros((1, nSegments))
        for i in range(nSegments):
            z1 = (nSegments - i - 1) * self.b.H / nSegments
            z2 = (nSegments - i) * self.b.H / nSegments
            dF4 = self._F4(z2) - self._F4(z1)
            dF5 = self._F5(z2) - self._F5(z1)
            a_b[0, i] = (dF4 + dF5) / (self._f3(self.b.H) - self._f2(self.b.H))
        return a_in, a_b

    def coefficients_temperature(self, z, m_flow, fluid, nSegments):
        """
        Build coefficient matrices to evaluate fluid temperatures.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{A_{T_f}} T_{f,in}
                + \\mathbf{A_{T_b}} \\mathbf{T_b}

        with:

            .. math::

                \\mathbf{A_{T_f}} &= \\mathbf{C_{T_f}}\\mathbf{B_{T_f}}

                \\mathbf{A_{T_b}} &= \\mathbf{C_{T_f}}\\mathbf{B_{T_b}}
                + \\mathbf{C_{T_b}}

        where the coefficient matrices :math:`\\mathbf{B}` are calculated
        from the matrices return by the coefficients_outlet_temperature method:

            .. math::

                \\mathbf{B_{T_f}} = \\left[ \\begin{matrix}
                1 \\\\
                \\frac{f_1(H)+f2(H)}{f_3(H)-f_2(H)}
                \\end{matrix} \\right]

                \\mathbf{B_{T_b}} = \\left[ \\begin{matrix}
                0 & \\cdots & 0 \\\\
                b_{0,T_b} & \\cdots & b_{N-1,T_b}
                \\end{matrix} \\right]

                b_{i,T_b} =
                \\int_{(N-i-1)\\frac{H}{N}}^{(N-i)\\frac{H}{N}}
                \\frac{f_4(z) + f_5(z)}{f_3(H)-f_2(H)} dz

        and the coefficient matrices :math:`\\mathbf{C}` are given by:

            .. math::

                \\mathbf{C_{T_f}} = \\left[ \\begin{matrix}
                f_1(z) & f_2(z) \\\\
                -f_2(z) & f_3(z)
                \\end{matrix} \\right]

                \\mathbf{C_{T_b}} = \\left[ \\begin{matrix}
                c_{0,0,T_b} & \\cdots & c_{0,N-1,T_b} \\\\
                c_{1,0,T_b} & \\cdots & c_{1,N-1,T_b}
                \\end{matrix} \\right]

                c_{0,i,T_b} =
                \\int_{max\\left(0,z-(i+1)\\frac{H}{N}\\right)}
                ^{max\\left(0,z-i\\frac{H}{N}\\right)}
                f_4(z) dz

                c_{1,i,T_b} =
                - \\int_{max\\left(0,z-(i+1)\\frac{H}{N}\\right)}
                ^{max\\left(0,z-i\\frac{H}{N}\\right)}
                f_5(z) dz

        Parameters
        ----------
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties.
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array for inlet fluid temperature.
        a_b : array
            Array for borehole wall temperatures.

        """
        self._hellstrom_coefficients(m_flow, fluid)
        # Intermediate matrices for [Tf](z=0) = [b_in] * Tin + [b_b] * [Tb]
        b0_in, b0_b = self.coefficients_outlet_temperature(m_flow,
                                                           fluid,
                                                           nSegments)
        b_in = np.concatenate([[[1.]], b0_in])
        b_b = np.concatenate([np.zeros((1, nSegments)), b0_b])
        # Intermediate matrices for [Tf](z) = [c_f] * [Tf](z=0) + [c_b] * [Tb]
        c_f = np.array([[self._f1(z), self._f2(z)],
                        [-self._f2(z), self._f3(z)]])
        c_b = np.zeros((2, nSegments))
        N = int(np.ceil(z/self.b.H*nSegments))
        for i in range(N):
            z1 = z - min((i+1)*self.b.H/nSegments, z)
            z2 = z - i * self.b.H / nSegments
            dF4 = self._F4(z2) - self._F4(z1)
            dF5 = self._F5(z2) - self._F5(z1)
            c_b[0, i] = dF4
            c_b[1, i] = -dF5
        # Final matrices for [Tf](z) = [a_in] * Tin + [a_b] * [Tb]
        a_in = c_f.dot(b_in)
        a_b = (c_f.dot(b_b) + c_b)
        return a_in, a_b

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

    def _hellstrom_coefficients(self, m_flow, fluid):
        """
        Evaluate dimensionless resistances for Hellstrom solution.
        """
        cp = fluid.get_SpecificIsobaricHeatCapacity()
        self._beta1 = 1./(self._Rd[0][0]*m_flow*cp)
        self._beta2 = 1./(self._Rd[1][1]*m_flow*cp)
        self._beta12 = 1./(self._Rd[0][1]*m_flow*cp)
        self._beta = 0.5*(self._beta2 - self._beta1)
        self._gamma = np.sqrt(0.25*(self._beta1+self._beta2)**2
                              + self._beta12*(self._beta1+self._beta2))
        self._delta = 1./self._gamma \
            * (self._beta12 + 0.5*(self._beta1+self._beta2))


class MultipleUTube(_BasePipe):
    """
    Class for multiple U-Tube boreholes.

    Contains information regarding the physical dimensions and thermal
    characteristics of the pipes and the grout material, as well as methods to
    evaluate fluid temperatures and heat extraction rates based on the work of
    Cimmino [#Cimmino2016]_ for boreholes with any number of U-Tubes.

    The outlet fluid temperatures are given by:

        .. math::

            \\mathbf{E_{out}}(H)\\mathbf{T_{f,out}} =
            \\mathbf{E_{in}}(H)\\mathbf{T_{f,in}}
            + \\int_{0}^{H} \\left[ \\mathbf{I_{n_{pipes}}},
            -\\mathbf{I_{n_{pipes}}} \\right]
            \\mathbf{E}(H-z) \\mathbf{A} \\mathbf{1_{2n_{pipes}\\times 1}}
            T_b(z) dz

    where:

        .. math::

            \\mathbf{E}(z) = exp(\\mathbf{A}z)
            = \\mathbf{V} exp(\\mathbf{D}z) \\mathbf{V}^{-1}

    is the matrix exponential of :math:`\\mathbf{A}z`, :math:`\\mathbf{V}`
    is the matrix of column eigenvectors of :math:`\\mathbf{A}` and
    :math:`\\mathbf{D}` is a diagonal matrix of the eigenvalues of
    :math:`\\mathbf{A}`. :math:`\\mathbf{T_{f,in}}` and
    :math:`\\mathbf{T_{f,out}}` are column vectors of the fluid inlet and
    outlet temperatures, :math:`\\mathbf{I_{n_{pipes}}}` is the
    :math:`n_{pipes} \\times n_{pipes}` identity matrix,
    :math:`\\mathbf{1_{2n_{pipes}}}` is column vector of ones of length
    :math:`n_{pipes}` and :math:`n_{pipes}` is the number of U-tubes in the
    borehole

    :math:`\\mathbf{A}=[A_{i,j}]` is the coefficient matrix of the linear
    system of differential equations for the fluid temperature, with:

        .. math::

            A_{i,j} = \\begin{cases}
            \\frac{-S_{ij}}{\\dot{m}_i c_{p,i}}
            &
            \\text{if} \\,\\, i \\leq n_{pipes}
            \\\\
            \\frac{S_{ij}}{\\dot{m}_{i-n_{pipes}} c_{p,i-n_{pipes}}}
            &
            \\text{if} \\,\\, n_{pipes} < i \\leq 2n_{pipes}
            \\end{cases}

    where :math:`\\mathbf{S} = [S_{i,j}] = \\mathbf{R}^{-1}` is the inverse of
    the thermal resistance matrix.

    :math:`\\mathbf{E_{in}}(z)=[E_{in,i,j}(z)]` and
    :math:`\\mathbf{E_{out}}(z)=[E_{out,i,j}(z)]` are given by:

        .. math::

            E_{in,i,j}(z) &= E_{i+n_{pipes},j}(z) - E_{i,j}

            E_{out,i,j}(z) &= E_{i,j+n_{pipes}}(z)
            - E_{i+n_{pipes},j+n_{pipes}}(z)

    The fluid temperature in each pipe are then calculated from:

        .. math::

            \\mathbf{T_{f}}(z) = \\mathbf{E}(z) \\mathbf{T_{f}}(0)
            -\\int_0^z \\mathbf{E}(z-\\zeta) \\mathbf{A}
            \\mathbf{1_{2n_{pipes}\\times 1}} T_b(\\zeta) d\\zeta

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
    config : str
        Configuration of the U-Tube pipes, parallel or series. Defaults to
        parallel.
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

    def coefficients_heat_extraction_rate(self, m_flow, fluid, nSegments):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_b} = \\mathbf{A_{T_f}} T_{f,in}
                + \\mathbf{A_{T_b}} \\mathbf{T_b}

        with:

            .. math::

                \\mathbf{A_{T_f}} = \\left(\\mathbf{M}
                (\\mathbf{B_{i,T_f}} - \\mathbf{B_{i+1,T_f}})\\right)^T

                \\mathbf{A_{T_b}} = \\left(\\mathbf{M}
                (\\mathbf{B_{i,T_b}} - \\mathbf{B_{i+1,T_b}})\\right)^T

        :math:`\\mathbf{B_{i,T_f}}` and :math:`\\mathbf{B_{i,T_b}}`
        the coefficient matrices returned by the
        coefficients_temperature evaluated at :math:`z=i\\frac{H}{N}`.

        :math:`\\mathbf{M}` is a configuration dependent mass flow rate vector.

        For U-tubes in parallel:

            .. math::

                \\mathbf{M} = \\frac{\\dot{m} c_p}{n_{pipes}}
                \\left[ \\begin{matrix}
                -\\mathbf{1 \\times 1_{n_{pipes}}} &
                \\mathbf{1 \\times 1_{n_{pipes}}}
                \\end{matrix} \\right]

        For U-tubes in series:

            .. math::

                \\mathbf{M} = {\\dot{m} c_p}
                \\left[ \\begin{matrix}
                -\\mathbf{1 \\times 1_{n_{pipes}}} &
                \\mathbf{1 \\times 1_{n_{pipes}}}
                \\end{matrix} \\right]

        Parameters
        ----------
        m_flow : array
            Inlet mass flow rate (in kg/s).
        fluid : fluid object
            Object with fluid properties.
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array for inlet fluid temperature.
        a_b : array
            Array for borehole wall temperatures.

        """
        if self.config.lower() == 'parallel':
            m_flow_pipe = m_flow / self.nPipes
        else:
            m_flow_pipe = m_flow
        cp = fluid.get_SpecificIsobaricHeatCapacity()
        M = m_flow_pipe*cp*np.concatenate([-np.ones((1, self.nPipes)),
                                           np.ones((1, self.nPipes))], axis=1)

        aQ = np.zeros((nSegments, self.nInlets))
        bQ = np.zeros((nSegments, nSegments))

        for i in range(nSegments):
            z1 = i * self.b.H / nSegments
            z2 = (i + 1) * self.b.H / nSegments
            aTf1, bTf1 = self.coefficients_temperature(z1,
                                                       m_flow,
                                                       fluid,
                                                       nSegments)
            aTf2, bTf2 = self.coefficients_temperature(z2,
                                                       m_flow,
                                                       fluid,
                                                       nSegments)
            aQ[i, :] = M.dot(aTf1 - aTf2)
            bQ[i, :] = M.dot(bTf1 - bTf2)
        return aQ, bQ

    def coefficients_temperature(self, z, m_flow, fluid, nSegments):
        """
        Build coefficient matrices to evaluate fluid temperatures.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{A_{T_f}} T_{f,in}
                + \\mathbf{A_{T_b}} \\mathbf{T_b}

        with:

            .. math::

                \\mathbf{A_{T_f}} &= \\mathbf{E}(z)\\mathbf{B_{T_f}}

                \\mathbf{A_{T_b}} &= \\mathbf{E}(z)\\mathbf{B_{T_b}}
                + \\mathbf{C_{T_b}}

        where :math:`\\mathbf{B_{T_f}}` and :math:`\\mathbf{B_{T_b}}` are
        configuration-specific coefficient matrices.

        For U-tubes in parallel:

            .. math::

                \\mathbf{B_{T_f}} = \\begin{bmatrix}
                \\mathbf{1_{n_{pipes} \\times 1}} \\\\
                \\mathbf{E_{out}^\\prime}^{-1} \\mathbf{E_{in}^\\prime}
                \\end{bmatrix}

            .. math::

                \\mathbf{B_{T_f}} = \\begin{bmatrix}
                \\mathbf{0_{n_{pipes} \\times N}} \\\\
                \\mathbf{E_{out}^\\prime}^{-1} \\mathbf{E_{in}^\\prime}
                \\end{bmatrix}

        For U-tubes in series:

            .. math::

                \\mathbf{B_{T_f}} = \\begin{bmatrix}
                \\begin{bmatrix}
                1 \\\\ 0 \\\\ \\vdots \\\\ 0
                \\end{bmatrix}
                + 
                \\begin{bmatrix}
                0 & 0 & \\cdots & 0 \\\\
                1 & 0 & \\cdots & 0 \\\\
                \\vdots & \\ddots & \\ddots & \\vdots \\\\
                0 & \\cdots & 1 & 0
                \\end{bmatrix}
                \\mathbf{E_{out}^\\prime}^{-1}
                \\mathbf{E_{in}^\\prime} \\\\
                \\mathbf{E_{out}^\\prime}^{-1}
                \\mathbf{E_{in}^\\prime}
                \\end{bmatrix}

                \\mathbf{B_{T_b}} = \\begin{bmatrix}
                \\begin{bmatrix}
                0 & 0 & \\cdots & 0 \\\\
                1 & 0 & \\cdots & 0 \\\\
                \\vdots & \\ddots & \\ddots & \\vdots \\\\
                0 & \\cdots & 1 & 0
                \\end{bmatrix}
                \\mathbf{E_{out}^\\prime}^{-1}
                \\mathbf{E_{b}} \\\\
                \\mathbf{E_{out}^\\prime}^{-1}
                \\mathbf{E_{b}}
                \\end{bmatrix}

        The coefficient matrix :math:`\\mathbf{C_{T_b}}` is given by:

            .. math::

                \\mathbf{C_{T_b}} &= -\\mathbf{V}
                \\mathbf{D}^{-1}
                \\mathbf{\\Delta E_i}(z)
                \\mathbf{V}^{-1}
                \\mathbf{A}
                \\mathbf{1_{2n_{pipes} \\times 1}}

                \\mathbf{\\Delta E_i}(z) &=
                exp\\left(\\mathbf{D} \\, max\\left(
                z-\\frac{iH}{N}, 0
                \\right)\\right)
                -exp\\left(\\mathbf{D} \\, max\\left(
                z-\\frac{(i+1)H}{N}, 0
                \\right)\\right)

        Parameters
        ----------
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties.
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array for inlet fluid temperature.
        a_b : array
            Array for borehole wall temperatures.

        """
        self._matrixExponentialsInOut(m_flow, fluid, nSegments)
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

    def coefficients_outlet_temperature(self, m_flow, fluid, nSegments):
        """
        Build coefficient matrices to evaluate outlet fluid temperature.

        Returns coefficients for the relation:

            .. math::

                T_{f,out} = \\mathbf{A_{T_f}} T_{f,in}
                + \\mathbf{A_{T_b}} \\mathbf{T_b}

        where :math:`\\mathbf{A_{T_f}}` and :math:`\\mathbf{A_{T_b}}` are
        configuration-specific matrices.

        For U-tubes in parallel:

            .. math::

                \\mathbf{A_{T_f}} &= \\frac{1}{n_{pipes}}
                \\mathbf{1_{1 \\times n_{pipes}}}
                \\mathbf{E_{out}^\\prime}^{-1}(H)
                \\mathbf{E_{in}^\\prime}(H)

                \\mathbf{A_{T_b}} &=
                \\mathbf{E_{out}^\\prime}^{-1}(H)
                \\mathbf{E_{b}}

                \mathbf{E_{in}^\\prime}(H) &= \mathbf{E_{in}}(H)
                \\mathbf{1_{n_{pipes} \\times 1}}

                \\mathbf{E_{out}^\\prime}(H) &= \\mathbf{E_{out}}(H)

        For U-tubes in series:

            .. math::

                \\mathbf{A_{T_f}} &=
                [\\mathbf{0_{1 \\times n_{pipes}-1}},
                \\mathbf{1_{1 \\times 1}}]
                \\mathbf{E_{out}^\\prime}^{-1}(H)
                \\mathbf{E_{in}^\\prime}(H)

                \\mathbf{A_{T_b}} &=
                [\\mathbf{0_{1 \\times n_{pipes}-1}},
                \\mathbf{1_{1 \\times 1}}]
                \\mathbf{E_{out}^\\prime}^{-1}(H)
                \\mathbf{E_{b}}

                \mathbf{E_{in}^\\prime}(H) &= \mathbf{E_{in}}(H)
                \\left[ \\begin{matrix}
                \\mathbf{0_{n_{pipes}-1 \\times 1}} \\\\
                \\mathbf{1_{1 \\times 1}}
                \\end{matrix} \\right]

                E_{out,i,j}^\\prime(H) &= E_{out,i,j}(H)
                - E_{in,i,j+1}(H)
                \\,\\text{for}\\,0 \\leq i \\leq n_{pipes}-1

        :math:`\\mathbf{E_{b}}` is the coefficient matrix for borehole wall
        temperatures:

            .. math::
                \\mathbf{E_{b}} &= [\\mathbf{E_{b,0}},
                \\cdots ,
                \\mathbf{E_{b,N-1}}]

                \\mathbf{E_{b,i}} &= [\\mathbf{I_{n_{pipes}}},
                -\\mathbf{I_{n_{pipes}}}]
                \\mathbf{V}
                \\mathbf{D}^{-1}
                \\mathbf{\\Delta E_i}(H)
                \\mathbf{V}^{-1}
                \\mathbf{A}
                \\mathbf{1_{2n_{pipes} \\times 1}}

                \\mathbf{\\Delta E_i}(H) &=
                exp\\left(\\mathbf{D}\\left(
                H-\\frac{iH}{N}
                \\right)\\right)
                -exp\\left(\\mathbf{D}\\left(
                H-\\frac{(i+1)H}{N}
                \\right)\\right)

        Parameters
        ----------
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties.
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array for inlet fluid temperature.
        a_b : array
            Array for borehole wall temperatures.

        """
        self._matrixExponentialsInOut(m_flow, fluid, nSegments)

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

    def _matrixExponentialsInOut(self, m_flow, fluid, nSegments):
        """ Evaluate configuration-specific coefficient matrices
        """
        cp = fluid.get_SpecificIsobaricHeatCapacity()
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

    def coefficients_heat_extraction_rate(self, m_flow, fluid, nSegments):
        """
        Build coefficient matrices to evaluate heat extraction rates.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{Q_b} = \\mathbf{A_{T_f}} T_{f,in}
                + \\mathbf{A_{T_b}} \\mathbf{T_b}

        with:

            .. math::

                \\mathbf{A_{T_f}} = \\left(\\mathbf{M}
                (\\mathbf{B_{i,T_f}} - \\mathbf{B_{i+1,T_f}})\\right)^T

                \\mathbf{A_{T_b}} = \\left(\\mathbf{M}
                (\\mathbf{B_{i,T_b}} - \\mathbf{B_{i+1,T_b}})\\right)^T

        :math:`\\mathbf{B_{i,T_f}}` and :math:`\\mathbf{B_{i,T_b}}`
        the coefficient matrices returned by the
        coefficients_temperature evaluated at :math:`z=i\\frac{H}{N}`.

        Parameters
        ----------
        m_flow : array
            Inlet mass flow rate (in kg/s).
        fluid : fluid object
            Object with fluid properties.
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array for inlet fluid temperature.
        a_b : array
            Array for borehole wall temperatures.

        """
        cp = fluid.get_SpecificIsobaricHeatCapacity()
        
        if not self._is_same_heat_extraction_rate(m_flow, cp, nSegments):
            m_flow_pipe = np.reshape(m_flow, (1, self.nPipes))
            M = cp*np.concatenate([-m_flow_pipe,
                                   m_flow_pipe], axis=1)
    
            aQ = np.zeros((nSegments, self.nInlets))
            bQ = np.zeros((nSegments, nSegments))

            z1 = 0.
            aTf1, bTf1 = self.coefficients_temperature(z1,
                                                       m_flow,
                                                       fluid,
                                                       nSegments)
            for i in range(nSegments):
                z2 = (i + 1) * self.b.H / nSegments
                aTf2, bTf2 = self.coefficients_temperature(z2,
                                                           m_flow,
                                                           fluid,
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

    def coefficients_temperature(self, z, m_flow, fluid, nSegments):
        """
        Build coefficient matrices to evaluate fluid temperatures.

        Returns coefficients for the relation:

            .. math::

                \\mathbf{T_f}(z) = \\mathbf{A_{T_f}} T_{f,in}
                + \\mathbf{A_{T_b}} \\mathbf{T_b}

        with:

            .. math::

                \\mathbf{A_{T_f}} &= \\mathbf{E}(z)\\mathbf{B_{T_f}}

                \\mathbf{A_{T_b}} &= \\mathbf{E}(z)\\mathbf{B_{T_b}}
                + \\mathbf{C_{T_b}}

        Parameters
        ----------
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties.
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array for inlet fluid temperature.
        a_b : array
            Array for borehole wall temperatures.

        """
        # Intermediate matrices for [Tf](z=0) = [b_in] * Tin + [b_b] * [Tb]
        b_in0, b_b0 = self.coefficients_outlet_temperature(m_flow,
                                                           fluid,
                                                           nSegments)
        b_in = np.concatenate((np.eye(self.nPipes),
                               b_in0), axis=0)
        b_b = np.concatenate((np.zeros((self.nPipes, nSegments)),
                              b_b0), axis=0)

        # Final matrices for [Tf](z) = [a_in] * Tin + [a_b] * [Tb]
        E, F = self._matrixExponentials(z,
                                        m_flow,
                                        fluid,
                                        nSegments)[-2:]
        a_in = E.dot(b_in)
        a_b = E.dot(b_b) - F

        return a_in, a_b

    def coefficients_outlet_temperature(self, m_flow, fluid, nSegments):
        """
        Build coefficient matrices to evaluate outlet fluid temperatures.

        Returns coefficients for the relation:

            .. math::

                T_{f,out} = \\mathbf{A_{T_f}} T_{f,in}
                + \\mathbf{A_{T_b}} \\mathbf{T_b}

        Parameters
        ----------
        m_flow : array
            Array inlet mass flow rates (in kg/s).
        fluid : fluid object
            Object with fluid properties.
        nSegments : int
            Number of borehole segments.

        Returns
        -------
        a_in : array
            Array for inlet fluid temperature.
        a_b : array
            Array for borehole wall temperatures.

        """
        cp = fluid.get_SpecificIsobaricHeatCapacity()

        if not self._is_same_outlet_temperature(m_flow, cp, nSegments):
            E_in, E_out, E_b, E, F = self._matrixExponentials(self.b.H,
                                                              m_flow,
                                                              fluid,
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

    def _matrixExponentials(self, z, m_flow, fluid, nSegments):
        """ Evaluate configuration-specific coefficient matrices
        """
        cp = fluid.get_SpecificIsobaricHeatCapacity()
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
        List of positions (tuple, in meters) (x,y) of pipes around the center
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
            'LineSource' : Line source approximation, from [#Hellstrom1991]_.

    Returns
    -------
    R : array
        Matrix of thermal resistances.
    Rd : array
        Matrix of delta-circuit thermal resistances.

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
    .. [#Hellstrom1991] Hellstrom, G. (1991). Ground heat storage. Thermal
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
