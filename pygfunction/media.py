from __future__ import absolute_import, division, print_function

from CoolProp.CoolProp import PropsSI
import warnings


class Fluid:
    """
        An object for handling the fluid properties

        Parameters
        ----------
        mixer: str
            The mixer for this application should be one of:
                - 'Water' - Complete water solution
                - 'MEG' - Ethylene glycol mixed with water
                - 'MPG' - Propylene glycol mixed with water
                - 'MEA' - Ethanol mixed with water
                - 'MMA' - Methanol mixed with water
        percent: float
            Mass fraction of the mixing fluid added to water (in %).
            Lower bound = 0. Upper bound is dependent on the mixture.
        T: float, optional
            The temperature of the fluid (in Celcius).
            Default is 20 degC.
        P: float, optional
            The pressure of the fluid (in Pa).
            Default is 101325 Pa.

        Examples
        ----------
        >>> import pygfunction as gt
        >>> T = 20.     # Temp at 20 C
        >>> gage_P = 20  # PsiG
        >>> atm_P = 14.69595
        >>> P = (gage_P + atm_P) * 6894.75728  # Pressure in Pa

        >>> # complete water solution
        >>> mix = 'Water'
        >>> percent = 0
        >>> fluid = gt.media.Fluid(mix, percent, T=T, P=P)
        >>> print(fluid)

        >>> # 20 % propylene glycol mixed with water
        >>> mix = 'MPG'
        >>> percent = 20
        >>> fluid = gt.media.Fluid(mix, percent, T=T, P=P)

        >>> # 60% ethylene glycol mixed with water
        >>> mix = 'MEG'
        >>> percent = 60
        >>> fluid = gt.media.Fluid(mix, percent, T=T, P=P)
        >>> print(fluid)

        >>> # 5% methanol mixed with water water
        >>> mix = 'MMA'
        >>> percent = 5
        >>> fluid = gt.media.Fluid(mix, percent, T=T, P=P)
        >>> print(fluid)

        >>> # ethanol / water
        >>> mix = 'MEA'
        >>> percent = 10
        >>> fluid = gt.media.Fluid(mix, percent, T=T, P=P)
        >>> print(fluid)
    """
    def __init__(self, mixer: str, percent: float,
                 T: float = 20., P: float = 101325.):
        if mixer == 'Water':
            self.fluid_mix = mixer
        elif mixer in ['MEG', 'MPG', 'MMA', 'MEA']:  # Expected brines
            self.fluid_mix = 'INCOMP::' + mixer + '-' + str(percent) + '%'
        else:
            warnings.warn('It is unknown whether or not cool props has the '
                          'mixing fluid requested, proceed with caution.')
        # Initialize all fluid properties
        # Temperature of the fluid (in Celsius)
        self.T_C = T
        # Temperature of the fluid (in Kelvin)
        self.T_K = T + 273.15
        # Pressure of the fluid (in Pa)
        self.P = P
        # Density (in kg/m3)
        self.rho = self.density()
        # Dynamic viscosity  (in Pa.s, or N.s/m2)
        self.mu = self.dynamic_viscosity()
        # Kinematic viscosity (in m2/s)
        self.nu = self.kinematic_viscosity()
        # Specific isobaric heat capacity (J/kg.K)
        self.cp = self.specific_heat_capacity()
        # Volumetric heat capacity (in J/m3.K)
        self.rhoCp = self.volumetric_heat_capacity()
        # Thermal conductivity (in W/m.K)
        self.k = self.thermal_conductivity()
        # Prandlt number
        self.Pr = self.Prandlt_number()

    def __repr__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            (str(item) + ' = ' + '{}'.format(
                self.__dict__[item]) for item in sorted(self.__dict__)))

    def append_to_dict(self, dnary):
        if len(list(dnary.keys())) == 0:
            for item in sorted(self.__dict__):
                dnary[item] = []
        for item in sorted(self.__dict__):
            dnary[item].append('{:.5E}'.format(self.__dict__[item]))

    def density(self):
        """
        Returns the density of the fluid (in kg/m3).

        Returns
        -------
        rho : float
            Density (in kg/m3).

        """
        return PropsSI('D', 'T', self.T_K, 'P', self.P, self.fluid_mix)

    def dynamic_viscosity(self):
        """
        Returns the dynamic viscosity of the fluid (in Pa.s, or N.s/m2).

        Returns
        -------
        mu : float
            Dynamic viscosity  (in Pa.s, or N.s/m2).

        """
        return PropsSI('V', 'T', self.T_K, 'P', self.P, self.fluid_mix)

    def kinematic_viscosity(self):
        """
        Returns the kinematic viscosity of the fluid (in m2/s).

        Returns
        -------
        nu : float
            Kinematic viscosity (in m2/s).

        """
        return self.mu / self.rho

    def specific_heat_capacity(self):
        """
        Returns the specific isobaric heat capacity of the fluid (J/kg.K).

        Returns
        -------
        cp : float
            Specific isobaric heat capacity (J/kg.K).

        """
        return PropsSI('C', 'T', self.T_K, 'P', self.P, self.fluid_mix)

    def volumetric_heat_capacity(self):
        """
        Returns the volumetric heat capacity of the fluid (J/m3.K).

        Returns
        -------
        rhoCp : float
            Volumetric heat capacity (in J/m3.K).

        """
        return self.rho * self.cp

    def thermal_conductivity(self):
        """
        Returns the thermal conductivity of the fluid (in W/m.K).

        Returns
        -------
        k : float
            Thermal conductivity (in W/m.K).

        """
        return PropsSI('L', 'T', self.T_K, 'P', self.P, self.fluid_mix)

    def Prandlt_number(self):
        """
        Returns the Prandtl of the fluid.

        Returns
        -------
        Pr : float
            Prandlt number.

        """
        return PropsSI('PRANDTL', 'T', self.T_K, 'P', self.P, self.fluid_mix)
