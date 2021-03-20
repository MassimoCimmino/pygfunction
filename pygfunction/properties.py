from __future__ import absolute_import, division, print_function

from CoolProp.CoolProp import PropsSI
import warnings


class Fluid:
    """
        An object for handling the fluid properties

        Parameters
        ----------
        T: float, optional
            The temperature of the fluid (Celcius)
        P: float, optional
            The pressure of the fluid (Pascal)
        mixer: str, optional
            The mixer for this application should be one of

                - 'MEG' - Ethylene glycol mixed with water
                - 'MPG' - Propylene glycol mixed with water
                - 'MMA' - Methanol mixed with water
                - 'MEA' - Ethanol mixed with water
        percent: int, optional
            The percentage of the mixing fluid added to the water.
            Lower bound = 0
            Upper bound is dependent on the mixture

        Examples
        ----------
        >>> import pygfunction as gt
        >>> T = 20.     # Temp at 20 C
        >>> gage_P = 20  # PsiG
        >>> atm_P = 14.69595
        >>> P = (gage_P + atm_P) * 6894.75728  # Pressure in Pa

        >>> # pure water
        >>> fluid = gt.properties.Fluid(T=T, P=P)
        >>> print(fluid)

        >>> # 20 % propylene glycol mixed with water
        >>> mix = 'MPG'
        >>> percent = 20
        >>> fluid = gt.properties.Fluid(T=T, P=P, mixer=mix, percent=percent)

        >>> # 60% ethylene glycol mixed with water
        >>> mix = 'MEG'
        >>> percent = 60
        >>> fluid = gt.properties.Fluid(T=T, P=P, mixer=mix, percent=percent)
        >>> print(fluid)

        >>> # 5% methanol mixed with water water
        >>> mix = 'MMA'
        >>> percent = 5
        >>> fluid = gt.properties.Fluid(T=T, P=P, mixer=mix, percent=percent)
        >>> print(fluid)

        >>> # ethanol / water
        >>> mix = 'MEA'
        >>> percent = 10
        >>> fluid = gt.properties.Fluid(T=T, P=P, mixer=mix, percent=percent)
        >>> print(fluid)
    """
    def __init__(self, T: float = 20., P: float = 239220., mixer: str = 'MEG', percent: float = 0):
        self.fluid_mix = 'INCOMP::' + mixer + '-' + str(percent) + '%'
        if mixer == 'MEG':
            pass
        elif mixer == 'MPG':
            pass
        elif mixer == 'MMA':
            pass
        elif mixer == 'MEA':
            pass
        # add in else if's
        else:
            warnings.warn('It is unknown whether or not cool props has the mixing fluid requested, '
                          'proceed with caution.')
        self.T_C = T                                    # temperature of the fluid {C}
        self.T_K = T + 273.15                           # temperature of the fluid {C}
        self.P = P                                      # pressure of the fluid in {Pa}
        self.rho = self.density()                       # Density, {kg/m^3}
        self.mu = self.dynamic_viscosity()              # Dynamic Viscosity, {Pa s} or {N s/ m^2}
        self.nu = self.kinematic_viscosity()            # Kinematic Viscosity, {m^2/s}
        self.cp = self.mass_heat_capacity()             # Mass specific Cp Specific Heat, {J/kg/K}
        self.rhoCp = self.volumetric_heat_capacity()    # Volumetric heat capacity, {kJ/m3/K}
        self.k = self.thermal_conductivity()            # Thermal conductivity, {W/m/K}
        self.Pr = self.Prandlt_number()                 # Prandlt number

    def __repr__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            (str(item) + ' = ' + '{}'.format(self.__dict__[item]) for item in sorted(self.__dict__)))

    def append_to_dict(self, dnary):
        if len(list(dnary.keys())) == 0:
            for item in sorted(self.__dict__):
                dnary[item] = []
        for item in sorted(self.__dict__):
            dnary[item].append('{:.5E}'.format(self.__dict__[item]))

    def density(self):
        return PropsSI('D', 'T', self.T_K, 'P', self.P, self.fluid_mix)

    def dynamic_viscosity(self):
        return PropsSI('V', 'T', self.T_K, 'P', self.P, self.fluid_mix)

    def kinematic_viscosity(self):
        return self.mu / self.rho

    def mass_heat_capacity(self):
        return PropsSI('C', 'T', self.T_K, 'P', self.P, self.fluid_mix)

    def volumetric_heat_capacity(self):
        return self.rho * self.cp / 1000

    def thermal_conductivity(self):
        return PropsSI('L', 'T', self.T_K, 'P', self.P, self.fluid_mix)

    def Prandlt_number(self):
        return PropsSI('PRANDTL', 'T', self.T_K, 'P', self.P, self.fluid_mix)
