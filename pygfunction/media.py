# -*- coding: utf-8 -*-
from scp.ethyl_alcohol import EthylAlcohol
from scp.ethylene_glycol import EthyleneGlycol
from scp.methyl_alcohol import MethylAlcohol
from scp.propylene_glycol import PropyleneGlycol
from scp.water import Water


class Fluid:
    """
        An object for handling the fluid properties

        Parameters
        ----------
        fluid_str: str
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

        Examples
        --------
        >>> import pygfunction as gt
        >>> T_f = 20.     # Temp at 20 C

        >>> # complete water solution
        >>> fluid_str = 'Water'
        >>> percent = 0
        >>> fluid = gt.media.Fluid(fluid_str, percent, T=T_f)
        >>> print(fluid)

        >>> # 20 % propylene glycol mixed with water
        >>> fluid_str = 'MPG'
        >>> percent = 20
        >>> fluid = gt.media.Fluid(fluid_str, percent, T=T_f)

        >>> # 60% ethylene glycol mixed with water
        >>> fluid_str = 'MEG'
        >>> percent = 60
        >>> fluid = gt.media.Fluid(fluid_str, percent, T=T_f)
        >>> print(fluid)

        >>> # 5% methanol mixed with water
        >>> fluid_str = 'MMA'
        >>> percent = 5
        >>> fluid = gt.media.Fluid(fluid_str, percent, T=T_f)
        >>> print(fluid)

        >>> # ethanol / water
        >>> fluid_str = 'MEA'
        >>> percent = 10
        >>> fluid = gt.media.Fluid(fluid_str, percent, T=T_f)
        >>> print(fluid)
    """
    def __init__(self, fluid_str: str, percent: float, T: float = 20.):
        # concentration fraction
        x_frac = percent / 100

        if fluid_str.upper() == 'WATER':
            self.fluid = Water()
        elif fluid_str.upper() in ['PROPYLENEGLYCOL', 'MPG']:
            self.fluid = PropyleneGlycol(x_frac)
        elif fluid_str.upper() in ['ETHYLENEGLYCOL', 'MEG']:
            self.fluid = EthyleneGlycol(x_frac)
        elif fluid_str.upper() in ['METHYLALCOHOL', 'MMA']:
            self.fluid = MethylAlcohol(x_frac)
        elif fluid_str.upper() in ['ETHYLALCOHOL', 'MEA']:
            self.fluid = EthylAlcohol(x_frac)
        else:
            raise ValueError(f'Unsupported fluid mixture: "{fluid_str}".')

        # Initialize all fluid properties
        # Name
        self.name = self.fluid.fluid_name
        # Temperature of the fluid (in Celsius)
        self.T_C = T
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
        return f'{self.__class__!s}\n' \
            + '\n'.join(
                (f'{item!s} = {self.__dict__[item]}'
                 for item in sorted(self.__dict__)))

    def append_to_dict(self, dnary):
        if len(list(dnary.keys())) == 0:
            for item in sorted(self.__dict__):
                dnary[item] = []
        for item in sorted(self.__dict__):
            dnary[item].append(f'{self.__dict__[item]:.5E}')

    def density(self):
        """
        Returns the density of the fluid (in kg/m3).

        Returns
        -------
        rho : float
            Density (in kg/m3).

        """
        return self.fluid.density(self.T_C)

    def dynamic_viscosity(self):
        """
        Returns the dynamic viscosity of the fluid (in Pa.s, or N.s/m2).

        Returns
        -------
        mu : float
            Dynamic viscosity  (in Pa.s, or N.s/m2).

        """
        return self.fluid.viscosity(self.T_C)

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
        return self.fluid.specific_heat(self.T_C)

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
        return self.fluid.conductivity(self.T_C)

    def Prandlt_number(self):
        """
        Returns the Prandtl of the fluid.

        Returns
        -------
        Pr : float
            Prandlt number.

        """
        return self.fluid.prandtl(self.T_C)
