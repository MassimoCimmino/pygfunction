from __future__ import absolute_import, division, print_function

from CoolProp.CoolProp import PropsSI
import warnings
import json
import os
import numpy as np


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


class Pipe:
    """
    Define a pipe from either standard definitions or user defined definitions.

    Parameters
    ----------
    k : float
        Pipe thermal conductivity (in W/m.K).
    standard : str, optional
        Either standard or user-defined.
        Default is 'user-defined'.
    schedule : str, optional
        Either SDR-11 or Schedule-40.
        Default is None
    nominal : str, optional
        The nominal size of the pipe (in inches).
    r_in : float, optional
        Pipe inner radius (in meters).
    r_out : float, optional
        Pipe outer radius (in meters).

    .. Note:
        If the standard is 'standard', then schedule and nominal are required.
        If the standard is 'user-defined' then r_in and r_out are required.

    Examples
    --------
    >>> import pygfunction as gt
    >>> pipe = gt.media.Pipe(k=0.4, standard='standard', schedule='SDR-11', \
        nominal='0.75')
    >>> pipe = gt.media.Pipe(k=0.4, standard='user-defined', r_in=0.0108, \
        r_out=0.0133)
    """
    def __init__(self, k, standard='user-defined', schedule=None,
                 nominal=None, r_in=None, r_out=None):
        if standard == 'standard':
            if schedule is None or nominal is None:
                raise ValueError('Please provide arguments for schedule and'
                                 'nominal.')
            self.schedule = schedule
            self.nominal = nominal
            self.HDPEPipeDimensions = {}
            self.access_pipe_dimensions()
            self.r_in, self.r_out = self.retrieve_pipe_dimensions()
        elif standard == 'user-defined':
            if r_in is None or r_out is None:
                raise ValueError('Please provide arguments for r_in and '
                                 'r_out.')
            self.r_in = r_in
            self.r_out = r_out
        else:
            raise ValueError('The options for standard are standard or user-'
                             'defined')
        self.k = k
        self.R = self.conduction_thermal_resistance_circular_pipe()

    def conduction_thermal_resistance_circular_pipe(self):
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
        R_pipe = np.log(self.r_out / self.r_in) / (2 * np.pi * self.k)

        return R_pipe

    def access_pipe_dimensions(self):
        path_to_hdpe = os.path.dirname(os.path.abspath(__file__))
        file_name = 'HDPEPipeDimensions.json'
        try:
            self.HDPEPipeDimensions = self.js_r(path_to_hdpe + r'/' +
                                                file_name)
        except:
            self.HDPEPipeDimensions = self.js_r(path_to_hdpe + r'\\' +
                                                file_name)

    def retrieve_pipe_dimensions(self):
        # get the specific pipe
        pipe = self.HDPEPipeDimensions[self.schedule]
        # get the index of the nominal pipe in the nominal pipe list
        idx = pipe['Nominal Size (in)'].index(self.nominal)
        r_in = pipe['Inside Diameter (mm)'][idx] / 2000.
        r_out = pipe['Outer Diameter (mm)'][idx] / 2000.
        return r_in, r_out

    @staticmethod
    def js_r(filename):
        with open(filename) as f_in:
            return json.load(f_in)
