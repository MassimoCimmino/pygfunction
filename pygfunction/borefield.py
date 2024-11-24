# -*- coding: utf-8 -*-
from typing import Union, List, Dict, Self

import numpy as np
import numpy.typing as npt

from .boreholes import Borehole

class Borefield:
    """This class represents the borehole field and can generate inputs for pygfunction"""

    def __init__(self, boreholes: Union[Borehole, List[Borehole]]):
        if isinstance(boreholes, Borehole):
            boreholes = [boreholes]
        self.boreholes = boreholes
        self.nBoreholes = len(boreholes)
        self.H = np.array([b.H for b in boreholes])
        self.D = np.array([b.D for b in boreholes])
        self.r_b = np.array([b.r_b for b in boreholes])
        self.tilt = np.array([b.tilt for b in boreholes])
        self.orientation = np.array([b.orientation for b in boreholes])
        self.x = np.array([b.x for b in boreholes])
        self.y = np.array([b.y for b in boreholes])
        self._is_tilted = np.array(
            [b._is_tilted for b in boreholes], dtype=bool)
        pass

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Borefield(self.boreholes[key])
        elif isinstance(key, int):
            return self.boreholes[key]
        elif isinstance(key, (list, np.ndarray)):
            return Borefield([self.boreholes[i] for i in key])
        return

    def __len__(self):
         return self.nBoreholes

    @classmethod
    def from_file(cls, filename: str) -> Self:
        # Load data from file
        data = np.loadtxt(filename, ndmin=2)
        # Build the bore field
        borefield = []
        for line in data:
            x = line[0]
            y = line[1]
            H = line[2]
            D = line[3]
            r_b = line[4]
            # Previous versions of pygfunction only required up to line[4].
            # Now check to see if tilt and orientation exist.
            if len(line) == 7:
                tilt = line[5]
                orientation = line[6]
            else:
                tilt = 0.
                orientation = 0.
        borefield = cls.from_lists(
            H, D, r_b, x, y, tilt=tilt, orientation=orientation)
        return borefield

    @classmethod
    def from_lists(
            cls, H: npt.ArrayLike, D: npt.ArrayLike, r_b: npt.ArrayLike,
            x: npt.ArrayLike, y: npt.ArrayLike, tilt: npt.ArrayLike = 0,
            orientation: npt.ArrayLike = 0) -> Self:
        # Convert x and y coordinates to arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        nBoreholes = np.maximum(len(x), len(y))

        def to_array(var):
            """Converts `var` to array of length `nBoreholes`"""
            if np.isscalar(var) or (
                    isinstance(var, (np.ndarray, list)) and len(var)==1):
                var = np.full(nBoreholes, var)
            assert len(var) == nBoreholes, \
                f"Expected list of length {nBoreholes}, but got {len(var)}."
            return var

        # Convert all variables to arrays of length `nBoreholes`
        H = to_array(H)
        D = to_array(D)
        r_b = to_array(r_b)
        x = to_array(x)
        y = to_array(y)
        tilt = to_array(tilt)
        orientation = to_array(orientation)

        # Create list of boreholes
        boreholes = [
            Borehole(
                _H, _D, _r_b, _x, _y, tilt=_tilt, orientation=_orientation)
            for _H, _D, _r_b, _x, _y, _tilt, _orientation in zip(
                    H, D, r_b, x, y, tilt, orientation)]
        borefield = cls(boreholes)
        return borefield

    def evaluate_g_function(
            self,
            alpha: float,
            time: npt.ArrayLike,
            options: Union[Dict[str, str], None] = None,
            method: str = "equivalent",
            boundary_condition: str = "UBWT"):
        """
        Generates g-function values

        Parameters
        ----------
        alpha: float
            soil thermal diffusivity, in m^2/s
        time: list[float]
            time interval values for computing g-function values
        options: Union[Dict[str, str], None]
            Optional argument, options dict containing options for g-function computation
        method: str
            optional argument, solver method for g-function computation. default: "equivalent".
            other options: "similarities" or "detailed"
        boundary_condition: str
            optional argument, boundary condition for g-function computation, default: "UBWT"
            other options: "UHTR" or "MIFT",

        Returns
        ----------
        list of g-function values
        """
        from .gfunction import gFunction
        if options is None:
            options = {}

        gfunc = gFunction(
            self,
            alpha,
            time=time,
            method=method,
            boundary_condition=boundary_condition,
            options=options,
        )

        return gfunc
