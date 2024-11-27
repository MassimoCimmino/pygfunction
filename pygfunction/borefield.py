# -*- coding: utf-8 -*-
from typing import Union, List, Dict, Self

import numpy as np
import numpy.typing as npt

from .boreholes import Borehole

class Borefield:
    """This class represents the borehole field and can generate inputs for pygfunction"""

    def __init__(
            self, H: npt.ArrayLike, D: npt.ArrayLike, r_b: npt.ArrayLike,
            x: npt.ArrayLike, y: npt.ArrayLike, tilt: npt.ArrayLike = 0,
            orientation: npt.ArrayLike = 0):
        # Convert x and y coordinates to arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        self.nBoreholes = np.maximum(len(x), len(y))

        # Broadcast all variables to arrays of length `nBoreholes`
        self.H = np.broadcast_to(H, self.nBoreholes)
        self.D = np.broadcast_to(D, self.nBoreholes)
        self.r_b = np.broadcast_to(r_b, self.nBoreholes)
        self.x = np.broadcast_to(x, self.nBoreholes)
        self.y = np.broadcast_to(y, self.nBoreholes)
        self.tilt = np.broadcast_to(tilt, self.nBoreholes)
        self.orientation = np.broadcast_to(orientation, self.nBoreholes)

        # Identify tilted boreholes
        self._is_tilted = np.greater(self.orientation, 1e-6)
        pass

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            output_class = Borehole
        else:
            output_class = Borefield
        return output_class(
            self.H[key], self.D[key], self.r_b[key], self.x[key], self.y[key],
            tilt=self.tilt[key], orientation=self.orientation[key])

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
    def from_boreholes(
            cls, boreholes: Union[Borehole, List[Borehole]]) -> Self:
        if isinstance(boreholes, Borehole):
            boreholes = [boreholes]
        H = np.array([b.H for b in boreholes])
        D = np.array([b.D for b in boreholes])
        r_b = np.array([b.r_b for b in boreholes])
        tilt = np.array([b.tilt for b in boreholes])
        orientation = np.array([b.orientation for b in boreholes])
        x = np.array([b.x for b in boreholes])
        y = np.array([b.y for b in boreholes])
        borefield = cls(H, D, r_b, x, y, tilt=tilt, orientation=orientation)
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
