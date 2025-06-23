from typing import List, Union, Dict

import numpy.typing as npt

from .borefield import Borefield
from .enums import PipeType
from .media import Fluid
from .networks import Network
from .pipes import get_pipes


class GroundHeatExchanger:

    def __init__(self,
                 H: npt.ArrayLike,
                 D: npt.ArrayLike,
                 r_b: npt.ArrayLike,
                 x: npt.ArrayLike,
                 y: npt.ArrayLike,
                 pipe_type: PipeType,
                 pos: List[tuple],
                 r_in: Union[float, tuple, list],
                 r_out: Union[float, tuple, list],
                 k_s: float,
                 k_g: float,
                 k_p: Union[float, tuple, list],
                 fluid_name: str,
                 fluid_concentration_pct: float,
                 m_flow_ghe,
                 epsilon,
                 tilt: npt.ArrayLike = 0.,
                 orientation: npt.ArrayLike = 0.,
                 reversible_flow: bool = True,
                 ):
        self.borefield = Borefield(H, D, r_b, x, y, tilt, orientation)
        self.boreholes = self.borefield.to_boreholes()
        self.fluid = Fluid(fluid_name, fluid_concentration_pct)
        self.pipes = get_pipes(
            self.boreholes,
            pipe_type,
            pos,
            r_in,
            r_out,
            k_s,
            k_g,
            k_p,
            m_flow_ghe,
            epsilon,
            self.fluid,
            reversible_flow
        )
        self.network = Network(self.boreholes, self.pipes, m_flow_network=m_flow_ghe, cp_f=self.fluid.cp)

    def evaluate_g_function(self, alpha: float,
                            time: npt.ArrayLike,
                            method: str = "equivalent",
                            boundary_condition: str = "UBWT",
                            options: Union[Dict[str, str], None] = None):
        if options is None:
            options = {}

        if boundary_condition in ["UBWT", "UHTR"]:
            return self.borefield.evaluate_g_function(alpha, time, method, boundary_condition, options)
        elif boundary_condition == "MIFT":
            return self.network.evaluate_g_function(alpha, time, method, boundary_condition, options)
        else:
            raise ValueError("Unsupported boundary_condition. Must be 'UBWT', 'UHTR' or 'MIFT'")
