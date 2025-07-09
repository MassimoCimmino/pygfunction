from enum import Enum, auto


class PipeType(Enum):
    COAXIAL_ANNULAR_IN = auto()
    COAXIAL_ANNULAR_OUT = auto()
    DOUBLE_UTUBE_PARALLEL = auto()
    DOUBLE_UTUBE_SERIES = auto()
    SINGLE_UTUBE = auto()
