from enum import Enum, auto


class GHEType(Enum):
    BOREHOLE = auto()
    BOREFIElD = auto()
    NETWORK = auto()


class PipeType(Enum):
    SINGLEUTUBE = auto()
    DOUBLEUTUBEPARALLEL = auto()
    DOUBLEUTUBESERIES = auto()
    COAXIALANNULARINLET = auto()
    COAXIALPIPEINLET = auto()
