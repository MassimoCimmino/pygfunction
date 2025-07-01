from enum import Enum, auto


class PipeType(Enum):
    SINGLEUTUBE = auto()
    DOUBLEUTUBEPARALLEL = auto()
    DOUBLEUTUBESERIES = auto()
    COAXIALANNULARINLET = auto()
    COAXIALPIPEINLET = auto()
