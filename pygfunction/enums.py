from enum import Enum, auto
from typing import Annotated


class PipeType(Enum):
    """Enumerator for pipe configuration type."""
    COAXIAL_ANNULAR_IN: Annotated[
        int, "Coaxial pipe (annular inlet)"
        ] = auto()
    COAXIAL_ANNULAR_OUT: Annotated[
        int, "Coaxial pipe (annular outlet)"
        ] = auto()
    DOUBLE_UTUBE_PARALLEL: Annotated[
        int, "Double U-tube (parallel)"
        ] = auto()
    DOUBLE_UTUBE_SERIES: Annotated[
        int, "Double U-tube (series)"
        ] = auto()
    SINGLE_UTUBE: Annotated[
        int, "Single U-tube"
        ] = auto()
