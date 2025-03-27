# -*- coding: utf-8 -*-
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
except ImportError as err:
    raise ImportError(
        "Matplotlib is required for plotting. Install it with "
        "`pip install pygfunction[plot]`"
    ) from err

__all__ = ["plt", "AutoMinorLocator"]
