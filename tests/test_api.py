import numpy as np
# from pprint import pprint

from pygfunction.api import PYG

def test_access_borehole_from_api():
    # -- Set up
    # borehole config
    H = 150.                    # Borehole length [m]
    D = 4.                      # Borehole buried depth [m]
    r_b = 0.075                 # Borehole radius [m]
    x = 0.                      # Borehole x-position [m]
    y = 0.                      # Borehole y-position [m]
    tilt = np.pi / 15           # Borehole tilt [rad]
    orientation = np.pi / 3     # Borehole orientation [rad]

    # g-function config
    alpha = 1e-6                # Ground thermal diffusivity [m2/s]

    ts = H**2 / (9 * alpha)

    time = np.array([0.1, 1., 10.]) * ts

    # Set up initial borehole config values
    complete_borehole_config = [H, D, r_b, x, y, tilt, orientation]
    # print(f"{complete_borehole_config=}=")
    simple_borehole_config = [H, D, r_b, x, y]
    # print(f"{simple_borehole_config=}")

    number_of_boreholes = 6
    # only change x, y coordinates between boreholes
    increment_indices = [3, 4]
    # Generate list of lists of borehole configs. Increment each value by 1 to avoid duplicates
    complete_borefield_config = [[x + i if idx in increment_indices else x for idx, x in enumerate(complete_borehole_config)] for i in range(number_of_boreholes)]
    # pprint(f"{complete_borefield_config=}=")
    simple_borefield_config = [[x + i if idx in increment_indices else x for idx, x in enumerate(simple_borehole_config)] for i in range(number_of_boreholes)]
    # pprint(f"{simple_borefield_config=}=")

    # -- Act
    # TODO: submit various options and test the output
    pyg_1 = PYG(complete_borefield_config, alpha, time, solver_method="detailed")
    pyg_2 = PYG(simple_borefield_config, alpha, time)


    # -- Assert
    assert isinstance(pyg_1.to_list(), list)
    assert isinstance(pyg_2.to_list(), list)
