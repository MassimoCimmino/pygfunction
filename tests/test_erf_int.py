from pygfunction.utilities import erf_int, erf_int_old
import numpy as np
from time import perf_counter_ns


def test_erf_int_error():
    print(f'test of erf error and performance')
    x = np.arange(-3.9, 3.9, 0.01)
    tic = perf_counter_ns()
    y_new = erf_int(x)
    toc = perf_counter_ns()
    dt_new1 = toc-tic
    tic = perf_counter_ns()
    y = erf_int_old(x)
    toc = perf_counter_ns()
    dt_old1 = toc - tic
    assert np.allclose(y, y_new, rtol=0.000_000_01)

    print(f'new time {dt_new1 / 1_000_000} ms; old time { dt_old1 / 1_000_000} ms')

    x = np.arange(-500, 500, 5)
    tic = perf_counter_ns()
    y_new = erf_int(x)
    toc = perf_counter_ns()
    dt_new2 = toc - tic
    tic = perf_counter_ns()
    y = erf_int_old(x)
    toc = perf_counter_ns()
    dt_old2 = toc - tic
    assert np.allclose(y, y_new, rtol=0.000_000_01)

    print(f'new time {dt_new2 / 1_000_000} ms; old time {dt_old2 / 1_000_000} ms')

    x = np.arange(-500, 500, 0.01)
    tic = perf_counter_ns()
    y_new = erf_int(x)
    toc = perf_counter_ns()
    dt_new3 = toc - tic
    tic = perf_counter_ns()
    y = erf_int_old(x)
    toc = perf_counter_ns()
    dt_old3 = toc - tic
    assert np.allclose(y, y_new, rtol=0.000_000_01)

    print(f'new time {dt_new3/1_000_000} ms; old time {dt_old3/1_000_000} ms')
