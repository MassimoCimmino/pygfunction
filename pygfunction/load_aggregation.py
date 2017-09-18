from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.interpolate as interp


class _LoadAggregation(object):

    def __init__(self, dt, tmax, nSources=1):
        self.dt = dt                # Simulation time step
        self.tmax = tmax            # Maximum simulation time
        self.nSources = nSources    # Number of heat sources
        return

    def initialize(self, time, g):
        raise NotImplementedError(
            'initialize class method not implemented, this '
            'method should do the start of simulation operations.')

    def get_times_for_simulation(self, time, g):
        raise NotImplementedError(
            'get_times_for_simulation class method not implemented, this '
            'method should return a list of time values at which the '
            'thermal response factors are needed.')

    def next_time_step(self, time):
        raise NotImplementedError(
            'next_time_step class method not implemented, this '
            'method should do the start of time step operations.')

    def set_current_load(self, Q):
        raise NotImplementedError(
            'set_current_load class method not implemented, this '
            'method should do the operations needed when setting current '
            'loads.')

    def temporal_superposition(self):
        raise NotImplementedError(
            'temporal_superposition class method not implemented, this '
            'method should return the borehole wall tempreatures at the '
            'current time step.')
