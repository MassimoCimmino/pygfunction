# -*- coding: utf-8 -*-
""" Test suite for gfunction module.
"""
import numpy as np
import pytest

import pygfunction as gt
from pygfunction.gfunction import Method


# =============================================================================
# Test gFunction (vertical boreholes)
# =============================================================================
# Test 'UBWT' g-functions for different bore fields using all solvers,
# unequal/uniform segments, and with/without the FLS approximation
@pytest.mark.parametrize("field, method, opts, expected", [
        #  'equivalent' solver - unequal segments
        ('single_borehole', Method.equivalent, 'unequal_segments', np.array([5.59717446, 6.36257605, 6.60517223])),
        ('single_borehole_short', Method.equivalent, 'unequal_segments', np.array([4.15784411, 4.98477603, 5.27975732])),
        ('ten_boreholes_rectangular', Method.equivalent, 'unequal_segments', np.array([10.89935004, 17.09864925, 19.0795435])),
        # 'equivalent' solver - uniform segments
        ('single_borehole', Method.equivalent, 'uniform_segments', np.array([5.6057331, 6.37369288, 6.61659795])),
        ('single_borehole_short', Method.equivalent, 'uniform_segments', np.array([4.16941861, 4.99989722, 5.29557193])),
        ('ten_boreholes_rectangular', Method.equivalent, 'uniform_segments', np.array([10.96118694, 17.24496533, 19.2536638])),
        #  'equivalent' solver - unequal segments, FLS approximation
        ('single_borehole', Method.equivalent, 'unequal_segments_approx', np.array([5.59717101, 6.36259907, 6.6050007])),
        ('single_borehole_short', Method.equivalent, 'unequal_segments_approx', np.array([4.15784584, 4.98478735, 5.27961509])),
        ('ten_boreholes_rectangular', Method.equivalent, 'unequal_segments_approx', np.array([10.8993464, 17.09872924, 19.0794071])),
        #  'equivalent' solver - uniform segments, FLS approximation
        ('single_borehole', Method.equivalent, 'uniform_segments_approx', np.array([5.60572735, 6.37371464, 6.61642409])),
        ('single_borehole_short', Method.equivalent, 'uniform_segments_approx', np.array([4.16941691, 4.99990922, 5.29542863])),
        ('ten_boreholes_rectangular', Method.equivalent, 'uniform_segments_approx', np.array([10.96117468, 17.2450427, 19.25351959])),
        # 'similarities' solver - unequal segments
        ('single_borehole', Method.similarities, 'unequal_segments', np.array([5.59717446, 6.36257605, 6.60517223])),
        ('single_borehole_short', Method.similarities, 'unequal_segments', np.array([4.15784411, 4.98477603, 5.27975732])),
        ('ten_boreholes_rectangular', Method.similarities, 'unequal_segments', np.array([10.89935004, 17.09864925, 19.0795435])),
        # 'similarities' solver - uniform segments
        ('single_borehole', Method.similarities, 'uniform_segments', np.array([5.6057331, 6.37369288, 6.61659795])),
        ('single_borehole_short', Method.similarities, 'uniform_segments', np.array([4.16941861, 4.99989722, 5.29557193])),
        ('ten_boreholes_rectangular', Method.similarities, 'uniform_segments', np.array([10.96118694, 17.24496533, 19.2536638])),
        # 'similarities' solver - unequal segments, FLS approximation
        ('single_borehole', Method.similarities, 'unequal_segments_approx', np.array([5.59717101, 6.36259907, 6.6050007])),
        ('single_borehole_short', Method.similarities, 'unequal_segments_approx', np.array([4.15784584, 4.98478735, 5.27961509])),
        ('ten_boreholes_rectangular', Method.similarities, 'unequal_segments_approx', np.array([10.89852244, 17.09793569, 19.07814962])),
        # 'similarities' solver - uniform segments, FLS approximation
        ('single_borehole', Method.similarities, 'uniform_segments_approx', np.array([5.60572735, 6.37371464, 6.61642409])),
        ('single_borehole_short', Method.similarities, 'uniform_segments_approx', np.array([4.16941691, 4.99990922, 5.29542863])),
        ('ten_boreholes_rectangular', Method.similarities, 'uniform_segments_approx', np.array([10.96035847, 17.24419784, 19.25220421])),
        # 'detailed' solver - unequal segments
        ('single_borehole', Method.detailed, 'unequal_segments', np.array([5.59717446, 6.36257605, 6.60517223])),
        ('single_borehole_short', Method.detailed, 'unequal_segments', np.array([4.15784411, 4.98477603, 5.27975732])),
        ('ten_boreholes_rectangular', Method.detailed, 'unequal_segments', np.array([10.89935004, 17.09864925, 19.0795435])),
        # 'detailed' solver - uniform segments
        ('single_borehole', Method.detailed, 'uniform_segments', np.array([5.6057331, 6.37369288, 6.61659795])),
        ('single_borehole_short', Method.detailed, 'uniform_segments', np.array([4.16941861, 4.99989722, 5.29557193])),
        ('ten_boreholes_rectangular', Method.detailed, 'uniform_segments', np.array([10.96118694, 17.24496533, 19.2536638])),
        # 'detailed' solver - unequal segments, FLS approximation
        ('single_borehole', Method.detailed, 'unequal_segments_approx', np.array([5.59717101, 6.36259907, 6.6050007])),
        ('single_borehole_short', Method.detailed, 'unequal_segments_approx', np.array([4.15784584, 4.98478735, 5.27961509])),
        ('ten_boreholes_rectangular', Method.detailed, 'unequal_segments_approx', np.array([10.89852244, 17.09793569, 19.07814962])),
        # 'detailed' solver - uniform segments, FLS approximation
        ('single_borehole', Method.detailed, 'uniform_segments_approx', np.array([5.60572735, 6.37371464, 6.61642409])),
        ('single_borehole_short', Method.detailed, 'uniform_segments_approx', np.array([4.16941691, 4.99990922, 5.29542863])),
        ('ten_boreholes_rectangular', Method.detailed, 'uniform_segments_approx', np.array([10.96035847, 17.24419784, 19.25220421])),
    ])
def test_gfunctions_UBWT(field, method, opts, expected, request):
    # Extract the bore field from the fixture
    boreholes = request.getfixturevalue(field)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean([b.h for b in boreholes])
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method=method, options=options,
        boundary_condition='UBWT')
    assert np.allclose(gFunc.gFunc, expected)


# Test 'UHTR' g-functions for different bore fields using all solvers,
# unequal/uniform segments, and with/without the FLS approximation
@pytest.mark.parametrize("field, method, opts, expected", [
        #  'equivalent' solver - unequal segments
        ('single_borehole', Method.equivalent, 'unequal_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', Method.equivalent, 'unequal_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', Method.equivalent, 'unequal_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'equivalent' solver - uniform segments
        ('single_borehole', Method.equivalent, 'uniform_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', Method.equivalent, 'uniform_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', Method.equivalent, 'uniform_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        #  'equivalent' solver - unequal segments, FLS approximation
        ('single_borehole', Method.equivalent, 'unequal_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', Method.equivalent, 'unequal_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', Method.equivalent, 'unequal_segments_approx', np.array([11.27831426, 18.48076919, 21.00650885])),
        #  'equivalent' solver - uniform segments, FLS approximation
        ('single_borehole', Method.equivalent, 'uniform_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', Method.equivalent, 'uniform_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', Method.equivalent, 'uniform_segments_approx', np.array([11.27831426, 18.48076919, 21.00650885])),
        # 'similarities' solver - unequal segments
        ('single_borehole', Method.similarities, 'unequal_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', Method.similarities, 'unequal_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', Method.similarities, 'unequal_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'similarities' solver - uniform segments
        ('single_borehole', Method.similarities, 'uniform_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', Method.similarities, 'uniform_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', Method.similarities, 'uniform_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'similarities' solver - unequal segments, FLS approximation
        ('single_borehole', Method.similarities, 'unequal_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', Method.similarities, 'unequal_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', Method.similarities, 'unequal_segments_approx', np.array([11.27751418, 18.47964006, 21.00475366])),
        # 'similarities' solver - uniform segments, FLS approximation
        ('single_borehole', Method.similarities, 'uniform_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', Method.similarities, 'uniform_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', Method.similarities, 'uniform_segments_approx', np.array([11.27751418, 18.47964006, 21.00475366])),
        # 'detailed' solver - unequal segments
        ('single_borehole', Method.detailed, 'unequal_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', Method.detailed, 'unequal_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', Method.detailed, 'unequal_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'detailed' solver - uniform segments
        ('single_borehole', Method.detailed, 'uniform_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', Method.detailed, 'uniform_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', Method.detailed, 'uniform_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'detailed' solver - unequal segments, FLS approximation
        ('single_borehole', Method.detailed, 'unequal_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', Method.detailed, 'unequal_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', Method.detailed, 'unequal_segments_approx', np.array([11.27751418, 18.47964006, 21.00475366])),
        # 'detailed' solver - uniform segments, FLS approximation
        ('single_borehole', Method.detailed, 'uniform_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', Method.detailed, 'uniform_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', Method.detailed, 'uniform_segments_approx', np.array([11.27751418, 18.47964006, 21.00475366])),
    ])
def test_gfunctions_UHTR(field, method, opts, expected, request):
    # Extract the bore field from the fixture
    boreholes = request.getfixturevalue(field)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean([b.h for b in boreholes])
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method=method, options=options,
        boundary_condition='UHTR')
    assert np.allclose(gFunc.gFunc, expected)


# Test 'MIFT' g-functions for different bore fields using all solvers,
# unequal/uniform segments, and with/without the FLS approximation
# The 'equivalent' solver is not applied to series-connected boreholes.
@pytest.mark.parametrize("field, method, opts, expected", [
        #  'equivalent' solver - unequal segments
        ('single_borehole_network', Method.equivalent, 'unequal_segments', np.array([5.76597302, 6.51058473, 6.73746895])),
        ('single_borehole_network_short', Method.equivalent, 'unequal_segments', np.array([4.17105954, 5.00930075, 5.30832133])),
        ('ten_boreholes_network_rectangular', Method.equivalent, 'unequal_segments', np.array([12.66229998, 18.57852681, 20.33535907])),
        # 'equivalent' solver - uniform segments
        ('single_borehole_network', Method.equivalent, 'uniform_segments', np.array([5.78644676, 6.5311583, 6.75699875])),
        ('single_borehole_network_short', Method.equivalent, 'uniform_segments', np.array([4.17553236, 5.01476781, 5.31381287])),
        ('ten_boreholes_network_rectangular', Method.equivalent, 'uniform_segments', np.array([12.931553, 18.8892892, 20.63810364])),
        #  'equivalent' solver - unequal segments, FLS approximation
        ('single_borehole_network', Method.equivalent, 'unequal_segments_approx', np.array([5.76596769, 6.51061169, 6.73731276])),
        ('single_borehole_network_short', Method.equivalent, 'unequal_segments_approx', np.array([4.17105984, 5.00931374, 5.30816983])),
        ('ten_boreholes_network_rectangular', Method.equivalent, 'unequal_segments_approx', np.array([12.66228879, 18.57863253, 20.33526092])),
        #  'equivalent' solver - uniform segments, FLS approximation
        ('single_borehole_network', Method.equivalent, 'uniform_segments_approx', np.array([5.78644007, 6.53117706, 6.75684456])),
        ('single_borehole_network_short', Method.equivalent, 'uniform_segments_approx', np.array([4.17553118, 5.01478084, 5.31366109])),
        ('ten_boreholes_network_rectangular', Method.equivalent, 'uniform_segments_approx', np.array([12.93153466, 18.88937176, 20.63801163])),
        # 'similarities' solver - unequal segments
        ('single_borehole_network', Method.similarities, 'unequal_segments', np.array([5.76597302, 6.51058473, 6.73746895])),
        ('single_borehole_network_short', Method.similarities, 'unequal_segments', np.array([4.17105954, 5.00930075, 5.30832133])),
        ('ten_boreholes_network_rectangular', Method.similarities, 'unequal_segments', np.array([12.66229998, 18.57852681, 20.33535907])),
        ('ten_boreholes_network_rectangular_series', Method.similarities, 'unequal_segments', np.array([3.19050169, 8.8595362, 10.84379419])),
        # 'similarities' solver - uniform segments
        ('single_borehole_network', Method.similarities, 'uniform_segments', np.array([5.78644676, 6.5311583, 6.75699875])),
        ('single_borehole_network_short', Method.similarities, 'uniform_segments', np.array([4.17553236, 5.01476781, 5.31381287])),
        ('ten_boreholes_network_rectangular', Method.similarities, 'uniform_segments', np.array([12.931553, 18.8892892, 20.63810364])),
        ('ten_boreholes_network_rectangular_series', Method.similarities, 'uniform_segments', np.array([3.22186337, 8.92628494, 10.91644607])),
        # 'similarities' solver - unequal segments, FLS approximation
        ('single_borehole_network', Method.similarities, 'unequal_segments_approx', np.array([5.76596769, 6.51061169, 6.73731276])),
        ('single_borehole_network_short', Method.similarities, 'unequal_segments_approx', np.array([4.17105984, 5.00931374, 5.30816983])),
        ('ten_boreholes_network_rectangular', Method.similarities, 'unequal_segments_approx', np.array([12.66136057, 18.57792276, 20.33429034])),
        ('ten_boreholes_network_rectangular_series', Method.similarities, 'unequal_segments_approx', np.array([3.19002567, 8.85783554, 10.84222402])),
        # 'similarities' solver - uniform segments, FLS approximation
        ('single_borehole_network', Method.similarities, 'uniform_segments_approx', np.array([5.78644007, 6.53117706, 6.75684456])),
        ('single_borehole_network_short', Method.similarities, 'uniform_segments_approx', np.array([4.17553118, 5.01478084, 5.31366109])),
        ('ten_boreholes_network_rectangular', Method.similarities, 'uniform_segments_approx', np.array([12.93064329, 18.88844718, 20.63710493])),
        ('ten_boreholes_network_rectangular_series', Method.similarities, 'uniform_segments_approx', np.array([3.22139028, 8.92448715, 10.91485947])),
        # 'detailed' solver - unequal segments
        ('single_borehole_network', Method.detailed, 'unequal_segments', np.array([5.76597302, 6.51058473, 6.73746895])),
        ('single_borehole_network_short', Method.detailed, 'unequal_segments', np.array([4.17105954, 5.00930075, 5.30832133])),
        ('ten_boreholes_network_rectangular', Method.detailed, 'unequal_segments', np.array([12.66229998, 18.57852681, 20.33535907])),
        ('ten_boreholes_network_rectangular_series', Method.detailed, 'unequal_segments', np.array([3.19050169, 8.8595362, 10.84379419])),
        # 'detailed' solver - uniform segments
        ('single_borehole_network', Method.detailed, 'uniform_segments', np.array([5.78644676, 6.5311583, 6.75699875])),
        ('single_borehole_network_short', Method.detailed, 'uniform_segments', np.array([4.17553236, 5.01476781, 5.31381287])),
        ('ten_boreholes_network_rectangular', Method.detailed, 'uniform_segments', np.array([12.931553, 18.8892892, 20.63810364])),
        ('ten_boreholes_network_rectangular_series', Method.detailed, 'uniform_segments', np.array([3.22186337, 8.92628494, 10.91644607])),
        # 'detailed' solver - unequal segments, FLS approximation
        ('single_borehole_network', Method.detailed, 'unequal_segments_approx', np.array([5.76596769, 6.51061169, 6.73731276])),
        ('single_borehole_network_short', Method.detailed, 'unequal_segments_approx', np.array([4.17105984, 5.00931374, 5.30816983])),
        ('ten_boreholes_network_rectangular', Method.detailed, 'unequal_segments_approx', np.array([12.66136057, 18.57792276, 20.33429034])),
        ('ten_boreholes_network_rectangular_series', Method.detailed, 'unequal_segments_approx', np.array([3.19002567, 8.85783554, 10.84222402])),
        # 'detailed' solver - uniform segments, FLS approximation
        ('single_borehole_network', Method.detailed, 'uniform_segments_approx', np.array([5.78644007, 6.53117706, 6.75684456])),
        ('single_borehole_network_short', Method.detailed, 'uniform_segments_approx', np.array([4.17553118, 5.01478084, 5.31366109])),
        ('ten_boreholes_network_rectangular', Method.detailed, 'uniform_segments_approx', np.array([12.93064329, 18.88844718, 20.63710493])),
        ('ten_boreholes_network_rectangular_series', Method.detailed, 'uniform_segments_approx', np.array([3.22139028, 8.92448715, 10.91485947])),
    ])
def test_gfunctions_MIFT(field, method, opts, expected, request):
    # Extract the bore field from the fixture
    network = request.getfixturevalue(field)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean([b.h for b in network.b])
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = gt.gfunction.gFunction(
        network, alpha, time=time, method=method, options=options,
        boundary_condition='MIFT')
    assert np.allclose(gFunc.gFunc, expected)


# =============================================================================
# Test gFunction (inclined boreholes)
# =============================================================================
# Test 'UBWT' g-functions for a field of inclined boreholes using all solvers,
# unequal/uniform segments, and with/without the FLS approximation
@pytest.mark.parametrize("method, opts, expected", [
        # 'similarities' solver - unequal segments
        (Method.similarities, 'unequal_segments', np.array([5.67249989, 6.72866814, 7.15134705])),
        # 'similarities' solver - uniform segments
        (Method.similarities, 'uniform_segments', np.array([5.68324619, 6.74356205, 7.16738741])),
        # 'similarities' solver - unequal segments, FLS approximation
        (Method.similarities, 'unequal_segments_approx', np.array([5.66984803, 6.72564218, 7.14826009])),
        # 'similarities' solver - uniform segments, FLS approximation
        (Method.similarities, 'uniform_segments_approx', np.array([5.67916493, 6.7395222 , 7.16339216])),
        # 'detailed' solver - unequal segments
        (Method.detailed, 'unequal_segments', np.array([5.67249989, 6.72866814, 7.15134705])),
        # 'detailed' solver - uniform segments
        (Method.detailed, 'uniform_segments', np.array([5.68324619, 6.74356205, 7.16738741])),
        # 'detailed' solver - unequal segments, FLS approximation
        (Method.detailed, 'unequal_segments_approx', np.array([5.66984803, 6.72564218, 7.14826009])),
        # 'detailed' solver - uniform segments, FLS approximation
        (Method.detailed, 'uniform_segments_approx', np.array([5.67916493, 6.7395222 , 7.16339216])),
    ])
def test_gfunctions_UBWT(two_boreholes_inclined, method, opts, expected, request):
    # Extract the bore field from the fixture
    boreholes = two_boreholes_inclined
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean([b.h for b in boreholes])
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method=method, options=options,
        boundary_condition='UBWT')
    assert np.allclose(gFunc.gFunc, expected)
