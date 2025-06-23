# -*- coding: utf-8 -*-
""" Test suite for gfunction module.
"""
import numpy as np
import pytest

import pygfunction as gt


# =============================================================================
# Test gFunction (vertical boreholes)
# =============================================================================
# Test 'UBWT' g-functions for different bore fields using all solvers,
# unequal/uniform segments, and with/without the FLS approximation
@pytest.mark.parametrize("field, method, opts, expected", [
        #  'equivalent' solver - unequal segments
        ('single_borehole', 'equivalent', 'unequal_segments', np.array([5.59717446, 6.36257605, 6.60517223])),
        ('single_borehole_short', 'equivalent', 'unequal_segments', np.array([4.15784411, 4.98477603, 5.27975732])),
        ('ten_boreholes_rectangular', 'equivalent', 'unequal_segments', np.array([10.89935004, 17.09864925, 19.0795435])),
        # 'equivalent' solver - uniform segments
        ('single_borehole', 'equivalent', 'uniform_segments', np.array([5.6057331, 6.37369288, 6.61659795])),
        ('single_borehole_short', 'equivalent', 'uniform_segments', np.array([4.16941861, 4.99989722, 5.29557193])),
        ('ten_boreholes_rectangular', 'equivalent', 'uniform_segments', np.array([10.96118694, 17.24496533, 19.2536638])),
        #  'equivalent' solver - unequal segments, FLS approximation
        ('single_borehole', 'equivalent', 'unequal_segments_approx', np.array([5.59717101, 6.36259907, 6.6050007])),
        ('single_borehole_short', 'equivalent', 'unequal_segments_approx', np.array([4.15784584, 4.98478735, 5.27961509])),
        ('ten_boreholes_rectangular', 'equivalent', 'unequal_segments_approx', np.array([10.8993464, 17.09872924, 19.0794071])),
        #  'equivalent' solver - uniform segments, FLS approximation
        ('single_borehole', 'equivalent', 'uniform_segments_approx', np.array([5.60572735, 6.37371464, 6.61642409])),
        ('single_borehole_short', 'equivalent', 'uniform_segments_approx', np.array([4.16941691, 4.99990922, 5.29542863])),
        ('ten_boreholes_rectangular', 'equivalent', 'uniform_segments_approx', np.array([10.96117468, 17.2450427, 19.25351959])),
        # 'similarities' solver - unequal segments
        ('single_borehole', 'similarities', 'unequal_segments', np.array([5.59717446, 6.36257605, 6.60517223])),
        ('single_borehole_short', 'similarities', 'unequal_segments', np.array([4.15784411, 4.98477603, 5.27975732])),
        ('ten_boreholes_rectangular', 'similarities', 'unequal_segments', np.array([10.89935004, 17.09864925, 19.0795435])),
        # 'similarities' solver - uniform segments
        ('single_borehole', 'similarities', 'uniform_segments', np.array([5.6057331, 6.37369288, 6.61659795])),
        ('single_borehole_short', 'similarities', 'uniform_segments', np.array([4.16941861, 4.99989722, 5.29557193])),
        ('ten_boreholes_rectangular', 'similarities', 'uniform_segments', np.array([10.96118694, 17.24496533, 19.2536638])),
        # 'similarities' solver - unequal segments, FLS approximation
        ('single_borehole', 'similarities', 'unequal_segments_approx', np.array([5.59717101, 6.36259907, 6.6050007])),
        ('single_borehole_short', 'similarities', 'unequal_segments_approx', np.array([4.15784584, 4.98478735, 5.27961509])),
        ('ten_boreholes_rectangular', 'similarities', 'unequal_segments_approx', np.array([10.89852244, 17.09793569, 19.07814962])),
        # 'similarities' solver - uniform segments, FLS approximation
        ('single_borehole', 'similarities', 'uniform_segments_approx', np.array([5.60572735, 6.37371464, 6.61642409])),
        ('single_borehole_short', 'similarities', 'uniform_segments_approx', np.array([4.16941691, 4.99990922, 5.29542863])),
        ('ten_boreholes_rectangular', 'similarities', 'uniform_segments_approx', np.array([10.96035847, 17.24419784, 19.25220421])),
        # 'detailed' solver - unequal segments
        ('single_borehole', 'detailed', 'unequal_segments', np.array([5.59717446, 6.36257605, 6.60517223])),
        ('single_borehole_short', 'detailed', 'unequal_segments', np.array([4.15784411, 4.98477603, 5.27975732])),
        ('ten_boreholes_rectangular', 'detailed', 'unequal_segments', np.array([10.89935004, 17.09864925, 19.0795435])),
        # 'detailed' solver - uniform segments
        ('single_borehole', 'detailed', 'uniform_segments', np.array([5.6057331, 6.37369288, 6.61659795])),
        ('single_borehole_short', 'detailed', 'uniform_segments', np.array([4.16941861, 4.99989722, 5.29557193])),
        ('ten_boreholes_rectangular', 'detailed', 'uniform_segments', np.array([10.96118694, 17.24496533, 19.2536638])),
        # 'detailed' solver - unequal segments, FLS approximation
        ('single_borehole', 'detailed', 'unequal_segments_approx', np.array([5.59717101, 6.36259907, 6.6050007])),
        ('single_borehole_short', 'detailed', 'unequal_segments_approx', np.array([4.15784584, 4.98478735, 5.27961509])),
        ('ten_boreholes_rectangular', 'detailed', 'unequal_segments_approx', np.array([10.89852244, 17.09793569, 19.07814962])),
        # 'detailed' solver - uniform segments, FLS approximation
        ('single_borehole', 'detailed', 'uniform_segments_approx', np.array([5.60572735, 6.37371464, 6.61642409])),
        ('single_borehole_short', 'detailed', 'uniform_segments_approx', np.array([4.16941691, 4.99990922, 5.29542863])),
        ('ten_boreholes_rectangular', 'detailed', 'uniform_segments_approx', np.array([10.96035847, 17.24419784, 19.25220421])),
    ])
def test_gfunctions_UBWT(field, method, opts, expected, request):
    # Extract the bore field from the fixture
    borefield = request.getfixturevalue(field)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean(borefield.H)
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = gt.gfunction.gFunction(
        borefield, alpha, time=time, method=method, options=options,
        boundary_condition='UBWT')
    assert np.allclose(gFunc.gFunc, expected)


# Test 'UHTR' g-functions for different bore fields using all solvers,
# unequal/uniform segments, and with/without the FLS approximation
@pytest.mark.parametrize("field, method, opts, expected", [
        #  'equivalent' solver - unequal segments
        ('single_borehole', 'equivalent', 'unequal_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', 'equivalent', 'unequal_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', 'equivalent', 'unequal_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'equivalent' solver - uniform segments
        ('single_borehole', 'equivalent', 'uniform_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', 'equivalent', 'uniform_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', 'equivalent', 'uniform_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        #  'equivalent' solver - unequal segments, FLS approximation
        ('single_borehole', 'equivalent', 'unequal_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', 'equivalent', 'unequal_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', 'equivalent', 'unequal_segments_approx', np.array([11.27831426, 18.48076919, 21.00650885])),
        #  'equivalent' solver - uniform segments, FLS approximation
        ('single_borehole', 'equivalent', 'uniform_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', 'equivalent', 'uniform_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', 'equivalent', 'uniform_segments_approx', np.array([11.27831426, 18.48076919, 21.00650885])),
        # 'similarities' solver - unequal segments
        ('single_borehole', 'similarities', 'unequal_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', 'similarities', 'unequal_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', 'similarities', 'unequal_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'similarities' solver - uniform segments
        ('single_borehole', 'similarities', 'uniform_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', 'similarities', 'uniform_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', 'similarities', 'uniform_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'similarities' solver - unequal segments, FLS approximation
        ('single_borehole', 'similarities', 'unequal_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', 'similarities', 'unequal_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', 'similarities', 'unequal_segments_approx', np.array([11.27751418, 18.47964006, 21.00475366])),
        # 'similarities' solver - uniform segments, FLS approximation
        ('single_borehole', 'similarities', 'uniform_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', 'similarities', 'uniform_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', 'similarities', 'uniform_segments_approx', np.array([11.27751418, 18.47964006, 21.00475366])),
        # 'detailed' solver - unequal segments
        ('single_borehole', 'detailed', 'unequal_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', 'detailed', 'unequal_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', 'detailed', 'unequal_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'detailed' solver - uniform segments
        ('single_borehole', 'detailed', 'uniform_segments', np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', 'detailed', 'uniform_segments', np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', 'detailed', 'uniform_segments', np.array([11.27831804, 18.48075762, 21.00669237])),
        # 'detailed' solver - unequal segments, FLS approximation
        ('single_borehole', 'detailed', 'unequal_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', 'detailed', 'unequal_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', 'detailed', 'unequal_segments_approx', np.array([11.27751418, 18.47964006, 21.00475366])),
        # 'detailed' solver - uniform segments, FLS approximation
        ('single_borehole', 'detailed', 'uniform_segments_approx', np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole_short', 'detailed', 'uniform_segments_approx', np.array([4.18276637, 5.03673008, 5.34353657])),
        ('ten_boreholes_rectangular', 'detailed', 'uniform_segments_approx', np.array([11.27751418, 18.47964006, 21.00475366])),
    ])
def test_gfunctions_UHTR(field, method, opts, expected, request):
    # Extract the bore field from the fixture
    borefield = request.getfixturevalue(field)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean(borefield.H)
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = gt.gfunction.gFunction(
        borefield, alpha, time=time, method=method, options=options,
        boundary_condition='UHTR')
    assert np.allclose(gFunc.gFunc, expected)


# Test 'MIFT' g-functions for different bore fields using all solvers,
# unequal/uniform segments, and with/without the FLS approximation
# The 'equivalent' solver is not applied to series-connected boreholes.
@pytest.mark.parametrize("field, method, m_flow_network, opts, expected", [
        #  'equivalent' solver - unequal segments
        ('single_borehole_network', 'equivalent', 0.05, 'unequal_segments', np.array([5.76597302, 6.51058473, 6.73746895])),
        ('single_borehole_network_short', 'equivalent', 0.05, 'unequal_segments', np.array([4.17105954, 5.00930075, 5.30832133])),
        ('ten_boreholes_network_rectangular', 'equivalent', 0.25, 'unequal_segments', np.array([12.66229998, 18.57852681, 20.33535907])),
        # 'equivalent' solver - uniform segments
        ('single_borehole_network', 'equivalent', 0.05, 'uniform_segments', np.array([5.78644676, 6.5311583, 6.75699875])),
        ('single_borehole_network_short', 'equivalent', 0.05, 'uniform_segments', np.array([4.17553236, 5.01476781, 5.31381287])),
        ('ten_boreholes_network_rectangular', 'equivalent', 0.25, 'uniform_segments', np.array([12.931553, 18.8892892, 20.63810364])),
        #  'equivalent' solver - unequal segments, FLS approximation
        ('single_borehole_network', 'equivalent', 0.05, 'unequal_segments_approx', np.array([5.76596769, 6.51061169, 6.73731276])),
        ('single_borehole_network_short', 'equivalent', 0.05, 'unequal_segments_approx', np.array([4.17105984, 5.00931374, 5.30816983])),
        ('ten_boreholes_network_rectangular', 'equivalent', 0.25, 'unequal_segments_approx', np.array([12.66228879, 18.57863253, 20.33526092])),
        #  'equivalent' solver - uniform segments, FLS approximation
        ('single_borehole_network', 'equivalent', 0.05, 'uniform_segments_approx', np.array([5.78644007, 6.53117706, 6.75684456])),
        ('single_borehole_network_short', 'equivalent', 0.05, 'uniform_segments_approx', np.array([4.17553118, 5.01478084, 5.31366109])),
        ('ten_boreholes_network_rectangular', 'equivalent', 0.25, 'uniform_segments_approx', np.array([12.93153466, 18.88937176, 20.63801163])),
        # 'similarities' solver - unequal segments
        ('single_borehole_network', 'similarities', 0.05, 'unequal_segments', np.array([5.76597302, 6.51058473, 6.73746895])),
        ('single_borehole_network_short', 'similarities', 0.05, 'unequal_segments', np.array([4.17105954, 5.00930075, 5.30832133])),
        ('ten_boreholes_network_rectangular', 'similarities', 0.25, 'unequal_segments', np.array([12.66229998, 18.57852681, 20.33535907])),
        ('ten_boreholes_network_rectangular_series', 'similarities', 0.05, 'unequal_segments', np.array([3.19050169, 8.8595362, 10.84379419])),
        # 'similarities' solver - uniform segments
        ('single_borehole_network', 'similarities', 0.05, 'uniform_segments', np.array([5.78644676, 6.5311583, 6.75699875])),
        ('single_borehole_network_short', 'similarities', 0.05, 'uniform_segments', np.array([4.17553236, 5.01476781, 5.31381287])),
        ('ten_boreholes_network_rectangular', 'similarities', 0.25, 'uniform_segments', np.array([12.931553, 18.8892892, 20.63810364])),
        ('ten_boreholes_network_rectangular_series', 'similarities', 0.05, 'uniform_segments', np.array([3.22186337, 8.92628494, 10.91644607])),
        # 'similarities' solver - unequal segments, FLS approximation
        ('single_borehole_network', 'similarities', 0.05, 'unequal_segments_approx', np.array([5.76596769, 6.51061169, 6.73731276])),
        ('single_borehole_network_short', 'similarities', 0.05, 'unequal_segments_approx', np.array([4.17105984, 5.00931374, 5.30816983])),
        ('ten_boreholes_network_rectangular', 'similarities', 0.25, 'unequal_segments_approx', np.array([12.66136057, 18.57792276, 20.33429034])),
        ('ten_boreholes_network_rectangular_series', 'similarities', 0.05, 'unequal_segments_approx', np.array([3.19002567, 8.85783554, 10.84222402])),
        # 'similarities' solver - uniform segments, FLS approximation
        ('single_borehole_network', 'similarities', 0.05, 'uniform_segments_approx', np.array([5.78644007, 6.53117706, 6.75684456])),
        ('single_borehole_network_short', 'similarities', 0.05, 'uniform_segments_approx', np.array([4.17553118, 5.01478084, 5.31366109])),
        ('ten_boreholes_network_rectangular', 'similarities', 0.25, 'uniform_segments_approx', np.array([12.93064329, 18.88844718, 20.63710493])),
        ('ten_boreholes_network_rectangular_series', 'similarities', 0.05, 'uniform_segments_approx', np.array([3.22139028, 8.92448715, 10.91485947])),
        # 'detailed' solver - unequal segments
        ('single_borehole_network', 'detailed', 0.05, 'unequal_segments', np.array([5.76597302, 6.51058473, 6.73746895])),
        ('single_borehole_network_short', 'detailed', 0.05, 'unequal_segments', np.array([4.17105954, 5.00930075, 5.30832133])),
        ('ten_boreholes_network_rectangular', 'detailed', 0.25, 'unequal_segments', np.array([12.66229998, 18.57852681, 20.33535907])),
        ('ten_boreholes_network_rectangular_series', 'detailed', 0.05, 'unequal_segments', np.array([3.19050169, 8.8595362, 10.84379419])),
        # 'detailed' solver - uniform segments
        ('single_borehole_network', 'detailed', 0.05, 'uniform_segments', np.array([5.78644676, 6.5311583, 6.75699875])),
        ('single_borehole_network_short', 'detailed', 0.05, 'uniform_segments', np.array([4.17553236, 5.01476781, 5.31381287])),
        ('ten_boreholes_network_rectangular', 'detailed', 0.25, 'uniform_segments', np.array([12.931553, 18.8892892, 20.63810364])),
        ('ten_boreholes_network_rectangular_series', 'detailed', 0.05, 'uniform_segments', np.array([3.22186337, 8.92628494, 10.91644607])),
        # 'detailed' solver - unequal segments, FLS approximation
        ('single_borehole_network', 'detailed', 0.05, 'unequal_segments_approx', np.array([5.76596769, 6.51061169, 6.73731276])),
        ('single_borehole_network_short', 'detailed', 0.05, 'unequal_segments_approx', np.array([4.17105984, 5.00931374, 5.30816983])),
        ('ten_boreholes_network_rectangular', 'detailed', 0.25, 'unequal_segments_approx', np.array([12.66136057, 18.57792276, 20.33429034])),
        ('ten_boreholes_network_rectangular_series', 'detailed', 0.05, 'unequal_segments_approx', np.array([3.19002567, 8.85783554, 10.84222402])),
        # 'detailed' solver - uniform segments, FLS approximation
        ('single_borehole_network', 'detailed', 0.05, 'uniform_segments_approx', np.array([5.78644007, 6.53117706, 6.75684456])),
        ('single_borehole_network_short', 'detailed', 0.05, 'uniform_segments_approx', np.array([4.17553118, 5.01478084, 5.31366109])),
        ('ten_boreholes_network_rectangular', 'detailed', 0.25, 'uniform_segments_approx', np.array([12.93064329, 18.88844718, 20.63710493])),
        ('ten_boreholes_network_rectangular_series', 'detailed', 0.05, 'uniform_segments_approx', np.array([3.22139028, 8.92448715, 10.91485947])),
    ])
def test_gfunctions_MIFT(
        field, method, m_flow_network, opts, expected, request):
    # Extract the bore field from the fixture
    network = request.getfixturevalue(field)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    # Mean borehole length [m]
    H_mean = np.mean([b.H for b in network.b])
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = gt.gfunction.gFunction(
        network, alpha, time=time, m_flow_network=m_flow_network,
        cp_f=fluid.cp, method=method, options=options, boundary_condition='MIFT')
    assert np.allclose(gFunc.gFunc, expected)


# Test 'MIFT' g-functions for different bore fields using all solvers, unequal
# segments, and with the FLS approximation for variable mass flow rate
# g-functions
@pytest.mark.parametrize("field, method, m_flow_network, opts, expected", [
        # 'detailed' solver - unequal segments, FLS approximation
        ('single_borehole_network', 'detailed', np.array([-0.25, 0.05]), 'unequal_segments_approx', np.array([[[5.60566489, 6.38071943, 6.62706326], [5.603355, 6.3615959, 6.5979712]], [[5.60284597, 6.36233529, 6.5996188], [5.76596769, 6.51061169, 6.73731276]]])),
        ('single_borehole_network_short', 'detailed', np.array([-0.25, 0.05]), 'unequal_segments_approx', np.array([[[4.17130904, 5.0108666, 5.31097968], [4.1708935, 5.00979555, 5.30928557]], [[4.1708873, 5.0097772, 5.30925837], [4.17105984, 5.00931374, 5.30816983]]])),
        ('ten_boreholes_network_rectangular', 'detailed', np.array([-0.5, 0.25]), 'unequal_segments_approx', np.array([[[11.19891898, 17.50417429, 19.51221751], [11.61378246, 17.69400368, 19.56564714]], [[11.44417287, 17.5662909, 19.47928482], [12.66136057, 18.57792276, 20.33429034]]])),
        ('ten_boreholes_network_rectangular_series', 'detailed', np.array([-0.25, 0.05]), 'unequal_segments_approx', np.array([[[7.57467211, 13.97746499, 16.08569783], [25.31574066, 31.796452, 33.86566759]], [[17.37750733, 23.69326533, 25.75080157], [3.19002567, 8.85783554, 10.84222402]]])),
        # 'similarities' solver - unequal segments, FLS approximation
        ('single_borehole_network', 'similarities', np.array([-0.25, 0.05]), 'unequal_segments_approx', np.array([[[5.60566489, 6.38071943, 6.62706326], [5.603355, 6.3615959, 6.5979712]], [[5.60284597, 6.36233529, 6.5996188], [5.76596769, 6.51061169, 6.73731276]]])),
        ('single_borehole_network_short', 'similarities', np.array([-0.25, 0.05]), 'unequal_segments_approx', np.array([[[4.17130904, 5.0108666, 5.31097968], [4.1708935, 5.00979555, 5.30928557]], [[4.1708873, 5.0097772, 5.30925837], [4.17105984, 5.00931374, 5.30816983]]])),
        ('ten_boreholes_network_rectangular', 'similarities', np.array([-0.5, 0.25]), 'unequal_segments_approx', np.array([[[11.19891898, 17.50417429, 19.51221751], [11.61378246, 17.69400368, 19.56564714]], [[11.44417287, 17.5662909, 19.47928482], [12.66136057, 18.57792276, 20.33429034]]])),
        ('ten_boreholes_network_rectangular_series', 'similarities', np.array([-0.25, 0.05]), 'unequal_segments_approx', np.array([[[7.57467211, 13.97746499, 16.08569783], [25.31574066, 31.796452, 33.86566759]], [[17.37750733, 23.69326533, 25.75080157], [3.19002567, 8.85783554, 10.84222402]]])),
        # 'equivalent' solver - unequal segments, FLS approximation
        ('single_borehole_network', 'equivalent', np.array([-0.25, 0.05]), 'unequal_segments_approx', np.array([[[5.60566489, 6.38071943, 6.62706326], [5.603355, 6.3615959, 6.5979712]], [[5.60284597, 6.36233529, 6.5996188], [5.76596769, 6.51061169, 6.73731276]]])),
        ('single_borehole_network_short', 'equivalent', np.array([-0.25, 0.05]), 'unequal_segments_approx', np.array([[[4.17130904, 5.0108666, 5.31097968], [4.1708935, 5.00979555, 5.30928557]], [[4.1708873, 5.0097772, 5.30925837], [4.17105984, 5.00931374, 5.30816983]]])),
        ('ten_boreholes_network_rectangular', 'equivalent', np.array([-0.5, 0.25]), 'unequal_segments_approx', np.array([[[11.19974879, 17.50502384, 19.51351783], [11.61465222, 17.69478305, 19.56679743]], [[11.44502591, 17.56709008, 19.48049408], [12.66228879, 18.57863253, 20.33526092]]])),
    ])
def test_gfunctions_MIFT_variable_mass_flow_rate(
        field, method, m_flow_network, opts, expected, request):
    # Extract the bore field from the fixture
    network = request.getfixturevalue(field)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    # Mean borehole length [m]
    H_mean = np.mean([b.H for b in network.b])
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = gt.gfunction.gFunction(
        network, alpha, time=time, m_flow_network=m_flow_network,
        cp_f=fluid.cp, method=method, options=options, boundary_condition='MIFT')
    assert np.allclose(gFunc.gFunc, expected)


# =============================================================================
# Test gFunction (inclined boreholes)
# =============================================================================
# Test 'UBWT' g-functions for a field of inclined boreholes using all solvers,
# unequal/uniform segments, and with/without the FLS approximation
@pytest.mark.parametrize("method, opts, expected", [
        # 'similarities' solver - unequal segments
        ('similarities', 'unequal_segments', np.array([5.67249989, 6.72866814, 7.15134705])),
        # 'similarities' solver - uniform segments
        ('similarities', 'uniform_segments', np.array([5.68324619, 6.74356205, 7.16738741])),
        # 'similarities' solver - unequal segments, FLS approximation
        ('similarities', 'unequal_segments_approx', np.array([5.66984803, 6.72564218, 7.14826009])),
        # 'similarities' solver - uniform segments, FLS approximation
        ('similarities', 'uniform_segments_approx', np.array([5.67916493, 6.7395222 , 7.16339216])),
        # 'detailed' solver - unequal segments
        ('detailed', 'unequal_segments', np.array([5.67249989, 6.72866814, 7.15134705])),
        # 'detailed' solver - uniform segments
        ('detailed', 'uniform_segments', np.array([5.68324619, 6.74356205, 7.16738741])),
        # 'detailed' solver - unequal segments, FLS approximation
        ('detailed', 'unequal_segments_approx', np.array([5.66984803, 6.72564218, 7.14826009])),
        # 'detailed' solver - uniform segments, FLS approximation
        ('detailed', 'uniform_segments_approx', np.array([5.67916493, 6.7395222 , 7.16339216])),
    ])
def test_gfunctions_UBWT(two_boreholes_inclined, method, opts, expected, request):
    # Extract the bore field from the fixture
    borefield = two_boreholes_inclined
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean(borefield.H)
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = gt.gfunction.gFunction(
        borefield, alpha, time=time, method=method, options=options,
        boundary_condition='UBWT')
    assert np.allclose(gFunc.gFunc, expected)


# =============================================================================
# Test gFunction linearization
# =============================================================================
# Test 'UBWT' g-functions for a single borehole at low time values
@pytest.mark.parametrize("field, method, opts, expected", [
        #  'equivalent' solver - unequal segments
        ('single_borehole', 'equivalent', 'unequal_segments', np.array([1.35223554e-05, 1.35223554e-04, 2.16066268e-01])),
    ])
def test_gfunctions_UBWT_linearization(field, method, opts, expected, request):
    # Extract the bore field from the fixture
    borefield = request.getfixturevalue(field)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * borefield[0].r_b**2 / (25 * alpha)
    # g-Function
    gFunc = gt.gfunction.gFunction(
        borefield, alpha, time=time, method=method, options=options,
        boundary_condition='UBWT')
    assert np.allclose(gFunc.gFunc, expected)

@pytest.mark.parametrize("field, boundary_condition, method, opts, pipe_type, m_flow_network, expected", [
        #  'equivalent' solver - unequal segments - UBWT - single u-tube
        ('single_borehole', 'UBWT', 'equivalent', 'unequal_segments', 'single_Utube', 0.05, np.array([5.59717446, 6.36257605, 6.60517223])),
        ('single_borehole_short', 'UBWT', 'equivalent', 'unequal_segments', 'single_Utube', 0.05, np.array([4.15784411, 4.98477603, 5.27975732])),
        ('ten_boreholes_rectangular', 'UBWT', 'equivalent', 'unequal_segments', 'single_Utube', 0.25, np.array([10.89935004, 17.09864925, 19.0795435])),
        #  'equivalent' solver - unequal segments - UHTR - single u-tube
        ('single_borehole', 'UHTR', 'equivalent', 'unequal_segments', 'single_Utube', 0.05, np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole_short', 'UHTR', 'equivalent', 'unequal_segments', 'single_Utube', 0.05, np.array([4.18276733, 5.03671562, 5.34369772])),
        ('ten_boreholes_rectangular', 'UHTR', 'equivalent', 'unequal_segments', 'single_Utube', 0.25, np.array([11.27831804, 18.48075762, 21.00669237])),
        #  'equivalent' solver - unequal segments - MIFT - single u-tube
        ('single_borehole', 'MIFT', 'equivalent', 'unequal_segments', 'single_Utube', 0.05,  np.array([5.76597302, 6.51058473, 6.73746895])),
        ('single_borehole_short', 'MIFT', 'equivalent', 'unequal_segments', 'single_Utube', 0.05, np.array([4.17105954, 5.00930075, 5.30832133])),
        ('ten_boreholes_rectangular', 'MIFT', 'equivalent', 'unequal_segments', 'single_Utube', 0.25, np.array([12.66229998, 18.57852681, 20.33535907])),
        #  'equivalent' solver - unequal segments - MIFT - double u-tube parallel
        ('single_borehole', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_parallel', 0.05, np.array([6.47497545, 7.18728277, 7.39167598])),
        ('single_borehole_short', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_parallel', 0.05, np.array([4.17080765, 5.00341368, 5.2989709])),
        ('ten_boreholes_rectangular', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_parallel', 0.25, np.array([15.96448954, 21.43320976, 22.90761598])),
        #  'equivalent' solver - unequal segments - MIFT - double u-tube series
        ('single_borehole', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_series', 0.05, np.array([5.69118368, 6.44386342, 6.67721347])),
        ('single_borehole_short', 'MIFT','equivalent', 'unequal_segments', 'double_Utube_series', 0.05, np.array([4.16750616, 5.00249502, 5.30038701])),
        ('ten_boreholes_rectangular', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_series', 0.25, np.array([11.94256058, 17.97858109, 19.83460231])),
        #  'equivalent' solver - unequal segments - MIFT - double u-tube series asymmetrical
        ('single_borehole', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_series_asymmetrical', 0.05, np.array([5.69174709, 6.4441862 , 6.67709693])),
        ('single_borehole_short', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_series_asymmetrical', 0.05, np.array([4.16851817, 5.00453267, 5.30282913])),
        ('ten_boreholes_rectangular', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_series_asymmetrical', 0.25, np.array([11.96927941, 18.00481705, 19.856554])),
        #  'equivalent' solver - unequal segments - MIFT - double u-tube series asymmetrical
        ('single_borehole', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_series_asymmetrical', 0.05, np.array([5.69174709, 6.4441862, 6.67709693])),
        ('single_borehole_short', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_series_asymmetrical', 0.05, np.array([4.16851817, 5.00453267, 5.30282913])),
        ('ten_boreholes_rectangular', 'MIFT', 'equivalent', 'unequal_segments', 'double_Utube_series_asymmetrical', 0.25, np.array([11.96927941, 18.00481705, 19.856554])),
        #  'equivalent' solver - unequal segments - MIFT - coaxial annular inlet
        ('single_borehole', 'MIFT', 'equivalent', 'unequal_segments', 'coaxial_annular_in', 0.05, np.array([6.10236427, 6.77069069, 6.95941276])),
        ('single_borehole_short', 'MIFT', 'equivalent', 'unequal_segments', 'coaxial_annular_in', 0.05, np.array([4.06874781, 4.89701125, 5.19157017])),
        ('ten_boreholes_rectangular', 'MIFT', 'equivalent', 'unequal_segments', 'coaxial_annular_in', 0.25, np.array([16.03433989, 21.18241954, 22.49479982])),
        #  'equivalent' solver - unequal segments - MIFT - coaxial annular outlet
        ('single_borehole', 'MIFT', 'equivalent', 'unequal_segments', 'coaxial_annular_out', 0.05, np.array([6.10236427, 6.77069069, 6.95941276])),
        ('single_borehole_short', 'MIFT', 'equivalent', 'unequal_segments', 'coaxial_annular_out', 0.05, np.array([4.06874781, 4.89701125, 5.19157017])),
        ('ten_boreholes_rectangular', 'MIFT', 'equivalent', 'unequal_segments', 'coaxial_annular_out', 0.25, np.array([16.03433989, 21.18241954, 22.49510883])),
    ])
def test_evaluate_ground_heat_exchanger_evaluate_g_function(field, boundary_condition, method, opts, pipe_type, m_flow_network, expected, request):
    # Extract the bore field from the fixture for convenience
    borefield = request.getfixturevalue(field)

    # convert to lists for testing
    H = list(borefield.H)
    D = list(borefield.D)
    r_b = list(borefield.r_b)
    x = list(borefield.x)
    y = list(borefield.y)

    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)

    # Extract the pipe options from the fixture, if needed
    pipe = request.getfixturevalue(pipe_type)
    pos = pipe.pos
    r_in = pipe.r_in
    r_out = pipe.r_out

    # replace pipe_type from fixture
    if pipe_type == 'single_Utube':
        pipe_type = gt.enums.PipeType.SINGLEUTUBE
        k_p = 0.4
    elif pipe_type == 'double_Utube_parallel':
        pipe_type = gt.enums.PipeType.DOUBLEUTUBEPARALLEL
        k_p = 0.4
    elif pipe_type in ['double_Utube_series', 'double_Utube_series_asymmetrical']:
        pipe_type = gt.enums.PipeType.DOUBLEUTUBESERIES
        k_p = 0.4
    elif pipe_type == 'coaxial_annular_in':
        pipe_type = gt.enums.PipeType.COAXIALANNULARINLET
        k_p = (0.4, 0.4)
    elif pipe_type == 'coaxial_annular_out':
        pipe_type = gt.enums.PipeType.COAXIALPIPEINLET
        k_p = (0.4, 0.4)
    else:
        raise ValueError(f"test pipe_type not recognized: '{pipe_type}'")

    # Static params
    k_s = 2.0
    k_g = 1.0
    epsilon = 1e-6
    fluid_name = 'MPG'
    fluid_pct = 20.

    # Mean borehole length [m]
    H_mean = np.mean(H)
    alpha = 1e-6  # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean ** 2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts

    ghe = gt.ground_heat_exchanger.GroundHeatExchanger(
        H=H,
        D=D,
        r_b=r_b,
        x=x,
        y=y,
        pipe_type=pipe_type,
        pos=pos,
        r_in=r_in,
        r_out=r_out,
        k_s=k_s,
        k_g=k_g,
        k_p=k_p,
        m_flow_ghe=m_flow_network,
        epsilon=epsilon,
        fluid_name=fluid_name,
        fluid_concentration_pct=fluid_pct,
    )

    # g-Function
    gFunc = ghe.evaluate_g_function(
        alpha=alpha,
        time=time,
        method=method,
        boundary_condition=boundary_condition,
        options=options,
    )
    assert np.allclose(gFunc, expected)
