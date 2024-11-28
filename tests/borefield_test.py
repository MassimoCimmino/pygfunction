import numpy as np
import pytest

import pygfunction as gt


# =============================================================================
# Test evaluate_g_function (vertical boreholes)
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
    boreholes = request.getfixturevalue(field)
    borefield = gt.borefield.Borefield.from_boreholes(boreholes)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean([b.H for b in boreholes])
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = borefield.evaluate_g_function(
        alpha, time, method=method, options=options, boundary_condition='UBWT')
    assert np.allclose(gFunc, expected)


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
    boreholes = request.getfixturevalue(field)
    borefield = gt.borefield.Borefield.from_boreholes(boreholes)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean([b.H for b in boreholes])
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = borefield.evaluate_g_function(
        alpha, time, method=method, options=options, boundary_condition='UHTR')
    assert np.allclose(gFunc, expected)


# =============================================================================
# Test evaluate_g_function (inclined boreholes)
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
def test_gfunctions_inclined_UBWT(two_boreholes_inclined, method, opts, expected, request):
    # Extract the bore field from the fixture
    boreholes = two_boreholes_inclined
    borefield = gt.borefield.Borefield.from_boreholes(boreholes)
    # Extract the g-function options from the fixture
    options = request.getfixturevalue(opts)
    # Mean borehole length [m]
    H_mean = np.mean([b.H for b in boreholes])
    alpha = 1e-6    # Ground thermal diffusivity [m2/s]
    # Bore field characteristic time [s]
    ts = H_mean**2 / (9 * alpha)
    # Times for the g-function [s]
    time = np.array([0.1, 1., 10.]) * ts
    # g-Function
    gFunc = borefield.evaluate_g_function(
        alpha, time, method=method, options=options, boundary_condition='UBWT')
    assert np.allclose(gFunc, expected)
