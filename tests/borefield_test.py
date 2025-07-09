import numpy as np
import pytest

import pygfunction as gt


# =============================================================================
# Test Borefield
# =============================================================================
# Compare __init__ and Borefield.from_boreholes for the initialization of
# Borefield objects
@pytest.mark.parametrize("field", [
        ('single_borehole'),
        ('single_borehole_short'),
        ('ten_boreholes_rectangular'),
        ('two_boreholes_inclined'),
    ])
def test_borefield_init(field, request):
    # Extract the bore field from the fixture
    boreholes = request.getfixturevalue(field).to_boreholes()
    # Borefield.from_boreholes
    borefield_from_boreholes = gt.borefield.Borefield.from_boreholes(boreholes)
    # Borefield.__init__
    H = np.array([b.H for b in boreholes])
    D = np.array([b.D for b in boreholes])
    r_b = np.array([b.r_b for b in boreholes])
    x = np.array([b.x for b in boreholes])
    y = np.array([b.y for b in boreholes])
    tilt = np.array([b.tilt for b in boreholes])
    orientation = np.array([b.orientation for b in boreholes])
    borefield = gt.borefield.Borefield(
        H, D, r_b, x, y, tilt=tilt, orientation=orientation)
    assert borefield == borefield_from_boreholes

# Test borefield comparison using __eq__
@pytest.mark.parametrize("field, other_field, expected", [
        # Fields that are equal
        ('single_borehole', 'single_borehole', True),
        ('single_borehole_short', 'single_borehole_short', True),
        ('ten_boreholes_rectangular', 'ten_boreholes_rectangular', True),
        ('two_boreholes_inclined', 'two_boreholes_inclined', True),
        # Fields that are not equal
        ('single_borehole', 'single_borehole_short', False),
        ('single_borehole', 'ten_boreholes_rectangular', False),
        ('single_borehole', 'two_boreholes_inclined', False),
        ('single_borehole_short', 'ten_boreholes_rectangular', False),
        ('single_borehole_short', 'two_boreholes_inclined', False),
        ('ten_boreholes_rectangular', 'two_boreholes_inclined', False),
    ])
def test_borefield_eq(field, other_field, expected, request):
    # Extract the bore field from the fixture
    borefield = request.getfixturevalue(field)
    other_field = request.getfixturevalue(other_field)
    assert (borefield == other_field) == expected

# Test borefield comparison using __ne__
@pytest.mark.parametrize("field, other_field, expected", [
        # Fields that are equal
        ('single_borehole', 'single_borehole', False),
        ('single_borehole_short', 'single_borehole_short', False),
        ('ten_boreholes_rectangular', 'ten_boreholes_rectangular', False),
        ('two_boreholes_inclined', 'two_boreholes_inclined', False),
        # Fields that are not equal
        ('single_borehole', 'single_borehole_short', True),
        ('single_borehole', 'ten_boreholes_rectangular', True),
        ('single_borehole', 'two_boreholes_inclined', True),
        ('single_borehole_short', 'ten_boreholes_rectangular', True),
        ('single_borehole_short', 'two_boreholes_inclined', True),
        ('ten_boreholes_rectangular', 'two_boreholes_inclined', True),
    ])
def test_borefield_ne(field, other_field, expected, request):
    # Extract the bore field from the fixture
    borefield = request.getfixturevalue(field)
    other_field = request.getfixturevalue(other_field)
    assert (borefield != other_field) == expected


def test_borefield_add():
    borehole = gt.boreholes.Borehole(100, 1, 0.075, 15, 10)
    borefield = gt.borefield.Borefield.rectangle_field(2, 1, 6, 6, 100, 1, 0.075)
    borefield_2 = gt.borefield.Borefield.from_boreholes([borehole, gt.boreholes.Borehole(110, 1, 0.075, 20, 15)])
    assert borefield + borehole == gt.borefield.Borefield.from_boreholes(borefield.to_boreholes() + [borehole])
    assert borehole + borefield == gt.borefield.Borefield.from_boreholes(borefield.to_boreholes() + [borehole])
    assert borefield + [borehole] == gt.borefield.Borefield.from_boreholes(borefield.to_boreholes() + [borehole])
    assert borefield + borefield_2 == gt.borefield.Borefield.from_boreholes(borefield.to_boreholes()+borefield_2.to_boreholes())


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
    gFunc = borefield.evaluate_g_function(
        alpha, time, method=method, options=options, boundary_condition='UBWT')
    assert np.allclose(gFunc, expected)


# =============================================================================
# Test borefield creation methods
# =============================================================================
# Test rectangle_field
@pytest.mark.parametrize("N_1, N_2, B_1, B_2", [
        (1, 1, 5., 5.),     # 1 by 1
        (2, 1, 5., 5.),     # 2 by 1
        (1, 2, 5., 5.),     # 1 by 2
        (2, 2, 5., 7.5),    # 2 by 2 (different x/y spacings)
        (10, 9, 7.5, 5.),   # 10 by 9 (different x/y spacings)
    ])
def test_rectangle_field(N_1, N_2, B_1, B_2):
    H = 150.        # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    # Generate the bore field
    borefield = gt.borefield.Borefield.rectangle_field(
        N_1, N_2, B_1, B_2, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = borefield.x
    y = borefield.y
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(borefield), dtype=bool)]
    assert np.all(
        [len(borefield) == N_1 * N_2,
         np.allclose(H, borefield.H),
         np.allclose(D, borefield.D),
         np.allclose(r_b, borefield.r_b),
         len(borefield) == 1 or np.isclose(np.min(dis), min(B_1, B_2)),
         ])


# Test staggered_rectangle_field
@pytest.mark.parametrize("N_1, N_2, B_1, B_2, include_last_element", [
        (1, 1, 5., 5., True),     # 1 by 1
        (2, 1, 5., 5., True),     # 2 by 1
        (1, 2, 5., 5., True),     # 1 by 2
        (2, 2, 5., 7.5, True),    # 2 by 2 (different x/y spacings)
        (10, 9, 7.5, 5., True),   # 10 by 9 (different x/y spacings),
        (1, 1, 5., 5., False),     # 1 by 1
        (2, 1, 5., 5., False),     # 2 by 1
        (1, 2, 5., 5., False),     # 1 by 2
        (2, 2, 5., 7.5, False),    # 2 by 2 (different x/y spacings)
        (10, 9, 7.5, 5., False),   # 10 by 9 (different x/y spacings)
])
def test_staggered_rectangular_field(N_1, N_2, B_1, B_2, include_last_element):
    H = 150.        # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    # Generate the bore field
    borefield = gt.borefield.Borefield.staggered_rectangle_field(
        N_1, N_2, B_1, B_2, H, D, r_b,
        include_last_borehole=include_last_element)
    # Evaluate the borehole to borehole distances
    x = borefield.x
    y = borefield.y
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(borefield), dtype=bool)]

    if include_last_element or N_1 == 1 or N_2 == 1:
        expected_nBoreholes = N_1 * N_2
    elif N_2 % 2 == 0:
        expected_nBoreholes = N_2 * (2 * N_1 - 1) / 2
    else:
        expected_nBoreholes = (N_2 - 1) * (2 * N_1 - 1) / 2 + N_1

    assert np.all(
        [len(borefield) == expected_nBoreholes,
         np.allclose(H, borefield.H),
         np.allclose(D, borefield.D),
         np.allclose(r_b, borefield.r_b),
         len(borefield) == 1 or np.isclose(
             np.min(dis), min(B_1, np.sqrt(B_2**2 + 0.25 * B_1**2))),
         ])


# Test dense_rectangle_field
@pytest.mark.parametrize("N_1, N_2, B, include_last_element", [
        (1, 1, 5., True),     # 1 by 1
        (2, 1, 5., True),     # 2 by 1
        (1, 2, 5., True),     # 1 by 2
        (2, 2, 5., True),     # 2 by 2
        (10, 9, 7.5, True),   # 10 by 9
        (10, 10, 7.5, True),   # 10 by 10
        (1, 1, 5., False),     # 1 by 1
        (2, 1, 5., False),     # 2 by 1
        (1, 2, 5., False),     # 1 by 2
        (2, 2, 5., False),     # 2 by 2
        (10, 9, 7.5, False),   # 10 by 9
        (10, 10, 7.5, False),  # 10 by 10
])
def test_dense_rectangle_field(N_1, N_2, B, include_last_element):
    H = 150.        # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    # Generate the bore field
    borefield = gt.borefield.Borefield.dense_rectangle_field(
        N_1, N_2, B, H, D, r_b, include_last_borehole=include_last_element)
    # Evaluate the borehole to borehole distances
    x = borefield.x
    y = borefield.y
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(borefield), dtype=bool)]

    if include_last_element or N_1 == 1 or N_2 == 1:
        expected_nBoreholes = N_1 * N_2
    elif N_2 % 2 == 0:
        expected_nBoreholes = N_2 * (2 * N_1 - 1) / 2
    else:
        expected_nBoreholes = (N_2 - 1) * (2 * N_1 - 1) / 2 + N_1

    assert np.all(
        [len(borefield) == expected_nBoreholes,
         np.allclose(H, borefield.H),
         np.allclose(D, borefield.D),
         np.allclose(r_b, borefield.r_b),
         len(borefield) == 1 or np.isclose(np.min(dis), B)
         ])


# Test L_shaped_field
@pytest.mark.parametrize("N_1, N_2, B_1, B_2", [
        (1, 1, 5., 5.),     # 1 by 1
        (2, 1, 5., 5.),     # 2 by 1
        (1, 2, 5., 5.),     # 1 by 2
        (2, 2, 5., 7.5),    # 2 by 2 (different x/y spacings)
        (10, 9, 7.5, 5.),   # 10 by 9 (different x/y spacings)
    ])
def test_L_shaped_field(N_1, N_2, B_1, B_2):
    H = 150.        # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    # Generate the bore field
    borefield = gt.borefield.Borefield.L_shaped_field(
        N_1, N_2, B_1, B_2, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = borefield.x
    y = borefield.y
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(borefield), dtype=bool)]
    assert np.all(
        [len(borefield) == N_1 + N_2 - 1,
         np.allclose(H, borefield.H),
         np.allclose(D, borefield.D),
         np.allclose(r_b, borefield.r_b),
         len(borefield) == 1 or np.isclose(np.min(dis), min(B_1, B_2)),
         ])


# Test U_shaped_field
@pytest.mark.parametrize("N_1, N_2, B_1, B_2", [
        (1, 1, 5., 5.),     # 1 by 1
        (2, 1, 5., 5.),     # 2 by 1
        (1, 2, 5., 5.),     # 1 by 2
        (2, 2, 5., 7.5),    # 2 by 2 (different x/y spacings)
        (10, 9, 7.5, 5.),   # 10 by 9 (different x/y spacings)
    ])
def test_U_shaped_field(N_1, N_2, B_1, B_2):
    H = 150.        # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    # Generate the bore field
    borefield = gt.borefield.Borefield.U_shaped_field(
        N_1, N_2, B_1, B_2, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = borefield.x
    y = borefield.y
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(borefield), dtype=bool)]
    assert np.all(
        [len(borefield) == N_1 + 2 * N_2 - 2 if N_1 > 1 else N_2,
         np.allclose(H, borefield.H),
         np.allclose(D, borefield.D),
         np.allclose(r_b, borefield.r_b),
         len(borefield) == 1 or np.isclose(np.min(dis), min(B_1, B_2)),
         ])


# Test box_shaped_field
@pytest.mark.parametrize("N_1, N_2, B_1, B_2", [
        (1, 1, 5., 5.),     # 1 by 1
        (2, 1, 5., 5.),     # 2 by 1
        (1, 2, 5., 5.),     # 1 by 2
        (2, 2, 5., 7.5),    # 2 by 2 (different x/y spacings)
        (10, 9, 7.5, 5.),   # 10 by 9 (different x/y spacings)
    ])
def test_box_shaped_field(N_1, N_2, B_1, B_2):
    H = 150.        # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    # Generate the bore field
    borefield = gt.borefield.Borefield.box_shaped_field(
        N_1, N_2, B_1, B_2, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = borefield.x
    y = borefield.y
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(borefield), dtype=bool)]
    if N_1 == 1 and N_2 == 1:
        nBoreholes_expected = 1
    elif N_1 == 1:
        nBoreholes_expected = N_2
    elif N_2 == 1:
        nBoreholes_expected = N_1
    else:
        nBoreholes_expected = 2 * (N_1 - 1) + 2 * (N_2 - 1)
    assert np.all(
        [len(borefield) == nBoreholes_expected,
         np.allclose(H, borefield.H),
         np.allclose(D, borefield.D),
         np.allclose(r_b, borefield.r_b),
         len(borefield) == 1 or np.isclose(np.min(dis), min(B_1, B_2)),
         ])


# Test circle_field
@pytest.mark.parametrize("N, R", [
        (1, 5.),    # 1 borehole
        (2, 5.),    # 2 boreholes
        (3, 7.5),   # 3 boreholes
        (10, 9.),   # 10 boreholes
    ])
def test_circle_field(N, R):
    H = 150.        # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    # Generate the bore field
    borefield = gt.borefield.Borefield.circle_field(N, R, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = borefield.x
    y = borefield.y
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(borefield), dtype=bool)]
    B_min = 2 * R * np.sin(np.pi / N)
    assert np.all(
        [len(borefield) == N,
         np.allclose(H, borefield.H),
         np.allclose(D, borefield.D),
         np.allclose(r_b, borefield.r_b),
         len(borefield) == 1 or np.isclose(np.min(dis), B_min),
         len(borefield) == 1 or np.max(dis) <= (2 + 1e-6) * R,
         ])
