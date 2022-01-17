# -*- coding: utf-8 -*-
""" Test suite for heat_transfer module.
"""
import pytest

import numpy as np

import pygfunction as gt


# =============================================================================
# Test finite_line_source
# =============================================================================
# Test finite_line_source for single borehole to single borehole solution over
# a single time step, using different combinations of reaSource, imgSource, and
# with/without the FLS approximation
@pytest.mark.parametrize("borehole1, borehole2, reaSource, imgSource, approximation, N, expected", [
        # Same borehole
        ('single_borehole', 'single_borehole', True, True, False, 10, 6.41336757901249),
        ('single_borehole', 'single_borehole', True, False, False, 10, 6.531908395192002),
        ('single_borehole', 'single_borehole', False, True, False, 10, -0.11854082),
        ('single_borehole', 'single_borehole', False, False, False, 10, 0.),
        # Same borehole - FLS approximation
        ('single_borehole', 'single_borehole', True, True, True, 10, 6.41337915),
        ('single_borehole', 'single_borehole', True, False, True, 10, 6.53191379),
        ('single_borehole', 'single_borehole', False, True, True, 10, -0.11853464),
        ('single_borehole', 'single_borehole', False, False, True, 10, 0.),
        # Two boreholes
        ('single_borehole', 'single_borehole_short', True, True, False, 10, 1.62660428),
        ('single_borehole', 'single_borehole_short', True, False, False, 10, 1.99622241),
        ('single_borehole', 'single_borehole_short', False, True, False, 10, -0.36961813),
        ('single_borehole', 'single_borehole_short', False, False, False, 10, 0.),
        # Two boreholes - FLS approximation
        ('single_borehole', 'single_borehole_short', True, True, True, 10, 1.62655937),
        ('single_borehole', 'single_borehole_short', True, False, True, 10, 1.99624209),
        ('single_borehole', 'single_borehole_short', False, True, True, 10, -0.36968273),
        ('single_borehole', 'single_borehole_short', False, False, True, 10, 0.),
    ])
def test_finite_line_source_one_to_one_single_time_step(
        borehole1, borehole2, reaSource, imgSource, approximation, N, expected,
        request):
        # Extract boreholes from fixtures
        b1 = request.getfixturevalue(borehole1)[0]
        b2 = request.getfixturevalue(borehole2)[0]
        alpha = 1e-6                # Ground thermal diffusivity [m2/s]
        ts = b1.H**2 / (9 * alpha)  # Borehole characteristic time [s]
        # Time for FLS calculation [s]
        time = ts
        # Evaluate FLS
        h = gt.heat_transfer.finite_line_source(
            time, alpha, b1, b2, reaSource=reaSource, imgSource=imgSource,
            approximation=approximation, N=N)
        assert np.isclose(h, expected)


# Test finite_line_source for single borehole to single borehole solution over
# a multiple time steps, using different combinations of reaSource, imgSource,
# and with/without the FLS approximation
@pytest.mark.parametrize("borehole1, borehole2, reaSource, imgSource, approximation, N, expected", [
        # Same borehole
        ('single_borehole', 'single_borehole', True, True, False, 10, np.array([5.61855789, 6.41336758, 6.66933682])),
        ('single_borehole', 'single_borehole', True, False, False, 10, np.array([5.63709751, 6.5319084, 7.0301884])),
        ('single_borehole', 'single_borehole', False, True, False, 10, np.array([-0.01853962, -0.11854082, -0.36085158])),
        ('single_borehole', 'single_borehole', False, False, False, 10, np.array([0., 0., 0.])),
        # Same borehole - FLS approximation
        ('single_borehole', 'single_borehole', True, True, True, 10, np.array([5.61855411, 6.41337915, 6.66915329])),
        ('single_borehole', 'single_borehole', True, False, True, 10, np.array([5.63709751, 6.53191379, 7.0301115])),
        ('single_borehole', 'single_borehole', False, True, True, 10, np.array([-0.01854339, -0.11853464, -0.3609582])),
        ('single_borehole', 'single_borehole', False, False, True, 10, np.array([0., 0., 0.])),
        # Two boreholes
        ('single_borehole', 'single_borehole_short', True, True, False, 10, np.array([1.1703042, 1.62660428, 1.72510234])),
        ('single_borehole', 'single_borehole_short', True, False, False, 10, np.array([1.24143841, 1.99622241, 2.46653331])),
        ('single_borehole', 'single_borehole_short', False, True, False, 10, np.array([-0.07113422, -0.36961813, -0.74143097])),
        ('single_borehole', 'single_borehole_short', False, False, False, 10, np.array([0., 0., 0.])),
        # Two boreholes - FLS approximation
        ('single_borehole', 'single_borehole_short', True, True, True, 10, np.array([1.17008899, 1.62655937, 1.72496239])),
        ('single_borehole', 'single_borehole_short', True, False, True, 10, np.array([1.24123405, 1.99624209, 2.46638582])),
        ('single_borehole', 'single_borehole_short', False, True, True, 10, np.array([-0.07114505, -0.36968273, -0.74142343])),
        ('single_borehole', 'single_borehole_short', False, False, True, 10, np.array([0., 0., 0.])),
    ])
def test_finite_line_source_one_to_one_multiple_time_steps(
        borehole1, borehole2, reaSource, imgSource, approximation, N, expected,
        request):
        # Extract boreholes from fixtures
        b1 = request.getfixturevalue(borehole1)[0]
        b2 = request.getfixturevalue(borehole2)[0]
        alpha = 1e-6                # Ground thermal diffusivity [m2/s]
        ts = b1.H**2 / (9 * alpha)  # Borehole characteristic time [s]
        # Times for FLS calculation [s]
        time = np.array([0.1, 1., 10.]) * ts
        # Evaluate FLS
        h = gt.heat_transfer.finite_line_source(
            time, alpha, b1, b2, reaSource=reaSource, imgSource=imgSource,
            approximation=approximation, N=N)
        assert np.allclose(h, expected)


# Test finite_line_source for single borehole to single borehole solution at
# steady-state, using different combinations of reaSource, imgSource
@pytest.mark.parametrize("borehole1, borehole2, reaSource, imgSource, expected", [
        # Same borehole
        ('single_borehole', 'single_borehole', True, True, 6.68879600214517),
        ('single_borehole', 'single_borehole', True, False, 7.29454957717299),
        ('single_borehole', 'single_borehole', False, True, -0.6057535750278188),
        ('single_borehole', 'single_borehole', False, False, 0.),
        # Two boreholes
        ('single_borehole', 'single_borehole_short', True, True, 1.7307294312317285),
        ('single_borehole', 'single_borehole_short', True, False, 2.729558982105579),
        ('single_borehole', 'single_borehole_short', False, True, -0.9988295508738515),
        ('single_borehole', 'single_borehole_short', False, False, 0.),
    ])
def test_finite_line_source_one_to_one_steady_state(
        borehole1, borehole2, reaSource, imgSource, expected, request):
        b1 = request.getfixturevalue(borehole1)[0]
        b2 = request.getfixturevalue(borehole2)[0]
        alpha = 1e-6     # Ground thermal diffusivity [m2/s]
        # Time for FLS calculation [s]
        time = np.inf
        # Evaluate FLS
        h = gt.heat_transfer.finite_line_source(
            time, alpha, b1, b2, reaSource=reaSource, imgSource=imgSource)
        assert np.isclose(h, expected)


# Test finite_line_source for multiple boreholes to multiple boreholes solution
# over a single time step, using different combinations of reaSource,
# imgSource, and with/without the FLS approximation
@pytest.mark.parametrize("reaSource, imgSource, approximation, N, expected", [
        # Without FLS approximation
        (True, True, False, 10, np.array(
            [[6.37361733, 1.00243165, 1.75179834],
             [1.70869031, 6.17773914, 1.78839136],
             [1.48457487, 0.88914372, 6.32534129]])),
        (True, False, False, 10, np.array(
            [[6.47984031, 1.11115916, 1.8478299],
             [1.8940213, 6.39079975, 1.96672666],
             [1.56595754, 0.9778076, 6.40728418]])),
        (False, True, False, 10, np.array(
            [[-0.10622298, -0.10872751, -0.09603156],
             [-0.18533099, -0.21306061, -0.1783353],
             [-0.08138268, -0.08866388, -0.0819429]])),
        (False, False, False, 10, np.zeros((3, 3))),
        # With FLS approximation
        (True, True, True, 10, np.array(
            [[6.37363678, 1.00243611, 1.75176654],
             [1.70869791, 6.17777608, 1.7884002],
             [1.48454791, 0.88914812, 6.32536189]])),
        (True, False, True, 10, np.array(
            [[6.4798501, 1.11115569, 1.84778942],
             [1.89401537, 6.3908279, 1.96672028],
             [1.56592324, 0.97780443, 6.40729653]])),
        (False, True, True, 10, np.array(
            [[-0.10621332, -0.10871958, -0.09602288],
             [-0.18531747, -0.21305182, -0.17832008],
             [-0.08137532, -0.08865631, -0.08193464]])),
        (False, False, True, 10, np.zeros((3, 3))),
    ])
def test_finite_line_source_multiple_boreholes_single_time_step(
        three_boreholes_unequal, reaSource, imgSource, approximation, N,
        expected):
        # Extract boreholes from fixture
        boreholes = three_boreholes_unequal
        alpha = 1e-6    # Ground thermal diffusivity [m2/s]
        # Bore field characteristic time [s]
        ts = np.mean([b.H for b in boreholes])**2 / (9 * alpha)
        # Time for FLS calculation [s]
        time = ts
        # Evaluate FLS
        h = gt.heat_transfer.finite_line_source(
            time, alpha, boreholes, boreholes, reaSource=reaSource,
            imgSource=imgSource, approximation=approximation, N=N)
        assert np.allclose(h, expected)


# Test finite_line_source for multiple boreholes to multiple boreholes solution
# over multiple time steps, using different combinations of reaSource,
# imgSource, and with/without the FLS approximation
@pytest.mark.parametrize("reaSource, imgSource, approximation, N, expected", [
        # Without FLS approximation
        (True, True, False, 10, np.array(
            [[[5.54962392, 6.37361733, 6.66453291],
              [0.56736312, 1.00243165, 1.13057208],
              [0.88843196, 1.75179834, 2.10314226]],
             [[0.96709622, 1.70869031, 1.9271115],
              [5.58527306, 6.17773914, 6.28409069],
              [1.03296293, 1.78839136, 2.04396068]],
             [[0.75290844, 1.48457487, 1.78232395],
              [0.51356349, 0.88914372, 1.01620644],
              [5.44663175, 6.32534129, 6.69487339]]])),
        (True, False, False, 10, np.array(
            [[[5.5653799, 6.47984031, 7.00849476],
              [0.58387558, 1.11115916, 1.42217343],
              [0.89939015, 1.8478299, 2.44818701]],
             [[0.99524247, 1.8940213, 2.42415926],
              [5.63143639, 6.39079975, 6.73519825],
              [1.05867673, 1.96672666, 2.55106101]],
             [[0.76219504, 1.56595754, 2.07473475],
              [0.52634775, 0.9778076, 1.26832412],
              [5.45693032, 6.40728418, 6.99916997]]])),
        (False, True, False, 10, np.array(
            [[[-0.01575598, -0.10622298, -0.34396185],
              [-0.01651247, -0.10872751, -0.29160135],
              [-0.01095819, -0.09603156, -0.34504475]],
             [[-0.02814625, -0.18533099, -0.49704776],
              [-0.04616333, -0.21306061, -0.45110757],
              [-0.02571379, -0.1783353, -0.50710034]],
             [[-0.0092866, -0.08138268, -0.2924108],
              [-0.01278426, -0.08866388, -0.25211768],
              [-0.01029857, -0.0819429, -0.30429658]]])),
        (False, False, False, 10, np.zeros((3, 3, 3))),
        # With FLS approximation
        (True, True, True, 10, np.array(
            [[[5.54962044, 6.37363678, 6.66437927],
              [0.56717846, 1.00243611, 1.13049774],
              [0.88813623, 1.75176654, 2.10302321]],
             [[0.96678146, 1.70869791, 1.92698479],
              [5.58527022, 6.17777608, 6.28433246],
              [1.03266465, 1.7884002, 2.04390268]],
             [[0.75265782, 1.48454791, 1.78222306],
              [0.51341519, 0.88914812, 1.0161776],
              [5.44663013, 6.32536189, 6.69481429]]])),
        (True, False, True, 10, np.array(
            [[[5.5653799, 6.4798501, 7.00842175],
              [0.58369262, 1.11115569, 1.42213585],
              [0.8990927, 1.84778942, 2.44811819]],
             [[0.9949306, 1.89401537, 2.42409521],
              [5.63143661, 6.3908279, 6.73530821],
              [1.05838072, 1.96672028, 2.55103547]],
             [[0.76194297, 1.56592324, 2.07467643],
              [0.52620058, 0.97780443, 1.26831142],
              [5.45693032, 6.40729653, 6.99913332]]])),
        (False, True, True, 10, np.array(
            [[[-0.01575946, -0.10621332, -0.34404249],
              [-0.01651416, -0.10871958, -0.29163811],
              [-0.01095647, -0.09602288, -0.34509498]],
             [[-0.02814914, -0.18531747, -0.49711042],
              [-0.0461664, -0.21305182, -0.45097575],
              [-0.02571607, -0.17832008, -0.50713279]],
             [[-0.00928515, -0.08137532, -0.29245337],
              [-0.01278539, -0.08865631, -0.25213381],
              [-0.01030019, -0.08193464, -0.30431902]]])),
        (False, False, True, 10, np.zeros((3, 3, 3))),
    ])
def test_finite_line_source_multiple_boreholes_multiple_time_steps(
        three_boreholes_unequal, reaSource, imgSource, approximation, N,
        expected):
        # Extract boreholes from fixture
        boreholes = three_boreholes_unequal
        alpha = 1e-6    # Ground thermal diffusivity [m2/s]
        # Bore field characteristic time [s]
        ts = np.mean([b.H for b in boreholes])**2 / (9 * alpha)
        # Times for FLS calculation [s]
        time = np.array([0.1, 1., 10.]) * ts
        # Evaluate FLS
        h = gt.heat_transfer.finite_line_source(
            time, alpha, boreholes, boreholes, reaSource=reaSource,
            imgSource=imgSource, approximation=approximation, N=N)
        assert np.allclose(h, expected)


# Test finite_line_source for multiple boreholes to multiple boreholes solution
# at steady-state, using different combinations of reaSource, imgSource
@pytest.mark.parametrize("reaSource, imgSource, expected", [
        (True, True, np.array(
            [[6.688796, 1.13927583, 2.13603998],
             [1.94194744, 6.28943104, 2.06405893],
             [1.81020337, 1.02619879, 6.73274814]])),
        (True, False, np.array(
            [[7.29454958, 1.59004241, 2.78444571],
             [2.71029956, 6.9045905, 2.8863451],
             [2.35969976, 1.43501903, 7.3348811]])),
        (False, True, np.array(
            [[-0.60575358, -0.45076657, -0.64840574],
             [-0.76835211, -0.61515946, -0.82228616],
             [-0.54949639, -0.40882024, -0.60213297]])),
        (False, False, np.zeros((3, 3))),
    ])
def test_finite_line_source_multiple_boreholes_steady_state(
        three_boreholes_unequal, reaSource, imgSource, expected):
        # Extract boreholes from fixture
        boreholes = three_boreholes_unequal
        alpha = 1e-6    # Ground thermal diffusivity [m2/s]
        # Time for FLS calculation [s]
        time = np.inf
        # Evaluate FLS
        h = gt.heat_transfer.finite_line_source(
            time, alpha, boreholes, boreholes, reaSource=reaSource,
            imgSource=imgSource)
        assert np.allclose(h, expected)
