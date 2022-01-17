# -*- coding: utf-8 -*-
""" Test suite for boreholes module.
"""
import numpy as np
import pytest

import pygfunction as gt


# =============================================================================
# Test Borehole class
# =============================================================================
# Test the initialization of the Borehole class
def test_borehole_init():
    H = 150.                    # Borehole length [m]
    D = 4.                      # Borehole buried depth [m]
    r_b = 0.075                 # Borehole radius [m]
    x = 0.                      # Borehole x-position [m]
    y = 0.                      # Borehole y-position [m]
    tilt = np.pi / 15           # Borehole tilt [rad]
    orientation = np.pi / 3     # Borehole orientation [rad]
    # Initialize borehole object
    borehole = gt.boreholes.Borehole(
        H, D, r_b, x, y, tilt=tilt, orientation=orientation)
    assert np.all(
        [H == borehole.H,
         D == borehole.D,
         r_b == borehole.r_b,
         x == borehole.x,
         y == borehole.y,
         orientation == borehole.orientation,
         tilt == borehole.tilt
         ])


# Test Borehole.distance
@pytest.mark.parametrize("borehole1, borehole2", [
        # Same borehole
        ('single_borehole', 'single_borehole'),
        # Two boreholes
        ('single_borehole', 'single_borehole_short'),
    ])
def test_borehole_distance(borehole1, borehole2, request):
    # Extract boreholes from fixtures
    b1 = request.getfixturevalue(borehole1)[0]
    b2 = request.getfixturevalue(borehole2)[0]
    # Expected distance [m]
    dis = max(b1.r_b, np.sqrt((b1.x - b2.x)**2 + (b1.y - b2.y)**2))
    assert np.isclose(dis, b1.distance(b2))


# Test Borehole.position
@pytest.mark.parametrize("borehole", [
        ('single_borehole'),
        ('single_borehole_short'),
    ])
def test_borehole_position(borehole, request):
    # Extract borehole from fixture
    b = request.getfixturevalue(borehole)[0]
    # Evaluate position ([m], [m])
    (x, y) = b.position()
    assert x == b.x and y == b.y


# =============================================================================
# Test functions
# =============================================================================
# Test rectangular_field
@pytest.mark.parametrize("N_1, N_2, B_1, B_2", [
        (1, 1, 5., 5.),     # 1 by 1
        (2, 1, 5., 5.),     # 2 by 1
        (1, 2, 5., 5.),     # 1 by 2
        (2, 2, 5., 7.5),    # 2 by 2 (different x/y spacings)
        (10, 9, 7.5, 5.),   # 10 by 9 (different x/y spacings)
    ])
def test_rectangular_field(N_1, N_2, B_1, B_2):
    H = 150.        # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    # Generate the bore field
    field = gt.boreholes.rectangle_field(N_1, N_2, B_1, B_2, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = np.array([b.x for b in field])
    y = np.array([b.y for b in field])
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(field), dtype=bool)]
    assert np.all(
        [len(field) == N_1 * N_2,
         np.allclose(H, [b.H for b in field]),
         np.allclose(D, [b.D for b in field]),
         np.allclose(r_b, [b.r_b for b in field]),
         len(field) == 1 or np.isclose(np.min(dis), min(B_1, B_2)),
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
    field = gt.boreholes.L_shaped_field(N_1, N_2, B_1, B_2, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = np.array([b.x for b in field])
    y = np.array([b.y for b in field])
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(field), dtype=bool)]
    assert np.all(
        [len(field) == N_1 + N_2 - 1,
         np.allclose(H, [b.H for b in field]),
         np.allclose(D, [b.D for b in field]),
         np.allclose(r_b, [b.r_b for b in field]),
         len(field) == 1 or np.isclose(np.min(dis), min(B_1, B_2)),
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
    field = gt.boreholes.U_shaped_field(N_1, N_2, B_1, B_2, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = np.array([b.x for b in field])
    y = np.array([b.y for b in field])
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(field), dtype=bool)]
    assert np.all(
        [len(field) == N_1 + 2 * N_2 - 2 if N_1 > 1 else N_2,
         np.allclose(H, [b.H for b in field]),
         np.allclose(D, [b.D for b in field]),
         np.allclose(r_b, [b.r_b for b in field]),
         len(field) == 1 or np.isclose(np.min(dis), min(B_1, B_2)),
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
    field = gt.boreholes.box_shaped_field(N_1, N_2, B_1, B_2, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = np.array([b.x for b in field])
    y = np.array([b.y for b in field])
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(field), dtype=bool)]
    if N_1 == 1 and N_2 == 1:
        nBoreholes_expected = 1
    elif N_1 == 1:
        nBoreholes_expected = N_2
    elif N_2 == 1:
        nBoreholes_expected = N_1
    else:
        nBoreholes_expected = 2 * (N_1 - 1) + 2 * (N_2 - 1)
    assert np.all(
        [len(field) == nBoreholes_expected,
         np.allclose(H, [b.H for b in field]),
         np.allclose(D, [b.D for b in field]),
         np.allclose(r_b, [b.r_b for b in field]),
         len(field) == 1 or np.isclose(np.min(dis), min(B_1, B_2)),
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
    field = gt.boreholes.circle_field(N, R, H, D, r_b)
    # Evaluate the borehole to borehole distances
    x = np.array([b.x for b in field])
    y = np.array([b.y for b in field])
    dis = np.sqrt(
        np.subtract.outer(x, x)**2 + np.subtract.outer(y, y)**2)[
            ~np.eye(len(field), dtype=bool)]
    B_min = 2 * R * np.sin(np.pi / N)
    assert np.all(
        [len(field) == N,
         np.allclose(H, [b.H for b in field]),
         np.allclose(D, [b.D for b in field]),
         np.allclose(r_b, [b.r_b for b in field]),
         len(field) == 1 or np.isclose(np.min(dis), B_min),
         len(field) == 1 or np.max(dis) <= (2 + 1e-6) * R,
         ])
