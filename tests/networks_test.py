# -*- coding: utf-8 -*-
""" Test suite for networks module.
"""
import pytest

import numpy as np

import pygfunction as gt


# =============================================================================
# Test network classes
# =============================================================================
# Test get_inlet_temperature
@pytest.mark.parametrize("network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected", [
    # Single borehole
    ('single_borehole_network', 0.2, None, 1., 1, np.array([5.])),
    ('single_borehole_network', 0.2, None, np.array([1., 2., 3., 1.]), 4, np.array([5.])),
    ('single_borehole_network', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, np.array([5.])),
    ('single_borehole_network', -0.2, None, 1., 1, np.array([5.])),
    ('single_borehole_network', -0.2, None, np.array([1., 2., 3., 1.]), 4, np.array([5.])),
    ('single_borehole_network', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, np.array([5.])),
    # Ten boreholes (Parallel-connected)
    ('ten_boreholes_network_rectangular', 2., None, 1., 1, np.full(10, 5.)),
    ('ten_boreholes_network_rectangular', 2., None, np.tile([1., 2., 3., 1.], 10), 4, np.full(10, 5.)),
    ('ten_boreholes_network_rectangular', 2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, np.full(10, 5.)),
    ('ten_boreholes_network_rectangular', -2., None, 1., 1, np.full(10, 5.)),
    ('ten_boreholes_network_rectangular', -2., None, np.tile([1., 2., 3., 1.], 10), 4, np.full(10, 5.)),
    ('ten_boreholes_network_rectangular', -2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, np.full(10, 5.)),
    # Three boreholes (Series-connected, unequal lengths)
    ('three_boreholes_network_series_unequal', 0.2, None, 1., 1, np.array([5., 2.85537985, 2.12376334])),
    ('three_boreholes_network_series_unequal', 0.2, None, np.tile([1., 2., 3., 1.], 3), 4, np.array([5., 3.25437509, 2.66026346])),
    ('three_boreholes_network_series_unequal', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, np.array([5., 3.46848051, 2.94756734])),
    ('three_boreholes_network_series_unequal', -0.2, None, 1., 1, np.array([2.053936, 2.74009201, 5.])),
    ('three_boreholes_network_series_unequal', -0.2, None, np.tile([1., 2., 3., 1.], 3), 4, np.array([2.60290661, 3.15967656, 5.])),
    ('three_boreholes_network_series_unequal', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, np.array([2.89712276, 3.38519442, 5.])),
    ])
def test_inlet_temperature(network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected, request):
    # Extract pipe from fixture
    network = request.getfixturevalue(network_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    T_f_in = 5.0           # Total heat transfer rate [W]
    # Inlet fluid temperature [degC]
    T_f_in_borehole = network.get_inlet_temperature(
        T_f_in, T_b, m_flow_network, fluid.cp, nSegments, segment_ratios=segment_ratios)
    assert np.allclose(T_f_in_borehole, expected)


# Test get_outlet_temperature
@pytest.mark.parametrize("network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected", [
    # Single borehole
    ('single_borehole_network', 0.2, None, 1., 1, 2.712371852688313),
    ('single_borehole_network', 0.2, None, np.array([1., 2., 3., 1.]), 4, 3.1377635748663573),
    ('single_borehole_network', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, 3.3661222931515784),
    ('single_borehole_network', -0.2, None, 1., 1, 2.712371852688313),
    ('single_borehole_network', -0.2, None, np.array([1., 2., 3., 1.]), 4, 3.1377635748663573),
    ('single_borehole_network', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, 3.3661222931515784),
    # Ten boreholes (Parallel-connected)
    ('ten_boreholes_network_rectangular', 2., None, 1., 1, np.full(10, 2.712371852688313)),
    ('ten_boreholes_network_rectangular', 2., None, np.tile([1., 2., 3., 1.], 10), 4, np.full(10, 3.1377635748663573)),
    ('ten_boreholes_network_rectangular', 2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, np.full(10, 3.3661222931515784)),
    ('ten_boreholes_network_rectangular', -2., None, 1., 1, np.full(10, 2.712371852688313)),
    ('ten_boreholes_network_rectangular', -2., None, np.tile([1., 2., 3., 1.], 10), 4, np.full(10, 3.1377635748663573)),
    ('ten_boreholes_network_rectangular', -2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, np.full(10, 3.3661222931515784)),
    # Three boreholes (Series-connected, unequal lengths)
    ('three_boreholes_network_series_unequal', 0.2, None, 1., 1, np.array([2.85537985, 2.12376334, 1.4888629])),
    ('three_boreholes_network_series_unequal', 0.2, None, np.tile([1., 2., 3., 1.], 3), 4, np.array([3.25437509, 2.66026346, 2.14183735])),
    ('three_boreholes_network_series_unequal', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, np.array([3.46848051, 2.94756734, 2.492339])),
    ('three_boreholes_network_series_unequal', -0.2, None, 1., 1, np.array([1.4888629, 2.053936, 2.74009201])),
    ('three_boreholes_network_series_unequal', -0.2, None, np.tile([1., 2., 3., 1.], 3), 4, np.array([2.1424954, 2.60290661, 3.15967656])),
    ('three_boreholes_network_series_unequal', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, np.array([2.49307149, 2.89712276, 3.38519442])),
    ])
def test_outlet_temperature(
        network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected, request):
    # Extract pipe from fixture
    network = request.getfixturevalue(network_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Outlet fluid temperature [degC]
    T_f_out_borehole = network.get_outlet_temperature(
        T_f_in, T_b, m_flow_network, fluid.cp, nSegments, segment_ratios=segment_ratios)
    assert np.allclose(T_f_out_borehole, expected)


# Test get_borehole_heat_extraction_rate
@pytest.mark.parametrize("network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected", [
    # Single borehole
    ('single_borehole_network', 0.2, None, 1., 1, np.array([-1819.4736348927008])),
    ('single_borehole_network', 0.2, None, np.array([1., 2., 3., 1.]), 4, np.array([-507.98022943, -330.29924271, -155.92399643, -486.93326314])),
    ('single_borehole_network', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, np.array([-212.48822505, -496.72428991, -284.6464931, -305.65175979])),
    ('single_borehole_network', -0.2, None, 1., 1, np.array([-1819.4736348927008])),
    ('single_borehole_network', -0.2, None, np.array([1., 2., 3., 1.]), 4, np.array([-507.98022943, -330.29924271, -155.92399643, -486.93326314])),
    ('single_borehole_network', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, np.array([-212.48822505, -496.72428991, -284.6464931, -305.65175979])),
    # Ten boreholes (Parallel-connected)
    ('ten_boreholes_network_rectangular', 2., None, 1., 1, np.full(10, -1819.4736348927008)),
    ('ten_boreholes_network_rectangular', 2., None, np.tile([1., 2., 3., 1.], 10), 4, np.tile([-507.98022943, -330.29924271, -155.92399643, -486.93326314], 10)),
    ('ten_boreholes_network_rectangular', 2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, np.tile([-212.48822505, -496.72428991, -284.6464931, -305.65175979], 10)),
    ('ten_boreholes_network_rectangular', -2., None, 1., 1, np.full(10, -1819.4736348927008)),
    ('ten_boreholes_network_rectangular', -2., None, np.tile([1., 2., 3., 1.], 10), 4, np.tile([-507.98022943, -330.29924271, -155.92399643, -486.93326314], 10)),
    ('ten_boreholes_network_rectangular', -2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, np.tile([-212.48822505, -496.72428991, -284.6464931, -305.65175979], 10)),
    # Three boreholes (Series-connected, unequal lengths)
    ('three_boreholes_network_series_unequal', 0.2, None, 1., 1, np.array([-1705.73168677, -581.89393685, -504.97044758])),
    ('three_boreholes_network_series_unequal', 0.2, None, np.tile([1., 2., 3., 1.], 3), 4, np.array([-472.32371371, -310.57683056, -151.75246159, -453.73631591, -194.1424052, -93.48480165, 6.82853736, -191.73025237, -231.10379372, -62.2454343, 105.87215406, -224.85507901])),
    ('three_boreholes_network_series_unequal', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, np.array([-196.92259261, -464.3542477, -272.99391166, -283.82902484, -87.82894928, -166.65785636, -29.55839836, -130.26503886, -114.15250293, -163.41153844, 82.98731428, -167.49084336])),
    ('three_boreholes_network_series_unequal', -0.2, None, 1., 1, np.array([-449.43300626, -545.73676086, -1797.42630407])),
    ('three_boreholes_network_series_unequal', -0.2, None, np.tile([1., 2., 3., 1.], 3), 4, np.array([-208.47432383, -53.86348191, 100.24050704, -204.09248464, -186.62226798, -86.04634283, 14.21271046, -184.37317225, -504.70429869, -326.58978133, -152.36194671, -480.05212833])),
    ('three_boreholes_network_series_unequal', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, np.array([-103.44758806, -145.80224228, 80.44510753, -152.55896958, -85.17323325, -157.46150512, -19.17059397, -126.38429075, -211.30877858, -492.28316225, -279.11269094, -301.63705046])),
    ])
def test_borehole_heat_extraction_rate(
        network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected, request):
    # Extract pipe from fixture
    network = request.getfixturevalue(network_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Borehole heat extraction rates [W]
    Q_b = network.get_borehole_heat_extraction_rate(
        T_f_in, T_b, m_flow_network, fluid.cp, nSegments, segment_ratios=segment_ratios)
    assert np.allclose(Q_b, expected)


# Test get_fluid_heat_extraction_rate
@pytest.mark.parametrize("network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected", [
    # Single borehole
    ('single_borehole_network', 0.2, None, 1., 1, np.array([-1819.4736348927008])),
    ('single_borehole_network', 0.2, None, np.array([1., 2., 3., 1.]), 4, np.array([-1481.1367317058312])),
    ('single_borehole_network', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, np.array([-1299.5107678418537])),
    ('single_borehole_network', -0.2, None, 1., 1, np.array([-1819.4736348927008])),
    ('single_borehole_network', -0.2, None, np.array([1., 2., 3., 1.]), 4, np.array([-1481.1367317058312])),
    ('single_borehole_network', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, np.array([-1299.5107678418537])),
    # Ten boreholes (Parallel-connected)
    ('ten_boreholes_network_rectangular', 2., None, 1., 1, np.full(10, -1819.4736348927008)),
    ('ten_boreholes_network_rectangular', 2., None, np.tile([1., 2., 3., 1.], 10), 4, np.full(10, -1481.1367317058312)),
    ('ten_boreholes_network_rectangular', 2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, np.full(10, -1299.5107678418537)),
    ('ten_boreholes_network_rectangular', -2., None, 1., 1, np.full(10, -1819.4736348927008)),
    ('ten_boreholes_network_rectangular', -2., None, np.tile([1., 2., 3., 1.], 10), 4, np.full(10, -1481.1367317058312)),
    ('ten_boreholes_network_rectangular', -2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, np.full(10, -1299.5107678418537)),
    # Three boreholes (Series-connected, unequal lengths)
    ('three_boreholes_network_series_unequal', 0.2, None, 1., 1, np.array([-1705.73168677, -581.89393685, -504.97044758])),
    ('three_boreholes_network_series_unequal', 0.2, None, np.tile([1., 2., 3., 1.], 3), 4, np.array([-1388.38932177, -472.52892185, -412.33215297])),
    ('three_boreholes_network_series_unequal', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, np.array([-1218.09977682, -414.31024285, -362.06757046])),
    ('three_boreholes_network_series_unequal', -0.2, None, 1., 1, np.array([-449.43300626, -545.73676086, -1797.42630407])),
    ('three_boreholes_network_series_unequal', -0.2, None, np.tile([1., 2., 3., 1.], 3), 4, np.array([-366.18978334, -442.82907261, -1463.70815506])),
    ('three_boreholes_network_series_unequal', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, np.array([-321.3636924, -388.18962309, -1284.34168224])),
    ])
def test_fluid_heat_extraction_rate(
        network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected, request):
    # Extract pipe from fixture
    network = request.getfixturevalue(network_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Fluid heat extraction rate [W]
    Q_f = network.get_fluid_heat_extraction_rate(
        T_f_in, T_b, m_flow_network, fluid.cp, nSegments, segment_ratios=segment_ratios)
    assert np.allclose(Q_f, expected)


# Test get_network_inlet_temperature
@pytest.mark.parametrize("network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected", [
    # Single borehole
    ('single_borehole_network', 0.2, None, 1., 1, 7.595314034714041),
    ('single_borehole_network', 0.2, None, np.array([1., 2., 3., 1.]), 4, 8.33912674339739),
    ('single_borehole_network', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, 8.73842016624424),
    ('single_borehole_network', -0.2, None, 1., 1, 7.595314034714041),
    ('single_borehole_network', -0.2, None, np.array([1., 2., 3., 1.]), 4, 8.33912674339739),
    ('single_borehole_network', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, 8.73842016624424),
    # Ten boreholes (Parallel-connected)
    ('ten_boreholes_network_rectangular', 2., None, 1., 1, 7.595314034714041),
    ('ten_boreholes_network_rectangular', 2., None, np.tile([1., 2., 3., 1.], 10), 4, 8.33912674339739),
    ('ten_boreholes_network_rectangular', 2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, 8.73842016624424),
    ('ten_boreholes_network_rectangular', -2., None, 1., 1, 7.595314034714041),
    ('ten_boreholes_network_rectangular', -2., None, np.tile([1., 2., 3., 1.], 10), 4, 8.33912674339739),
    ('ten_boreholes_network_rectangular', -2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, 8.73842016624424),
    # Three boreholes (Series-connected, unequal lengths)
    ('three_boreholes_network_series_unequal', 0.2, None, 1., 1, 13.891230626356375),
    ('three_boreholes_network_series_unequal', 0.2, None, np.tile([1., 2., 3., 1.], 3), 4, 14.635120055915039),
    ('three_boreholes_network_series_unequal', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, 15.034422782654072),
    ('three_boreholes_network_series_unequal', -0.2, None, 1., 1, 13.891230626356379),
    ('three_boreholes_network_series_unequal', -0.2, None, np.tile([1., 2., 3., 1.], 3), 4, 14.635869731958419),
    ('three_boreholes_network_series_unequal', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, 15.03525726407867),
    ])
def test_network_inlet_temperature(network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected, request):
    # Extract pipe from fixture
    network = request.getfixturevalue(network_fixture)
    nBoreholes = len(network.b)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    Q_f = -3000.0 * nBoreholes  # Total heat transfer rate [W]
    # Inlet fluid temperature [degC]
    T_f_in = network.get_network_inlet_temperature(
        Q_f, T_b, m_flow_network, fluid.cp, nSegments, segment_ratios=segment_ratios)
    assert np.isclose(T_f_in, expected)


# Test get_network_outlet_temperature
@pytest.mark.parametrize("network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected", [
    # Single borehole
    ('single_borehole_network', 0.2, None, 1., 1, 2.712371852688313),
    ('single_borehole_network', 0.2, None, np.array([1., 2., 3., 1.]), 4, 3.1377635748663573),
    ('single_borehole_network', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, 3.3661222931515784),
    ('single_borehole_network', -0.2, None, 1., 1, 2.712371852688313),
    ('single_borehole_network', -0.2, None, np.array([1., 2., 3., 1.]), 4, 3.1377635748663573),
    ('single_borehole_network', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, 3.3661222931515784),
    # Ten boreholes (Parallel-connected)
    ('ten_boreholes_network_rectangular', 2., None, 1., 1, 2.712371852688313),
    ('ten_boreholes_network_rectangular', 2., None, np.tile([1., 2., 3., 1.], 10), 4, 3.1377635748663573),
    ('ten_boreholes_network_rectangular', 2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, 3.3661222931515784),
    ('ten_boreholes_network_rectangular', -2., None, 1., 1, 2.712371852688313),
    ('ten_boreholes_network_rectangular', -2., None, np.tile([1., 2., 3., 1.], 10), 4, 3.1377635748663573),
    ('ten_boreholes_network_rectangular', -2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, 3.3661222931515784),
    # Three boreholes (Series-connected, unequal lengths)
    ('three_boreholes_network_series_unequal', 0.2, None, 1., 1, 1.4888629029742049),
    ('three_boreholes_network_series_unequal', 0.2, None, np.tile([1., 2., 3., 1.], 3), 4, 2.141837346026401),
    ('three_boreholes_network_series_unequal', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, 2.492339000225644),
    ('three_boreholes_network_series_unequal', -0.2, None, 1., 1, 1.4888629029742049),
    ('three_boreholes_network_series_unequal', -0.2, None, np.tile([1., 2., 3., 1.], 3), 4, 2.142495399868063),
    ('three_boreholes_network_series_unequal', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, 2.4930714948973103),
    ])
def test_network_outlet_temperature(
        network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected, request):
    # Extract pipe from fixture
    network = request.getfixturevalue(network_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Outlet fluid temperature [degC]
    T_f_out = network.get_network_outlet_temperature(
        T_f_in, T_b, m_flow_network, fluid.cp, nSegments, segment_ratios=segment_ratios)
    assert np.isclose(T_f_out, expected)


# Test get_network_heat_extraction_rate
@pytest.mark.parametrize("network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected", [
    # Single borehole
    ('single_borehole_network', 0.2, None, 1., 1, -1819.4736348927008),
    ('single_borehole_network', 0.2, None, np.array([1., 2., 3., 1.]), 4, -1481.1367317058312),
    ('single_borehole_network', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, -1299.5107678418537),
    ('single_borehole_network', -0.2, None, 1., 1, -1819.4736348927008),
    ('single_borehole_network', -0.2, None, np.array([1., 2., 3., 1.]), 4, -1481.1367317058312),
    ('single_borehole_network', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 4, -1299.5107678418537),
    # Ten boreholes (Parallel-connected)
    ('ten_boreholes_network_rectangular', 2., None, 1., 1, -18194.736348927008),
    ('ten_boreholes_network_rectangular', 2., None, np.tile([1., 2., 3., 1.], 10), 4, -14811.367317058312),
    ('ten_boreholes_network_rectangular', 2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, -12995.107678418537),
    ('ten_boreholes_network_rectangular', -2., None, 1., 1, -18194.736348927008),
    ('ten_boreholes_network_rectangular', -2., None, np.tile([1., 2., 3., 1.], 10), 4, -14811.367317058312),
    ('ten_boreholes_network_rectangular', -2., np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 10), 4, -12995.107678418537),
    # Three boreholes (Series-connected, unequal lengths)
    ('three_boreholes_network_series_unequal', 0.2, None, 1., 1, -2792.5960711925586),
    ('three_boreholes_network_series_unequal', 0.2, None, np.tile([1., 2., 3., 1.], 3), 4, -2273.2503965957585),
    ('three_boreholes_network_series_unequal', 0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, -1994.477590118784),
    ('three_boreholes_network_series_unequal', -0.2, None, 1., 1, -2792.596071192558),
    ('three_boreholes_network_series_unequal', -0.2, None, np.tile([1., 2., 3., 1.], 3), 4, -2272.7270110024083),
    ('three_boreholes_network_series_unequal', -0.2, np.array([0.1, 0.35, 0.40, 0.15]), np.tile([1., 2., 3., 1.], 3), 4, -1993.8949977318334),
    ])
def test_total_heat_extraction_rate(
        network_fixture, m_flow_network, segment_ratios, T_b, nSegments, expected, request):
    # Extract pipe from fixture
    network = request.getfixturevalue(network_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Total heat extraction rate [W]
    Q_t = network.get_network_heat_extraction_rate(
        T_f_in, T_b, m_flow_network, fluid.cp, nSegments, segment_ratios=segment_ratios)
    assert np.isclose(Q_t, expected)
