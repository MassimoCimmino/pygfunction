import numpy as np
import pytest

import pygfunction as gt


# =============================================================================
# boreholes fixtures
# =============================================================================
@pytest.fixture
def single_borehole():
    H = 150.        # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    x = 0.          # Borehole x-position [m]
    y = 0.          # Borehole y-position [m]
    return [gt.boreholes.Borehole(H, D, r_b, x, y)]


@pytest.fixture
def single_borehole_short():
    H = 35.         # Borehole length [m]
    D = 4.          # Borehole buried depth [m]
    r_b = 0.075     # Borehole radius [m]
    x = 3.          # Borehole x-position [m]
    y = 5.          # Borehole y-position [m]
    return [gt.boreholes.Borehole(H, D, r_b, x, y)]


@pytest.fixture
def ten_boreholes_rectangular():
    H = 150.            # Borehole length [m]
    D = 4.              # Borehole buried depth [m]
    r_b = 0.075         # Borehole radius [m]
    B_1 = B_2 = 7.5     # Borehole spacing [m]
    return gt.boreholes.rectangle_field(5, 2, B_1, B_2, H, D, r_b)


# =============================================================================
# gfunction fixtures
# =============================================================================
@pytest.fixture
def uniform_segments():
    options = {'nSegments': 12,
               'segment_ratios': None,
               'approximate_FLS': False}
    return options


@pytest.fixture
def uniform_segments_approx():
    options = {'nSegments': 12,
               'segment_ratios': None,
               'approximate_FLS': True,
               'nFLS': 10}
    return options


@pytest.fixture
def unequal_segments():
    segment_ratios = gt.utilities.segment_ratios(8, end_length_ratio=0.02)
    options = {'nSegments': 8,
               'segment_ratios': segment_ratios,
               'approximate_FLS': False}
    return options


@pytest.fixture
def unequal_segments_approx():
    segment_ratios = gt.utilities.segment_ratios(8, end_length_ratio=0.02)
    options = {'nSegments': 8,
               'segment_ratios': segment_ratios,
               'approximate_FLS': True,
               'nFLS': 10}
    return options


# =============================================================================
# networks fixtures
# =============================================================================
@pytest.fixture
def single_borehole_network(single_borehole):
    # Extract borehole from fixture
    boreField = single_borehole
    # Pipe positions [m]
    D_s = 0.05
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]
    k_s = 2.0               # Ground thermal conductivity [W/m.K]
    k_g = 1.0               # Grout thermal conductivity [W/m.K]
    k_p = 0.4               # Pipe thermal conductivity [W/m.K]
    r_out = 0.02            # Pipe outer radius [m]
    r_in = 0.015            # Pipe inner radius [m]
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
    m_flow_network = m_flow_borehole
    # Fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    # Pipe thermal resistance [m.K/W]
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in, r_out, k_p)
    # Convection heat transfer coefficient [W/m2.K]
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, fluid.mu, fluid.rho, fluid.k, fluid.cp,
        epsilon)
    # Film thermal resistance [m.K/W]
    R_f = 1.0 / (h_f * 2 * np.pi * r_in)

    # Build network of parallel-connected boreholes
    bore_connectivity = [-1] * len(boreField)
    UTubes = []
    for borehole in boreField:
        UTube = gt.pipes.SingleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p)
        UTubes.append(UTube)
    # Initialize network
    network = gt.networks.Network(
        boreField, UTubes, bore_connectivity=bore_connectivity,
        m_flow_network=m_flow_network, cp_f=fluid.cp)
    return network


@pytest.fixture
def single_borehole_network_short(single_borehole_short):
    # Extract borehole from fixture
    boreField = single_borehole_short
    # Pipe positions [m]
    D_s = 0.05
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]
    k_s = 2.0               # Ground thermal conductivity [W/m.K]
    k_g = 1.0               # Grout thermal conductivity [W/m.K]
    k_p = 0.4               # Pipe thermal conductivity [W/m.K]
    r_out = 0.02            # Pipe outer radius [m]
    r_in = 0.015            # Pipe inner radius [m]
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
    m_flow_network = m_flow_borehole
    # Fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    # Pipe thermal resistance [m.K/W]
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in, r_out, k_p)
    # Convection heat transfer coefficient [W/m2.K]
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, fluid.mu, fluid.rho, fluid.k, fluid.cp,
        epsilon)
    # Film thermal resistance [m.K/W]
    R_f = 1.0 / (h_f * 2 * np.pi * r_in)

    # Build network of parallel-connected boreholes
    bore_connectivity = [-1] * len(boreField)
    UTubes = []
    for borehole in boreField:
        UTube = gt.pipes.SingleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p)
        UTubes.append(UTube)
    # Initialize network
    network = gt.networks.Network(
        boreField, UTubes, bore_connectivity=bore_connectivity,
        m_flow_network=m_flow_network, cp_f=fluid.cp)
    return network


@pytest.fixture
def ten_boreholes_network_rectangular(ten_boreholes_rectangular):
    # Extract bore field from fixture
    boreField = ten_boreholes_rectangular
    # Pipe positions [m]
    D_s = 0.05
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]
    k_s = 2.0               # Ground thermal conductivity [W/m.K]
    k_g = 1.0               # Grout thermal conductivity [W/m.K]
    k_p = 0.4               # Pipe thermal conductivity [W/m.K]
    r_out = 0.02            # Pipe outer radius [m]
    r_in = 0.015            # Pipe inner radius [m]
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
    m_flow_network = m_flow_borehole * 5
    # Fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    # Pipe thermal resistance [m.K/W]
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in, r_out, k_p)
    # Convection heat transfer coefficient [W/m2.K]
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, fluid.mu, fluid.rho, fluid.k, fluid.cp,
        epsilon)
    # Film thermal resistance [m.K/W]
    R_f = 1.0 / (h_f * 2 * np.pi * r_in)

    # Build network of parallel-connected boreholes
    bore_connectivity = [-1] * len(boreField)
    UTubes = []
    for borehole in boreField:
        UTube = gt.pipes.SingleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p)
        UTubes.append(UTube)
    # Initialize network
    network = gt.networks.Network(
        boreField, UTubes, bore_connectivity=bore_connectivity,
        m_flow_network=m_flow_network, cp_f=fluid.cp)
    return network


@pytest.fixture
def ten_boreholes_network_rectangular_series(ten_boreholes_rectangular):
    # Extract bore field from fixture
    boreField = ten_boreholes_rectangular
    # Pipe positions [m]
    D_s = 0.05
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]
    k_s = 2.0               # Ground thermal conductivity [W/m.K]
    k_g = 1.0               # Grout thermal conductivity [W/m.K]
    k_p = 0.4               # Pipe thermal conductivity [W/m.K]
    r_out = 0.02            # Pipe outer radius [m]
    r_in = 0.015            # Pipe inner radius [m]
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
    m_flow_network = m_flow_borehole
    # Fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    # Pipe thermal resistance [m.K/W]
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in, r_out, k_p)
    # Convection heat transfer coefficient [W/m2.K]
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, fluid.mu, fluid.rho, fluid.k, fluid.cp,
        epsilon)
    # Film thermal resistance [m.K/W]
    R_f = 1.0 / (h_f * 2 * np.pi * r_in)

    # Build network of series-connected boreholes
    bore_connectivity = list(range(-1, len(boreField)-1))
    UTubes = []
    for borehole in boreField:
        UTube = gt.pipes.SingleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p)
        UTubes.append(UTube)
    # Initialize network
    network = gt.networks.Network(
        boreField, UTubes, bore_connectivity=bore_connectivity,
        m_flow_network=m_flow_network, cp_f=fluid.cp)
    return network
