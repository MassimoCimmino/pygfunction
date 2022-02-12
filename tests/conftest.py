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


@pytest.fixture
def three_boreholes_unequal():
    return [gt.boreholes.Borehole(150., 4., 0.075, -1., -2.),
            gt.boreholes.Borehole(88., 2., 0.065, 5., 3.),
            gt.boreholes.Borehole(177., 5., 0.085, -1., 7.),]


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
# pipes fixtures
# =============================================================================
@pytest.fixture
def single_Utube(single_borehole):
    # Extract borehole from fixture
    borehole = single_borehole[0]
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
    R_f = 1.0/(h_f*2*np.pi*r_in)
    # Initialize pipe
    singleUTube = gt.pipes.SingleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p)
    return singleUTube


@pytest.fixture
def double_Utube_parallel(single_borehole):
    # Extract borehole from fixture
    borehole = single_borehole[0]
    # Pipe positions [m]
    D_s = 0.05
    pos_pipes = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]
    k_s = 2.0               # Ground thermal conductivity [W/m.K]
    k_g = 1.0               # Grout thermal conductivity [W/m.K]
    k_p = 0.4               # Pipe thermal conductivity [W/m.K]
    r_out = 0.02            # Pipe outer radius [m]
    r_in = 0.015            # Pipe inner radius [m]
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
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
    # Initialize pipe
    doubleUTube = gt.pipes.MultipleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p, 2,
            config='parallel')
    return doubleUTube


@pytest.fixture
def double_Utube_series(single_borehole):
    # Extract borehole from fixture
    borehole = single_borehole[0]
    # Pipe positions [m]
    D_s = 0.05
    pos_pipes = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]
    k_s = 2.0               # Ground thermal conductivity [W/m.K]
    k_g = 1.0               # Grout thermal conductivity [W/m.K]
    k_p = 0.4               # Pipe thermal conductivity [W/m.K]
    r_out = 0.02            # Pipe outer radius [m]
    r_in = 0.015            # Pipe inner radius [m]
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
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
    # Initialize pipe
    doubleUTube = gt.pipes.MultipleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p, 2,
            config='series')
    return doubleUTube


@pytest.fixture
def double_Utube_independent(single_borehole):
    # Extract borehole from fixture
    borehole = single_borehole[0]
    # Pipe positions [m]
    D_s = 0.05
    pos_pipes = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]
    k_s = 2.0               # Ground thermal conductivity [W/m.K]
    k_g = 1.0               # Grout thermal conductivity [W/m.K]
    k_p = 0.4               # Pipe thermal conductivity [W/m.K]
    r_out = 0.02            # Pipe outer radius [m]
    r_in = 0.015            # Pipe inner radius [m]
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
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
    # Initialize pipe
    doubleUTube = gt.pipes.IndependentMultipleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p, 2)
    return doubleUTube


@pytest.fixture
def triple_Utube_independent(single_borehole):
    # Extract borehole from fixture
    borehole = single_borehole[0]
    # Pipe positions [m]
    D_s = 0.05
    pos_pipes = [(-D_s, 0.),
                 (-D_s*np.cos(np.pi/3), D_s*np.sin(np.pi/3)),
                 (-D_s*np.cos(np.pi/3), -D_s*np.sin(np.pi/3)),
                 (D_s, 0.),
                 (D_s*np.cos(np.pi/3), -D_s*np.sin(np.pi/3)),
                 (D_s*np.cos(np.pi/3), D_s*np.sin(np.pi/3)),]
    k_s = 2.0               # Ground thermal conductivity [W/m.K]
    k_g = 1.0               # Grout thermal conductivity [W/m.K]
    k_p = 0.4               # Pipe thermal conductivity [W/m.K]
    r_out = 0.02            # Pipe outer radius [m]
    r_in = 0.015            # Pipe inner radius [m]
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
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
    # Initialize pipe
    tripleUTube = gt.pipes.IndependentMultipleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p, 3)
    return tripleUTube


@pytest.fixture
def coaxial_annular_in(single_borehole):
    # Extract borehole from fixture
    borehole = single_borehole[0]
    pos = (0., 0.)      # Pipe position [m]
    k_s = 2.0           # Ground thermal conductivity [W/m.K]
    k_g = 1.0           # Grout thermal conductivity [W/m.K]
    k_p = 0.4           # Pipe thermal conductivity [W/m.K]
    r_in_in = 0.0221    # Inside pipe inner radius [m]
    r_in_out = 0.025    # Inside pipe outer radius [m]
    r_out_in = 0.0487   # Outer pipe inside radius [m]
    r_out_out = 0.055   # Outer pipe outside radius [m]
    r_inner = np.array([r_out_in, r_in_in])
    r_outer = np.array([r_out_out, r_in_out])
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
    # Fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    # Pipe thermal resistances [m.K/W]
    # Inner pipe
    R_p_in = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in_in, r_in_out, k_p)
    # Outer pipe
    R_p_out = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_out_in, r_out_out, k_p)
    # Fluid-to-fluid thermal resistance [m.K/W]
    # Inner pipe
    h_f_in = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in_in, fluid.mu, fluid.rho, fluid.k, fluid.cp, epsilon)
    R_f_in = 1.0 / (h_f_in * 2 * np.pi * r_in_in)
    # Outer pipe
    h_f_a_in, h_f_a_out = \
        gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
            m_flow_borehole, r_in_out, r_out_in, fluid.mu, fluid.rho, fluid.k,
            fluid.cp, epsilon)
    R_f_out_in = 1.0 / (h_f_a_in * 2 * np.pi * r_in_out)
    R_ff = R_f_in + R_p_in + R_f_out_in
    # Coaxial GHE in borehole
    R_f_out_out = 1.0 / (h_f_a_out * 2 * np.pi * r_out_in)
    R_fp = R_p_out + R_f_out_out
    # Initialize pipe
    pipe = gt.pipes.Coaxial(
        pos, r_inner, r_outer, borehole, k_s, k_g, R_ff, R_fp)
    return pipe


@pytest.fixture
def coaxial_annular_out(single_borehole):
    # Extract borehole from fixture
    borehole = single_borehole[0]
    pos = (0., 0.)      # Pipe position [m]
    k_s = 2.0           # Ground thermal conductivity [W/m.K]
    k_g = 1.0           # Grout thermal conductivity [W/m.K]
    k_p = 0.4           # Pipe thermal conductivity [W/m.K]
    r_in_in = 0.0221    # Inside pipe inner radius [m]
    r_in_out = 0.025    # Inside pipe outer radius [m]
    r_out_in = 0.0487   # Outer pipe inside radius [m]
    r_out_out = 0.055   # Outer pipe outside radius [m]
    r_inner = np.array([r_in_in, r_out_in])
    r_outer = np.array([r_in_out, r_out_out])
    epsilon = 1.0e-06       # Pipe surface roughness [m]
    m_flow_borehole = 0.05  # Nominal fluid mass flow rate [kg/s]
    m_flow_pipe = m_flow_borehole
    # Fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    # Pipe thermal resistances [m.K/W]
    # Inner pipe
    R_p_in = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in_in, r_in_out, k_p)
    # Outer pipe
    R_p_out = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_out_in, r_out_out, k_p)
    # Fluid-to-fluid thermal resistance [m.K/W]
    # Inner pipe
    h_f_in = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in_in, fluid.mu, fluid.rho, fluid.k, fluid.cp, epsilon)
    R_f_in = 1.0 / (h_f_in * 2 * np.pi * r_in_in)
    # Outer pipe
    h_f_a_in, h_f_a_out = \
        gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
            m_flow_borehole, r_in_out, r_out_in, fluid.mu, fluid.rho, fluid.k,
            fluid.cp, epsilon)
    R_f_out_in = 1.0 / (h_f_a_in * 2 * np.pi * r_in_out)
    R_ff = R_f_in + R_p_in + R_f_out_in
    # Coaxial GHE in borehole
    R_f_out_out = 1.0 / (h_f_a_out * 2 * np.pi * r_out_in)
    R_fp = R_p_out + R_f_out_out
    # Initialize pipe
    pipe = gt.pipes.Coaxial(
        pos, r_inner, r_outer, borehole, k_s, k_g, R_ff, R_fp)
    return pipe


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
