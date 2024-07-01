# -*- coding: utf-8 -*-
""" Test suite for pipes module.
"""
import pytest

import numpy as np

import pygfunction as gt


# =============================================================================
# Test functions
# =============================================================================
# Test convective_heat_transfer_coefficient_circular_pipe
@pytest.mark.parametrize("m_flow, expected", [
    (0.05, 90.07260000000001),  # Laminar flow
    (0.10, 572.256944167273),   # Transition flow
    (0.50, 4036.196217814895)   # Turbulent flow
    ])
def test_convective_heat_transfer_coefficient_circular_pipe(m_flow, expected):
    r_in = 0.01         # Inner radius [m]
    epsilon = 1.5e-6    # Pipe surface roughness [m]
    visc = 0.00203008   # Fluid viscosity [kg/m.s]
    den = 1014.78       # Fluid density [kg/m3]
    cp = 3977.          # Fluid specific heat capacity [J/kg.K]
    k = 0.4922          # Fluid thermal conductivity [W/m.K]
    # Convective heat transfer coefficient [W/m2.K]
    h = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow, r_in, visc, den, k, cp, epsilon)
    assert np.isclose(h, expected)


# Test convective_heat_transfer_coefficient_concentric_annulus
@pytest.mark.parametrize("m_flow, expected", [
    (0.05, (141.4907984705223, 110.95487746200112)),    # Laminar flow
    (0.40, (904.4869811625874, 904.4869811625874)),     # Transition flow
    (0.60, (1411.2063074288633, 1411.2063074288633))    # Turbulent flow
    ])
def test_convective_heat_transfer_coefficient_concentric_annulus(
        m_flow, expected):
    r_in = 0.01         # Inner radius [m]
    r_out = 0.02        # Outer radius [m]
    epsilon = 1.5e-6    # Pipe surface roughness [m]
    visc = 0.00203008   # Fluid viscosity [kg/m.s]
    den = 1014.78       # Fluid density [kg/m3]
    cp = 3977.          # Fluid specific heat capacity [J/kg.K]
    k = 0.4922          # Fluid thermal conductivity [W/m.K]
    # Convective heat transfer coefficients [W/m2.K]
    h = gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
        m_flow, r_in, r_out, visc, den, k, cp, epsilon)
    assert np.allclose(h, expected)


# Test conduction_thermal_resistance_circular_pipe
def test_conduction_thermal_resistance_circular_pipe():
    r_in = 0.01     # Inner radius [m]
    r_out = 0.02    # Outer radius [m]
    k = 0.6         # Fluid thermal conductivity [W/m.K]
    # Conduction thermal resistance [m.K/W]
    expected = 0.18386300012720966
    R = gt.pipes.conduction_thermal_resistance_circular_pipe(r_in, r_out, k)
    assert np.isclose(R, expected)


# Test fluid_friction_factor_circular_pipe
@pytest.mark.parametrize("m_flow, expected", [
    (0.05, 0.04081718025087723),    # Laminar flow
    (0.50, 0.027641340780182006)    # Turbulent flow
    ])
def test_fluid_friction_factor_circular_pipe(m_flow, expected):
    r_in = 0.01         # Inner radius [m]
    epsilon = 1.5e-6    # Pipe surface roughness [m]
    visc = 0.00203008   # Fluid viscosity [kg/m.s]
    den = 1014.78       # Fluid density [kg/m3]
    # Fluid Darcy friction factor [-]
    f = gt.pipes.fluid_friction_factor_circular_pipe(
        m_flow, r_in, visc, den, epsilon)
    assert np.isclose(f, expected)


# Test thermal_resistances
@pytest.mark.parametrize("J, R_expected, Rd_expected", [
    (0,     # Zero-th order multipoles
     np.array([[0.25486306, 0.01538038],
               [0.01538038, 0.25206829]]),
     np.array([[0.27042505, 4.16155713],
               [4.16155713, 0.26726918]])),
    (1,     # First order multipoles
     np.array([[0.25569372, 0.01562313],
               [0.01562313, 0.25288076]]),
     np.array([[0.27150208, 4.12311447],
               [4.12311447, 0.26832082]])),
    (2,     # Second order multipoles
     np.array([[0.25590404, 0.01560503],
               [0.01560503, 0.25308681]]),
     np.array([[0.27169419, 4.13472001],
               [4.13472001, 0.26850888]])),
    (3,     # Third order multipoles
     np.array([[0.25592405, 0.01560826],
               [0.01560826, 0.25310667]]),
     np.array([[0.27171747, 4.1345064 ],
               [4.1345064, 0.26853194]])),
    ])
def test_thermal_resistances(J, R_expected, Rd_expected):
    # Pipe positions [m]
    pos_2pipes = [(0.03, 0.00), (-0.03, 0.02)]
    r_out = 0.02    # Pipe outer radius [m]
    r_b = 0.07      # Borehole radius [m]
    k_s = 2.5       # Ground thermal conductivity [W/m.K]
    k_g = 1.5       # Grout thermal conductivity [W/m.K]
    # Fluid to outer pipe wall thermal resistance [m.K/W]
    beta = 1.2
    Rfp = beta / (2 * np.pi * k_g)
    # Thermal resistances [m.k/W]
    R, Rd = gt.pipes.thermal_resistances(
        pos_2pipes, r_out, r_b, k_s, k_g, Rfp, J=J)
    assert np.allclose(R, R_expected) and np.allclose(Rd, Rd_expected)


# Test multipole
@pytest.mark.parametrize("J, expected", [
    (0, np.array([2.67263436, 2.47271955, 2.15219567, 1.95228086])),
    (1, np.array([2.71748588, 2.51729508, 2.19369127, 1.99350047])),
    (2, np.array([2.71914947, 2.51894712, 2.19312216, 1.99291981])),
    (3, np.array([2.71942944, 2.51913631, 2.19328373, 1.9929906])),
    ])
def test_multipole(J, expected):
    # Pipe positions [m]
    pos_4pipes = [
        (0.03, 0.03), (-0.03, 0.03), (-0.03, -0.03), (0.03, -0.03)]
    r_out = 0.02    # Pipe outer radius [m]
    r_b = 0.07      # Borehole radius [m]
    k_s = 2.5       # Ground thermal conductivity [W/m.K]
    k_g = 1.5       # Grout thermal conductivity [W/m.K]
    T_b = 0.0       # Borehole wall temperature [degC]
    # Pipe heat transfer rates [W/m]
    q_p = np.array([10., 9., 8., 7.])
    # Fluid to outer pipe wall thermal resistance [m.K/W]
    beta = 1.2
    R_fp = beta / (2 * np.pi * k_g)
    # Fluid temperatures [degC]
    T_f = gt.pipes.multipole(
        pos_4pipes, r_out, r_b, k_s, k_g, R_fp, T_b, q_p, J)[0]
    assert np.allclose(T_f, expected)


# =============================================================================
# Test pipe classes
# =============================================================================
# Test get_temperature
@pytest.mark.parametrize("pipe_fixture, segment_ratios, T_b, z, expected", [
    # Single U-tube
    ('single_Utube', None, 1., 65., np.array([4.34676755, 3.07354134])),
    ('single_Utube', None, np.array([1., 2., 3., 1.]), 65., np.array([4.41754093, 3.49949295])),
    ('single_Utube', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 65., np.array([4.47282344, 3.70132653])),
    ('single_Utube', None, 1., np.array([65., 75., 150.]), np.array([[4.34676755, 3.07354134], [4.25566624, 3.13435325], [3.64199359, 3.64199359]])),
    ('single_Utube', None, np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.41754093, 3.49949295], [4.35173147, 3.54346564], [3.89942492, 3.89942492]])),
    ('single_Utube', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.47282344, 3.70132653], [4.4261555, 3.72955733], [4.03655472, 4.03655472]])),
    # Double U-tube (Parallel)
    ('double_Utube_parallel', None, 1., 65., np.array([3.87525104, 3.87525104, 2.20313908, 2.20313908])),
    ('double_Utube_parallel', None, np.array([1., 2., 3., 1.]), 65., np.array([4.00464852, 4.00464852, 2.84788608, 2.84788608])),
    ('double_Utube_parallel', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 65., np.array([4.09844256, 4.09844256, 3.14231047, 3.14231047])),
    ('double_Utube_parallel', None, 1., np.array([65., 75., 150.]), np.array([[3.87525104, 3.87525104, 2.20313908, 2.20313908], [3.73265141, 3.73265141, 2.26719823, 2.26719823], [2.86396102, 2.86396102, 2.86396102, 2.86396102]])),
    ('double_Utube_parallel', None, np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.00464852, 4.00464852, 2.84788608, 2.84788608], [3.90522192, 3.90522192, 2.89301847, 2.89301847], [3.26970513, 3.26970513, 3.26970513, 3.26970513]])),
    ('double_Utube_parallel', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.09844256, 4.09844256, 3.14231047, 3.14231047], [4.03189362, 4.03189362, 3.16367201, 3.16367201], [3.48739254, 3.48739254, 3.48739254, 3.48739254]])),
    # Double U-tube (Series)
    ('double_Utube_series', None, 1., 65., np.array([4.36908096, 2.53231146, 3.13441957, 2.03763963])),
    ('double_Utube_series', None, np.array([1., 2., 3., 1.]), 65., np.array([4.44022419, 2.94528677, 3.54323578, 2.65057213])),
    ('double_Utube_series', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 65., np.array([4.49417249, 3.18261918, 3.73818275, 2.95502223])),
    ('double_Utube_series', None, 1., np.array([65., 75., 150.]), np.array([[4.36908096, 2.53231146, 3.13441957, 2.03763963], [4.28094228, 2.49706752, 3.19348974, 2.0612353], [3.68632591, 2.25874399, 3.68632591, 2.25874399]])),
    ('double_Utube_series', None, np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.44022419, 2.94528677, 3.54323578, 2.65057213], [4.37608674, 2.92420008, 3.58625745, 2.66472128], [3.93523525, 2.77333354, 3.93523525, 2.77333354]])),
    ('double_Utube_series', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.49417249, 3.18261918, 3.73818275, 2.95502223], [4.44797829, 3.17419469, 3.76650271, 2.95801146], [4.06780952, 3.04843839, 4.06780952, 3.04843839]])),
    # Coaxial (Annular pipe is inlet pipe)
    ('coaxial_annular_in', None, 1., 65., np.array([3.15203088, 2.18408362])),
    ('coaxial_annular_in', None, np.array([1., 2., 3., 1.]), 65., np.array([3.4176666 , 2.73205968])),
    ('coaxial_annular_in', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 65., np.array([3.57440691, 3.0183749])),
    ('coaxial_annular_in', None, 1., np.array([65., 75., 150.]), np.array([[3.15203088, 2.18408362], [2.96401382, 2.15051705], [2.0474566, 2.0474566]])),
    ('coaxial_annular_in', None, np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[3.4176666, 2.73205968], [3.2920645, 2.7081367], [2.59520774, 2.59520774]])),
    ('coaxial_annular_in', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[3.57440691, 3.0183749], [3.49639377, 2.99877788], [2.90174463, 2.90174463]])),
    # Coaxial (Annular pipe is outlet pipe)
    ('coaxial_annular_out', None, 1., 65., np.array([4.50649998, 2.92933532])),
    ('coaxial_annular_out', None, np.array([1., 2., 3., 1.]), 65., np.array([4.62416027, 3.50307539])),
    ('coaxial_annular_out', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 65., np.array([4.67083279, 3.72110166])),
    ('coaxial_annular_out', None, 1., np.array([65., 75., 150.]), np.array([[4.50649998, 2.92933532], [4.44976224, 3.02086677], [4.23165721, 4.23165721]])),
    ('coaxial_annular_out', None, np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.62416027, 3.50307539], [4.58402116, 3.57860389], [4.39160319, 4.39160319]])),
    ('coaxial_annular_out', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.67083279, 3.72110166], [4.63666726, 3.76190344], [4.47159269, 4.47159269]])),
    # Coaxial (Annular pipe is inlet pipe, no grout)
    ('coaxial_no_grout', None, 1., 140., np.array([2.8355954, 1.48415788])),
    ('coaxial_no_grout', None, np.array([1., 2., 3., 1.]), 140., np.array([3.2799902, 1.60137175])),
    ('coaxial_no_grout', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 140., np.array([3.51716507, 1.66392916])),
    ('coaxial_no_grout', None, 1., np.array([65., 140., 150.]), np.array([[3.77463898, 1.10936191], [2.8355954, 1.48415788], [2.78756714, 2.78756714]])),
    ('coaxial_no_grout', None, np.array([1., 2., 3., 1.]), np.array([65., 140., 150.]), np.array([[3.95937381, 2.30543206], [3.2799902, 1.60137175], [3.22033437, 3.22033437]])),
    ('coaxial_no_grout', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([65., 140., 150.]), np.array([[4.07193759, 2.7536371], [3.51716507, 1.66392916], [3.45130357, 3.45130357]])),
    ])
def test_temperature(
        pipe_fixture, segment_ratios, T_b, z, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    m_flow_borehole = 0.2   # Fluid mass flow rate [kg/s]
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Fluid temperatures [degC]
    T_f = pipe.get_temperature(
        z, T_f_in, T_b, m_flow_borehole, fluid.cp,
        segment_ratios=segment_ratios)
    assert np.allclose(T_f, expected)


# Test get_outlet_temperature
@pytest.mark.parametrize("pipe_fixture, segment_ratios, T_b, expected", [
    # Single U-tube
    ('single_Utube', None, 1., 2.712371852688313),
    ('single_Utube', None, np.array([1., 2., 3., 1.]), 3.1377635748663573),
    ('single_Utube', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 3.3661222931515784),
    # Double U-tube (Parallel)
    ('double_Utube_parallel', None, 1., 1.8553031331306218),
    ('double_Utube_parallel', None, np.array([1., 2., 3., 1.]), 2.4278457017624655),
    ('double_Utube_parallel', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 2.7404064201268588),
    # Double U-tube (Series)
    ('double_Utube_series', None, 1., 1.8983711735742066),
    ('double_Utube_series', None, np.array([1., 2., 3., 1.]), 2.4755999700741578),
    ('double_Utube_series', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 2.7852683378456216),
    # Coaxial (Annular pipe is inlet pipe)
    ('coaxial_annular_in', None, 1., 2.581130521333567),
    ('coaxial_annular_in', None, np.array([1., 2., 3., 1.]), 3.0276625795763357),
    ('coaxial_annular_in', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 3.2740053393585953),
    # Coaxial (Annular pipe is outlet pipe)
    ('coaxial_annular_out', None, 1., 2.5811305213335674),
    ('coaxial_annular_out', None, np.array([1., 2., 3., 1.]), 2.981638747649938),
    ('coaxial_annular_out', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 3.216388652724723),
    # Coaxial (Annular pipe is inlet pipe, no grout)
    ('coaxial_no_grout', None, 1., 1.157646033266169),
    ('coaxial_no_grout', None, np.array([1., 2., 3., 1.]), 1.1622084785087514),
    ('coaxial_no_grout', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 1.2705296410278433),
    ])
def test_outlet_temperature(
        pipe_fixture, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    m_flow_borehole = 0.2   # Fluid mass flow rate [kg/s]
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Outlet fluid temperature [degC]
    T_f_out = pipe.get_outlet_temperature(
        T_f_in, T_b, m_flow_borehole, fluid.cp, segment_ratios=segment_ratios)
    assert np.isclose(T_f_out, expected)


# Test get_inlet_temperature
@pytest.mark.parametrize("pipe_fixture, segment_ratios, T_b, expected", [
    # Single U-tube
    ('single_Utube', None, 1., 7.595314034714041),
    ('single_Utube', None, np.array([1., 2., 3., 1.]), 8.33912674339739),
    ('single_Utube', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 8.73842016624424),
    # Double U-tube (Parallel)
    ('double_Utube_parallel', None, 1., 5.7977998086638305),
    ('double_Utube_parallel', None, np.array([1., 2., 3., 1.]), 6.526064048901171),
    ('double_Utube_parallel', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 6.923635874226969),
    # Double U-tube (Series)
    ('double_Utube_series', None, 1., 5.864420235466437),
    ('double_Utube_series', None, np.array([1., 2., 3., 1.]), 6.608840446656091),
    ('double_Utube_series', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 7.008202698042327),
    # Coaxial (Annular pipe is inlet pipe)
    ('coaxial_annular_in', None, 1., 7.237470090568812),
    ('coaxial_annular_in', None, np.array([1., 2., 3., 1.]), 7.97588456424095),
    ('coaxial_annular_in', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 8.383252984827369),
    # Coaxial (Annular pipe is outlet pipe)
    ('coaxial_annular_out', None, 1., 7.237470090568813),
    ('coaxial_annular_out', None, np.array([1., 2., 3., 1.]), 7.899776560345228),
    ('coaxial_annular_out', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 8.287974281876208),
    # Coaxial (Annular pipe is inlet pipe, no grout)
    ('coaxial_no_grout', None, 1., 4.926662185940351),
    ('coaxial_no_grout', None, np.array([1., 2., 3., 1.]), 4.93141182148365),
    ('coaxial_no_grout', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 5.044177239200924),
    ])
def test_inlet_temperature(pipe_fixture, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    m_flow_borehole = 0.2   # Fluid mass flow rate [kg/s]
    Q_f = -3000.0           # Total heat transfer rate [W]
    # Inlet fluid temperature [degC]
    T_f_in = pipe.get_inlet_temperature(
        Q_f, T_b, m_flow_borehole, fluid.cp, segment_ratios=segment_ratios)
    assert np.isclose(T_f_in, expected)


# Test get_borehole_heat_extraction_rate
@pytest.mark.parametrize("pipe_fixture, segment_ratios, T_b, expected", [
    # Single U-tube
    ('single_Utube', None, 1., -1819.4736348927008),
    ('single_Utube', None, np.array([1., 2., 3., 1.]), np.array([-507.98022943, -330.29924271, -155.92399643, -486.93326314])),
    ('single_Utube', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-212.48822505, -496.72428991, -284.6464931, -305.65175979])),
    # Double U-tube (Parallel)
    ('double_Utube_parallel', None, 1., -2501.14645849),
    ('double_Utube_parallel', None, np.array([1., 2., 3., 1.]), np.array([-796.48662356, -444.22614316, -108.02227066, -697.03753979])),
    ('double_Utube_parallel', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-345.6258447, -724.00374747, -270.96795891, -456.57868535])),
    # Double U-tube (Series)
    ('double_Utube_series', None, 1., -2466.89213085),
    ('double_Utube_series', None, np.array([1., 2., 3., 1.]), np.array([-745.16518357, -428.05472293, -114.8035859, -719.7675482])),
    ('double_Utube_series', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-323.00538071, -688.58422816, -281.10403131, -468.80150159])),
    # Coaxial (Annular pipe is inlet pipe)
    ('coaxial_annular_in', None, 1., -1923.85692048),
    ('coaxial_annular_in', None, np.array([1., 2., 3., 1.]), np.array([-757.51176437, -346.76503548, -48.92829119, -415.50088061])),
    ('coaxial_annular_in', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-335.1152758, -618.6742642, -139.8377907, -279.14900216])),
    # Coaxial (Annular pipe is outlet pipe)
    ('coaxial_annular_out', None, 1., -1923.85692048),
    ('coaxial_annular_out', None, np.array([1., 2., 3., 1.]), np.array([-480.81667849, -324.83211948, -133.10520419, -666.55719699])),
    ('coaxial_annular_out', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-205.74675004, -483.36307319, -299.14468223, -430.34747541])),
    # Coaxial (Annular pipe is inlet pipe, no grout)
    ('coaxial_no_grout', None, 1., -3056.03065193),
    ('coaxial_no_grout', None, np.array([1., 2., 3., 1.]), np.array([-1316.35767661, -1055.61151182, 1327.31501919, -2007.74772473])),
    ('coaxial_no_grout', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-896.16349915, -1263.97717837, 1166.37595544, -1972.48352864])),
    ])
def test_borehole_heat_extraction_rate(
        pipe_fixture, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    m_flow_borehole = 0.2   # Fluid mass flow rate [kg/s]
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Borehole heat extraction rates [W]
    Q_b = pipe.get_borehole_heat_extraction_rate(
        T_f_in, T_b, m_flow_borehole, fluid.cp, segment_ratios=segment_ratios)
    assert np.allclose(Q_b, expected)


# Test get_fluid_heat_extraction_rate
@pytest.mark.parametrize("pipe_fixture, segment_ratios, T_b, expected", [
    # Single U-tube
    ('single_Utube', None, 1., -1819.4736348927008),
    ('single_Utube', None, np.array([1., 2., 3., 1.]), -1481.1367317058312),
    ('single_Utube', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1299.5107678418537),
    # Double U-tube (Parallel)
    ('double_Utube_parallel', None, 1., -2501.146458493431),
    ('double_Utube_parallel', None, np.array([1., 2., 3., 1.]), -2045.7725771641726),
    ('double_Utube_parallel', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1797.176236436576),
    # Double U-tube (Series)
    ('double_Utube_series', None, 1., -2466.892130845958),
    ('double_Utube_series', None, np.array([1., 2., 3., 1.]), -2007.7910405893476),
    ('double_Utube_series', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1761.49514176394),
    # Coaxial (Annular pipe is inlet pipe)
    ('coaxial_annular_in', None, 1., -1923.85692048),
    ('coaxial_annular_in', None, np.array([1., 2., 3., 1.]), -1568.705971637178),
    ('coaxial_annular_in', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1372.776332855085),
    # Coaxial (Annular pipe is outlet pipe)
    ('coaxial_annular_out', None, 1., -1923.85692048),
    ('coaxial_annular_out', None, np.array([1., 2., 3., 1.]), -1605.3111991367698),
    ('coaxial_annular_out', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1418.6019808667165),
    # Coaxial (Annular pipe is inlet pipe, no grout)
    ('coaxial_no_grout', None, 1., -3056.0306519279184),
    ('coaxial_no_grout', None, np.array([1., 2., 3., 1.]), -3052.4018939764674),
    ('coaxial_no_grout', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -2966.2482507211057),
    ])
def test_fluid_heat_extraction_rate(
        pipe_fixture, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    m_flow_borehole = 0.2   # Fluid mass flow rate [kg/s]
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Fluid heat extraction rate [W]
    Q_f = pipe.get_fluid_heat_extraction_rate(
        T_f_in, T_b, m_flow_borehole, fluid.cp, segment_ratios=segment_ratios)
    assert np.isclose(Q_f, expected)


# Test get_total_heat_extraction_rate
@pytest.mark.parametrize("pipe_fixture, segment_ratios, T_b, expected", [
    # Single U-tube
    ('single_Utube', None, 1., -1819.4736348927008),
    ('single_Utube', None, np.array([1., 2., 3., 1.]), -1481.1367317058312),
    ('single_Utube', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1299.5107678418537),
    # Double U-tube (Parallel)
    ('double_Utube_parallel', None, 1., -2501.146458493431),
    ('double_Utube_parallel', None, np.array([1., 2., 3., 1.]), -2045.7725771641726),
    ('double_Utube_parallel', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1797.176236436576),
    # Double U-tube (Series)
    ('double_Utube_series', None, 1., -2466.892130845958),
    ('double_Utube_series', None, np.array([1., 2., 3., 1.]), -2007.7910405893476),
    ('double_Utube_series', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1761.49514176394),
    # Coaxial (Annular pipe is inlet pipe)
    ('coaxial_annular_in', None, 1., -1923.85692048),
    ('coaxial_annular_in', None, np.array([1., 2., 3., 1.]), -1568.705971637178),
    ('coaxial_annular_in', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1372.776332855085),
    # Coaxial (Annular pipe is outlet pipe)
    ('coaxial_annular_out', None, 1., -1923.85692048),
    ('coaxial_annular_out', None, np.array([1., 2., 3., 1.]), -1605.3111991367698),
    ('coaxial_annular_out', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1418.6019808667165),
    # Coaxial (Annular pipe is inlet pipe, no grout)
    ('coaxial_no_grout', None, 1., -3056.0306519279184),
    ('coaxial_no_grout', None, np.array([1., 2., 3., 1.]), -3052.4018939764674),
    ('coaxial_no_grout', np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -2966.2482507211057),
    ])
def test_total_heat_extraction_rate(
        pipe_fixture, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    m_flow_borehole = 0.2   # Fluid mass flow rate [kg/s]
    T_f_in = 5.0            # Inlet fluid temperature [degC]
    # Total heat extraction rate [W]
    Q_t = pipe.get_total_heat_extraction_rate(
        T_f_in, T_b, m_flow_borehole, fluid.cp, segment_ratios=segment_ratios)
    assert np.isclose(Q_t, expected)


# =============================================================================
# Test IndependentMultipleUTube class
# =============================================================================
# Test get_temperature
@pytest.mark.parametrize(
    "pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, z, expected", [
    # Double U-tube
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, 1., 65., np.array([4.33561246, -0.53401739, 3.03985865, 0.28974217])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, np.array([1., 2., 3., 1.]), 65., np.array([4.40351925, -0.44632268, 3.43990994, 0.77984857])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 65., np.array([4.45572396, -0.37988472, 3.63010142, 1.00960735])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, 1., np.array([65., 75., 150.]), np.array([[4.33561246, -0.53401739, 3.03985865, 0.28974217], [4.2430049, -0.47133142, 3.10196228, 0.25289182], [3.61910357, -0.06417533, 3.61910357, -0.06417533]])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.40351925, -0.44632268, 3.43990994, 0.77984857], [4.33450267, -0.35334337, 3.48624558, 0.72533714], [3.86172238, 0.2372006, 3.86172238, 0.2372006]])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.45572396, -0.37988472, 3.63010142, 1.00960735], [4.40442447, -0.26400763, 3.66189091, 0.93718219], [3.99089068, 0.39795939, 3.99089068, 0.39795939]])),
    # Triple U-tube
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, 1., 65., np.array([4.42233734, -0.45836746, 5.35867251, 3.21820626, 0.54732768, 2.80410243])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, np.array([1., 2., 3., 1.]), 65., np.array([4.47430559, -0.3726503, 5.47798812, 3.56864252, 0.98391184, 3.36084568])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), 65., np.array([4.51695038, -0.3115071, 5.56146368, 3.73728694, 1.18959709, 3.6163127])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, 1., np.array([65., 75., 150.]), np.array([[4.42233734, -0.45836746, 5.35867251, 3.21820626, 0.54732768, 2.80410243], [4.33960106, -0.38478969, 5.148416, 3.27807117, 0.49886893, 2.90891002], [3.76853744, 0.10098669,  3.84628411, 3.76853744, 0.10098669, 3.84628411]])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.47430559, -0.3726503, 5.47798812, 3.56864252, 0.98391184, 3.36084568], [4.41115886, -0.27214867, 5.30491836, 3.61292754, 0.92062819, 3.45008461], [3.9754362, 0.37402984, 4.20147852, 3.9754362, 0.37402984, 4.20147852]])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([65., 75., 150.]), np.array([[4.51695038, -0.3115071, 5.56146368, 3.73728694, 1.18959709, 3.6163127], [4.46863202, -0.19118263, 5.41633523, 3.76847629, 1.11127473, 3.68580011], [4.08533665, 0.51957984, 4.39206115, 4.08533665, 0.51957984, 4.39206115]])),
    ])
def test_temperature_independent(
        pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, z, expected,
        request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    # Fluid temperatures [degC]
    T_f = pipe.get_temperature(
        z, T_f_in, T_b, m_flow, fluid.cp, segment_ratios=segment_ratios)
    assert np.allclose(T_f, expected)


# Test get_outlet_temperature
@pytest.mark.parametrize(
    "pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, expected", [
    # Double U-tube
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, 1., np.array([2.66975268, 0.50433911])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, np.array([1., 2., 3., 1.]), np.array([3.07152772, 0.9760115])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([3.28701848, 1.23027312])),
    # Triple U-tube
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, 1., np.array([2.85759687, 0.84655363, 2.20811155])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, np.array([1., 2., 3., 1.]), np.array([3.22566425, 1.26691386, 2.70350023])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([3.42207691, 1.49331263, 2.97388805])),
    ])
def test_outlet_temperature_independent(
        pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    # Outlet fluid temperatures [degC]
    T_f_out = pipe.get_outlet_temperature(
        T_f_in, T_b, m_flow, fluid.cp, segment_ratios=segment_ratios)
    assert np.allclose(T_f_out, expected)


# Test get_inlet_temperature
@pytest.mark.parametrize(
    "pipe_fixture, m_flow, Q_f, segment_ratios, T_b, expected", [
    # Double U-tube
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([-3000., 2000.]), None, 1., np.array([7.40595748, -3.59946781])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([-3000., 2000.]), None, np.array([1., 2., 3., 1.]), np.array([8.15037424, -2.85931237])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([-3000., 2000.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([8.54973576, -2.4604302])),
    # Triple U-tube
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([-3000., 2000., -2500.]), None, 1., np.array([7.87321014, -2.88443189, 8.87646527])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([-3000., 2000., -2500.]), None, np.array([1., 2., 3., 1.]), np.array([8.62102769, -2.14384786, 9.60745667])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([-3000., 2000., -2500.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([9.02076799, -1.74491241, 10.00533859])),
    ])
def test_inlet_temperature_independent(
        pipe_fixture, m_flow, Q_f, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    # Inlet fluid temperatures [degC]
    T_f_in = pipe.get_inlet_temperature(
        Q_f, T_b, m_flow, fluid.cp, segment_ratios=segment_ratios)
    assert np.allclose(T_f_in, expected)


# Test get_borehole_heat_extraction_rate
@pytest.mark.parametrize(
    "pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, expected", [
    # Double U-tube
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, 1., -956.00963184),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, np.array([1., 2., 3., 1.]), np.array([-311.65264343, -12.22888682, 289.43868666, -320.65369703])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-138.6403424, -69.24200957, 394.70262906, -218.85448613])),
    # Triple U-tube
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, 1., -2508.09408199),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, np.array([1., 2., 3., 1.]), np.array([-754.8446691, -351.75861521, 43.23971775, -704.23081162])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-323.63535083, -573.51851525, -16.07605399, -455.56959127])),
    ])
def test_borehole_heat_extraction_rate_independent(
        pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    # Borehole heat extraction rates [W]
    Q_b = pipe.get_borehole_heat_extraction_rate(
        T_f_in, T_b, m_flow, fluid.cp, segment_ratios=segment_ratios)
    assert np.allclose(Q_b, expected)


# Test get_fluid_heat_extraction_rate
@pytest.mark.parametrize(
    "pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, expected", [
    # Double U-tube
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, 1., np.array([-1853.37094997, 897.36131814])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, np.array([1., 2., 3., 1.]), np.array([-1533.81766537, 1178.72112475])),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-1362.42628578, 1330.39207674])),
    # Triple U-tube
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, 1., np.array([-1703.96836927, 1101.4975216, -1905.62323433])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, np.array([1., 2., 3., 1.]), np.array([-1411.22460372, 1352.24883733, -1708.61861179])),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), np.array([-1255.0070539, 1487.29916766, -1601.09162509])),
    ])
def test_fluid_heat_extraction_rate_independent(
        pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    # Fluid heat extraction rates [W]
    Q_f = pipe.get_fluid_heat_extraction_rate(
        T_f_in, T_b, m_flow, fluid.cp, segment_ratios=segment_ratios)
    assert np.allclose(Q_f, expected)


# Test get_total_heat_extraction_rate
@pytest.mark.parametrize(
    "pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, expected", [
    # Double U-tube
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, 1., -956.0096318353369),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), None, np.array([1., 2., 3., 1.]), -355.0965406202847),
    ('double_Utube_independent', np.array([0.2, 0.15]), np.array([5., -1.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -32.03420904269501),
    # Triple U-tube
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, 1., -2508.094081991296),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), None, np.array([1., 2., 3., 1.]), -1767.5943781810613),
    ('triple_Utube_independent', np.array([0.2, 0.15, 0.10]), np.array([5., -1., 7.]), np.array([0.1, 0.35, 0.40, 0.15]), np.array([1., 2., 3., 1.]), -1368.7995113394977),
    ])
def test_total_heat_extraction_rate_independent(
        pipe_fixture, m_flow, T_f_in, segment_ratios, T_b, expected, request):
    # Extract pipe from fixture
    pipe = request.getfixturevalue(pipe_fixture)
    # Fluid is propylene-glycol 20%
    fluid = gt.media.Fluid('MPG', 20.)
    # Total heat extraction rate [W]
    Q_t = pipe.get_total_heat_extraction_rate(
        T_f_in, T_b, m_flow, fluid.cp, segment_ratios=segment_ratios)
    assert np.isclose(Q_t, expected)
