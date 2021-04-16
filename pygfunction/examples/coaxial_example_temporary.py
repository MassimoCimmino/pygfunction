import pygfunction as gt


def main():
    # Fluid properties
    mix = 'MEA'
    percent = 23.
    fluid = gt.media.Fluid(T=3., mixer=mix, percent=percent)
    print(fluid)
    print(50 * '-')

    # Flow rate
    V_dot = 3.1  # m^3/hour
    m_flow = V_dot / 3600 * fluid.rho
    print('Mass flow rate (kg/s) = {}'.format(m_flow))
    print('Volumetric flow rate (L/s) = {}'.format(V_dot / 3600 * 1000))

    # Borehole Specification (GLHEPRO provides these values in Diameter)
    # Borehole dimensions
    H = 100.  # Borehole length (m)
    D = 5.  # Borehole buried depth (m)
    r_b = 150 / 1000 / 2  # borehole radius (m)
    # Inner tube
    r_in_in = 44.2 / 1000 / 2  # Inner pipe inner radius (m)
    r_in_out = 50 / 1000 / 2  # Inner pipe outer radius (m)
    # Outer tube
    r_out_in = 97.4 / 1000 / 2  # Outer pipe inner radius (m)
    r_out_out = 110 / 1000 / 2  # Outer pipe outer radius (m)

    epsilon_in = 1.0e-06

    # Pipe position
    # Coaxial (Concentric) pipe
    pos = [(0, 0)]

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    # Thermal physical properties
    # Pipe properties
    k_p_inner = 0.4  # Inner Pipe thermal conductivity (W/m.K)
    k_p_outer = 0.4  # Outer Pipe thermal conductivity (W/m.K)
    # Soil properties
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    # Grout properties
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    concentric_pipe = gt.pipes.SingleCoaxialPipe(pos, r_in_in, r_in_out, r_out_in, r_out_out,
                               borehole, k_p_inner, k_p_outer, k_s, k_g, epsilon_in)
    Rb = concentric_pipe.compute_effective_borehole_resistance(m_flow,
                                                               fluid.mu,
                                                               fluid.rho,
                                                               fluid.k,
                                                               fluid.cp,
                                                               'AVG',
                                                               disp=True)
    print('Effective borehole thermal resistance (K/(W/m)) = {}'.format(Rb))


if __name__ == '__main__':
    main()
