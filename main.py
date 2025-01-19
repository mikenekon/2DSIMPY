from multiconcalc import MultiConductorCalculator

def test1():
  calculator = MultiConductorCalculator(epsilon_r=2.0)

  radius = 100e-9
  height = 300e-9
  spacing = 3 * radius
  n=20

  calculator.add_conductor(radius=radius, height=height, N_points=n, x_offset=0.0)
  calculator.add_conductor(radius=radius, height=height, N_points=n, x_offset=spacing)
  calculator.add_conductor(radius=radius, height=height, N_points=n, x_offset=spacing*2)

  voltages = [1.0,-1.0, 2.0]
  charge_density = calculator.solve_charge_density(voltages)

  C = calculator.calculate_capacitance_matrix()
  print("\n C Matrix [F/m]:")
  print(C)

  calculator.plot_chage_distribution(charge_density)

  calculator.plot_potential(charge_density)

# RECTのテスト
def test2():
    calculator = MultiConductorCalculator(epsilon_r=2.0)
    
    w = 20e-9
    h = 50e-9
    n = 12
    bh1 = 300e-9
    bh2 = 400e-9
    bh3 = 500e-9
    bh4 = 600e-9
    bh5 = 700e-9
    xo1 = 0
    xo2 = 400e-9
    xo3 = 800e-9
    xo4 = 1200e-9
    xo5 = 1600e-9
    
        # Add rectangular conductor
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo3  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo5  )

    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo3  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo5  )

    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo3  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo5  )

    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo3  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo5  )

    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo3  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo5  )

    # Modified voltages array to match the number of conductors
    voltages = [1.0 ,1,0, 1.0, 1.0, 1.0,
                1.0 ,1,0, 1.0, 1.0, 1.0,
                1.0 ,1,0, 1.0, 1.0, 1.0,
                1.0 ,1,0, 1.0, 1.0, 1.0,
                1.0 ,1,0, 1.0, 1.0, 1.0
                ]  # Only one voltage for one conductor
    
    charge_density = calculator.solve_charge_density(voltages)
    calculator.plot_chage_distribution(charge_density)
    calculator.plot_potential(charge_density) 
       
    C = calculator.calculate_capacitance_matrix()
    print("\n C Matrix [F/m]:")
    print(C)
    
    # calculator.plot_chage_distribution(charge_density)
    # calculator.plot_potential(charge_density)



if __name__ == "__main__":
  # test1()
  test2()
