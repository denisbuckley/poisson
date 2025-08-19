
def main():
    """
    Main function to run the simulation based on user choice.
    """
    print("Welcome to the Glider Thermal Simulation Script!")
    print("Please choose a mode of operation:")
    print("1. Single Simulation (Hexagonal Model) with Plotting")
    print("2. Monte Carlo Simulation with User Input")
    print("3. Nested Loop Simulation (Poisson Model) and save to CSV")
    print("4. Nested Loop Simulation (Hexagonal Model) and save to CSV")

    try:
        choice = int(input("Enter your choice (1, 2, 3, or 4): "))
        if choice == 1:
            print(f"\n--- Mode {choice}: Single Simulation with Plotting (Hexagonal) ---")

            # Use fixed parameters for this mode for consistency
            z_cbl = 2500.0
            lambda_thermals = 0.02
            lambda_strength = 2.0
            mc_band1 = 4.0
            mc_band2 = 2.0
            end_point = (-50000.0, -100000.0)
            search_arc = 30.0
            initial_bearing = 210.0  # From a previous successful run
            thermal_model = 'hexagonal'

            params = {
                'z_cbl': z_cbl,
                'lambda_thermals': lambda_thermals,
                'lambda_strength': lambda_strength,
                'mc_band1': mc_band1,
                'mc_band2': mc_band2,
                'search_arc': search_arc
            }

            result = simulate_intercept_experiment_dynamic(
                z_cbl_meters=z_cbl,
                lambda_thermals_per_sq_km=lambda_thermals,
                lambda_strength=lambda_strength,
                mc_sniff_band1=mc_band1,
                mc_sniff_band2=mc_band2,
                end_point=end_point,
                search_arc_angle=search_arc,
                plot_simulation=True,
                thermal_model=thermal_model
            )

            print_detailed_single_flight_results(result, initial_bearing, params)
            if result.get('plot'):
                plt.show()

        elif choice == 2:
            print(f"\n--- Mode {choice}: Monte Carlo Simulation ---")
            run_monte_carlo_with_user_input()

        elif choice == 3:
            run_nested_loop_simulation_and_save_to_csv('poisson')

        elif choice == 4:
            run_nested_loop_simulation_and_save_to_csv('hexagonal')

        else:
            print("Invalid choice. Please run the script again and select 1, 2, 3, or 4.")
    except Exception as e:
        print(f"\n--- An unexpected error occurred! Please see the error message below. ---")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


def simulate_intercept_experiment_dynamic(
        z_cbl_meters,
        lambda_thermals_per_sq_km,
        lambda_strength,
        mc_sniff_band1,
        mc_sniff_band2,
        end_point,
        search_arc_angle,
        plot_simulation=False,
        thermal_model='hexagonal'
):
    """
    Simulates a single glider flight with dynamic Macready settings and plotting.

    Args:
        z_cbl_meters (float): The height of the cloud base in meters.
        lambda_thermals_per_sq_km (float): The lambda for thermal density.
        lambda_strength (float): The lambda for thermal strength.
        mc_sniff_band1 (float): The Macready setting for altitude band 1.
        mc_sniff_band2 (float): The Macready setting for altitude band 2.
        end_point (tuple): The (x, y) coordinates of the destination.
        search_arc_angle (float): The angle of the search arc in degrees.
        plot_simulation (bool): Whether to generate a plot of the simulation.
        thermal_model (str): The model for thermal placement ('poisson' or 'hexagonal').

    Returns:
        dict: A dictionary of simulation results, including plotting data.
    """
    # --- SIMULATION INITIALIZATION ---
    initial_glider_pos = (0.0, 0.0)
    current_glider_pos = initial_glider_pos
    current_altitude = z_cbl_meters
    time_elapsed = 0.0
    thermal_found = False
    climb_rate = 0.0

    path_points = [current_glider_pos]

    # Generate thermals
    if thermal_model == 'poisson':
        thermals = generate_thermals_poisson(
            area_size=100_000,
            lambda_thermals=lambda_thermals_per_sq_km,
            lambda_strength=lambda_strength
        )
    elif thermal_model == 'hexagonal':
        thermals = generate_thermals_hexagonal(
            area_size=100_000,
            lambda_thermals=lambda_thermals_per_sq_km,
            lambda_strength=lambda_strength
        )
    else:
        raise ValueError("Invalid thermal model specified.")

    # --- SIMULATION LOOP ---
    while current_altitude > GLOBAL_MIN_ALTITUDE and time_elapsed < MAX_SIMULATION_TIME_SECONDS:

        # Determine Macready setting based on altitude
        if current_altitude > z_cbl_meters * 0.5:
            current_macready = mc_sniff_band1
        else:
            current_macready = mc_sniff_band2

        # Calculate search distance based on Macready setting
        glider_sink_rate = calculate_glider_sink_rate(GLIDER_SPEED)
        search_glide_ratio = GLIDER_SPEED / (glider_sink_rate + current_macready)
        search_distance = current_altitude * search_glide_ratio

        # Find the nearest thermal in the search arc
        nearest_thermal = find_nearest_thermal_in_arc(current_glider_pos, thermals, search_arc_angle)

        if nearest_thermal:
            thermal_dist = calculate_distance(current_glider_pos, (nearest_thermal['x'], nearest_thermal['y']))

            if thermal_dist <= search_distance:
                # Glider intercepts the thermal
                time_to_thermal = thermal_dist / GLIDER_SPEED
                time_elapsed += time_to_thermal
                current_altitude -= (glider_sink_rate + current_macready) * time_to_thermal
                current_glider_pos = (nearest_thermal['x'], nearest_thermal['y'])
                path_points.append(current_glider_pos)

                # Glider decides to climb or not
                if nearest_thermal['strength'] >= current_macready:
                    thermal_found = True
                    climb_rate = nearest_thermal['strength']

                    # Climb until cloud base or max time
                    while current_altitude < z_cbl_meters and time_elapsed < MAX_SIMULATION_TIME_SECONDS:
                        climb_time_step = 10.0  # Climb in 10-second steps
                        altitude_change = climb_rate * climb_time_step
                        current_altitude += altitude_change
                        time_elapsed += climb_time_step

                    # Cap altitude at cloud base
                    current_altitude = min(current_altitude, z_cbl_meters)

                    # Now glide to the end point or next thermal
                    glide_dist_to_end = calculate_distance(current_glider_pos, end_point)
                    if glide_dist_to_end <= search_distance:
                        time_to_end = glide_dist_to_end / GLIDER_SPEED
                        time_elapsed += time_to_end
                        current_altitude -= (glider_sink_rate) * time_to_end
                        current_glider_pos = end_point
                        path_points.append(current_glider_pos)
                        break  # End of flight

                # In either case (climb or no climb), remove the thermal so it is not considered again.
                # This prevents the infinite loop.
                thermals.remove(nearest_thermal)

            else:
                # No thermals found in range, glide towards the end point
                dist_to_end = calculate_distance(current_glider_pos, end_point)
                time_to_end = dist_to_end / GLIDER_SPEED
                time_elapsed += time_to_end
                current_altitude -= (glider_sink_rate + current_macready) * time_to_end
                current_glider_pos = end_point
                path_points.append(current_glider_pos)
                break  # End of flight
        else:
            # No thermals, glide towards end point
            dist_to_end = calculate_distance(current_glider_pos, end_point)
            time_to_end = dist_to_end / GLIDER_SPEED
            time_elapsed += time_to_end
            current_altitude -= (glider_sink_rate) * time_to_end  # No Macready sink as there's no sniffing
            current_glider_pos = end_point
            path_points.append(current_glider_pos)
            break  # End of flight

    # --- PLOTTING ---
    if plot_simulation:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot thermals with red updraft core and green downdraft annulus
        for thermal in thermals:
            # Downdraft ring (green annulus)
            downdraft_radius = thermal['radius'] + DOWNDRAFT_RING_WIDTH
            downdraft_circle = plt.Circle(
                (thermal['x'], thermal['y']),
                downdraft_radius,
                color='green',
                fill=False,
                linestyle='--',
                alpha=0.5,
                label='_nolegend_'
            )
            ax.add_patch(downdraft_circle)

            # Updraft core (red circle)
            updraft_circle = plt.Circle(
                (thermal['x'], thermal['y']),
                thermal['radius'],
                color='red',
                label='_nolegend_'
            )
            ax.add_patch(updraft_circle)

        # Plot glider path
        path_x = [p[0] for p in path_points]
        path_y = [p[1] for p in path_points]
        ax.plot(path_x, path_y, 'g-', marker='^', markersize=8, label='Glider Path')

        # Plot start and end points
        # Start point (Green circle)
        ax.plot(path_points[0][0], path_points[0][1], 'go', markersize=10, label='Start')
        # End point (Red X)
        ax.plot(path_points[-1][0], path_points[-1][1], 'rx', markersize=10, label='End')

        # Add labels, title, and legend
        ax.set_title(f'Glider Thermal Interception Simulation ({thermal_model} model)')
        ax.set_xlabel('East-West (m)')
        ax.set_ylabel('North-South (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True)

        # Store the plot object in the results dictionary to be shown later
        plot = fig

    # --- RETURN RESULTS ---
    results = {
        'total_time': time_elapsed,
        'final_altitude': current_altitude,
        'thermal_found': thermal_found,
        'path_points': path_points
    }

    if plot_simulation:
        results['plot'] = plot

    return results
