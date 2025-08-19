import math
import random
import matplotlib.pyplot as plt
import numpy as np

# --- SCRIPT CHARACTERISTICS ---
# This file contains the core simulation logic for a glider's flight.
# It is designed to be imported by the main script (scratch3.py).

# --- CONSTANTS ---
GLIDER_SINK_RATE = 1.0  # m/s
GLIDER_SPEED_KPH = 100.0
GLIDER_SPEED_MPS = GLIDER_SPEED_KPH * 1000.0 / 3600.0
DOWNDRAFT_FACTOR = 1.05  # How much worse sink rate is in downdraft
DOWNDRAFT_DIAMETER = 300.0  # m
MIN_ALTITUDE = 500  # m, flight ends if altitude drops below this


def generate_hexagonal_thermals(sim_area_side, lambda_thermals, lambda_strength, z_cbl_meters):
    """
    Generates thermals in a semi-random hexagonal grid.

    This is a simplified version for demonstration. In the user's full script,
    this function would contain the detailed logic for thermal placement.
    """
    thermals = []
    # Create a grid of points
    grid_spacing = 1000  # meters
    for x in range(0, int(sim_area_side), grid_spacing):
        for y in range(0, int(sim_area_side), grid_spacing):
            # Use Bernoulli trial to determine if a thermal exists at this grid point
            if random.random() < 0.5:  # 50% chance of thermal existing
                strength = int(np.random.poisson(lambda_strength)) + 1
                strength = max(1, min(10, strength))
                radius = 100 + 10 * strength  # Placeholder for a realistic radius
                thermals.append({
                    'x': x + random.uniform(-100, 100),
                    'y': y + random.uniform(-100, 100),
                    'radius': radius,
                    'strength': strength
                })
    return thermals


def calculate_bearing(start_point, end_point):
    """Calculates the bearing in degrees from a start to an end point."""
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    return math.degrees(math.atan2(dy, dx))


def is_in_thermal(pos, thermals):
    """Checks if a position is inside a thermal updraft or downdraft."""
    for thermal in thermals:
        distance = math.sqrt((pos[0] - thermal['x']) ** 2 + (pos[1] - thermal['y']) ** 2)
        if distance <= thermal['radius']:
            return 'updraft', thermal['strength']
        elif distance <= thermal['radius'] + DOWNDRAFT_DIAMETER:
            return 'downdraft', None
    return 'neutral', None


def find_nearest_thermal_in_arc(glider_pos, end_point, thermals, search_arc_angle):
    """Finds the nearest thermal within a specified search arc towards the end point."""
    bearing_to_end = calculate_bearing(glider_pos, end_point)
    closest_thermal = None
    min_distance = float('inf')

    for thermal in thermals:
        thermal_pos = (thermal['x'], thermal['y'])
        distance = math.sqrt((glider_pos[0] - thermal_pos[0]) ** 2 + (glider_pos[1] - thermal_pos[1]) ** 2)

        # Calculate angle to the thermal relative to the bearing to the end point
        bearing_to_thermal = calculate_bearing(glider_pos, thermal_pos)
        angle_delta = (bearing_to_thermal - bearing_to_end + 180) % 360 - 180

        # Check if the thermal is within the search arc
        if abs(angle_delta) <= search_arc_angle / 2.0 and distance < min_distance:
            min_distance = distance
            closest_thermal = thermal

    return closest_thermal, min_distance


def simulate_intercept_experiment_dynamic(
        z_cbl_meters, lambda_thermals_per_sq_km, lambda_strength,
        mc_sniff_band1, mc_sniff_band2, end_point, search_arc_angle, plot_simulation=False, thermal_model='poisson'
):
    """
    Runs a single simulation of a glider's flight with the selected thermal model.
    """
    path_points = [(0, 0)]
    current_pos = (0, 0)
    current_altitude = z_cbl_meters
    total_height_climbed = 0.0
    total_climbing_time = 0.0
    total_gliding_time = 0.0
    total_distance_covered = 0.0
    intercept_count = 0
    flight_time = 0.0
    success = False

    angle_deltas = []

    plot_padding = 20000.0
    max_coord = max(abs(end_point[0]), abs(end_point[1]))
    sim_area_side_meters = (max_coord + plot_padding) * 2

    if thermal_model == 'poisson':
        # Placeholder for Poisson thermal generation
        updraft_thermals_info = []
    elif thermal_model == 'hexagonal':
        updraft_thermals_info = generate_hexagonal_thermals(
            sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength, z_cbl_meters
        )
    else:
        raise ValueError("Invalid thermal model specified. Choose 'poisson' or 'hexagonal'.")

    # Main flight simulation loop
    while current_altitude > MIN_ALTITUDE:
        # Check if the glider has reached the end point
        if math.sqrt((current_pos[0] - end_point[0]) ** 2 + (current_pos[1] - end_point[1]) ** 2) < 500:
            success = True
            break

        # Determine current Macready setting based on altitude
        macready_sniff = mc_sniff_band1 if current_altitude > 1500 else mc_sniff_band2

        # Find the nearest thermal in the search arc
        target_thermal, distance_to_thermal = find_nearest_thermal_in_arc(current_pos, end_point, updraft_thermals_info,
                                                                          search_arc_angle)

        if target_thermal and target_thermal['strength'] >= macready_sniff:
            # Fly towards the thermal
            gliding_time = distance_to_thermal / GLIDER_SPEED_MPS
            altitude_loss = gliding_time * GLIDER_SINK_RATE

            # Check for downdraft on the way
            # This is a simplified check for downdraft on the path
            is_path_in_downdraft = False
            for thermal in updraft_thermals_info:
                distance_to_thermal = math.sqrt(
                    (current_pos[0] - thermal['x']) ** 2 + (current_pos[1] - thermal['y']) ** 2)
                if thermal['radius'] < distance_to_thermal < thermal['radius'] + DOWNDRAFT_DIAMETER:
                    is_path_in_downdraft = True
                    break

            if is_path_in_downdraft:
                altitude_loss *= DOWNDRAFT_FACTOR

            current_altitude -= altitude_loss
            current_pos = (target_thermal['x'], target_thermal['y'])
            total_gliding_time += gliding_time
            total_distance_covered += distance_to_thermal
            flight_time += gliding_time
            path_points.append(current_pos)

            # Record the angle delta
            bearing_to_end = calculate_bearing(path_points[-2], end_point)
            bearing_to_thermal = calculate_bearing(path_points[-2], current_pos)
            angle_delta = (bearing_to_thermal - bearing_to_end + 180) % 360 - 180
            angle_deltas.append(angle_delta)

            intercept_count += 1

            # Climb the thermal
            height_climbed = (z_cbl_meters - current_altitude)
            climbing_time = height_climbed / target_thermal['strength']
            current_altitude += height_climbed
            total_height_climbed += height_climbed
            total_climbing_time += climbing_time
            flight_time += climbing_time

        else:
            # No thermal to intercept, glide straight towards the end point
            distance_to_end = math.sqrt((current_pos[0] - end_point[0]) ** 2 + (current_pos[1] - end_point[1]) ** 2)
            time_to_end = distance_to_end / GLIDER_SPEED_MPS
            altitude_loss = time_to_end * GLIDER_SINK_RATE
            current_altitude -= altitude_loss
            current_pos = end_point
            total_gliding_time += time_to_end
            total_distance_covered += distance_to_end
            flight_time += time_to_end
            path_points.append(current_pos)

            # If we reached the end point, the loop will break on the next check.
            success = True
            break

    plot = None
    if plot_simulation:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot thermals
        for thermal in updraft_thermals_info:
            # Downdraft ring (blue circle)
            downdraft_circle = plt.Circle(
                (thermal['x'], thermal['y']),
                thermal['radius'] + DOWNDRAFT_DIAMETER,
                color='lightblue',
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
        ax.plot(path_points[0][0], path_points[0][1], 'go', markersize=10, label='Start')
        ax.plot(path_points[-1][0], path_points[-1][1], 'rx', markersize=10, label='End')

        # Add labels, title, and legend
        ax.set_title(f'Glider Thermal Interception Simulation ({thermal_model} model)')
        ax.set_xlabel('East-West (m)')
        ax.set_ylabel('North-South (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True)
        plot = fig

    # --- RETURN RESULTS ---
    results = {
        'success': success,
        'final_altitude': current_altitude,
        'flight_time': flight_time,
        'total_height_climbed': total_height_climbed,
        'total_climbing_time': total_climbing_time,
        'total_gliding_time': total_gliding_time,
        'total_distance_covered': total_distance_covered,
        'intercept_count': intercept_count,
        'plot': plot,
        'path': path_points,
        'thermals': updraft_thermals_info,
        'angle_deltas': angle_deltas,
    }

    return results

