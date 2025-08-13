# --- SCRIPT CHARACTERISTICS ---
# This is a comprehensive Monte Carlo simulation script for a glider's thermal interception.
# The simulation environment models a 2D plane with a hexagonal grid-based thermal placement.
# This version features an interactive menu to select from four different simulation modes:
# 1. Single simulation with a detailed plot.
# 2. Monte Carlo simulation with default parameters for statistical analysis.
# 3. Nested loop simulation over a grid of parameters, with results saved to a CSV file.
# 4. Nested loop simulation specifically for the new hexagonal grid pattern parameters,
#    now with user-defined thermal density and search arc values.
#
# Key Features:
# - Thermal Placement: Uses a hexagonal grid with a random component to place thermals.
# - Thermal Properties: Thermal updraft strength is also Poisson-distributed
#   (clamped 1-10 m/s), and the updraft radius is derived from this strength.
# - Dynamic Flight Path: The glider's path is an iterative, multi-segment path.
# - The Macready settings change based on altitude bands.
# - The glider flies to the nearest thermal in the arc, chooses to climb if
#   the thermal strength is >= the current Macready, otherwise it continues to glide.
# - The flight ends if the glider's altitude drops below 500m.
#
# --- Version 2: Debugging Final Glide Logic ---
# The logic for determining a "successful" flight has a bug. This version adds
# detailed print statements to the simulate_intercept_experiment_dynamic function to
# track altitude and final glide calculations, specifically in scenarios where
# the glider should fail but is incorrectly marked as successful.
# -----------------------------

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
from tqdm import tqdm
import csv
import pandas as pd
import sys

# --- Constants ---
KNOT_TO_MS = 0.514444
FT_TO_M = 0.3048
MS_TO_KMH = 3.6  # Conversion constant for meters per second to kilometers per hour

C_UPDRAFT_STRENGTH_DECREMENT = 5.9952e-7
FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS = 1200.0
FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS = FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS / 2
K_DOWNDRAFT_STRENGTH = 0.042194
EPSILON = 1e-9
MIN_SAFE_ALTITUDE = 500.0  # Minimum altitude for a safe landing

# --- LS10 Glider Polar Data ---
# Glider polar data points: airspeed vs sink rate.
pol_v = np.array([45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120])
pol_w = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.5, 6.0])

# --- Default Scenario Parameters ---
DEFAULT_Z_CBL = 2500.0
DEFAULT_MC_SNIFF_BAND1 = 4.0
DEFAULT_MC_SNIFF_BAND2 = 2.0
DEFAULT_LAMBDA_THERMALS_PER_SQ_KM = 0.02
DEFAULT_LAMBDA_STRENGTH = 2
DEFAULT_SEARCH_ARC_ANGLE_DEGREES = 30.0
RANDOM_END_POINT_DISTANCE = 100000.0


# --- Helper functions ---
def calculate_sniffing_radius(Wt_ms_ambient, MC_for_sniffing_ms, thermal_type="NORMAL"):
    """
    Calculates the sniffing radius based on ambient thermal strength and pilot's Macready setting.
    """
    if MC_for_sniffing_ms < 0:
        MC_for_sniffing_ms = 0.0

    C_thermal = 0.033 if thermal_type == "NORMAL" else 0.10
    Wt_knots_for_sniff = Wt_ms_ambient / KNOT_TO_MS
    MC_knots_for_sniff = MC_for_sniffing_ms / KNOT_TO_MS
    y_MC_knots_for_sniff = Wt_knots_for_sniff - MC_knots_for_sniff
    if y_MC_knots_for_sniff / C_thermal > 0:
        R_sniffing_feet = 100 * ((y_MC_knots_for_sniff / C_thermal) ** (1 / 3))
    else:
        R_sniffing_feet = 0
    D_sniffing_meters = 2 * (R_sniffing_feet * FT_TO_M)
    return D_sniffing_meters / 2


def distance_from_point_to_line_segment(point, line_start, line_end):
    """
    Calculates the distance from a point to a line segment.
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx, dy = x2 - x1, y2 - y1
    line_segment_length_sq = dx * dx + dy * dy
    if line_segment_length_sq == 0:
        closest_x, closest_y = x1, y1
        distance = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        return distance, (closest_x, closest_y)
    t = ((px - x1) * dx + (py - y1) * dy) / line_segment_length_sq
    if t < 0:
        closest_x, closest_y = x1, y1
    elif t > 1:
        closest_x, closest_y = x2, y2
    else:
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

    distance = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    return distance, (closest_x, closest_y)


def generate_hexagonal_grid_thermals(sim_area_side_meters, thermal_density_per_sq_km, lambda_strength):
    """
    Generates thermal locations and properties using a hexagonal grid with a random component.
    """
    updraft_thermals = []

    avg_area_per_thermal_sq_km = 1 / thermal_density_per_sq_km
    grid_spacing = math.sqrt(avg_area_per_thermal_sq_km) * 1000

    hex_height = grid_spacing
    hex_width = math.sqrt(3) / 2 * hex_height

    x_min, x_max = -sim_area_side_meters / 2, sim_area_side_meters / 2
    y_min, y_max = -sim_area_side_meters / 2, sim_area_side_meters / 2

    row = 0
    y = y_min
    while y <= y_max:
        col = 0
        x_start = x_min + (hex_width / 2 if row % 2 == 1 else 0)
        x = x_start
        while x <= x_max:
            probability_of_thermal = thermal_density_per_sq_km * (grid_spacing / 1000) ** 2

            if random.random() < probability_of_thermal:
                center_x = x + random.uniform(-hex_width / 4, hex_width / 4)
                center_y = y + random.uniform(-hex_height / 4, hex_height / 4)
                updraft_strength_magnitude = 0
                while updraft_strength_magnitude == 0:
                    updraft_strength_magnitude = np.random.poisson(lambda_strength)
                updraft_strength_magnitude = min(10, updraft_strength_magnitude)
                updraft_radius = (updraft_strength_magnitude / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)
                updraft_thermals.append({
                    'center': (center_x, center_y),
                    'updraft_radius': updraft_radius,
                    'updraft_strength': updraft_strength_magnitude
                })

            x += hex_width
            col += 1
        y += hex_height * 3 / 4
        row += 1

    return updraft_thermals


def get_glider_parameters(mc_setting_ms):
    """
    Calculates the optimum airspeed, sink rate, and glide ratio for a given
    Macready setting using the glider polar.
    """
    mc_setting_knots = mc_setting_ms / KNOT_TO_MS
    effective_sink_rate_knots = pol_w - mc_setting_knots
    positive_indices = np.where(effective_sink_rate_knots > 0)[0]

    if len(positive_indices) == 0:
        max_glide_ratio_index = len(pol_v) - 1
    else:
        relevant_pol_v = pol_v[positive_indices]
        relevant_effective_sink_rate = effective_sink_rate_knots[positive_indices]
        effective_glide_ratio = relevant_pol_v / relevant_effective_sink_rate
        max_glide_ratio_relative_index = np.argmax(effective_glide_ratio)
        max_glide_ratio_index = positive_indices[max_glide_ratio_relative_index]

    airspeed_knots = pol_v[max_glide_ratio_index]
    sink_rate_knots = pol_w[max_glide_ratio_index]

    airspeed_ms = airspeed_knots * KNOT_TO_MS
    sink_rate_ms = sink_rate_knots * FT_TO_M

    if sink_rate_ms > 0:
        glide_ratio = airspeed_ms / sink_rate_ms
    else:
        glide_ratio = float('inf')

    return airspeed_ms, sink_rate_ms, glide_ratio, airspeed_knots, sink_rate_knots


def calculate_bearing_delta(start_point, next_point, end_point):
    """
    Calculates the angle (delta) between the bearing to the next point and the bearing to the end point.
    """
    bearing_to_next = math.degrees(math.atan2(next_point[1] - start_point[1], next_point[0] - start_point[0]))
    bearing_to_end = math.degrees(math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0]))

    if bearing_to_next < 0: bearing_to_next += 360
    if bearing_to_end < 0: bearing_to_end += 360

    delta = bearing_to_next - bearing_to_end
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360

    return delta


def get_band_info(altitude, scenario_z_cbl, scenario_mc_band1, scenario_mc_band2):
    """
    Determines the current altitude band, Macready setting, and altitude to the next band boundary.
    """
    min_alt_band2 = (scenario_z_cbl / 3) * 2
    min_alt_band3 = scenario_z_cbl / 3

    if altitude > min_alt_band2:
        return "Band 1", scenario_mc_band1, altitude - min_alt_band2, min_alt_band2
    elif altitude > min_alt_band3:
        return "Band 2", scenario_mc_band2, altitude - min_alt_band3, min_alt_band3
    else:
        return "Band 3", 0.0, altitude - MIN_SAFE_ALTITUDE, MIN_SAFE_ALTITUDE


def simulate_intercept_experiment_dynamic(
        z_cbl_meters, lambda_thermals_per_sq_km, lambda_strength,
        mc_sniff_band1, mc_sniff_band2, end_point, search_arc_angle
):
    """
    Runs a single simulation of a glider's flight.
    """
    plot_padding = 20000.0
    max_coord = max(abs(end_point[0]), abs(end_point[1]))
    sim_area_side_meters = (max_coord + plot_padding) * 2

    # Use the hexagonal thermal generation function
    updraft_thermals_info = generate_hexagonal_grid_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    path_points = [(0, 0)]
    current_pos = (0, 0)
    current_altitude = z_cbl_meters
    total_height_climbed = 0.0
    total_climbing_time = 0.0
    total_gliding_time = 0.0
    total_distance_covered = 0.0
    path_details = []

    remaining_thermals = list(updraft_thermals_info)

    while math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]) > EPSILON:
        # --- DEBUG PRINT: Altitude Check ---
        print(f"DEBUG: Current Altitude: {current_altitude:.2f}m")
        if current_altitude <= MIN_SAFE_ALTITUDE:
            print(
                f"DEBUG: Altitude ({current_altitude:.2f}m) is below minimum safe altitude ({MIN_SAFE_ALTITUDE}m). Flight failed.")
            straight_line_dist_origin = math.hypot(current_pos[0], current_pos[1])
            return {'success': False, 'distance_to_land': straight_line_dist_origin, 'path': path_points,
                    'path_details': path_details, 'total_distance_covered': total_distance_covered}

        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        current_band, current_mc_sniff_ms, altitude_to_next_band, next_band_alt = get_band_info(current_altitude,
                                                                                                z_cbl_meters,
                                                                                                mc_sniff_band1,
                                                                                                mc_sniff_band2)
        airspeed_ms, sink_rate_ms, glide_ratio, airspeed_knots, sink_rate_knots = get_glider_parameters(
            current_mc_sniff_ms)

        direct_glide_dist = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio

        # --- DEBUG PRINT: Final Glide Calculation ---
        print(f"DEBUG: Current distance to end: {distance_to_end:.2f}m")
        print(f"DEBUG: Max possible glide distance from current altitude: {direct_glide_dist:.2f}m")

        if distance_to_end < direct_glide_dist:
            print("DEBUG: Condition for Final Glide met.")
            time_to_travel = distance_to_end / airspeed_ms
            altitude_drop = time_to_travel * sink_rate_ms
            current_altitude -= altitude_drop
            total_gliding_time += time_to_travel
            total_distance_covered += distance_to_end
            current_pos = end_point
            path_points.append(current_pos)

            delta = calculate_bearing_delta(path_start, current_pos, end_point)
            path_details.append({
                'dist': total_distance_covered,
                'alt': current_altitude,
                'delta': delta,
                'band': current_band,
                'mc_set': current_mc_sniff_ms,
                'updraft': "N/A",
                'speed_knots': airspeed_knots,
                'sink_rate_ms': sink_rate_ms,
                'action': "Final Glide"
            })

            straight_line_dist_origin = math.hypot(end_point[0], end_point[1])
            return {
                'success': True,
                'total_height_climbed': total_height_climbed,
                'total_climbing_time': total_climbing_time,
                'total_gliding_time': total_gliding_time,
                'total_time': total_climbing_time + total_gliding_time,
                'straight_line_distance': straight_line_dist_origin,
                'path': path_points,
                'path_details': path_details,
                'total_distance_covered': total_distance_covered,
            }

        glide_dist_to_band = altitude_to_next_band * glide_ratio

        segment_length = min(distance_to_end, glide_dist_to_band + EPSILON)
        nearest_thermal_in_arc = None
        min_dist_to_thermal = float('inf')

        bearing_to_end_radians = math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0])
        bearing_to_end_degrees = math.degrees(bearing_to_end_radians)
        if bearing_to_end_degrees < 0: bearing_to_end_degrees += 360
        arc_half_angle_degrees = search_arc_angle / 2

        sniffing_radius_meters_base = calculate_sniffing_radius(lambda_strength, current_mc_sniff_ms)

        for thermal in remaining_thermals:
            thermal_center = thermal['center']
            dist_to_thermal = math.hypot(thermal_center[0] - path_start[0], thermal_center[1] - path_start[1])

            if dist_to_thermal > direct_glide_dist + sniffing_radius_meters_base:
                continue

            thermal_bearing_from_start = math.degrees(
                math.atan2(thermal_center[1] - path_start[1], thermal_center[0] - path_start[0]))
            if thermal_bearing_from_start < 0: thermal_bearing_from_start += 360

            angle_from_line = thermal_bearing_from_start - bearing_to_end_degrees
            if angle_from_line > 180:
                angle_from_line -= 360
            elif angle_from_line <= -180:
                angle_from_line += 360

            is_in_arc = abs(angle_from_line) <= arc_half_angle_degrees

            if is_in_arc and dist_to_thermal < min_dist_to_thermal:
                min_dist_to_thermal = dist_to_thermal
                nearest_thermal_in_arc = thermal

        travel_distance_to_thermal = float('inf')
        if nearest_thermal_in_arc:
            travel_distance_to_thermal = min_dist_to_thermal

        travel_distance = min(segment_length, travel_distance_to_thermal)

        if travel_distance <= EPSILON:
            travel_distance = EPSILON

        next_pos = None
        is_thermal_intercept = abs(travel_distance - travel_distance_to_thermal) < EPSILON and nearest_thermal_in_arc

        if is_thermal_intercept:
            next_pos = nearest_thermal_in_arc['center']
        else:
            next_pos = (path_start[0] + travel_distance * math.cos(bearing_to_end_radians),
                        path_start[1] + travel_distance * math.sin(math.radians(bearing_to_end_degrees)))

        glide_time = travel_distance / airspeed_ms
        altitude_after_glide = current_altitude - glide_time * sink_rate_ms
        delta_after_glide = calculate_bearing_delta(path_start, next_pos, end_point)

        path_details.append({
            'dist': total_distance_covered,
            'alt': current_altitude,
            'delta': delta_after_glide,
            'band': current_band,
            'mc_set': current_mc_sniff_ms,
            'updraft': "N/A",
            'speed_knots': airspeed_knots,
            'sink_rate_ms': sink_rate_ms,
            'action': "Glide"
        })

        current_altitude = altitude_after_glide
        current_pos = next_pos
        path_points.append(current_pos)
        total_gliding_time += glide_time
        total_distance_covered += travel_distance

        if is_thermal_intercept:
            remaining_thermals.remove(nearest_thermal_in_arc)
            updraft_val = nearest_thermal_in_arc['updraft_strength']
            if updraft_val >= current_mc_sniff_ms:
                height_climbed = z_cbl_meters - current_altitude
                climbing_time = height_climbed / updraft_val
                total_height_climbed += height_climbed
                total_climbing_time += climbing_time
                current_altitude = z_cbl_meters

                path_details.append({
                    'dist': total_distance_covered,
                    'alt': current_altitude,
                    'delta': delta_after_glide,
                    'band': current_band,
                    'mc_set': current_mc_sniff_ms,
                    'updraft': updraft_val,
                    'speed_knots': "N/A",  # Glider is circling, not moving forward
                    'sink_rate_ms': -updraft_val,  # Negative sink is climb
                    'action': "Climb"
                })

    return {
        'success': True,
        'total_height_climbed': total_height_climbed,
        'total_climbing_time': total_climbing_time,
        'total_gliding_time': total_gliding_time,
        'total_time': total_climbing_time + total_gliding_time,
        'straight_line_distance': math.hypot(end_point[0], end_point[1]),
        'path': path_points,
        'path_details': path_details,
        'total_distance_covered': total_distance_covered,
    }


def run_single_simulation_with_plot():
    """
    Runs a single simulation and visualizes the results.
    """
    random_angle = random.uniform(0, 360)
    end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
    end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
    random_end_point = (end_point_x, end_point_y)

    print("Running single simulation...")
    result = simulate_intercept_experiment_dynamic(
        z_cbl_meters=DEFAULT_Z_CBL,
        lambda_thermals_per_sq_km=DEFAULT_LAMBDA_THERMALS_PER_SQ_KM,
        lambda_strength=DEFAULT_LAMBDA_STRENGTH,
        mc_sniff_band1=DEFAULT_MC_SNIFF_BAND1,
        mc_sniff_band2=DEFAULT_MC_SNIFF_BAND2,
        end_point=random_end_point,
        search_arc_angle=DEFAULT_SEARCH_ARC_ANGLE_DEGREES
    )

    updraft_thermals = generate_hexagonal_grid_thermals(
        max(abs(random_end_point[0]), abs(random_end_point[1])) * 2 + 40000,
        DEFAULT_LAMBDA_THERMALS_PER_SQ_KM,
        DEFAULT_LAMBDA_STRENGTH
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Glider Flight Path Simulation")
    ax.set_xlabel("X-coordinate (m)")
    ax.set_ylabel("Y-coordinate (m)")

    # Plot thermals
    for thermal in updraft_thermals:
        center_x, center_y = thermal['center']
        updraft_radius = thermal['updraft_radius']
        updraft_circle = patches.Circle((center_x, center_y), updraft_radius, color='orange', alpha=0.5,
                                        label='Updraft Thermal')
        ax.add_patch(updraft_circle)

    # Plot flight path
    path_x = [p[0] for p in result['path']]
    path_y = [p[1] for p in result['path']]
    ax.plot(path_x, path_y, 'b-', label='Glider Path')
    ax.plot(path_x[0], path_y[0], 'go', label='Start')
    ax.plot(path_x[-1], path_y[-1], 'ro', label='End')
    ax.legend()
    plt.show()

    print("\n--- Simulation Results ---")
    if result['success']:
        print("Flight was successful!")
        print(f"Total time: {result['total_time']:.2f} s")
        print(
            f"Average Speed Made Good (VMG): {result['straight_line_distance'] / result['total_time'] * MS_TO_KMH:.2f} km/h")
        print(f"Total distance covered: {result['total_distance_covered']:.2f} m")
    else:
        print("Flight was unsuccessful.")
        print(f"Failed at distance: {result['distance_to_land']:.2f} m from origin.")


def run_monte_carlo_simulation():
    """
    Runs a Monte Carlo simulation with default parameters and prints a summary.
    """
    num_simulations = 1000
    successful_flights = 0
    total_time_successful = 0.0
    total_vmg_successful = 0.0
    total_failed_distance = 0.0
    failed_count = 0

    print("Starting Monte Carlo simulation...")
    for _ in tqdm(range(num_simulations), desc="Running simulations"):
        random_angle = random.uniform(0, 360)
        end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
        end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
        random_end_point = (end_point_x, end_point_y)

        result = simulate_intercept_experiment_dynamic(
            z_cbl_meters=DEFAULT_Z_CBL,
            lambda_thermals_per_sq_km=DEFAULT_LAMBDA_THERMALS_PER_SQ_KM,
            lambda_strength=DEFAULT_LAMBDA_STRENGTH,
            mc_sniff_band1=DEFAULT_MC_SNIFF_BAND1,
            mc_sniff_band2=DEFAULT_MC_SNIFF_BAND2,
            end_point=random_end_point,
            search_arc_angle=DEFAULT_SEARCH_ARC_ANGLE_DEGREES
        )

        if result['success']:
            successful_flights += 1
            total_time_successful += result['total_time']
            total_vmg_successful += (result['straight_line_distance'] / result['total_time'])
        else:
            failed_count += 1
            total_failed_distance += result['distance_to_land']

    print("\n--- Monte Carlo Simulation Summary ---")
    probability = successful_flights / num_simulations
    print(f"Total simulations run: {num_simulations}")
    print(f"Successful flights: {successful_flights} ({probability:.2%})")
    print(f"Unsuccessful flights: {failed_count} ({(1 - probability):.2%})")

    if successful_flights > 0:
        avg_time = total_time_successful / successful_flights
        avg_vmg_ms = total_vmg_successful / successful_flights
        print(f"Average total time for successful flights: {avg_time:.2f} s")
        print(f"Average VMG for successful flights: {avg_vmg_ms * MS_TO_KMH:.2f} km/h")

    if failed_count > 0:
        avg_failed_dist = total_failed_distance / failed_count
        print(f"Average distance from origin for failed flights: {avg_failed_dist:.2f} m")


def run_nested_loop_simulation():
    """
    Runs a nested loop simulation over a grid of parameters and saves the results to a CSV.
    This version now takes user input for thermal density and search arc values.
    """
    print("Starting Nested Loop Simulation with user-defined parameters...")

    default_densities = [0.000625, 0.005, 0.02]
    user_density_input = input(
        f"Enter thermal density values (per km^2) separated by commas (e.g., {','.join(map(str, default_densities))}), or press Enter for the defaults: ")

    if user_density_input.strip() == "":
        lambda_thermals_values = default_densities
        print("Using default thermal densities.")
    else:
        try:
            lambda_thermals_values = [float(x.strip()) for x in user_density_input.split(',')]
            print(f"Using user-defined thermal densities: {lambda_thermals_values}")
        except ValueError:
            print("Invalid input. Please enter a comma-separated list of numbers. Using default values instead.")
            lambda_thermals_values = default_densities

    default_arcs = [30, 45, 60]
    user_arc_input = input(
        f"Enter search arc angles (degrees) separated by commas (e.g., {','.join(map(str, default_arcs))}), or press Enter for the defaults: ")

    if user_arc_input.strip() == "":
        search_arc_values = default_arcs
        print("Using default search arc angles.")
    else:
        try:
            search_arc_values = [float(x.strip()) for x in user_arc_input.split(',')]
            print(f"Using user-defined search arc angles: {search_arc_values}")
        except ValueError:
            print("Invalid input. Please enter a comma-separated list of numbers. Using default values instead.")
            search_arc_values = default_arcs

    z_cbl_values = [2500]
    lambda_strength_values = [0.6638]
    mc_band1_values = list(np.arange(1, 6, 1))
    mc_band2_values = list(np.arange(1, 6, 1))

    num_simulations_per_scenario = 1000
    all_scenario_results = []

    for z_cbl in z_cbl_values:
        for lambda_thermals in lambda_thermals_values:
            for lambda_strength in lambda_strength_values:
                for mc_band1 in mc_band1_values:
                    for mc_band2 in mc_band2_values:
                        for search_arc in search_arc_values:
                            print(
                                f"\nRunning scenario: Z_CBL={z_cbl}m, λ_thermals={lambda_thermals}, λ_strength={lambda_strength}, MC1={mc_band1}m/s, MC2={mc_band2}m/s, Arc={search_arc}deg")

                            successful_flights = 0
                            successful_metrics = []
                            failed_distances = []

                            tqdm_desc = f"Trials for this scenario ({num_simulations_per_scenario})"
                            for _ in tqdm(range(num_simulations_per_scenario), desc=tqdm_desc):
                                random_angle = random.uniform(0, 360)
                                end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
                                end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
                                random_end_point = (end_point_x, end_point_y)

                                result = simulate_intercept_experiment_dynamic(
                                    z_cbl_meters=z_cbl,
                                    lambda_thermals_per_sq_km=lambda_thermals,
                                    lambda_strength=lambda_strength,
                                    mc_sniff_band1=mc_band1,
                                    mc_sniff_band2=mc_band2,
                                    end_point=random_end_point,
                                    search_arc_angle=search_arc
                                )

                                if result['success']:
                                    successful_flights += 1
                                    successful_metrics.append(result)
                                else:
                                    failed_distances.append(result['distance_to_land'])

                            probability = successful_flights / num_simulations_per_scenario

                            avg_Ht = sum(m['total_height_climbed'] for m in successful_metrics) / len(
                                successful_metrics) if successful_flights > 0 else 0
                            avg_Tc = sum(m['total_climbing_time'] for m in successful_metrics) / len(
                                successful_metrics) if successful_flights > 0 else 0
                            avg_Tg = sum(m['total_gliding_time'] for m in successful_metrics) / len(
                                successful_metrics) if successful_flights > 0 else 0
                            avg_T = avg_Tc + avg_Tg
                            avg_Wc = avg_Ht / avg_Tc if avg_Tc > 0 else 0
                            avg_Vmg_ms = RANDOM_END_POINT_DISTANCE / avg_T if avg_T > 0 else 0
                            avg_Vmg_kmh = avg_Vmg_ms * MS_TO_KMH
                            avg_failed_dist = sum(failed_distances) / len(failed_distances) if failed_distances else 0

                            scenario_results = {
                                'Z_CBL (m)': z_cbl,
                                'Thermal Density (per km^2)': lambda_thermals,
                                'Thermal Strength Lambda': lambda_strength,
                                'MC_SNIFF_BAND1 (m/s)': mc_band1,
                                'MC_SNIFF_BAND2 (m/s)': mc_band2,
                                'Search Arc Angle (deg)': search_arc,
                                'Successful Flights': successful_flights,
                                'Probability': probability,
                                'Avg Total Height Climbed (m)': avg_Ht,
                                'Avg Total Climbing Time (s)': avg_Tc,
                                'Avg Total Gliding Time (s)': avg_Tg,
                                'Avg Total Time (s)': avg_T,
                                'Avg Rate of Climb (m/s)': avg_Wc,
                                'Avg Speed Made Good (km/h)': avg_Vmg_kmh,
                                'Avg Failed Distance (m)': avg_failed_dist
                            }
                            all_scenario_results.append(scenario_results)

    results_df = pd.DataFrame(all_scenario_results)
    output_filename = "thermal_sim_results_grid_search_default.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Results successfully exported to '{output_filename}'")


def run_nested_loop_hexagonal_simulation():
    """
    Runs a nested loop simulation with the specific hexagonal thermal parameters and saves the results to a CSV.
    The thermal densities and search arc values are now user-defined inputs.
    """
    print("Starting Nested Loop Simulation for Hexagonal Pattern...")

    default_densities = [0.000625, 0.005, 0.02, 0.01]
    user_density_input = input(
        f"Enter thermal density values (per km^2) separated by commas (e.g., {','.join(map(str, default_densities))}), or press Enter for the defaults: ")

    if user_density_input.strip() == "":
        lambda_thermals_values = default_densities
        print("Using default thermal densities.")
    else:
        try:
            lambda_thermals_values = [float(x.strip()) for x in user_density_input.split(',')]
            print(f"Using user-defined thermal densities: {lambda_thermals_values}")
        except ValueError:
            print("Invalid input. Please enter a comma-separated list of numbers. Using default values instead.")
            lambda_thermals_values = default_densities

    default_arcs = [30, 45, 60]
    user_arc_input = input(
        f"Enter search arc angles (degrees) separated by commas (e.g., {','.join(map(str, default_arcs))}), or press Enter for the defaults: ")

    if user_arc_input.strip() == "":
        search_arc_values = default_arcs
        print("Using default search arc angles.")
    else:
        try:
            search_arc_values = [float(x.strip()) for x in user_arc_input.split(',')]
            print(f"Using user-defined search arc angles: {search_arc_values}")
        except ValueError:
            print("Invalid input. Please enter a comma-separated list of numbers. Using default values instead.")
            search_arc_values = default_arcs

    z_cbl_values = [2500]
    lambda_strength_values = [0.6638]
    mc_band1_values = list(np.arange(1, 6, 1))
    mc_band2_values = list(np.arange(1, 6, 1))

    num_simulations_per_scenario = 1000
    all_scenario_results = []

    for z_cbl in z_cbl_values:
        for lambda_thermals in lambda_thermals_values:
            for lambda_strength in lambda_strength_values:
                for mc_band1 in mc_band1_values:
                    for mc_band2 in mc_band2_values:
                        for search_arc in search_arc_values:
                            print(
                                f"\nRunning scenario: Z_CBL={z_cbl}m, λ_thermals={lambda_thermals}, λ_strength={lambda_strength}, MC1={mc_band1}m/s, MC2={mc_band2}m/s, Arc={search_arc}deg")

                            successful_flights = 0
                            successful_metrics = []
                            failed_distances = []

                            tqdm_desc = f"Trials for this scenario ({num_simulations_per_scenario})"
                            for _ in tqdm(range(num_simulations_per_scenario), desc=tqdm_desc):
                                random_angle = random.uniform(0, 360)
                                end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
                                end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
                                random_end_point = (end_point_x, end_point_y)

                                result = simulate_intercept_experiment_dynamic(
                                    z_cbl_meters=z_cbl,
                                    lambda_thermals_per_sq_km=lambda_thermals,
                                    lambda_strength=lambda_strength,
                                    mc_sniff_band1=mc_band1,
                                    mc_sniff_band2=mc_band2,
                                    end_point=random_end_point,
                                    search_arc_angle=search_arc
                                )

                                if result['success']:
                                    successful_flights += 1
                                    successful_metrics.append(result)
                                else:
                                    failed_distances.append(result['distance_to_land'])

                            probability = successful_flights / num_simulations_per_scenario

                            avg_Ht = sum(m['total_height_climbed'] for m in successful_metrics) / len(
                                successful_metrics) if successful_flights > 0 else 0
                            avg_Tc = sum(m['total_climbing_time'] for m in successful_metrics) / len(
                                successful_metrics) if successful_flights > 0 else 0
                            avg_Tg = sum(m['total_gliding_time'] for m in successful_metrics) / len(
                                successful_metrics) if successful_flights > 0 else 0
                            avg_T = avg_Tc + avg_Tg
                            avg_Wc = avg_Ht / avg_Tc if avg_Tc > 0 else 0
                            avg_Vmg_ms = RANDOM_END_POINT_DISTANCE / avg_T if avg_T > 0 else 0
                            avg_Vmg_kmh = avg_Vmg_ms * MS_TO_KMH
                            avg_failed_dist = sum(failed_distances) / len(failed_distances) if failed_distances else 0

                            scenario_results = {
                                'Z_CBL (m)': z_cbl,
                                'Thermal Density (per km^2)': lambda_thermals,
                                'Thermal Strength Lambda': lambda_strength,
                                'MC_SNIFF_BAND1 (m/s)': mc_band1,
                                'MC_SNIFF_BAND2 (m/s)': mc_band2,
                                'Search Arc Angle (deg)': search_arc,
                                'Successful Flights': successful_flights,
                                'Probability': probability,
                                'Avg Total Height Climbed (m)': avg_Ht,
                                'Avg Total Climbing Time (s)': avg_Tc,
                                'Avg Total Gliding Time (s)': avg_Tg,
                                'Avg Total Time (s)': avg_T,
                                'Avg Rate of Climb (m/s)': avg_Wc,
                                'Avg Speed Made Good (km/h)': avg_Vmg_kmh,
                                'Avg Failed Distance (m)': avg_failed_dist
                            }
                            all_scenario_results.append(scenario_results)

    results_df = pd.DataFrame(all_scenario_results)
    output_filename = "thermal_sim_results_hexagonal_pattern.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Results successfully exported to '{output_filename}'")


if __name__ == '__main__':
    print("Select a simulation mode:")
    print("1: Single simulation with plot and detailed printout")
    print("2: Monte Carlo simulation with default parameters")
    print("3: Nested loop simulation with user-defined parameters")
    print("4: Nested loop simulation with specific hexagonal pattern and user-defined parameters")

    choice = input("Enter your choice (1, 2, 3, or 4): ")

    if choice == '1':
        run_single_simulation_with_plot()
    elif choice == '2':
        run_monte_carlo_simulation()
    elif choice == '3':
        run_nested_loop_simulation()
    elif choice == '4':
        run_nested_loop_hexagonal_simulation()
    else:
        print("Invalid choice. Please run the script again and select 1, 2, 3, or 4.")
