# --- SCRIPT CHARACTERISTICS ---
# This is a comprehensive Monte Carlo simulation script for a glider's thermal interception.
# The simulation environment now models a 2D plane with thermals arranged in a semi-random
# hexagonal pattern, with each thermal consisting of a core updraft region and a downdraft ring.
#
# This version includes extra print statements and error handling to help debug issues
# that may be preventing the script from running correctly.
#
# Key Features:
# - Thermal Placement: Uses a new function to place thermals in a hexagonal grid, with
#   a random offset applied to each thermal's position to add a realistic random component.
#   A Bernoulli trial now determines if a thermal exists at all at a given grid point.
# - Thermal Properties: Updraft strength is also Poisson-distributed (clamped 1-10 m/s),
#   and the updraft radius is derived from this strength.
# - Downdraft Modeling: A fixed-diameter downdraft ring encircles each updraft.
# - Dynamic Flight Path: The glider's path is an iterative, multi-segment path.
# - The Macready settings change based on altitude bands.
# - The glider flies to the nearest thermal in the arc, chooses to climb if
#   the thermal strength is >= the current Macready, otherwise it continues to glide.
# - The flight ends if the glider's altitude drops below 500m.
# - The script offers four modes of operation:
#   1. Single simulation with a visual plot and detailed printout (Hexagonal).
#   2. Monte Carlo simulation with user-defined parameters (Poisson).
#   3. Nested loop simulation and save to CSV (Poisson).
#   4. Nested loop simulation and save to CSV (Hexagonal).
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
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

# --- Constants ---
KNOT_TO_MS = 0.514444
FT_TO_M = 0.3048
MS_TO_KMH = 3.6  # Conversion constant for meters per second to kilometers per hour

# --- NEW: Corrected formula constant based on user's input ---
# The original formula was y = 0.033 * (x/100)^3, where y is in knots and x is in feet.
# Converting to metric (y in m/s, x in meters):
# y_ms = (0.033 * KNOT_TO_MS) * ((x_m / FT_TO_M) / 100)^3
# y_ms = (0.033 * 0.514444) * (x_m / (100 * 0.3048))^3
# y_ms = 0.016976652 * (x_m / 30.48)^3
# y_ms = 0.016976652 * (x_m^3 / 30.48^3)
# y_ms = (0.016976652 / 28285.49) * x_m^3
# y_ms = 5.9899e-7 * x_m^3
# The constant for the metric formula is therefore 5.9899e-7
C_UPDRAFT_STRENGTH_DECREMENT = 5.9899e-7

FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS = 1200.0
FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS = FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS / 2
K_DOWNDRAFT_STRENGTH = 0.042194
EPSILON = 1e-9
MIN_SAFE_ALTITUDE = 500.0  # Minimum altitude for a safe landing

# --- LS10 Glider Polar Data (from thermal_sim_poisson_segmented.py) ---
# Glider polar data points: airspeed vs sink rate.
pol_v = np.array([45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120])
pol_w = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.5, 6.0])

# --- Default Scenario Parameters ---
# These are the default values for the simulation. You can change these to match
# your specific "real world" data.
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


def generate_poisson_updraft_thermals(sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength):
    """
    Generates thermal locations and properties using a Poisson distribution.
    This is the original function.
    """
    sim_area_sq_km = (sim_area_side_meters / 1000) ** 2
    expected_num_thermals = lambda_thermals_per_sq_km * sim_area_sq_km
    num_thermals = np.random.poisson(expected_num_thermals)
    updraft_thermals = []
    for _ in range(num_thermals):
        center_x = random.uniform(-sim_area_side_meters / 2, sim_area_side_meters / 2)
        center_y = random.uniform(-sim_area_side_meters / 2, sim_area_side_meters / 2)
        updraft_strength_magnitude = 0
        while updraft_strength_magnitude == 0:
            updraft_strength_magnitude = np.random.poisson(lambda_strength)
        updraft_strength_magnitude = min(10, updraft_strength_magnitude)

        # --- NEW: Updated updraft radius calculation based on the user's formula ---
        # The radius is the distance x where the decrement y equals the thermal strength.
        # W_updraft = (C_UPDRAFT_STRENGTH_DECREMENT) * x_m^3
        # x_m = (W_updraft / C_UPDRAFT_STRENGTH_DECREMENT)^(1/3)
        updraft_radius = (updraft_strength_magnitude / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)

        updraft_thermals.append({
            'center': (center_x, center_y),
            'updraft_radius': updraft_radius,
            'updraft_strength': updraft_strength_magnitude
        })
    return updraft_thermals


def generate_hexagonal_thermals(sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength, z_cbl_meters):
    """
    Generates thermal locations in a hexagonal pattern with a random offset.
    This is the new function to model more realistic thermal fields.

    The dimensions of the hexagonal grid are now based on the Z_CBL (Convective Boundary Layer)
    height, and the jitter is determined by a Bernoulli trial.
    """
    updraft_thermals = []

    # Calculate hexagonal spacing based on Z_CBL
    hex_spacing = 1.5 * z_cbl_meters

    # Define a jitter magnitude as a percentage of the hexagonal spacing
    jitter_magnitude = hex_spacing * 0.1

    # Define grid boundaries
    x_min = -sim_area_side_meters / 2
    x_max = sim_area_side_meters / 2
    y_min = -sim_area_side_meters / 2
    y_max = sim_area_side_meters / 2

    # Calculate probability of a thermal existing at each grid point
    # Area of a single hex cell in km^2
    cell_area_sq_km = (hex_spacing * hex_spacing * math.sqrt(3) / 2) / 1e6
    probability = lambda_thermals_per_sq_km * cell_area_sq_km

    row_count = 0
    y = y_min
    while y < y_max:
        x = x_min
        if row_count % 2 == 1:
            x += hex_spacing / 2

        while x < x_max:
            # New: Use a Bernoulli trial to decide if a thermal should be placed
            if random.random() < probability:
                # Apply a random offset to the ideal grid point using a Bernoulli trial for the sign
                offset_x = jitter_magnitude * random.choice([-1, 1])
                offset_y = jitter_magnitude * random.choice([-1, 1])
                center_x = x + offset_x
                center_y = y + offset_y

                # Ensure the thermal is within the simulation boundaries
                if x_min < center_x < x_max and y_min < center_y < y_max:
                    updraft_strength_magnitude = 0
                    while updraft_strength_magnitude == 0:
                        updraft_strength_magnitude = np.random.poisson(lambda_strength)
                    updraft_strength_magnitude = min(10, updraft_strength_magnitude)

                    # --- NEW: Updated updraft radius calculation based on the user's formula ---
                    updraft_radius = (updraft_strength_magnitude / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)

                    updraft_thermals.append({
                        'center': (center_x, center_y),
                        'updraft_radius': updraft_radius,
                        'updraft_strength': updraft_strength_magnitude
                    })

            x += hex_spacing

        y += hex_spacing * math.sqrt(3) / 2
        row_count += 1

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

    # Fix: Corrected typo. This line should use KNOT_TO_MS, not an undefined KNOT_TO_M.
    sink_rate_ms = sink_rate_knots * KNOT_TO_MS

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
        mc_sniff_band1, mc_sniff_band2, end_point, search_arc_angle, plot_simulation=False, thermal_model='poisson'
):
    """
    Runs a single simulation of a glider's flight. If plot_simulation is True,
    it returns the plot object for later display. The thermal_model parameter
    allows the user to choose between the 'poisson' and 'hexagonal' models.
    """
    plot_padding = 20000.0
    max_coord = max(abs(end_point[0]), abs(end_point[1]))
    sim_area_side_meters = (max_coord + plot_padding) * 2

    if thermal_model == 'poisson':
        updraft_thermals_info = generate_poisson_updraft_thermals(
            sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
        )
        model_name = "Poisson"
    elif thermal_model == 'hexagonal':
        updraft_thermals_info = generate_hexagonal_thermals(
            sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength, z_cbl_meters
        )
        model_name = "Hexagonal"
    else:
        raise ValueError("Invalid thermal_model specified. Choose 'poisson' or 'hexagonal'.")

    fig, ax = (None, None)
    if plot_simulation:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"Glider Flight Path Simulation ({model_name} Model)\n"
                     f"Z_CBL={z_cbl_meters}m, λ_thermals={lambda_thermals}, λ_strength={lambda_strength}\n"
                     f"MC1={mc_sniff_band1}m/s, MC2={mc_sniff_band2}m/s, Arc={search_arc_angle}deg")
        ax.set_xlabel("X-coordinate (m)")
        ax.set_ylabel("Y-coordinate (m)")
        ax.set_xlim(-sim_area_side_meters / 2, sim_area_side_meters / 2)
        ax.set_ylim(-sim_area_side_meters / 2, sim_area_side_meters / 2)
        ax.grid(True)
        ax.plot(0, 0, 'go', markersize=10, label='Start')
        ax.plot(end_point[0], end_point[1], 'rx', markersize=10, label='End Point')

        # Plot all thermals with red updraft core and green downdraft annulus
        first_updraft = True
        first_downdraft = True
        for thermal in updraft_thermals_info:
            # Plot downdraft annulus (green)
            downdraft_circle = plt.Circle(thermal['center'], FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS,
                                          color='green', alpha=0.1, fill=False, linewidth=2.0,
                                          label='Downdraft' if first_downdraft else "")
            ax.add_patch(downdraft_circle)
            first_downdraft = False

            # Plot updraft core (red)
            updraft_circle = plt.Circle(thermal['center'], thermal['updraft_radius'],
                                        color='red', alpha=0.5,
                                        label='Updraft' if first_updraft else "")
            ax.add_patch(updraft_circle)
            first_updraft = False

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
        if current_altitude <= MIN_SAFE_ALTITUDE:
            if plot_simulation:
                ax.plot([p[0] for p in path_points], [p[1] for p in path_points], 'r-')
            straight_line_dist_origin = math.hypot(current_pos[0], current_pos[1])
            return {'success': False, 'distance_to_land': straight_line_dist_origin, 'path': path_points,
                    'path_details': path_details, 'total_distance_covered': total_distance_covered, 'plot': fig}

        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        current_band, current_mc_sniff_ms, altitude_to_next_band, next_band_alt = get_band_info(current_altitude,
                                                                                                z_cbl_meters,
                                                                                                mc_sniff_band1,
                                                                                                mc_sniff_band2)
        airspeed_ms, sink_rate_ms, glide_ratio, airspeed_knots, sink_rate_knots = get_glider_parameters(
            current_mc_sniff_ms)

        direct_glide_dist = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio

        if distance_to_end < direct_glide_dist:
            time_to_travel = distance_to_end / airspeed_ms
            altitude_drop = time_to_travel * sink_rate_ms
            current_altitude -= altitude_drop
            total_gliding_time += time_to_travel
            total_distance_covered += distance_to_end
            current_pos = end_point
            path_points.append(current_pos)

            if plot_simulation:
                ax.plot([p[0] for p in path_points], [p[1] for p in path_points], 'g-')

            # Record final segment details
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
                'plot': fig
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

        # MODIFIED: Log the glide action at the start of the segment
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

        # Update state after glide
        current_altitude = altitude_after_glide
        current_pos = next_pos
        path_points.append(current_pos)
        total_gliding_time += glide_time
        total_distance_covered += travel_distance

        if is_thermal_intercept:
            remaining_thermals.remove(nearest_thermal_in_arc)
            updraft_val = nearest_thermal_in_arc['updraft_strength']
            if updraft_val >= current_mc_sniff_ms:
                # MODIFIED: Log the climb action after the glide
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

                if plot_simulation:
                    ax.plot(current_pos[0], current_pos[1], 'm^', markersize=7, label='Climb')

    if plot_simulation:
        ax.plot([p[0] for p in path_points], [p[1] for p in path_points], 'g-')
        ax.legend()

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
        'plot': fig
    }


def print_detailed_single_flight_results(result, initial_bearing, params):
    """
    Prints a detailed table of a single flight simulation result.
    """
    print("\n--- Generating Single Plot with Dynamic Path Simulation ---")
    print("--- Single Flight Simulation Printout ---")
    print(f"Initial Bearing (Origin to End): {initial_bearing:.2f} deg")
    print(f"Macready Settings: Band 1={params['mc_band1']:.1f}m/s, Band 2={params['mc_band2']:.1f}m/s, Band 3=0.0m/s")
    print("-" * 120)
    print(
        f"{'Dist (km)':<12} | {'Alt (m)':<10}| {'Delta (deg)':<12}| {'Band':<12}| {'MC Set (m/s)':<15}| {'Updraft (m/s)':<17}| {'Speed (knots)':<17}| {'Sink Rate (m/s)':<15}| {'Action':<12}")
    print("-" * 120)

    for detail in result['path_details']:
        dist_km = detail['dist'] / 1000.0
        updraft_val = f"{detail['updraft']:.1f}" if isinstance(detail['updraft'], (int, float)) else detail['updraft']

        speed_knots_str = f"{detail['speed_knots']:.1f}" if isinstance(detail['speed_knots'], (int, float)) else detail[
            'speed_knots']
        sink_rate_ms_str = f"{detail['sink_rate_ms']:.2f}" if isinstance(detail['sink_rate_ms'], (int, float)) else \
        detail['sink_rate_ms']

        print(
            f"{dist_km:<12.3f} | {detail['alt']:<10.0f}| {detail['delta']:<12.2f}| {detail['band']:<12}| {detail['mc_set']:<15.1f}| {updraft_val:<17}| {speed_knots_str:<17}| {sink_rate_ms_str:<15}| {detail['action']:<12}")

    print("-" * 120)

    if result['success']:
        print(f"\n--- Simulation Result: SUCCESS. Glider reached destination. ---")
        print(f"Total distance flown: {result['total_distance_covered'] / 1000.0:.3f} km.")
    else:
        print(f"\n--- Simulation Result: FAILURE. Glider landed short. ---")
        print(f"Distance to landing point at failure: {result['distance_to_land']:.2f} m.")
        print(f"Total distance flown: {result['total_distance_covered'] / 1000.0:.3f} km.")


def run_monte_carlo_with_user_input():
    """
    Runs a Monte Carlo simulation with user-defined parameters and prints/plots the results.
    """
    print("\n--- Running Monte Carlo Simulation with User-Defined Parameters ---")
    num_simulations = 1000  # This is fixed for this mode.

    z_cbl = float(input(f"Enter Z_CBL (m) [default {DEFAULT_Z_CBL}]: ") or DEFAULT_Z_CBL)
    mc_band1 = float(input(f"Enter MC Band 1 (m/s) [default {DEFAULT_MC_SNIFF_BAND1}]: ") or DEFAULT_MC_SNIFF_BAND1)
    mc_band2 = float(input(f"Enter MC Band 2 (m/s) [default {DEFAULT_MC_SNIFF_BAND2}]: ") or DEFAULT_MC_SNIFF_BAND2)
    lambda_thermals = float(input(
        f"Enter Thermal Density (per km^2) [default {DEFAULT_LAMBDA_THERMALS_PER_SQ_KM}]: ") or DEFAULT_LAMBDA_THERMALS_PER_SQ_KM)
    lambda_strength = int(
        input(f"Enter Thermal Strength Lambda [default {DEFAULT_LAMBDA_STRENGTH}]: ") or DEFAULT_LAMBDA_STRENGTH)
    search_arc = float(input(
        f"Enter Search Arc Angle (deg) [default {DEFAULT_SEARCH_ARC_ANGLE_DEGREES}]: ") or DEFAULT_SEARCH_ARC_ANGLE_DEGREES)

    print(f"\nRunning with: Z_CBL={z_cbl}m, λ_thermals={lambda_thermals}, λ_strength={lambda_strength}\n"
          f"MC Band 1: {mc_band1}m/s, MC Band 2: {mc_band2}m/s, Search Arc: {search_arc}deg")

    successful_flights = 0
    successful_metrics = []
    failed_distances = []
    plot_for_first_success = None

    tqdm_desc = f"Trials for this scenario ({num_simulations})"
    for _ in tqdm(range(num_simulations), desc=tqdm_desc):
        random_angle = random.uniform(0, 360)
        end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
        end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
        random_end_point = (end_point_x, end_point_y)

        # Only create a plot for the first successful simulation
        plot_this_run = False
        if not plot_for_first_success:
            plot_this_run = True

        result = simulate_intercept_experiment_dynamic(
            z_cbl_meters=z_cbl,
            lambda_thermals_per_sq_km=lambda_thermals,
            lambda_strength=lambda_strength,
            mc_sniff_band1=mc_band1,
            mc_sniff_band2=mc_band2,
            end_point=random_end_point,
            search_arc_angle=search_arc,
            plot_simulation=plot_this_run
        )

        if result['success']:
            successful_flights += 1
            successful_metrics.append(result)
            if not plot_for_first_success:
                plot_for_first_success = result['plot']
        else:
            failed_distances.append(result['distance_to_land'])

    probability = successful_flights / num_simulations

    avg_Ht = sum(m['total_height_climbed'] for m in successful_metrics) / len(
        successful_metrics) if successful_metrics else 0
    avg_Tc = sum(m['total_climbing_time'] for m in successful_metrics) / len(
        successful_metrics) if successful_metrics else 0
    avg_Tg = sum(m['total_gliding_time'] for m in successful_metrics) / len(
        successful_metrics) if successful_metrics else 0
    avg_T = avg_Tc + avg_Tg
    avg_Wc = avg_Ht / avg_Tc if avg_Tc > 0 else 0
    avg_Vmg_ms = RANDOM_END_POINT_DISTANCE / avg_T if avg_T > 0 else 0
    avg_Vmg_kmh = avg_Vmg_ms * MS_TO_KMH
    avg_failed_dist = sum(failed_distances) / len(failed_distances) if failed_distances else 0

    print("\n--- Simulation Results ---")
    print(f"Total simulations: {num_simulations}")
    print(f"Successful flights: {successful_flights} (Probability: {probability:.2f})")
    print(f"Avg Total Height Climbed (m): {avg_Ht:.2f}")
    print(f"Avg Total Climbing Time (s): {avg_Tc:.2f}")
    print(f"Avg Total Gliding Time (s): {avg_Tg:.2f}")
    print(f"Avg Total Time (s): {avg_T:.2f}")
    print(f"Avg Rate of Climb (m/s): {avg_Wc:.2f}")
    print(f"Avg Speed Made Good (km/h): {avg_Vmg_kmh:.2f}")
    print(f"Avg Failed Distance (m): {avg_failed_dist:.2f}")

    if plot_for_first_success:
        print("\nDisplaying plot for the first successful flight...")
        plt.show()


def run_nested_loop_simulation_and_save_to_csv(thermal_model):
    """
    Performs a nested loop simulation over a specific parameter set and saves the results to a CSV file.
    The thermal_model parameter allows the user to choose between 'poisson' and 'hexagonal'.
    """
    # Define the fixed parameters as per the user's request
    z_cbl = 2500

    # Define the parameters to iterate over
    lambda_thermals_values = [0.01] + list(np.arange(0.02, 0.11, 0.02))
    lambda_strength_values = list(np.arange(1, 6, 1))
    mc_band1_values = list(np.arange(1, 6, 1))
    mc_band2_values = list(np.arange(1, 6, 1))
    search_arc_values = [30]

    num_simulations_per_scenario = 1000
    all_scenario_results = []

    print(f"--- Starting Nested Loop Simulation ({thermal_model} model) ---")

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
                                search_arc_angle=search_arc,
                                thermal_model=thermal_model
                            )

                            if result['success']:
                                successful_flights += 1
                                successful_metrics.append(result)
                            else:
                                failed_distances.append(result['distance_to_land'])

                        probability = successful_flights / num_simulations_per_scenario

                        avg_Ht = sum(m['total_height_climbed'] for m in successful_metrics) / len(
                            successful_metrics) if successful_metrics else 0
                        avg_Tc = sum(m['total_climbing_time'] for m in successful_metrics) / len(
                            successful_metrics) if successful_metrics else 0
                        avg_Tg = sum(m['total_gliding_time'] for m in successful_metrics) / len(
                            successful_metrics) if successful_metrics else 0
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

    print("\n--- Simulation Complete. Generating CSV file. ---")

    results_df = pd.DataFrame(all_scenario_results)
    output_filename = "nested_loop_simulation_results.csv"
    results_df.to_csv(output_filename, index=False)

    print(f"Results successfully exported to '{output_filename}'")


if __name__ == '__main__':
    print("Select a simulation mode:")
    print("1: Single simulation with plot and detailed printout (Hexagonal)")
    print("2: Monte Carlo simulation with user-defined parameters (Poisson)")
    print("3: Nested loop simulation and save to CSV (Poisson)")
    print("4: Nested loop simulation and save to CSV (Hexagonal)")

    try:
        choice = int(input("Enter your choice (1, 2, 3, or 4): "))
    except ValueError:
        print("Invalid input. Please enter 1, 2, 3, or 4.")
        sys.exit()

    try:
        if choice == 1:
            thermal_model = 'hexagonal'
            print(f"\n--- Mode {choice}: Single Simulation ({thermal_model} model) ---")

            z_cbl = float(input(f"Enter Z_CBL (m) [default {DEFAULT_Z_CBL}]: ") or DEFAULT_Z_CBL)
            lambda_thermals = float(input(
                f"Enter Thermal Density (per km^2) [default {DEFAULT_LAMBDA_THERMALS_PER_SQ_KM}]: ") or DEFAULT_LAMBDA_THERMALS_PER_SQ_KM)
            lambda_strength = int(input(
                f"Enter Thermal Strength Lambda [default {DEFAULT_LAMBDA_STRENGTH}]: ") or DEFAULT_LAMBDA_STRENGTH)
            mc_band1 = float(
                input(f"Enter MC Band 1 (m/s) [default {DEFAULT_MC_SNIFF_BAND1}]: ") or DEFAULT_MC_SNIFF_BAND1)
            mc_band2 = float(
                input(f"Enter MC Band 2 (m/s) [default {DEFAULT_MC_SNIFF_BAND2}]: ") or DEFAULT_MC_SNIFF_BAND2)
            search_arc = float(input(
                f"Enter Search Arc Angle (deg) [default {DEFAULT_SEARCH_ARC_ANGLE_DEGREES}]: ") or DEFAULT_SEARCH_ARC_ANGLE_DEGREES)

            random_angle = random.uniform(0, 360)
            end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
            end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
            end_point = (end_point_x, end_point_y)
            initial_bearing = math.degrees(math.atan2(end_point[1], end_point[0]))

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
        print(f"Please copy this error message and share it with me so I can help debug it.")
