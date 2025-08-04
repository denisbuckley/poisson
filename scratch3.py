# --- SCRIPT CHARACTERISTICS ---
# This is a comprehensive Monte Carlo simulation script for a glider's thermal interception.
# The simulation environment models a 2D plane with Poisson-distributed thermals.
# Each thermal consists of a core updraft region and an encircling downdraft ring.
# The script can run in two modes:
#   1. A single-run visualization mode that generates a plot of the thermal field,
#      the glider's dynamic flight path (multiple segments), and any intercepts.
#   2. A large-scale Monte Carlo simulation mode that calculates the statistical
#      probability of a successful thermal intercept over many trials.
#
# Key Features:
# - Thermal Placement: Uses a Poisson distribution to place thermals randomly and
#   realistically throughout the simulation area.
# - Thermal Properties: Thermal updraft strength is also Poisson-distributed
#   (clamped 1-10 m/s), and the updraft radius is derived from this strength.
# - Downdraft Modeling: A fixed-diameter downdraft ring encircles each updraft.
# - Dynamic Flight Path: The glider's path is an iterative, multi-segment path.
# - The Macready settings change based on altitude bands.
# - The glider flies to the nearest thermal in the arc, chooses to climb if
#   the thermal strength is >= the current Macready, otherwise it continues to glide.
# - The flight ends if the glider's altitude drops below 500m.
# - The single-run visualization mode (Option 1) provides a detailed running
#   printout at each intercept, including distance, height, delta angle, band, updraft,
#   speed, sink rate, and the action taken (climb/glide).
# - The printout for Option 1 also includes the straight-line distance from
#   the origin if the flight ends at 500m and the total distance flown.
# - The Monte Carlo simulation (Option 2) summary is formatted to be
#   concise and readable, with headers `Z`, `Arc`, `Prob` at the top and a more
#   detailed table below.
# -----------------------------

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
from tqdm import tqdm
import csv

# --- Constants ---
KNOT_TO_MS = 0.514444
FT_TO_M = 0.3048

C_UPDRAFT_STRENGTH_DECREMENT = 5.9952e-7
FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS = 1200.0
FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS = FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS / 2
K_DOWNDRAFT_STRENGTH = 0.042194
EPSILON = 1e-9
MIN_SAFE_ALTITUDE = 500.0  # Minimum altitude for a safe landing

# --- LS10 Glider Polar Data (from thermal_sim_poisson_segmented.py) ---
# Glider polar data points: airspeed vs sink rate.
# The `pol_v` is airspeed in knots, `pol_w` is sink rate in knots.
pol_v = np.array([45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120])
pol_w = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.5, 6.0])

# --- Scenario Parameters ---
SCENARIO_Z_CBL = 2500.0
SCENARIO_MC_SNIFF_BAND1 = 3.0  # User-defined MC setting for the top band
SCENARIO_MC_SNIFF_BAND2 = 1.0  # User-defined MC setting for the middle band
# Altitude boundary for the middle band - now calculated as (2/3) * CBL
SCENARIO_MIN_ALT_BAND2 = (SCENARIO_Z_CBL / 3) * 2
# Altitude boundary for the lowest band - now calculated as (1/3) * CBL
SCENARIO_MIN_ALT_BAND3 = SCENARIO_Z_CBL / 3
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.01
SCENARIO_LAMBDA_STRENGTH = 3
SEARCH_ARC_ANGLE_DEGREES = 30.0
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
    t = max(0, min(1, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    distance = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    return distance, (closest_x, closest_y)


def generate_poisson_updraft_thermals(sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength):
    """
    Generates thermal locations and properties using a Poisson distribution.
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
        updraft_radius = (updraft_strength_magnitude / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)
        updraft_thermals.append({
            'center': (center_x, center_y),
            'updraft_radius': updraft_radius,
            'updraft_strength': updraft_strength_magnitude
        })
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


def get_band_info(altitude):
    """
    Determines the current altitude band, Macready setting, and altitude to the next band boundary.
    """
    if altitude >= SCENARIO_MIN_ALT_BAND2:
        return "Band 1", SCENARIO_MC_SNIFF_BAND1, altitude - SCENARIO_MIN_ALT_BAND2
    elif altitude >= SCENARIO_MIN_ALT_BAND3:
        return "Band 2", SCENARIO_MC_SNIFF_BAND2, altitude - SCENARIO_MIN_ALT_BAND3
    else:
        return "Band 3", 0.0, altitude - MIN_SAFE_ALTITUDE


# --- Main Dynamic Simulation Function for Visualization (Option 1) ---
def simulate_dynamic_glide_path_and_draw(
        z_cbl_meters, lambda_thermals_per_sq_km, lambda_strength,
        end_point, fig_width=12, fig_height=12
):
    """
    Simulates a glider's flight and visualizes the path.
    """
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.set_aspect('equal')

    plot_padding = 20000.0
    max_coord = max(abs(end_point[0]), abs(end_point[1]))
    sim_area_side_meters = (max_coord + plot_padding) * 2

    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )
    original_thermals = list(updraft_thermals_info)

    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']
        updraft_radius = thermal_info['updraft_radius']
        updraft_circle = patches.Circle(updraft_center, updraft_radius, facecolor='red', alpha=0.6, edgecolor='black',
                                        linewidth=0.5)
        ax.add_patch(updraft_circle)
        downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS
        if downdraft_outer_radius > updraft_radius:
            downdraft_annulus = patches.Circle(updraft_center, downdraft_outer_radius, facecolor='green', alpha=0.05,
                                               edgecolor='green', linewidth=0.5, fill=True, hatch='/')
            ax.add_patch(downdraft_annulus)

    current_pos = (0, 0)
    current_altitude = z_cbl_meters
    path_segments = []
    total_distance_covered = 0.0

    ax.plot(end_point[0], end_point[1], 's', color='black', markersize=10, label='End Point')
    ax.plot(current_pos[0], current_pos[1], 'o', color='blue', markersize=10, label='Start Point')

    start_point = current_pos
    initial_bearing = math.degrees(math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0]))
    if initial_bearing < 0:
        initial_bearing += 360

    print("--- Single Flight Simulation Printout ---")
    print(f"Initial Bearing (Origin to End): {initial_bearing:.2f} deg")
    print(
        f"Macready Settings: Band 1={SCENARIO_MC_SNIFF_BAND1:.1f}m/s, Band 2={SCENARIO_MC_SNIFF_BAND2:.1f}m/s, Band 3=0.0m/s")
    print("-" * 150)
    # Changed 'New Band' to 'Band'
    print(
        f"{'Dist (km)':<12} | {'Alt (m)':<9} | {'Delta (deg)':<12} | {'Band':<10} | {'MC Set (m/s)':<15} | {'Updraft (m/s)':<15} | {'Speed (knots)':<15} | {'Sink Rate (m/s)':<15} | {'Action':<10}")
    print("-" * 150)

    # Variables to track previous state for concise printing
    previous_band = None
    previous_action = None

    # --- Corrected logic for main glide loop ---
    while math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]) > EPSILON:
        if current_altitude <= MIN_SAFE_ALTITUDE:
            break

        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        # Check if the remaining distance to the end is within a direct glide
        current_band, current_mc_sniff_ms, altitude_to_next_band = get_band_info(current_altitude)
        airspeed_ms, sink_rate_ms, glide_ratio, airspeed_knots, sink_rate_knots = get_glider_parameters(
            current_mc_sniff_ms)
        direct_glide_dist = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio

        if distance_to_end < direct_glide_dist:
            # Final direct glide
            travel_distance = distance_to_end
            next_pos = end_point

            time_to_travel = travel_distance / airspeed_ms
            altitude_drop = time_to_travel * sink_rate_ms
            current_altitude -= altitude_drop
            current_pos = next_pos
            path_segments.append((path_start, current_pos))
            total_distance_covered += travel_distance

            # FIX: Calculate delta here before printing
            delta = calculate_bearing_delta(path_start, current_pos, end_point)

            # Final printout
            print(
                f"{total_distance_covered / 1000:<12.3f} | {current_altitude:<9.0f} | {delta:<12.2f} | {'Band 3':<10} | {current_mc_sniff_ms:<15.1f} | {'N/A':<15} | {airspeed_knots:<15.1f} | {sink_rate_ms:<15.2f} | {'Final Glide':<10}")

            if current_altitude > MIN_SAFE_ALTITUDE:
                print("\n--- Simulation Result: SUCCESS. Glider reached destination. ---")
            else:
                print("\n--- Simulation Result: FAILURE. Glider landed before destination. ---")
            break

        # If not on a final glide, proceed with thermal search
        glide_dist_to_band = altitude_to_next_band * glide_ratio
        glide_dist_to_safe_landing = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio

        # FIX: Ensure segment_length is always positive unless a termination condition is met.
        # This prevents the simulation from breaking prematurely at a band boundary.
        segment_length = min(distance_to_end, glide_dist_to_band, glide_dist_to_safe_landing)
        if segment_length <= 0:
            segment_length = distance_to_end  # Fallback to prevent infinite loops if other distances are zero

        bearing_to_end_radians = math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0])
        bearing_to_end_degrees = math.degrees(bearing_to_end_radians)
        if bearing_to_end_degrees < 0: bearing_to_end_degrees += 360

        arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2
        arc_start_angle = bearing_to_end_degrees - arc_half_angle_degrees
        arc_end_angle = bearing_to_end_degrees + arc_half_angle_degrees

        arc_line_upper_end = (path_start[0] + segment_length * math.cos(math.radians(arc_end_angle)),
                              path_start[1] + segment_length * math.sin(math.radians(arc_end_angle)))
        arc_line_lower_end = (path_start[0] + segment_length * math.cos(math.radians(arc_start_angle)),
                              path_start[1] + segment_length * math.sin(math.radians(arc_end_angle)))

        ax.plot([path_start[0], arc_line_upper_end[0]], [path_start[1], arc_line_upper_end[1]],
                'g--', linewidth=0.5, alpha=0.7, label='Search Arc' if len(path_segments) == 0 else '')
        ax.plot([path_start[0], arc_line_lower_end[0]], [path_start[1], arc_line_lower_end[1]],
                'g--', linewidth=0.5, alpha=0.7)

        nearest_thermal = None
        min_dist_to_thermal = float('inf')

        sniffing_radius_meters_base = calculate_sniffing_radius(SCENARIO_LAMBDA_STRENGTH, current_mc_sniff_ms)

        for thermal in updraft_thermals_info:
            thermal_center = thermal['center']
            dist_to_thermal = math.hypot(thermal_center[0] - path_start[0], thermal_center[1] - path_start[1])

            if dist_to_thermal > segment_length + sniffing_radius_meters_base:
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
            dist_to_upper_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_upper_end)
            dist_to_lower_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_lower_end)
            is_near_arc_edge = (dist_to_upper_line <= sniffing_radius_meters_base) or (
                        dist_to_lower_line <= sniffing_radius_meters_base)

            if (is_in_arc or is_near_arc_edge) and dist_to_thermal < min_dist_to_thermal:
                min_dist_to_thermal = dist_to_thermal
                nearest_thermal = thermal

        if nearest_thermal and min_dist_to_thermal < segment_length:
            travel_distance = min_dist_to_thermal
            next_pos = nearest_thermal['center']
        else:
            travel_distance = segment_length
            next_pos = (path_start[0] + segment_length * math.cos(bearing_to_end_radians),
                        path_start[1] + segment_length * math.sin(bearing_to_end_radians))

        time_to_travel = travel_distance / airspeed_ms
        altitude_drop = time_to_travel * sink_rate_ms

        current_altitude -= altitude_drop
        current_pos = next_pos
        path_segments.append((path_start, current_pos))
        total_distance_covered += travel_distance

        # Determine the action and print values
        action = "Glide"
        updraft_val = "N/A"
        if nearest_thermal and min_dist_to_thermal < segment_length:
            updraft_val = nearest_thermal['updraft_strength']
            updraft_thermals_info.remove(nearest_thermal)
            ax.plot(nearest_thermal['center'][0], nearest_thermal['center'][1], 'x', color='blue', markersize=8,
                    markeredgecolor='black', linewidth=1)
            if updraft_val >= current_mc_sniff_ms:
                current_altitude = z_cbl_meters
                action = "Climb"

        # Logic for printing only when values change
        band_to_print = current_band if current_band != previous_band else ''
        action_to_print = action if action != previous_action else ''

        updraft_print_val = f"{updraft_val:<15.1f}" if isinstance(updraft_val, (int, float)) else f"{updraft_val:<15}"

        delta = calculate_bearing_delta(path_start, current_pos, end_point)
        print(
            f"{total_distance_covered / 1000:<12.3f} | {current_altitude:<9.0f} | {delta:<12.2f} | {band_to_print:<10} | {current_mc_sniff_ms:<15.1f} | {updraft_print_val} | {airspeed_knots:<15.1f} | {sink_rate_ms:<15.2f} | {action_to_print:<10}")

        # Update previous state for next iteration
        previous_band = current_band
        previous_action = action

    # Final flight state
    if current_altitude > MIN_SAFE_ALTITUDE:
        print("\n--- Simulation Result: SUCCESS. Glider reached destination. ---")
    else:
        straight_line_dist_origin = math.hypot(current_pos[0], current_pos[1])
        print(f"\n--- Simulation Result: FAILURE. Glider landed before reaching destination. ---")
        print(f"Landed at a straight-line distance of {straight_line_dist_origin / 1000:.3f} km from origin.")

    print(f"Total distance flown: {total_distance_covered / 1000:.3f} km.")

    path_coords_x = []
    path_coords_y = []
    for start, end in path_segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b--', alpha=0.5, linewidth=1)
        path_coords_x.extend([start[0], end[0]])
        path_coords_y.extend([start[1], end[1]])

    ax.plot(path_coords_x, path_coords_y, color='blue', linewidth=2, label='Glider Path')
    ax.plot([], [], 'x', color='blue', markersize=8, markeredgecolor='black', linewidth=1, label='Intercepted Thermal')

    all_x = path_coords_x + [t['center'][0] for t in original_thermals]
    all_y = path_coords_y + [t['center'][1] for t in original_thermals]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    plot_padding_x = (max_x - min_x) * 0.1
    plot_padding_y = (max_y - min_y) * 0.1

    ax.set_xlim(min_x - plot_padding_x, max_x + plot_padding_x)
    ax.set_ylim(min_y - plot_padding_y, max_y + plot_padding_y)
    ax.set_xlabel('East-West (m)')
    ax.set_ylabel('North-South (m)')
    ax.set_title("Dynamic Glider Path Simulation with Thermal Intercepts")
    ax.legend()
    plt.show()


# --- Main Dynamic Simulation Function for Monte Carlo (Option 2) ---
def simulate_intercept_experiment_dynamic(
        z_cbl_meters, lambda_thermals_per_sq_km, lambda_strength,
        end_point
):
    """
    Runs a single, non-visual simulation of a glider's flight.
    """
    plot_padding = 20000.0
    max_coord = max(abs(end_point[0]), abs(end_point[1]))
    sim_area_side_meters = (max_coord + plot_padding) * 2

    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    current_pos = (0, 0)
    current_altitude = z_cbl_meters

    while math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]) > EPSILON:
        if current_altitude <= MIN_SAFE_ALTITUDE:
            return False

        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        # Check for final direct glide
        current_band, current_mc_sniff_ms, _ = get_band_info(current_altitude)
        airspeed_ms, sink_rate_ms, glide_ratio, _, _ = get_glider_parameters(current_mc_sniff_ms)
        direct_glide_dist = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio

        if distance_to_end < direct_glide_dist:
            final_glide_altitude_drop = (distance_to_end / airspeed_ms) * sink_rate_ms
            return current_altitude - final_glide_altitude_drop > MIN_SAFE_ALTITUDE

        # If not on a final glide, proceed with thermal search
        current_band, current_mc_sniff_ms, altitude_to_next_band = get_band_info(current_altitude)
        airspeed_ms, sink_rate_ms, glide_ratio, _, _ = get_glider_parameters(current_mc_sniff_ms)

        glide_dist_to_band = altitude_to_next_band * glide_ratio
        glide_dist_to_safe_landing = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio
        segment_length = min(distance_to_end, glide_dist_to_band, glide_dist_to_safe_landing)

        # FIX: If segment length is zero or less (due to floating point math at boundary),
        # continue to the next iteration with the new band info.
        if segment_length <= 0:
            segment_length = distance_to_end
            if segment_length <= 0:  # If both distances are zero, we are at the end point
                return True

        bearing_to_end_radians = math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0])
        bearing_to_end_degrees = math.degrees(bearing_to_end_radians)
        if bearing_to_end_degrees < 0: bearing_to_end_degrees += 360

        arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2

        nearest_thermal = None
        min_dist_to_thermal = float('inf')

        sniffing_radius_meters_base = calculate_sniffing_radius(SCENARIO_LAMBDA_STRENGTH, current_mc_sniff_ms)

        for thermal in updraft_thermals_info:
            thermal_center = thermal['center']
            dist_to_thermal = math.hypot(thermal_center[0] - path_start[0], thermal_center[1] - path_start[1])

            if dist_to_thermal > segment_length + sniffing_radius_meters_base:
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
                nearest_thermal = thermal

        if nearest_thermal and min_dist_to_thermal < segment_length:
            travel_distance = min_dist_to_thermal
            next_pos = nearest_thermal['center']
        else:
            travel_distance = segment_length
            next_pos = (path_start[0] + segment_length * math.cos(bearing_to_end_radians),
                        path_start[1] + segment_length * math.sin(bearing_to_end_radians))

        time_to_travel = travel_distance / airspeed_ms
        altitude_drop = time_to_travel * sink_rate_ms

        current_altitude -= altitude_drop
        current_pos = next_pos

        if nearest_thermal and min_dist_to_thermal < segment_length:
            updraft_thermals_info.remove(nearest_thermal)
            if float(nearest_thermal['updraft_strength']) >= float(current_mc_sniff_ms):
                current_altitude = z_cbl_meters

    return current_altitude > MIN_SAFE_ALTITUDE


# --- Main execution block ---
if __name__ == '__main__':
    print("Choose an option:")
    print("1. Generate a single plot (visualize dynamic glider path)")
    print("2. Run Monte Carlo simulation (compute probability for a single scenario)")

    choice = input("Enter 1 or 2: ")

    if choice == '1':
        print("\n--- Generating Single Plot with Dynamic Path Simulation ---")
        random_angle = random.uniform(0, 360)
        end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
        end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
        random_end_point = (end_point_x, end_point_y)
        simulate_dynamic_glide_path_and_draw(
            z_cbl_meters=SCENARIO_Z_CBL,
            lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            lambda_strength=SCENARIO_LAMBDA_STRENGTH,
            end_point=random_end_point
        )
    elif choice == '2':
        num_simulations = 1000
        print(f"\n--- Running Monte Carlo Simulation for a Single Scenario ({num_simulations} trials) ---")

        successful_flights = 0
        tqdm_desc = "Running Monte Carlo Trials"
        for _ in tqdm(range(num_simulations), desc=tqdm_desc):
            random_angle = random.uniform(0, 360)
            end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
            end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
            random_end_point = (end_point_x, end_point_y)
            if simulate_intercept_experiment_dynamic(
                    z_cbl_meters=SCENARIO_Z_CBL,
                    lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
                    lambda_strength=SCENARIO_LAMBDA_STRENGTH,
                    end_point=random_end_point
            ):
                successful_flights += 1

        probability = successful_flights / num_simulations

        # Calculate dynamic parameters once for the summary printout
        airspeed1_ms, sink_rate1_ms, glide_ratio1, airspeed1_knots, sink_rate1_knots = get_glider_parameters(
            SCENARIO_MC_SNIFF_BAND1)
        airspeed2_ms, sink_rate2_ms, glide_ratio2, airspeed2_knots, sink_rate2_knots = get_glider_parameters(
            SCENARIO_MC_SNIFF_BAND2)
        airspeed3_ms, sink_rate3_ms, glide_ratio3, airspeed3_knots, sink_rate3_knots = get_glider_parameters(0.0)

        all_results = [{
            'Z (m)': SCENARIO_Z_CBL,
            'Calc. Glide Ratio (B1)': glide_ratio1,
            'Calc. Glide Ratio (B2)': glide_ratio2,
            'Calc. Glide Ratio (B3)': glide_ratio3,
            'Calc. Airspeed (B1) (knots)': airspeed1_knots,
            'Calc. Airspeed (B2) (knots)': airspeed2_knots,
            'Calc. Airspeed (B3) (knots)': airspeed3_knots,
            'Calc. Sink Rate (B1) (m/s)': sink_rate1_ms,
            'Calc. Sink Rate (B2) (m/s)': sink_rate2_ms,
            'Calc. Sink Rate (B3) (m/s)': sink_rate3_ms,
            'Search Arc Angle (deg)': SEARCH_ARC_ANGLE_DEGREES,
            'Thermal Density (per km^2)': SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            'Thermal Strength Lambda': SCENARIO_LAMBDA_STRENGTH,
            'Successful Flights': successful_flights,
            'Probability': probability
        }]

        print("\n" + "=" * 110)
        print("--- Monte Carlo Simulation Results for Single Scenario ---")
        print(f"Z: {SCENARIO_Z_CBL:<8} | Arc: {SEARCH_ARC_ANGLE_DEGREES:<8.1f} | Prob: {probability:<8.4f}")
        print("-" * 110)

        print("\nPerformance Parameters per Band:")
        print("-" * 110)

        # Headers for the band-specific performance table, including upper height
        band_headers = ['Band', 'Upper Height (m)', 'MC (m/s)', 'Airspeed (knots)', 'Sink Rate (m/s)', 'Glide Ratio']
        print(
            f"{band_headers[0]:<10} | {band_headers[1]:<18} | {band_headers[2]:<12} | {band_headers[3]:<17} | {band_headers[4]:<17} | {band_headers[5]:<15}")
        print("-" * 110)

        # Print data for each band
        row = all_results[0]
        print(
            f"{'Band 1':<10} | {SCENARIO_Z_CBL:<18.0f} | {SCENARIO_MC_SNIFF_BAND1:<12.1f} | {row['Calc. Airspeed (B1) (knots)']:<17.1f} | {row['Calc. Sink Rate (B1) (m/s)']:<17.2f} | {row['Calc. Glide Ratio (B1)']:<15.2f}")
        print(
            f"{'Band 2':<10} | {SCENARIO_MIN_ALT_BAND2:<18.0f} | {SCENARIO_MC_SNIFF_BAND2:<12.1f} | {row['Calc. Airspeed (B2) (knots)']:<17.1f} | {row['Calc. Sink Rate (B2) (m/s)']:<17.2f} | {row['Calc. Glide Ratio (B2)']:<15.2f}")
        print(
            f"{'Band 3':<10} | {SCENARIO_MIN_ALT_BAND3:<18.0f} | {0.0:<12.1f} | {row['Calc. Airspeed (B3) (knots)']:<17.1f} | {row['Calc. Sink Rate (B3) (m/s)']:<17.2f} | {row['Calc. Glide Ratio (B3)']:<15.2f}")

        print("\nSimulation Parameters & Results:")
        print("-" * 110)
        print(f"{'Thermal Density (per km^2)':<27} | {SCENARIO_LAMBDA_THERMALS_PER_SQ_KM:<25.2f}")
        print(f"{'Thermal Strength Lambda':<27} | {SCENARIO_LAMBDA_STRENGTH:<25.1f}")
        print(f"{'Successful Flights':<27} | {successful_flights:<25}")
        print("-" * 110)

        csv_filename = "thermal_intercept_simulation_results_poisson_dist_arc_search_dynamic_path_corrected.csv"
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'Z (m)', 'Calc. Glide Ratio (B1)', 'Calc. Glide Ratio (B2)', 'Calc. Glide Ratio (B3)',
                    'Calc. Airspeed (B1) (knots)', 'Calc. Airspeed (B2) (knots)', 'Calc. Airspeed (B3) (knots)',
                    'Calc. Sink Rate (B1) (m/s)', 'Calc. Sink Rate (B2) (m/s)', 'Calc. Sink Rate (B3) (m/s)',
                    'Search Arc Angle (deg)',
                    'Thermal Density (per km^2)', 'Thermal Strength Lambda',
                    'Successful Flights', 'Probability'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(all_results[0])
            print(f"\nResults successfully exported to '{csv_filename}'")
        except IOError as e:
            print(f"\nError writing to CSV file '{csv_filename}': {e}")
    else:
        print("Invalid choice. Please enter 1 or 2.")
