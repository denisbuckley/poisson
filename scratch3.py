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

# --- Default Scenario Parameters ---
DEFAULT_Z_CBL = 2500.0
DEFAULT_MC_SNIFF_BAND1 = 4.0
DEFAULT_MC_SNIFF_BAND2 = 2.0
DEFAULT_LAMBDA_THERMALS_PER_SQ_KM = 0.1
DEFAULT_LAMBDA_STRENGTH = 3
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
    bearing_to_end = math.degrees(math.atan2(end_point[1] - start_point[1], end_point[0] - end_point[0]))

    if bearing_to_next < 0: bearing_to_next += 360
    if bearing_to_end < 0: bearing_to_end += 360

    delta = bearing_to_next - bearing_to_end
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360

    return delta


def calculate_bearing_delta_to_thermal(start_point, thermal_point, end_point):
    """
    Calculates the delta angle between the bearing to a thermal and the bearing to the end point.
    """
    bearing_to_thermal = math.degrees(math.atan2(thermal_point[1] - start_point[1], thermal_point[0] - start_point[0]))
    bearing_to_end = math.degrees(math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0]))

    if bearing_to_thermal < 0: bearing_to_thermal += 360
    if bearing_to_end < 0: bearing_to_end += 360

    delta = bearing_to_thermal - bearing_to_end
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

    if altitude >= min_alt_band2:
        return "Band 1", scenario_mc_band1, altitude - min_alt_band2, min_alt_band2
    elif altitude >= min_alt_band3:
        return "Band 2", scenario_mc_band2, altitude - min_alt_band3, min_alt_band3
    else:
        return "Band 3", 0.0, altitude - MIN_SAFE_ALTITUDE, MIN_SAFE_ALTITUDE


# --- Main Dynamic Simulation Function for Visualization (Option 1) ---
def simulate_dynamic_glide_path_and_draw(
        z_cbl_meters, lambda_thermals_per_sq_km, lambda_strength,
        mc_sniff_band1, mc_sniff_band2, end_point, fig_width=12, fig_height=12
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

    print("--- Single Flight Simulation Printout ---")
    print(f"Macready Settings: Band 1={mc_sniff_band1:.1f}m/s, Band 2={mc_sniff_band2:.1f}m/s, Band 3=0.0m/s")
    print("-" * 150)
    print(
        f"{'Dist (km)':<12} | {'Alt (m)':<9} | {'Delta (deg)':<12} | {'Band':<10} | {'MC Set (m/s)':<15} | {'Updraft (m/s)':<15} | {'Speed (knots)':<15} | {'Sink Rate (m/s)':<15} | {'Action':<10}")
    print("-" * 150)

    previous_band = None
    previous_action = None

    # --- Initial Printout at the start of the flight ---
    current_band, current_mc_sniff_ms, _, _ = get_band_info(current_altitude, z_cbl_meters, mc_sniff_band1,
                                                            mc_sniff_band2)
    airspeed_ms, sink_rate_ms, _, airspeed_knots, sink_rate_knots = get_glider_parameters(current_mc_sniff_ms)

    bearing_to_end_degrees = math.degrees(math.atan2(end_point[1] - current_pos[1], end_point[0] - current_pos[0]))
    if bearing_to_end_degrees < 0: bearing_to_end_degrees += 360
    arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2

    initial_delta = 0.0
    min_dist_to_thermal_initial = float('inf')
    sniffing_radius_meters_base = calculate_sniffing_radius(lambda_strength, current_mc_sniff_ms)

    start_pos_temp = current_pos
    distance_to_end_temp = math.hypot(end_point[0] - start_pos_temp[0], end_point[1] - start_pos_temp[1])
    glide_dist_to_safe_landing_temp = (current_altitude - MIN_SAFE_ALTITUDE) * \
                                      get_glider_parameters(current_mc_sniff_ms)[2]
    segment_length_temp = min(distance_to_end_temp, glide_dist_to_safe_landing_temp)

    for thermal in updraft_thermals_info:
        thermal_center = thermal['center']
        dist_to_thermal = math.hypot(thermal_center[0] - start_pos_temp[0], thermal_center[1] - start_pos_temp[1])
        if dist_to_thermal > segment_length_temp + sniffing_radius_meters_base:
            continue
        thermal_bearing_from_start = math.degrees(
            math.atan2(thermal_center[1] - start_pos_temp[1], thermal_center[0] - start_pos_temp[0]))
        if thermal_bearing_from_start < 0: thermal_bearing_from_start += 360
        angle_from_line = thermal_bearing_from_start - bearing_to_end_degrees
        if angle_from_line > 180:
            angle_from_line -= 360
        elif angle_from_line <= -180:
            angle_from_line += 360
        if abs(angle_from_line) <= arc_half_angle_degrees and dist_to_thermal < min_dist_to_thermal_initial:
            min_dist_to_thermal_initial = dist_to_thermal
            initial_delta = angle_from_line

    print(
        f"{0:<12.3f} | {current_altitude:<9.0f} | {initial_delta:<12.2f} | {'Band 1':<10} | {current_mc_sniff_ms:<15.1f} | {'N/A':<15} | {airspeed_knots:<15.1f} | {sink_rate_ms:<15.2f} | {'Glide':<10}")

    previous_band = "Band 1"
    previous_action = "Glide"

    remaining_thermals = list(updraft_thermals_info)

    while True:
        if current_altitude <= MIN_SAFE_ALTITUDE:
            print("\n--- Simulation Result: FAILURE. Glider landed before reaching destination. ---")
            break

        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        current_band, current_mc_sniff_ms, _, _ = get_band_info(current_altitude, z_cbl_meters, mc_sniff_band1,
                                                                mc_sniff_band2)
        airspeed_ms, sink_rate_ms, glide_ratio, airspeed_knots, sink_rate_knots = get_glider_parameters(
            current_mc_sniff_ms)
        direct_glide_dist = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio

        if distance_to_end < direct_glide_dist:
            travel_distance = distance_to_end
            next_pos = end_point
            time_to_travel = travel_distance / airspeed_ms
            altitude_drop = time_to_travel * sink_rate_ms
            current_altitude -= altitude_drop
            current_pos = next_pos
            path_segments.append((path_start, current_pos))
            total_distance_covered += travel_distance

            band_to_print = "Band 3" if "Band 3" != previous_band else ''
            action_to_print = "Final Glide" if "Final Glide" != previous_action else ''

            print(
                f"{total_distance_covered / 1000:<12.3f} | {current_altitude:<9.0f} | {0.0:<12.2f} | {band_to_print:<10} | {current_mc_sniff_ms:<15.1f} | {'N/A':<15} | {airspeed_knots:<15.1f} | {sink_rate_ms:<15.2f} | {action_to_print:<10}")

            print("\n--- Simulation Result: SUCCESS. Glider reached destination. ---")
            break

        bearing_to_end_radians = math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0])
        bearing_to_end_degrees = math.degrees(bearing_to_end_radians)
        if bearing_to_end_degrees < 0: bearing_to_end_degrees += 360
        arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2

        nearest_thermal = None
        min_dist_to_thermal = float('inf')
        sniffing_radius_meters_base = calculate_sniffing_radius(lambda_strength, current_mc_sniff_ms)

        for thermal in remaining_thermals:
            thermal_center = thermal['center']
            dist_to_thermal = math.hypot(thermal_center[0] - path_start[0], thermal_center[1] - path_start[1])
            thermal_bearing_from_start = math.degrees(
                math.atan2(thermal_center[1] - path_start[1], thermal_center[0] - path_start[0]))
            if thermal_bearing_from_start < 0: thermal_bearing_from_start += 360

            angle_from_line = thermal_bearing_from_start - bearing_to_end_degrees
            if angle_from_line > 180:
                angle_from_line -= 360
            elif angle_from_line <= -180:
                angle_from_line += 360

            is_in_arc = abs(angle_from_line) <= arc_half_angle_degrees

            if is_in_arc and dist_to_thermal < min_dist_to_thermal and dist_to_thermal < direct_glide_dist:
                min_dist_to_thermal = dist_to_thermal
                nearest_thermal = thermal

        if nearest_thermal:
            travel_distance = min_dist_to_thermal
            next_pos = nearest_thermal['center']
            time_to_travel = travel_distance / airspeed_ms
            altitude_drop = time_to_travel * sink_rate_ms
            current_altitude -= altitude_drop
            current_pos = next_pos
            path_segments.append((path_start, current_pos))
            total_distance_covered += travel_distance

            updraft_val = nearest_thermal['updraft_strength']
            remaining_thermals.remove(nearest_thermal)
            ax.plot(nearest_thermal['center'][0], nearest_thermal['center'][1], 'x', color='blue', markersize=8,
                    markeredgecolor='black', linewidth=1)

            if updraft_val >= current_mc_sniff_ms:
                current_altitude = z_cbl_meters
                action = "Climb"
            else:
                action = "Glide"

            # Determine the delta angle to the next nearest thermal from the new position
            delta_to_print = 0.0
            min_dist_to_thermal_next = float('inf')

            # Recalculate parameters for the new altitude
            current_band_at_intercept, current_mc_sniff_ms_at_intercept, _, _ = get_band_info(current_altitude,
                                                                                              z_cbl_meters,
                                                                                              mc_sniff_band1,
                                                                                              mc_sniff_band2)
            airspeed_ms_new, _, glide_ratio_new, _, _ = get_glider_parameters(current_mc_sniff_ms_at_intercept)

            path_start_new = current_pos
            distance_to_end_new = math.hypot(end_point[0] - path_start_new[0], end_point[1] - path_start_new[1])
            glide_dist_to_safe_landing_new = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio_new
            segment_length_new = min(distance_to_end_new, glide_dist_to_safe_landing_new)

            bearing_to_end_degrees_new = math.degrees(
                math.atan2(end_point[1] - path_start_new[1], end_point[0] - path_start_new[0]))
            if bearing_to_end_degrees_new < 0: bearing_to_end_degrees_new += 360

            sniffing_radius_meters_base_new = calculate_sniffing_radius(lambda_strength,
                                                                        current_mc_sniff_ms_at_intercept)

            for thermal in remaining_thermals:  # Check remaining thermals
                thermal_center = thermal['center']
                dist_to_thermal = math.hypot(thermal_center[0] - path_start_new[0],
                                             thermal_center[1] - path_start_new[1])
                if dist_to_thermal > segment_length_new + sniffing_radius_meters_base_new:
                    continue
                thermal_bearing_from_start = math.degrees(
                    math.atan2(thermal_center[1] - path_start_new[1], thermal_center[0] - path_start_new[0]))
                if thermal_bearing_from_start < 0: thermal_bearing_from_start += 360
                angle_from_line = thermal_bearing_from_start - bearing_to_end_degrees_new
                if angle_from_line > 180:
                    angle_from_line -= 360
                elif angle_from_line <= -180:
                    angle_from_line += 360
                if abs(angle_from_line) <= arc_half_angle_degrees and dist_to_thermal < min_dist_to_thermal_next:
                    min_dist_to_thermal_next = dist_to_thermal
                    delta_to_print = angle_from_line

            band_to_print = current_band_at_intercept if current_band_at_intercept != previous_band else ''
            action_to_print = action if action != previous_action else ''
            updraft_print_val = f"{updraft_val:<15.1f}"

            print(
                f"{total_distance_covered / 1000:<12.3f} | {current_altitude:<9.0f} | {delta_to_print:<12.2f} | {band_to_print:<10} | {current_mc_sniff_ms_at_intercept:<15.1f} | {updraft_print_val} | {airspeed_knots:<15.1f} | {sink_rate_ms:<15.2f} | {action_to_print:<10}")

            previous_band = current_band_at_intercept
            previous_action = action
        else:
            # If no thermal is found in the arc, the glider glides straight towards the destination
            travel_distance = distance_to_end
            next_pos = end_point
            time_to_travel = travel_distance / airspeed_ms
            altitude_drop = time_to_travel * sink_rate_ms
            current_altitude -= altitude_drop
            current_pos = next_pos
            path_segments.append((path_start, current_pos))
            total_distance_covered += travel_distance

    if current_altitude <= MIN_SAFE_ALTITUDE:
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

    min_x, max_x = min(all_x) if all_x else -sim_area_side_meters / 2, max(all_x) if all_x else sim_area_side_meters / 2
    min_y, max_y = min(all_y) if all_y else -sim_area_side_meters / 2, max(all_y) if all_y else sim_area_side_meters / 2

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
        mc_sniff_band1, mc_sniff_band2, end_point
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
    remaining_thermals = list(updraft_thermals_info)

    while math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]) > EPSILON:
        if current_altitude <= MIN_SAFE_ALTITUDE:
            return False

        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        current_band, current_mc_sniff_ms, _, _ = get_band_info(current_altitude, z_cbl_meters, mc_sniff_band1,
                                                                mc_sniff_band2)
        airspeed_ms, sink_rate_ms, glide_ratio, _, _ = get_glider_parameters(current_mc_sniff_ms)
        direct_glide_dist = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio

        if distance_to_end < direct_glide_dist:
            final_glide_altitude_drop = (distance_to_end / airspeed_ms) * sink_rate_ms
            return current_altitude - final_glide_altitude_drop > MIN_SAFE_ALTITUDE

        glide_dist_to_band = (get_band_info(current_altitude, z_cbl_meters, mc_sniff_band1, mc_sniff_band2)[
                                  3] - MIN_SAFE_ALTITUDE) * glide_ratio
        segment_length = min(distance_to_end, glide_dist_to_band, direct_glide_dist)

        if segment_length <= 0:
            segment_length = distance_to_end
            if segment_length <= 0:
                return True

        bearing_to_end_radians = math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0])
        bearing_to_end_degrees = math.degrees(bearing_to_end_radians)
        if bearing_to_end_degrees < 0: bearing_to_end_degrees += 360

        arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2

        nearest_thermal = None
        min_dist_to_thermal = float('inf')

        sniffing_radius_meters_base = calculate_sniffing_radius(lambda_strength, current_mc_sniff_ms)

        for thermal in remaining_thermals:
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
            remaining_thermals.remove(nearest_thermal)
            if float(nearest_thermal['updraft_strength']) >= float(current_mc_sniff_ms):
                current_altitude = z_cbl_meters

    return current_altitude > MIN_SAFE_ALTITUDE


def get_user_input_for_parameters():
    """
    Prompts the user for key simulation parameters with defaults.
    """
    print("\n--- Customize Simulation Parameters (Press Enter to use default) ---")

    # Get Thermal Density
    thermal_density = input(f"Enter Thermal Density (per km^2) [{DEFAULT_LAMBDA_THERMALS_PER_SQ_KM}]: ")
    if thermal_density == "":
        thermal_density = DEFAULT_LAMBDA_THERMALS_PER_SQ_KM
    else:
        thermal_density = float(thermal_density)

    # Get CBL Height
    cbl_height = input(f"Enter Cloud Base Level (m) [{DEFAULT_Z_CBL}]: ")
    if cbl_height == "":
        cbl_height = DEFAULT_Z_CBL
    else:
        cbl_height = float(cbl_height)

    # Get MC setting for Band 1
    mc_band1 = input(f"Enter Macready Setting for Band 1 (m/s) [{DEFAULT_MC_SNIFF_BAND1}]: ")
    if mc_band1 == "":
        mc_band1 = DEFAULT_MC_SNIFF_BAND1
    else:
        mc_band1 = float(mc_band1)

    # Get MC setting for Band 2
    mc_band2 = input(f"Enter Macready Setting for Band 2 (m/s) [{DEFAULT_MC_SNIFF_BAND2}]: ")
    if mc_band2 == "":
        mc_band2 = DEFAULT_MC_SNIFF_BAND2
    else:
        mc_band2 = float(mc_band2)

    # Get Thermal Strength Lambda
    thermal_strength_lambda = input(f"Enter Thermal Strength Lambda (λ) [{DEFAULT_LAMBDA_STRENGTH}]: ")
    if thermal_strength_lambda == "":
        thermal_strength_lambda = DEFAULT_LAMBDA_STRENGTH
    else:
        thermal_strength_lambda = float(thermal_strength_lambda)

    return {
        'thermal_density': thermal_density,
        'cbl_height': cbl_height,
        'mc_band1': mc_band1,
        'mc_band2': mc_band2,
        'thermal_strength_lambda': thermal_strength_lambda
    }


# --- Main execution block ---
if __name__ == '__main__':
    # Get user-defined or default parameters
    params = get_user_input_for_parameters()

    # Set the scenario parameters based on user input
    SCENARIO_Z_CBL = params['cbl_height']
    SCENARIO_MC_SNIFF_BAND1 = params['mc_band1']
    SCENARIO_MC_SNIFF_BAND2 = params['mc_band2']
    SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = params['thermal_density']
    SCENARIO_LAMBDA_STRENGTH = params['thermal_strength_lambda']

    # Recalculate band boundaries based on the new CBL height
    SCENARIO_MIN_ALT_BAND2 = (SCENARIO_Z_CBL / 3) * 2
    SCENARIO_MIN_ALT_BAND3 = SCENARIO_Z_CBL / 3

    print("\nChoose an option:")
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
            mc_sniff_band1=SCENARIO_MC_SNIFF_BAND1,
            mc_sniff_band2=SCENARIO_MC_SNIFF_BAND2,
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
                    mc_sniff_band1=SCENARIO_MC_SNIFF_BAND1,
                    mc_sniff_band2=SCENARIO_MC_SNIFF_BAND2,
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
        print(f"Z: {SCENARIO_Z_CBL:<8.1f} | Arc: {SEARCH_ARC_ANGLE_DEGREES:<8.1f} | Prob: {probability:<8.4f}")
        print("-" * 110)

        print("\nPerformance Parameters per Band:")
        print("-" * 110)

        band_headers = ['Band', 'Upper Height (m)', 'MC (m/s)', 'Airspeed (knots)', 'Sink Rate (m/s)', 'Glide Ratio']
        print(
            f"{band_headers[0]:<10} | {band_headers[1]:<18} | {band_headers[2]:<12} | {band_headers[3]:<17} | {band_headers[4]:<17} | {band_headers[5]:<15}")
        print("-" * 110)

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