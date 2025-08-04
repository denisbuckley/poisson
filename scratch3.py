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
# - Dynamic Flight Path: The glider's path is no longer a single straight line.
#   It is an iterative, multi-segment path where the glider "moves" to an intercepted
#   thermal before continuing its flight towards a final destination.
# - **MODIFIED:** The script now uses a glider polar to dynamically calculate the
#   glide ratio and sink rate based on the pilot's Macready setting, removing the
#   previous fixed glide ratio.
# - **FIXED:** The Monte Carlo simulation now correctly determines a "successful" flight
#   by checking if the glider reaches its destination with sufficient altitude.
# - Climb and Restart Logic: If an intercepted updraft's strength is greater
#   than or equal to the current MC_Sniff setting, the glider instantaneously
#   "climbs" back to CBL height and "restarts" its glide from that horizontal position.
# - Interception Logic: A thermal is considered a potential intercept if its center is within the
#   search arc OR if its sniffing radius overlaps with the search arc's boundary lines.
# - **FIXED:** The bug causing an infinite loop has been resolved. Intercepted thermals are now
#   removed from the list of available thermals, forcing the glider to always move forward.
# - **NEW:** The initial line orientation is now randomized by generating a random
#   end point at a fixed distance from the origin.
# - **FIXED:** The final step is now labeled 'Final Glide' with distance and angle relative to the initial glide path.
# - **NEW:** The single-run visualization mode now explicitly calculates and prints the distance from the origin when the glider lands (i.e., its altitude drops to 500m).
# - Visualization: The search arc is now visually drawn on the plot to demonstrate
#   the search area for each path segment, and the specific intercepted thermal
#   is highlighted with a unique marker.
# - Data Output: The output for the single plot mode is now simplified to only
#   show the distance from the origin and the relative bearing, without redundant labels.
# - **MODIFIED:** The printout for the Monte Carlo simulation (option 2) has been
#   updated to remove the 'Sniffing Radius (Base)(m)', 'Initial Glide Path Length (m)',
#   and 'Wt_Ambient (m/s)' columns, as requested.
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
# NOTE: SCENARIO_GLIDE_RATIO is no longer a fixed value; it's dynamically calculated.
SCENARIO_MC_SNIFF = 6
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.01
SCENARIO_LAMBDA_STRENGTH = 3
SEARCH_ARC_ANGLE_DEGREES = 30.0
RANDOM_END_POINT_DISTANCE = 100000.0


# --- Helper functions ---
def calculate_sniffing_radius(Wt_ms_ambient, MC_for_sniffing_ms, thermal_type="NORMAL"):
    """
    Calculates the sniffing radius based on ambient thermal strength and pilot's Macready setting.
    This function remains the same.
    """
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
    This function remains the same.
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
    This function remains the same.
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


def calculate_glider_parameters_from_polar(MC_setting_ms):
    """
    Calculates the optimum airspeed, sink rate, and glide ratio for a given
    Macready setting using the glider polar.

    Args:
        MC_setting_ms (float): The pilot's Macready setting in m/s.

    Returns:
        tuple: (airspeed_ms, sink_rate_ms, glide_ratio)
    """
    MC_setting_knots = MC_setting_ms / KNOT_TO_MS

    # Calculate effective sink rate (pol_w - MC_setting) for each airspeed.
    effective_sink_rate_knots = pol_w - MC_setting_knots

    # Calculate effective glide ratio (pol_v / effective_sink_rate)
    effective_glide_ratio = pol_v / effective_sink_rate_knots

    # Find the maximum effective glide ratio, which corresponds to the best speed-to-fly.
    max_glide_ratio_index = np.argmax(effective_glide_ratio)

    airspeed_knots = pol_v[max_glide_ratio_index]
    sink_rate_knots = pol_w[max_glide_ratio_index]

    airspeed_ms = airspeed_knots * KNOT_TO_MS
    sink_rate_ms = sink_rate_knots * KNOT_TO_MS
    glide_ratio = airspeed_ms / sink_rate_ms

    return airspeed_ms, sink_rate_ms, glide_ratio


# --- Main Dynamic Simulation Function for Visualization ---
def simulate_dynamic_glide_path_and_draw(
        z_cbl_meters, mc_for_sniffing_ms,
        lambda_thermals_per_sq_km, lambda_strength,
        end_point, fig_width=12, fig_height=12
):
    """
    Simulates a glider's flight and visualizes the path.
    Now uses dynamic glide ratio and sink rate.
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
    bearing_to_end_initial = math.degrees(math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0]))
    if bearing_to_end_initial < 0:
        bearing_to_end_initial += 360

    print(f"Starting simulation from (0,0) to random end point {end_point}")
    print(f"Initial Bearing from origin to end point: {bearing_to_end_initial:.2f}°")

    successful_flight = False

    while math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]) > EPSILON:
        if current_altitude <= MIN_SAFE_ALTITUDE:
            print(
                f"Landing: Altitude dropped below {MIN_SAFE_ALTITUDE} m. Landed at {total_distance_covered / 1000:.3f} km from origin.")
            if len(path_segments) > 0:
                path_segments.append((path_segments[-1][1], current_pos))
            break

        # Dynamically calculate glider parameters based on Macready setting
        airspeed_ms, sink_rate_ms, glide_ratio = calculate_glider_parameters_from_polar(mc_for_sniffing_ms)

        path_start = current_pos

        available_glide_height = current_altitude - MIN_SAFE_ALTITUDE
        # The distance flown to land safely is available height * glide ratio
        segment_length = available_glide_height * glide_ratio

        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        if distance_to_end <= 0:
            break

        bearing_to_end_radians = math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0])
        bearing_to_end_degrees = math.degrees(bearing_to_end_radians)
        if bearing_to_end_degrees < 0:
            bearing_to_end_degrees += 360

        arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2
        search_distance = min(distance_to_end, segment_length)

        arc_start_angle = bearing_to_end_degrees - arc_half_angle_degrees
        arc_end_angle = bearing_to_end_degrees + arc_half_angle_degrees
        wedge = patches.Wedge(
            path_start, search_distance,
            arc_start_angle, arc_end_angle,
            facecolor='gray', alpha=0.1, edgecolor='none'
        )
        ax.add_patch(wedge)

        path_end = (path_start[0] + search_distance * math.cos(bearing_to_end_radians),
                    path_start[1] + search_distance * math.sin(bearing_to_end_radians))

        nearest_thermal = None
        min_dist_to_thermal = float('inf')

        ambient_wt_for_sniff_calc = lambda_strength
        sniffing_radius_meters_base = calculate_sniffing_radius(ambient_wt_for_sniff_calc, mc_for_sniffing_ms)

        for thermal in updraft_thermals_info:
            thermal_center = thermal['center']
            dist_to_thermal = math.hypot(thermal_center[0] - path_start[0], thermal_center[1] - path_start[1])

            if dist_to_thermal > search_distance:
                continue

            thermal_bearing_from_start = math.degrees(
                math.atan2(thermal_center[1] - path_start[1], thermal_center[0] - path_start[0]))
            if thermal_bearing_from_start < 0:
                thermal_bearing_from_start += 360

            angle_from_line = thermal_bearing_from_start - bearing_to_end_degrees
            if angle_from_line > 180:
                angle_from_line -= 360
            elif angle_from_line <= -180:
                angle_from_line += 360

            is_in_arc = abs(angle_from_line) <= arc_half_angle_degrees

            arc_line_upper_end = (path_start[0] + search_distance * math.cos(math.radians(arc_end_angle)),
                                  path_start[1] + search_distance * math.sin(math.radians(arc_end_angle)))
            arc_line_lower_end = (path_start[0] + search_distance * math.cos(math.radians(arc_start_angle)),
                                  path_start[1] + search_distance * math.sin(math.radians(arc_start_angle)))

            dist_to_upper_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_upper_end)
            dist_to_lower_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_lower_end)

            is_near_arc_edge = (dist_to_upper_line <= sniffing_radius_meters_base) or (
                    dist_to_lower_line <= sniffing_radius_meters_base)

            if (is_in_arc or is_near_arc_edge):
                if dist_to_thermal < min_dist_to_thermal:
                    min_dist_to_thermal = dist_to_thermal
                    nearest_thermal = thermal

        if nearest_thermal:
            thermal_center = nearest_thermal['center']
            time_to_thermal = min_dist_to_thermal / airspeed_ms
            altitude_drop = time_to_thermal * sink_rate_ms

            if current_altitude - altitude_drop < MIN_SAFE_ALTITUDE:
                glide_dist_to_safe_alt = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio
                bearing_to_thermal_radians = math.atan2(thermal_center[1] - path_start[1],
                                                        thermal_center[0] - path_start[0])
                landing_pos_x = path_start[0] + glide_dist_to_safe_alt * math.cos(bearing_to_thermal_radians)
                landing_pos_y = path_start[1] + glide_dist_to_safe_alt * math.sin(bearing_to_thermal_radians)
                landing_pos = (landing_pos_x, landing_pos_y)
                total_distance_at_landing = total_distance_covered + glide_dist_to_safe_alt

                print(
                    f"Landing: Altitude dropped below {MIN_SAFE_ALTITUDE} m. Landed at {total_distance_at_landing / 1000:.3f} km from origin.")
                path_segments.append((path_start, landing_pos))
                current_pos = landing_pos
                current_altitude = MIN_SAFE_ALTITUDE - 1
                break

            current_altitude -= altitude_drop
            path_segments.append((path_start, thermal_center))
            current_pos = thermal_center
            total_distance_covered += min_dist_to_thermal

            updraft_thermals_info.remove(nearest_thermal)

            thermal_dist = math.hypot(thermal_center[0] - start_point[0], thermal_center[1] - start_point[1])
            thermal_bearing = math.degrees(
                math.atan2(thermal_center[1] - start_point[1], thermal_center[0] - start_point[0]))
            if thermal_bearing < 0:
                thermal_bearing += 360

            relative_bearing = thermal_bearing - bearing_to_end_initial
            if relative_bearing > 180:
                relative_bearing -= 360
            elif relative_bearing <= -180:
                relative_bearing += 360

            if float(nearest_thermal['updraft_strength']) >= float(mc_for_sniffing_ms):
                print(
                    f"Total Distance: {total_distance_covered / 1000:.3f} km, Rel. Bearing: {relative_bearing:.2f}° @ {current_altitude:.0f} m, Updraft: {nearest_thermal['updraft_strength']:.1f} m/s, Climbing.")
                current_altitude = z_cbl_meters
            else:
                print(
                    f"Total Distance: {total_distance_covered / 1000:.3f} km, Rel. Bearing: {relative_bearing:.2f}° @ {current_altitude:.0f} m, Updraft: {nearest_thermal['updraft_strength']:.1f} m/s, Continuing Glide.")

        else:
            time_to_travel = search_distance / airspeed_ms
            altitude_drop = time_to_travel * sink_rate_ms

            if current_altitude - altitude_drop < MIN_SAFE_ALTITUDE:
                glide_dist_to_safe_alt = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio

                landing_pos_x = path_start[0] + glide_dist_to_safe_alt * math.cos(bearing_to_end_radians)
                landing_pos_y = path_start[1] + glide_dist_to_safe_alt * math.sin(bearing_to_end_radians)
                landing_pos = (landing_pos_x, landing_pos_y)
                total_distance_at_landing = total_distance_covered + glide_dist_to_safe_alt

                print(
                    f"Landing: Altitude dropped below {MIN_SAFE_ALTITUDE} m. Landed at {total_distance_at_landing / 1000:.3f} km from origin.")
                path_segments.append((path_start, landing_pos))
                current_pos = landing_pos
                current_altitude = MIN_SAFE_ALTITUDE - 1
                break

            current_altitude -= altitude_drop
            current_pos = path_end
            path_segments.append((path_start, path_end))
            total_distance_covered += search_distance

    if current_altitude > MIN_SAFE_ALTITUDE:
        final_glide_distance = math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1])
        current_pos_at_final_glide = current_pos

        airspeed_ms, sink_rate_ms, glide_ratio = calculate_glider_parameters_from_polar(mc_for_sniffing_ms)
        time_to_travel = final_glide_distance / airspeed_ms
        final_glide_altitude_drop = time_to_travel * sink_rate_ms

        if current_altitude - final_glide_altitude_drop < MIN_SAFE_ALTITUDE:
            glide_dist_to_safe_alt = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio
            total_distance_at_landing = total_distance_covered + glide_dist_to_safe_alt

            final_glide_bearing_radians = math.atan2(end_point[1] - current_pos_at_final_glide[1],
                                                     end_point[0] - current_pos_at_final_glide[0])
            landing_pos_x = current_pos_at_final_glide[0] + glide_dist_to_safe_alt * math.cos(
                final_glide_bearing_radians)
            landing_pos_y = current_pos_at_final_glide[1] + glide_dist_to_safe_alt * math.sin(
                final_glide_bearing_radians)
            landing_pos = (landing_pos_x, landing_pos_y)

            path_segments.append((current_pos_at_final_glide, landing_pos))
            current_pos = landing_pos
            current_altitude = MIN_SAFE_ALTITUDE - 1
            print(
                f"Final Glide Landing: Altitude dropped below {MIN_SAFE_ALTITUDE} m. Landed at {total_distance_at_landing / 1000:.3f} km from origin. Final distance to end point: {final_glide_distance - glide_dist_to_safe_alt:.0f} m")

        else:
            current_altitude -= final_glide_altitude_drop
            current_pos = end_point
            path_segments.append((current_pos_at_final_glide, current_pos))
            total_distance_covered += final_glide_distance
            successful_flight = True

            final_glide_bearing = math.degrees(
                math.atan2(end_point[1] - current_pos_at_final_glide[1], end_point[0] - current_pos_at_final_glide[0]))
            if final_glide_bearing < 0:
                final_glide_bearing += 360

            final_glide_relative_bearing = final_glide_bearing - bearing_to_end_initial
            if final_glide_relative_bearing > 180:
                final_glide_relative_bearing -= 360
            elif final_glide_relative_bearing <= -180:
                final_glide_relative_bearing += 360

            print(
                f"Final Glide: {final_glide_distance / 1000:.3f} km, Rel. Bearing: {final_glide_relative_bearing:.2f}° to end point. Total distance: {total_distance_covered / 1000:.3f} km")

    if current_altitude > MIN_SAFE_ALTITUDE and successful_flight:
        print(f"\nSimulation Result: SUCCESS! Glider reached destination with altitude {current_altitude:.0f} m.")
    else:
        print(f"\nSimulation Result: FAILURE. Glider landed before reaching destination.")

    path_coords_x = []
    path_coords_y = []

    for i, (start, end) in enumerate(path_segments):
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b--', alpha=0.5, linewidth=1)
        path_coords_x.append(start[0])
        path_coords_y.append(start[1])
        if i < len(path_segments) - 1:
            path_coords_x.append(end[0])
            path_coords_y.append(end[1])
            ax.plot(end[0], end[1], 'x', color='red', markersize=8, markeredgecolor='black', linewidth=1)

    if len(path_segments) > 0:
        path_coords_x.append(path_segments[-1][1][0])
        path_coords_y.append(path_segments[-1][1][1])

    ax.plot(path_coords_x, path_coords_y, color='blue', linewidth=2, label='Glider Path')
    ax.plot([], [], 'x', color='red', markersize=8, markeredgecolor='black', linewidth=1, label='Intercepted Thermal')

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
    ax.set_title(f"Dynamic Glider Path Simulation with Thermal Intercepts")
    ax.legend()
    plt.show()

    return successful_flight


# --- Main Dynamic Simulation Function for Monte Carlo ---
def simulate_intercept_experiment_dynamic(
        z_cbl_meters, mc_for_sniffing_ms,
        lambda_thermals_per_sq_km, lambda_strength,
        end_point
):
    """
    Runs a single, non-visual simulation of a glider's flight.
    Now uses dynamic glide ratio and sink rate.
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

        # Dynamically calculate glider parameters based on Macready setting
        airspeed_ms, sink_rate_ms, glide_ratio = calculate_glider_parameters_from_polar(mc_for_sniffing_ms)

        path_start = current_pos

        available_glide_height = current_altitude - MIN_SAFE_ALTITUDE
        # The distance flown to land safely is available height * glide ratio
        segment_length = available_glide_height * glide_ratio

        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        if distance_to_end <= 0:
            break

        bearing_to_end_radians = math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0])
        bearing_to_end_degrees = math.degrees(bearing_to_end_radians)
        if bearing_to_end_degrees < 0:
            bearing_to_end_degrees += 360

        arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2
        arc_start_angle = bearing_to_end_degrees - arc_half_angle_degrees
        arc_end_angle = bearing_to_end_degrees + arc_half_angle_degrees

        search_distance = min(distance_to_end, segment_length)

        path_end = (path_start[0] + search_distance * math.cos(bearing_to_end_radians),
                    path_start[1] + search_distance * math.sin(bearing_to_end_radians))

        nearest_thermal = None
        min_dist_to_thermal = float('inf')

        ambient_wt_for_sniff_calc = lambda_strength
        sniffing_radius_meters_base = calculate_sniffing_radius(ambient_wt_for_sniff_calc, mc_for_sniffing_ms)

        for thermal in updraft_thermals_info:
            thermal_center = thermal['center']
            dist_to_thermal = math.hypot(thermal_center[0] - path_start[0], thermal_center[1] - path_start[1])

            if dist_to_thermal > search_distance:
                continue

            thermal_bearing_from_start = math.degrees(
                math.atan2(thermal_center[1] - path_start[1], thermal_center[0] - path_start[0]))
            if thermal_bearing_from_start < 0:
                thermal_bearing_from_start += 360

            angle_from_line = thermal_bearing_from_start - bearing_to_end_degrees
            if angle_from_line > 180:
                angle_from_line -= 360
            elif angle_from_line <= -180:
                angle_from_line += 360

            is_in_arc = abs(angle_from_line) <= arc_half_angle_degrees

            arc_line_upper_end = (path_start[0] + search_distance * math.cos(math.radians(arc_end_angle)),
                                  path_start[1] + search_distance * math.sin(math.radians(arc_end_angle)))
            arc_line_lower_end = (path_start[0] + search_distance * math.cos(math.radians(arc_start_angle)),
                                  path_start[1] + search_distance * math.sin(math.radians(arc_start_angle)))

            dist_to_upper_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_upper_end)
            dist_to_lower_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_lower_end)

            is_near_arc_edge = (dist_to_upper_line <= sniffing_radius_meters_base) or (
                    dist_to_lower_line <= sniffing_radius_meters_base)

            if (is_in_arc or is_near_arc_edge):
                if dist_to_thermal < min_dist_to_thermal:
                    min_dist_to_thermal = dist_to_thermal
                    nearest_thermal = thermal

        if nearest_thermal:
            thermal_center = nearest_thermal['center']
            time_to_thermal = min_dist_to_thermal / airspeed_ms
            altitude_drop = time_to_thermal * sink_rate_ms
            if current_altitude - altitude_drop < MIN_SAFE_ALTITUDE:
                return False
            current_altitude -= altitude_drop
            current_pos = thermal_center
            updraft_thermals_info.remove(nearest_thermal)
            if float(nearest_thermal['updraft_strength']) >= float(mc_for_sniffing_ms):
                current_altitude = z_cbl_meters
        else:
            time_to_travel = search_distance / airspeed_ms
            altitude_drop = time_to_travel * sink_rate_ms
            if current_altitude - altitude_drop < MIN_SAFE_ALTITUDE:
                return False
            current_altitude -= altitude_drop
            current_pos = path_end

    return current_altitude > MIN_SAFE_ALTITUDE


# --- Main execution block ---
if __name__ == '__main__':
    print("Choose an option:")
    print("1. Generate a single plot (visualize dynamic glider path)")
    print("2. Run Monte Carlo simulation (compute probability for a single scenario and export CSV)")

    choice = input("Enter 1 or 2: ")

    if choice == '1':
        print("\n--- Generating Single Plot with Dynamic Path Simulation ---")
        random_angle = random.uniform(0, 360)
        end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
        end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
        random_end_point = (end_point_x, end_point_y)
        simulate_dynamic_glide_path_and_draw(
            z_cbl_meters=SCENARIO_Z_CBL,
            mc_for_sniffing_ms=SCENARIO_MC_SNIFF,
            lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            lambda_strength=SCENARIO_LAMBDA_STRENGTH,
            end_point=random_end_point
        )
    elif choice == '2':
        num_simulations = 100000
        # Calculate dynamic parameters once for the summary printout
        airspeed, sink_rate, glide_ratio = calculate_glider_parameters_from_polar(SCENARIO_MC_SNIFF)
        print(f"\n--- Running Monte Carlo Simulation for a Single Scenario ({num_simulations} trials) ---")
        print(f"Scenario Parameters:")
        print(f"  Z (CBL Height): {SCENARIO_Z_CBL} m")
        print(f"  Pilot MC Sniff: {SCENARIO_MC_SNIFF} m/s")
        print(f"  Calculated Glide Ratio: {glide_ratio:.2f}:1")
        print(f"  Calculated Airspeed: {airspeed:.2f} m/s")
        print(f"  Calculated Sink Rate: {sink_rate:.2f} m/s")
        print(f"  Thermal Density (Lambda): {SCENARIO_LAMBDA_THERMALS_PER_SQ_KM} thermals/km²")
        print(f"  Thermal Strength Mean (Lambda): {SCENARIO_LAMBDA_STRENGTH} m/s (clamped 1-10 m/s)")
        print(f"  Search Arc Angle: +/- {SEARCH_ARC_ANGLE_DEGREES / 2}° (Total {SEARCH_ARC_ANGLE_DEGREES}°)")
        print("-" * 50)

        successful_flights = 0
        tqdm_desc = "Running Monte Carlo Trials"
        for _ in tqdm(range(num_simulations), desc=tqdm_desc):
            random_angle = random.uniform(0, 360)
            end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
            end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
            random_end_point = (end_point_x, end_point_y)
            if simulate_intercept_experiment_dynamic(
                    z_cbl_meters=SCENARIO_Z_CBL,
                    mc_for_sniffing_ms=SCENARIO_MC_SNIFF,
                    lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
                    lambda_strength=SCENARIO_LAMBDA_STRENGTH,
                    end_point=random_end_point
            ):
                successful_flights += 1

        probability = successful_flights / num_simulations

        all_results = [{
            'Z (m)': SCENARIO_Z_CBL,
            'MC_Sniff (m/s)': SCENARIO_MC_SNIFF,
            'Calc. Glide Ratio': glide_ratio,
            'Calc. Airspeed (m/s)': airspeed,
            'Calc. Sink Rate (m/s)': sink_rate,
            'Search Arc Angle (deg)': SEARCH_ARC_ANGLE_DEGREES,
            'Thermal Density (per km^2)': SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            'Thermal Strength Lambda': SCENARIO_LAMBDA_STRENGTH,
            'Successful Flights': successful_flights,
            'Probability': probability
        }]

        print("\n" + "=" * 120)
        print("\n--- Monte Carlo Simulation Results for Single Scenario ---")
        headers = [
            'Z (m)', 'MC_Sniff (m/s)', 'Calc. Glide Ratio', 'Calc. Airspeed (m/s)',
            'Calc. Sink Rate (m/s)', 'Search Arc Angle (deg)',
            'Thermal Density (per km^2)', 'Thermal Strength Lambda',
            'Successful Flights', 'Probability'
        ]

        print(
            f"{headers[0]:<8} | {headers[1]:<15} | {headers[2]:<20} | {headers[3]:<23} | {headers[4]:<23} |"
            f"{headers[5]:<23} | {headers[6]:<25} | {headers[7]:<25} |"
            f"{headers[8]:<25} | {headers[9]:<15}"
        )
        print("-" * 300)

        for row in all_results:
            print(
                f"{row['Z (m)']:<8} | {row['MC_Sniff (m/s)']:<15.1f} | {row['Calc. Glide Ratio']:<20.2f} | "
                f"{row['Calc. Airspeed (m/s)']:<23.2f} | {row['Calc. Sink Rate (m/s)']:<23.2f} |"
                f"{row['Search Arc Angle (deg)']:<23.2f} | "
                f"{row['Thermal Density (per km^2)']:<25.2f} | {row['Thermal Strength Lambda']:<25.1f} | "
                f"{row['Successful Flights']:<25} | {row['Probability']:<15.4f}"
            )

        csv_filename = "thermal_intercept_simulation_results_poisson_dist_arc_search_dynamic_path.csv"
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = headers
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_results:
                    writer.writerow(row)
            print(f"\nResults successfully exported to '{csv_filename}'")
        except IOError as e:
            print(f"\nError writing to CSV file '{csv_filename}': {e}")
    else:
        print("Invalid choice. Please enter 1 or 2.")
