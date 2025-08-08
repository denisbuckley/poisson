# --- SCRIPT CHARACTERISTICS ---
# This is a comprehensive Monte Carlo simulation script for a glider's thermal interception.
# The simulation environment models a 2D plane with Poisson-distributed thermals.
# Each thermal consists of a core updraft region and an encircling downdraft ring.
# This version is designed to run a large-scale Monte Carlo simulation over a grid
# of parameters and save the aggregated results to a CSV file.
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
# - The script iterates over a defined grid of Z_CBL, Thermal Density, Thermal
#   Strength, Macready settings for two bands, and search arc angle.
# - Results for each parameter combination are collected and saved to a single CSV file.
# -----------------------------

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
from tqdm import tqdm
import csv
import pandas as pd

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

# --- LS10 Glider Polar Data (from thermal_sim_poisson_segmented.py) ---
# Glider polar data points: airspeed vs sink rate.
# The `pol_v` is airspeed in knots, `pol_w` is sink rate in knots.
pol_v = np.array([45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120])
pol_w = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.5, 6.0])

# --- Default Scenario Parameters ---
DEFAULT_Z_CBL = 2500.0
DEFAULT_MC_SNIFF_BAND1 = 4.0
DEFAULT_MC_SNIFF_BAND2 = 2.0
DEFAULT_LAMBDA_THERMALS_PER_SQ_KM = 0.01
DEFAULT_LAMBDA_STRENGTH = 2
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


# --- Main Dynamic Simulation Function for Monte Carlo (Option 2) ---
def simulate_intercept_experiment_dynamic(
        z_cbl_meters, lambda_thermals_per_sq_km, lambda_strength,
        mc_sniff_band1, mc_sniff_band2, end_point, search_arc_angle
):
    """
    Runs a single, non-visual simulation of a glider's flight and returns key metrics.
    """
    plot_padding = 20000.0
    max_coord = max(abs(end_point[0]), abs(end_point[1]))
    sim_area_side_meters = (max_coord + plot_padding) * 2

    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    current_pos = (0, 0)
    current_altitude = z_cbl_meters
    total_height_climbed = 0.0
    total_climbing_time = 0.0
    total_gliding_time = 0.0
    total_distance_covered = 0.0

    remaining_thermals = list(updraft_thermals_info)

    while math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]) > EPSILON:
        if current_altitude <= MIN_SAFE_ALTITUDE:
            straight_line_dist_origin = math.hypot(current_pos[0], current_pos[1])
            return {'success': False, 'distance_to_land': straight_line_dist_origin}

        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        current_band, current_mc_sniff_ms, altitude_to_next_band, next_band_alt = get_band_info(current_altitude,
                                                                                                z_cbl_meters,
                                                                                                mc_sniff_band1,
                                                                                                mc_sniff_band2)
        airspeed_ms, sink_rate_ms, glide_ratio, _, _ = get_glider_parameters(current_mc_sniff_ms)

        direct_glide_dist = (current_altitude - MIN_SAFE_ALTITUDE) * glide_ratio

        if distance_to_end < direct_glide_dist:
            time_to_travel = distance_to_end / airspeed_ms
            altitude_drop = time_to_travel * sink_rate_ms
            current_altitude -= altitude_drop
            total_gliding_time += time_to_travel
            total_distance_covered += distance_to_end

            final_altitude = current_altitude
            straight_line_dist_origin = math.hypot(end_point[0], end_point[1])

            return {
                'success': True,
                'total_height_climbed': total_height_climbed,
                'total_climbing_time': total_climbing_time,
                'total_gliding_time': total_gliding_time,
                'total_time': total_climbing_time + total_gliding_time,
                'straight_line_distance': straight_line_dist_origin
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
                        path_start[1] + travel_distance * math.sin(bearing_to_end_radians))

        time_to_travel = travel_distance / airspeed_ms
        altitude_drop = time_to_travel * sink_rate_ms

        current_altitude -= altitude_drop
        current_pos = next_pos
        total_gliding_time += time_to_travel
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

    return {
        'success': True,
        'total_height_climbed': total_height_climbed,
        'total_climbing_time': total_climbing_time,
        'total_gliding_time': total_gliding_time,
        'total_time': total_climbing_time + total_gliding_time,
        'straight_line_distance': math.hypot(end_point[0], end_point[1])
    }


if __name__ == '__main__':
    # Define the parameter grid
    z_cbl_values = np.arange(1000, 3001, 500)
    lambda_thermals_values = np.arange(0.01, 0.11, 0.01)
    lambda_strength_values = np.arange(1, 6, 1)
    mc_band1_values = np.arange(1, 6, 1)
    mc_band2_values = np.arange(1, 6, 1)
    search_arc_values = np.arange(0, 61, 10)

    num_simulations_per_scenario = 1000
    all_scenario_results = []

    print("--- Starting Monte Carlo Grid Search ---")

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
    output_filename = "thermal_sim_results_grid_search.csv"
    results_df.to_csv(output_filename, index=False)

    print(f"Results successfully exported to '{output_filename}'")