#!/usr/bin/env python3

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
# - This script is designed to run a large-scale Monte Carlo simulation over a grid
#   of parameters and save the aggregated results to a CSV file.
#
# --- PLOTTING CHANGES (Per User Request) ---
# - Thermal Updrafts are now plotted as red circles with radii proportional to their strength.
# - Downdraft Annuli are now plotted as green circles around the updrafts.
# - Intercepted thermals that are NOT climbed are marked with a black 'X'.
# - Intercepted thermals that ARE climbed are marked with a red 'X'.
# -----------------------------

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
from tqdm import tqdm  # For progress bar
import csv  # For CSV export
import pandas as pd
import sys

# --- Constants ---
KNOT_TO_MS = 0.514444
FT_TO_M = 0.3048
MS_TO_KMH = 3.6  # Conversion constant for meters per second to kilometers per hour
C_UPDRAFT_STRENGTH_DECREMENT = 5.9899e-7
FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS = 1200.0
FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS = FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS / 2
EPSILON = 1e-9
MIN_SAFE_ALTITUDE = 500.0  # Minimum altitude for a safe landing

# --- LS10 Glider Polar Data ---
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
NUM_SIMULATIONS_PER_SCENARIO = 1000


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

    return airspeed_ms, sink_rate_ms, glide_ratio


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


def generate_hexagonal_thermals(sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength, z_cbl_meters):
    """
    Generates thermal locations in a hexagonal pattern with a random offset.
    """
    updraft_thermals = []
    hex_spacing = 1.5 * z_cbl_meters
    jitter_magnitude = hex_spacing * 0.1
    x_min, x_max = -sim_area_side_meters / 2, sim_area_side_meters / 2
    y_min, y_max = -sim_area_side_meters / 2, sim_area_side_meters / 2
    cell_area_sq_km = (hex_spacing * hex_spacing * math.sqrt(3) / 2) / 1e6
    probability = lambda_thermals_per_sq_km * cell_area_sq_km

    row_count = 0
    y = y_min
    while y < y_max:
        x = x_min
        if row_count % 2 == 1:
            x += hex_spacing / 2
        while x < x_max:
            if random.random() < probability:
                offset_x = jitter_magnitude * random.choice([-1, 1])
                offset_y = jitter_magnitude * random.choice([-1, 1])
                center_x = x + offset_x
                center_y = y + offset_y
                if x_min < center_x < x_max and y_min < center_y < y_max:
                    updraft_strength_magnitude = max(1, np.random.poisson(lambda_strength))
                    updraft_strength_magnitude = min(10, updraft_strength_magnitude)
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


# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson
import random
import time
import math
import concurrent.futures

# --- GLOBAL CONSTANTS ---
GRAVITY = 9.81
MAX_SIMULATION_TIME_SECONDS = 3600 * 24 * 10  # 10 days
GLOBAL_MIN_ALTITUDE = 500.0  # m
DOWNDRAFT_FACTOR = 0.5  # Fraction of updraft strength
DOWNDRAFT_RING_WIDTH = 250.0  # m
GLIDER_SINK_RATE = 1.0  # m/s
GLIDER_SPEED = 25.0  # m/s
MIN_THERMAL_STRENGTH = 1.0  # m/s
MAX_THERMAL_STRENGTH = 10.0  # m/s
UPWIND_DRIFT_FACTOR = 0.5  # The glider will drift upwind during a climb by this amount
GLIDER_PERFORMANCE_FACTOR = 1.0  # Efficiency of glider in turning
MIN_SEARCH_ARC_ANGLE = 15  # deg
MAX_SEARCH_ARC_ANGLE = 180  # deg


# --- THERMAL PLACEMENT FUNCTIONS ---
def generate_thermals_hexagonal(
        area_size,
        lambda_thermals,
        lambda_strength,
        num_thermals=None,
        random_offset_std_dev=200
):
    """
    Generates thermal locations in a hexagonal pattern with a random offset.

    Args:
        area_size (float): The side length of the square area in meters.
        lambda_thermals (float): The mean number of thermals per square km for the Bernoulli trial.
        lambda_strength (float): The lambda parameter for the Poisson distribution of thermal strength.
        num_thermals (int, optional): An explicit number of thermals to generate. Defaults to None.
        random_offset_std_dev (float, optional): Standard deviation of the random offset for each thermal. Defaults to 200.

    Returns:
        list: A list of thermal dictionaries.
    """
    thermals = []

    # Calculate grid spacing based on thermal density
    thermal_density_per_sq_m = lambda_thermals / 1_000_000.0
    s = np.sqrt(2.0 / (np.sqrt(3.0) * thermal_density_per_sq_m))  # Side length of a hexagon

    # Determine the number of grid points
    cols = int(area_size / s) + 2
    rows = int(area_size / (s * np.sqrt(3.0) / 2.0)) + 2

    for i in range(rows):
        for j in range(cols):
            x = j * s + (i % 2) * s * 0.5
            y = i * s * np.sqrt(3.0) / 2.0

            # Apply a Bernoulli trial to determine if a thermal exists at this grid point
            if np.random.rand() < (lambda_thermals * (s ** 2) / 1_000_000):
                # Apply random offset
                x += np.random.normal(0, random_offset_std_dev)
                y += np.random.normal(0, random_offset_std_dev)

                # Check if within the defined area
                if -area_size / 2 < x < area_size / 2 and -area_size / 2 < y < area_size / 2:
                    strength = poisson.rvs(lambda_strength)
                    strength = max(MIN_THERMAL_STRENGTH, min(MAX_THERMAL_STRENGTH, strength))
                    radius = strength * 100  # A simple model where radius scales with strength
                    thermals.append({
                        'x': x,
                        'y': y,
                        'strength': strength,
                        'radius': radius
                    })

    return thermals


# --- GLIDER FLIGHT FUNCTIONS ---
def calculate_glider_sink_rate(glider_speed):
    """
    A placeholder function for a more complex glider polar curve.
    For now, it returns a fixed sink rate.

    Args:
        glider_speed (float): The current airspeed of the glider in m/s.

    Returns:
        float: The sink rate in m/s.
    """
    return GLIDER_SINK_RATE


def calculate_distance(p1, p2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        p1 (tuple): A tuple containing the (x, y) coordinates of the first point.
        p2 (tuple): A tuple containing the (x, y) coordinates of the second point.

    Returns:
        float: The distance between the two points.
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_nearest_thermal_in_arc(glider_pos, thermals, search_arc_angle):
    """
    Finds the nearest thermal within the specified search arc.

    Args:
        glider_pos (tuple): (x, y) coordinates of the glider.
        thermals (list): List of thermal dictionaries.
        search_arc_angle (float): The total search arc angle in degrees.

    Returns:
        dict or None: The nearest thermal dictionary, or None if no thermals are found.
    """
    if not thermals:
        return None

    distances = []

    # Simple search for now, assuming glider is always heading along a straight line towards the end point
    # We will refine this later if needed.

    for thermal in thermals:
        dist = calculate_distance(glider_pos, (thermal['x'], thermal['y']))
        distances.append(dist)

    if not distances:
        return None

    nearest_thermal_index = np.argmin(distances)
    return thermals[nearest_thermal_index]

def simulate_intercept_experiment_dynamic(
        z_cbl_meters, lambda_thermals_per_sq_km, lambda_strength,
        mc_sniff_band1, mc_sniff_band2, end_point, search_arc_angle, plot_simulation=False, thermal_model='poisson'
):
    """
    Runs a single simulation of a glider's flight. The thermal_model parameter
    allows the user to choose between the 'poisson' and 'hexagonal' models.
    """
    path_points = [(0, 0)]
    current_pos = (0, 0)
    current_altitude = z_cbl_meters
    total_height_climbed = 0.0
    total_climbing_time = 0.0
    total_gliding_time = 0.0
    total_distance_covered = 0.0
    plot_padding = 20000.0

    max_coord = max(abs(end_point[0]), abs(end_point[1]))
    sim_area_side_meters = (max_coord + plot_padding) * 2

    if thermal_model == 'poisson':
        # Logic for Poisson thermal placement (not used in this version but kept for clarity)
        updraft_thermals_info = []
    elif thermal_model == 'hexagonal':
        updraft_thermals_info = generate_hexagonal_thermals(
            sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength, z_cbl_meters
        )
    else:
        raise ValueError("Invalid thermal_model specified. Choose 'poisson' or 'hexagonal'.")

    remaining_thermals = list(updraft_thermals_info)
    intercepted_thermals = []

    while math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]) > EPSILON:
        if current_altitude <= MIN_SAFE_ALTITUDE:
            straight_line_dist_origin = math.hypot(current_pos[0], current_pos[1])
            if plot_simulation:
                plot_simulation_results(
                    path_points, updraft_thermals_info, intercepted_thermals, end_point,
                    'Flight Failed: Altitude dropped below 500m'
                )
            return {'success': False, 'distance_to_land': straight_line_dist_origin,
                    'total_distance_covered': total_distance_covered}

        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        current_band, current_mc_sniff_ms, altitude_to_next_band, next_band_alt = get_band_info(current_altitude,
                                                                                                z_cbl_meters,
                                                                                                mc_sniff_band1,
                                                                                                mc_sniff_band2)
        airspeed_ms, sink_rate_ms, glide_ratio = get_glider_parameters(current_mc_sniff_ms)
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
                plot_simulation_results(
                    path_points, updraft_thermals_info, intercepted_thermals, end_point,
                    'Flight Succeeded'
                )
            return {
                'success': True,
                'total_height_climbed': total_height_climbed,
                'total_climbing_time': total_climbing_time,
                'total_gliding_time': total_gliding_time,
                'total_time': total_climbing_time + total_gliding_time,
                'straight_line_distance': math.hypot(end_point[0], end_point[1]),
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

                # Add to intercepted list
                intercepted_thermals.append({'center': nearest_thermal_in_arc['center'], 'climbed': True})
            else:
                intercepted_thermals.append({'center': nearest_thermal_in_arc['center'], 'climbed': False})

    if plot_simulation:
        plot_simulation_results(
            path_points, updraft_thermals_info, intercepted_thermals, end_point,
            'Flight Failed: Loop terminated without reaching destination'
        )
    return {
        'success': False,
        'distance_to_land': math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]),
        'total_distance_covered': total_distance_covered,
    }


def print_detailed_single_flight_results(result, initial_bearing, params):
    """
    Prints the detailed results of a single flight simulation.

    Args:
        result (dict): The result dictionary from a single simulation run.
        initial_bearing (float): The initial bearing of the glider.
        params (dict): The parameters used for the simulation.
    """
    print("\n--- Simulation Results ---")
    print(f"Initial Bearing: {initial_bearing:.2f} degrees")
    print(f"Initial Parameters: {params}")
    print(f"Total Flight Time: {result['total_time']:.2f} seconds")
    print(f"Final Altitude: {result['final_altitude']:.2f} meters")
    print(f"Thermal Intercepted: {'Yes' if result['thermal_found'] else 'No'}")
    if result['thermal_found']:
        print(f"Number of Path Segments: {len(result['path_points']) - 1}")


def run_monte_carlo_with_user_input():
    """
    Runs a Monte Carlo simulation based on user-provided parameters.
    """
    try:
        num_simulations = int(input("Enter number of simulations: "))
        z_cbl = float(input("Enter cloud base height (m): "))
        lambda_thermals = float(input("Enter thermal density (lambda per sq km): "))
        lambda_strength = float(input("Enter thermal strength (lambda): "))
        mc_band1 = float(input("Enter Macready setting for band 1 (m/s): "))
        mc_band2 = float(input("Enter Macready setting for band 2 (m/s): "))
        search_arc = float(input("Enter search arc angle (degrees): "))
        thermal_model = input("Enter thermal model ('poisson' or 'hexagonal'): ")

        # Validate inputs
        if not (0 <= search_arc <= 360):
            print("Search arc angle must be between 0 and 360 degrees.")
            return

        results = []
        start_time = time.time()

        for _ in range(num_simulations):
            end_point_x = np.random.uniform(-50000, 50000)
            end_point_y = np.random.uniform(-50000, 50000)
            end_point = (end_point_x, end_point_y)

            result = simulate_intercept_experiment_dynamic(
                z_cbl_meters=z_cbl,
                lambda_thermals_per_sq_km=lambda_thermals,
                lambda_strength=lambda_strength,
                mc_sniff_band1=mc_band1,
                mc_sniff_band2=mc_band2,
                end_point=end_point,
                search_arc_angle=search_arc,
                plot_simulation=False,
                thermal_model=thermal_model
            )
            results.append(result)

        end_time = time.time()
        print(f"\n--- Monte Carlo Simulation Complete ---")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        # Process and print results
        num_intercepts = sum(1 for r in results if r['thermal_found'])
        thermal_intercept_rate = (num_intercepts / num_simulations) * 100
        print(f"Number of Simulations: {num_simulations}")
        print(f"Thermal Intercept Rate: {thermal_intercept_rate:.2f}%")

        avg_time = np.mean([r['total_time'] for r in results if r['thermal_found']])
        print(f"Average time to complete (for successful flights): {avg_time:.2f} seconds")

    except ValueError:
        print("Invalid input. Please enter valid numbers.")


def run_nested_loop_simulation_and_save_to_csv(thermal_model_type):
    """
    Runs a nested loop simulation for different parameters and saves results to a CSV.

    Args:
        thermal_model_type (str): The thermal model to use ('poisson' or 'hexagonal').
    """
    print(f"\n--- Mode 3/4: Nested Loop Simulation ({thermal_model_type.capitalize()} Model) ---\n")

    # Define parameter ranges for the nested loops
    lambda_thermals_range = [0.01, 0.02]
    lambda_strength_range = [1.0, 2.0]
    mc_band1_range = [1.0, 2.0]
    mc_band2_range = [0.5, 1.0]
    search_arc_range = [30.0, 60.0]
    num_simulations_per_run = 100

    results_list = []

    # Use a ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for lambda_thermals in lambda_thermals_range:
            for lambda_strength in lambda_strength_range:
                for mc_band1 in mc_band1_range:
                    for mc_band2 in mc_band2_range:
                        for search_arc in search_arc_range:

                            # Create a list of futures to be executed
                            for _ in range(num_simulations_per_run):
                                end_point_x = np.random.uniform(-50000, 50000)
                                end_point_y = np.random.uniform(-50000, 50000)
                                end_point = (end_point_x, end_point_y)

                                future = executor.submit(
                                    simulate_intercept_experiment_dynamic,
                                    z_cbl_meters=2500,
                                    lambda_thermals_per_sq_km=lambda_thermals,
                                    lambda_strength=lambda_strength,
                                    mc_sniff_band1=mc_band1,
                                    mc_sniff_band2=mc_band2,
                                    end_point=end_point,
                                    search_arc_angle=search_arc,
                                    plot_simulation=False,
                                    thermal_model=thermal_model_type
                                )
                                futures.append(future)

        # Wait for all futures to complete and collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results_list.append(result)
            except Exception as e:
                print(f"A simulation run generated an exception: {e}")

    df = pd.DataFrame(results_list)
    filename = f"simulation_results_{thermal_model_type}_{int(time.time())}.csv"
    df.to_csv(filename, index=False)
    print(f"\n--- Nested loop simulation complete. Results saved to {filename} ---")
    print(f"Saved {len(df)} simulation results.")



def plot_simulation_results(path_points, updraft_thermals_info, intercepted_thermals, end_point, title):
    """
    Plots the glider's flight path and thermal locations with detailed visuals.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot thermals: updrafts (red) and downdrafts (green)
    plotted_updraft_label = False
    plotted_downdraft_label = False
    for thermal in updraft_thermals_info:
        updraft_radius = thermal['updraft_radius']
        center_x, center_y = thermal['center']

        # Plot updraft circle
        updraft_circle = patches.Circle(
            (center_x, center_y), updraft_radius, color='red', alpha=0.5,
            label='Thermal Updraft' if not plotted_updraft_label else ""
        )
        ax.add_patch(updraft_circle)
        plotted_updraft_label = True

        # Plot downdraft annulus
        downdraft_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS
        downdraft_circle = patches.Circle(
            (center_x, center_y), downdraft_radius, color='green', fill=False, linestyle='--',
            label='Downdraft Annulus' if not plotted_downdraft_label else ""
        )
        ax.add_patch(downdraft_circle)
        plotted_downdraft_label = True

    # Plot glider path, start, and destination
    path_points = np.array(path_points)
    ax.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=2, label='Glider Path')
    ax.plot(path_points[0, 0], path_points[0, 1], 'go', label='Start', markersize=10)
    ax.plot(end_point[0], end_point[1], 'ro', label='Destination', markersize=10)

    # Plot intercepted thermals
    plotted_climbed_label = False
    plotted_not_climbed_label = False
    for thermal in intercepted_thermals:
        center_x, center_y = thermal['center']
        if thermal['climbed']:
            ax.plot(center_x, center_y, 'rx', markersize=12, mew=2,
                    label='Climbed Thermal' if not plotted_climbed_label else "")
            plotted_climbed_label = True
        else:
            ax.plot(center_x, center_y, 'kx', markersize=12, mew=2,
                    label='Intercepted (Not Climbed)' if not plotted_not_climbed_label else "")
            plotted_not_climbed_label = True

    ax.set_title(title)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best')
    plt.show()


def run_single_simulation():
    """
    Prompts the user for parameters and runs a single simulation with a plot.
    """
    print("\n--- Single Simulation with Plot ---", flush=True)
    try:
        z_cbl = float(input(f"Enter Cloud Base Height (m) [default {DEFAULT_Z_CBL}]: ") or DEFAULT_Z_CBL)
        thermal_density = float(input(
            f"Enter Thermal Density (per km^2) [default {DEFAULT_LAMBDA_THERMALS_PER_SQ_KM}]: ") or DEFAULT_LAMBDA_THERMALS_PER_SQ_KM)
        thermal_strength = float(
            input(f"Enter Thermal Strength Lambda [default {DEFAULT_LAMBDA_STRENGTH}]: ") or DEFAULT_LAMBDA_STRENGTH)
        mc_band1 = float(
            input(f"Enter MC Sniff Band 1 (m/s) [default {DEFAULT_MC_SNIFF_BAND1}]: ") or DEFAULT_MC_SNIFF_BAND1)
        mc_band2 = float(
            input(f"Enter MC Sniff Band 2 (m/s) [default {DEFAULT_MC_SNIFF_BAND2}]: ") or DEFAULT_MC_SNIFF_BAND2)
        search_arc = float(input(
            f"Enter Search Arc Angle (deg) [default {DEFAULT_SEARCH_ARC_ANGLE_DEGREES}]: ") or DEFAULT_SEARCH_ARC_ANGLE_DEGREES)

        # Fixed end point for consistent plotting
        end_point = (RANDOM_END_POINT_DISTANCE, 0)

        print("\nStarting single simulation...", flush=True)
        result = simulate_intercept_experiment_dynamic(
            z_cbl_meters=z_cbl,
            lambda_thermals_per_sq_km=thermal_density,
            lambda_strength=thermal_strength,
            mc_sniff_band1=mc_band1,
            mc_sniff_band2=mc_band2,
            end_point=end_point,
            search_arc_angle=search_arc,
            plot_simulation=True,
            thermal_model='hexagonal'
        )

        print("\n--- Single Simulation Results ---", flush=True)
        if result['success']:
            print("Flight was successful!")
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            print("Flight failed.")
            print(f"Distance to land from start: {result['distance_to_land']:.2f} meters")
            print(f"Total distance covered: {result['total_distance_covered']:.2f} meters")

    except ValueError:
        print("Invalid input. Please enter a valid number.")


def run_default_monte_carlo_simulation():
    """
    Runs a Monte Carlo simulation with a fixed set of default parameters.
    """
    print("\n--- Monte Carlo Simulation with Default Parameters ---", flush=True)
    print(f"Running {NUM_SIMULATIONS_PER_SCENARIO} simulations with the following defaults:")
    print(f"  Z_CBL: {DEFAULT_Z_CBL} m")
    print(f"  Thermal Density: {DEFAULT_LAMBDA_THERMALS_PER_SQ_KM} per km^2")
    print(f"  Thermal Strength: {DEFAULT_LAMBDA_STRENGTH}")
    print(f"  MC Sniff Band 1: {DEFAULT_MC_SNIFF_BAND1} m/s")
    print(f"  MC Sniff Band 2: {DEFAULT_MC_SNIFF_BAND2} m/s")
    print(f"  Search Arc: {DEFAULT_SEARCH_ARC_ANGLE_DEGREES} degrees")

    successful_flights = 0
    successful_metrics = []
    failed_distances = []

    for _ in tqdm(range(NUM_SIMULATIONS_PER_SCENARIO), desc="Simulating"):
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
            search_arc_angle=DEFAULT_SEARCH_ARC_ANGLE_DEGREES,
            thermal_model='hexagonal'
        )

        if result['success']:
            successful_flights += 1
            successful_metrics.append(result)
        else:
            failed_distances.append(result['distance_to_land'])

    probability = successful_flights / NUM_SIMULATIONS_PER_SCENARIO
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

    print("\n--- Simulation Results (Default Parameters) ---")
    print(f"Successful Flights: {successful_flights}")
    print(f"Probability of Success: {probability:.4f}")
    print(f"Avg Total Height Climbed (m): {avg_Ht:.2f}")
    print(f"Avg Total Climbing Time (s): {avg_Tc:.2f}")
    print(f"Avg Total Gliding Time (s): {avg_Tg:.2f}")
    print(f"Avg Total Time (s): {avg_T:.2f}")
    print(f"Avg Rate of Climb (m/s): {avg_Wc:.2f}")
    print(f"Avg Speed Made Good (km/h): {avg_Vmg_kmh:.2f}")
    print(f"Avg Failed Distance (m): {avg_failed_dist:.2f}")


def run_nested_loop_simulation():
    """
    Runs a nested loop simulation for a specific set of parameters, saving results to a CSV.
    This function simulates the process that would have generated a file like the
    thermal_sim_results_hexagonal_pattern.csv you provided.
    """
    print("\n--- Nested Loop Simulation for CSV Output ---", flush=True)

    # Define the parameter ranges to loop through
    z_cbl_range = [2500, 3000]
    thermal_density_range = [0.000625, 0.01, 0.05]
    thermal_strength_range = [0.6638, 2.0, 4.0]
    mc_band1_range = [1.0, 2.0, 3.0, 4.0]
    mc_band2_range = [1.0, 2.0, 3.0, 4.0]
    search_arc_angle = 30.0  # Keeping this fixed for this simulation
    num_sims = NUM_SIMULATIONS_PER_SCENARIO

    csv_filename = "thermal_intercept_simulation_results_nested_loop.csv"
    headers = [
        'Z_CBL (m)', 'Thermal Density (per km^2)', 'Thermal Strength Lambda',
        'MC_SNIFF_BAND1 (m/s)', 'MC_SNIFF_BAND2 (m/s)', 'Search Arc Angle (deg)',
        'Successful Flights', 'Probability', 'Avg Total Height Climbed (m)',
        'Avg Total Climbing Time (s)', 'Avg Total Gliding Time (s)',
        'Avg Total Time (s)', 'Avg Rate of Climb (m/s)',
        'Avg Speed Made Good (km/h)', 'Avg Failed Distance (m)'
    ]

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        total_iterations = (
                len(z_cbl_range) * len(thermal_density_range) * len(thermal_strength_range) * len(mc_band1_range) * len(
            mc_band2_range)
        )
        pbar = tqdm(total=total_iterations, desc="Nested Loop Simulation")

        for z_cbl in z_cbl_range:
            for thermal_density in thermal_density_range:
                for thermal_strength in thermal_strength_range:
                    for mc_band1 in mc_band1_range:
                        for mc_band2 in mc_band2_range:
                            successful_flights = 0
                            successful_metrics = []
                            failed_distances = []

                            for _ in range(num_sims):
                                random_angle = random.uniform(0, 360)
                                end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
                                end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
                                random_end_point = (end_point_x, end_point_y)

                                result = simulate_intercept_experiment_dynamic(
                                    z_cbl_meters=z_cbl,
                                    lambda_thermals_per_sq_km=thermal_density,
                                    lambda_strength=thermal_strength,
                                    mc_sniff_band1=mc_band1,
                                    mc_sniff_band2=mc_band2,
                                    end_point=random_end_point,
                                    search_arc_angle=search_arc_angle,
                                    thermal_model='hexagonal'
                                )

                                if result['success']:
                                    successful_flights += 1
                                    successful_metrics.append(result)
                                else:
                                    failed_distances.append(result['distance_to_land'])

                            probability = successful_flights / num_sims
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

                            row = {
                                'Z_CBL (m)': z_cbl,
                                'Thermal Density (per km^2)': thermal_density,
                                'Thermal Strength Lambda': thermal_strength,
                                'MC_SNIFF_BAND1 (m/s)': mc_band1,
                                'MC_SNIFF_BAND2 (m/s)': mc_band2,
                                'Search Arc Angle (deg)': search_arc_angle,
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
                            writer.writerow(row)
                            pbar.update(1)
        pbar.close()
    print(f"\nNested loop simulation complete. Results saved to {csv_filename}")


def run_mc_sniff_varied_simulation():
    """
    Runs a simulation varying only the MC sniffing parameters.
    """
    print("\n--- Monte Carlo Simulation with Varying Sniffing MC ---", flush=True)

    # Define the parameter ranges to loop through
    mc_band1_range = [1.0, 2.0, 3.0, 4.0]
    mc_band2_range = [1.0, 2.0, 3.0, 4.0]

    # Fixed parameters
    z_cbl = DEFAULT_Z_CBL
    thermal_density = DEFAULT_LAMBDA_THERMALS_PER_SQ_KM
    thermal_strength = DEFAULT_LAMBDA_STRENGTH
    search_arc_angle = DEFAULT_SEARCH_ARC_ANGLE_DEGREES
    num_sims = NUM_SIMULATIONS_PER_SCENARIO

    csv_filename = "thermal_intercept_simulation_results_mc_sniff.csv"
    headers = [
        'MC_SNIFF_BAND1 (m/s)', 'MC_SNIFF_BAND2 (m/s)', 'Z_CBL (m)',
        'Thermal Density (per km^2)', 'Thermal Strength Lambda', 'Search Arc Angle (deg)',
        'Successful Flights', 'Probability', 'Avg Total Height Climbed (m)',
        'Avg Total Climbing Time (s)', 'Avg Total Gliding Time (s)',
        'Avg Total Time (s)', 'Avg Rate of Climb (m/s)',
        'Avg Speed Made Good (km/h)', 'Avg Failed Distance (m)'
    ]

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        total_iterations = len(mc_band1_range) * len(mc_band2_range)
        pbar = tqdm(total=total_iterations, desc="Varying Sniffing MC")

        for mc_band1 in mc_band1_range:
            for mc_band2 in mc_band2_range:
                successful_flights = 0
                successful_metrics = []
                failed_distances = []

                for _ in range(num_sims):
                    random_angle = random.uniform(0, 360)
                    end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
                    end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
                    random_end_point = (end_point_x, end_point_y)

                    result = simulate_intercept_experiment_dynamic(
                        z_cbl_meters=z_cbl,
                        lambda_thermals_per_sq_km=thermal_density,
                        lambda_strength=thermal_strength,
                        mc_sniff_band1=mc_band1,
                        mc_sniff_band2=mc_band2,
                        end_point=random_end_point,
                        search_arc_angle=search_arc_angle,
                        thermal_model='hexagonal'
                    )

                    if result['success']:
                        successful_flights += 1
                        successful_metrics.append(result)
                    else:
                        failed_distances.append(result['distance_to_land'])

                probability = successful_flights / num_sims
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

                row = {
                    'MC_SNIFF_BAND1 (m/s)': mc_band1,
                    'MC_SNIFF_BAND2 (m/s)': mc_band2,
                    'Z_CBL (m)': z_cbl,
                    'Thermal Density (per km^2)': thermal_density,
                    'Thermal Strength Lambda': thermal_strength,
                    'Search Arc Angle (deg)': search_arc_angle,
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
                writer.writerow(row)
                pbar.update(1)
        pbar.close()
    print(f"\nVarying sniffing MC simulation complete. Results saved to {csv_filename}")


def run_hexagonal_simulation_with_user_params():
    """
    Prompts the user for all parameters and runs a Monte Carlo simulation
    for a hexagonal thermal pattern with those parameters.
    """
    print("\n--- User-defined Hexagonal Pattern Simulation ---", flush=True)

    try:
        z_cbl = float(input(f"Enter Cloud Base Height (m) [default {DEFAULT_Z_CBL}]: ") or DEFAULT_Z_CBL)
        thermal_density = float(input(
            f"Enter Thermal Density (per km^2) [default {DEFAULT_LAMBDA_THERMALS_PER_SQ_KM}]: ") or DEFAULT_LAMBDA_THERMALS_PER_SQ_KM)
        thermal_strength = float(
            input(f"Enter Thermal Strength Lambda [default {DEFAULT_LAMBDA_STRENGTH}]: ") or DEFAULT_LAMBDA_STRENGTH)
        mc_band1 = float(
            input(f"Enter MC Sniff Band 1 (m/s) [default {DEFAULT_MC_SNIFF_BAND1}]: ") or DEFAULT_MC_SNIFF_BAND1)
        mc_band2 = float(
            input(f"Enter MC Sniff Band 2 (m/s) [default {DEFAULT_MC_SNIFF_BAND2}]: ") or DEFAULT_MC_SNIFF_BAND2)
        search_arc = float(input(
            f"Enter Search Arc Angle (deg) [default {DEFAULT_SEARCH_ARC_ANGLE_DEGREES}]: ") or DEFAULT_SEARCH_ARC_ANGLE_DEGREES)
        num_sims = int(input(
            f"Enter number of simulations [default {NUM_SIMULATIONS_PER_SCENARIO}]: ") or NUM_SIMULATIONS_PER_SCENARIO)
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return

    print("\nStarting Monte Carlo simulation with user-defined parameters...", flush=True)

    successful_flights = 0
    successful_metrics = []
    failed_distances = []

    for _ in tqdm(range(num_sims), desc="Simulating"):
        random_angle = random.uniform(0, 360)
        end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
        end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
        random_end_point = (end_point_x, end_point_y)

        result = simulate_intercept_experiment_dynamic(
            z_cbl_meters=z_cbl,
            lambda_thermals_per_sq_km=thermal_density,
            lambda_strength=thermal_strength,
            mc_sniff_band1=mc_band1,
            mc_sniff_band2=mc_band2,
            end_point=random_end_point,
            search_arc_angle=search_arc,
            thermal_model='hexagonal'
        )

        if result['success']:
            successful_flights += 1
            successful_metrics.append(result)
        else:
            failed_distances.append(result['distance_to_land'])

    probability = successful_flights / num_sims

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

    final_results = {
        'Z_CBL (m)': z_cbl,
        'Thermal Density (per km^2)': thermal_density,
        'Thermal Strength Lambda': thermal_strength,
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

    print("\n--- Simulation Results (User-defined Hexagonal Pattern) ---")
    for key, value in final_results.items():
        print(f"{key:<30}: {value}")

    csv_filename = "thermal_intercept_simulation_results_hexagons.csv"
    try:
        with open(csv_filename, 'a', newline='') as csvfile:
            headers = list(final_results.keys())
            writer = csv.DictWriter(csvfile, fieldnames=headers)

            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow(final_results)
        print(f"\nResults appended to {csv_filename}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")


def main():
    """
    Presents a menu to the user and calls the appropriate simulation function.
    """
    print("--- Monte Carlo Simulation for Glider Thermal Interception ---")
    print("Select a simulation option:")
    print("1. Single simulation with a visual plot and detailed printout")
    print("2. Monte Carlo simulation using a fixed set of default parameters")
    print("3. A nested loop simulation for a specific set of parameters, saving results to a CSV")
    print("4. Monte Carlo simulation with varying sniffing MC")
    print("5. User-defined parameters (Hexagonal)")
    print("6. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-6): ")
            if choice == '1':
                run_single_simulation()
            elif choice == '2':
                run_default_monte_carlo_simulation()
            elif choice == '3':
                run_nested_loop_simulation()
            elif choice == '4':
                run_mc_sniff_varied_simulation()
            elif choice == '5':
                run_hexagonal_simulation_with_user_params()
            elif choice == '6':
                print("Exiting.")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter a number from 1 to 6.")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)


if __name__ == "__main__":
    main()
