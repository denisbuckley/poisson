# --- SCRIPT CHARACTERISTICS ---
# This is a comprehensive Monte Carlo simulation script for a glider's thermal interception.
# The simulation environment models a 2D plane with Poisson-distributed thermals.
# Each thermal consists of a core updraft region and an encircling downdraft ring.
# The script can run in two modes:
#   1. A single-run visualization mode that generates a plot of the thermal field,
#      the glider's flight path, and any intercepts.
#   2. A large-scale Monte Carlo simulation mode that calculates the statistical
#      probability of a successful thermal intercept over many trials.
#
# Key Features:
# - Thermal Placement: Uses a Poisson distribution to place thermals randomly and
#   realistically throughout the simulation area.
# - Thermal Properties: Thermal updraft strength is also Poisson-distributed
#   (clamped 1-10 m/s), and the updraft radius is derived from this strength.
# - Downdraft Modeling: A fixed-diameter downdraft ring encircles each updraft.
# - Flight Path: The glider's path is a straight-line corridor of a specified width,
#   originating from (0,0) and randomly oriented for each trial.
# - Interception Logic: An intercept is considered successful if the glider's
#   flight path corridor overlaps with the thermal's "sniffing radius," which is
#   a combination of a pilot-defined Macready setting and the corridor width.
# - Data Output: Generates detailed printouts of intercept data (distance, bearing,
#   and relative angle to the flight path) for the visualization mode and a CSV
#   file with statistical results for the Monte Carlo mode.
# - Modularity: The script is designed with a clear separation of concerns,
#   using functions for thermal generation, intersection checks, and visualization,
#   making it easy to understand and modify parameters.
# -----------------------------

# --- USER INPUTS ---
# This script is configured with the following user-settable parameters for defining a simulation scenario:
# - SCENARIO_Z_CBL: The height of the Convective Boundary Layer (CBL) in meters. This is a crucial factor
#   for determining the maximum possible glide distance and the simulation's scale.
# - SCENARIO_GLIDE_RATIO: The glider's glide ratio (e.g., 40:1). This directly affects the horizontal
#   distance the glider can travel from a given altitude.
# - SCENARIO_MC_SNIFF: The pilot's Macready setting for sniffing in m/s. This value influences the size of the
#   "sniffing radius" around each thermal, which determines how far from a thermal the glider can detect it.
# - SCENARIO_LAMBDA_THERMALS_PER_SQ_KM: The average number of thermals per square kilometer. This is the
#   Poisson distribution's lambda value for thermal density, controlling how many thermals are generated.
# - SCENARIO_LAMBDA_STRENGTH: The mean strength of the thermals in m/s. This is the Poisson distribution's
#   lambda value for thermal strength, controlling the average updraft strength. The strength is capped at 10 m/s.
# - GLIDEPATH_CORRIDOR_WIDTH_METERS: The width of the glider's flight path corridor in meters. This
#   parameter, along with the Macready sniffing setting, defines the total intercept radius.
# - MAX_SEARCH_DISTANCE_METERS: The maximum distance the glider will search for thermals. This
#   simulates a practical limit to a cross-country flight.
# -------------------

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
from tqdm import tqdm  # For progress bar
import csv  # For CSV export

# --- Constants ---
KNOT_TO_MS = 0.514444  # 1 knot = 0.514444 m/s
FT_TO_M = 0.3048  # 1 foot = 0.3048 m

# Constant 'C' derived from the updraft strength formula: y_m/s = C * x_m^3
# Where C = 0.033 * KNOT_TO_MS / (100 * FT_TO_M)**3
C_UPDRAFT_STRENGTH_DECREMENT = 5.9952e-7  # Approximately 5.995248074266928e-07

# Fixed outer diameter for the entire thermal system (updraft + downdraft ring)
# This means the downdraft ring extends from the updraft's edge to this outer radius.
FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS = 1200.0
FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS = FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS / 2  # 600m

# Constant for downdraft strength calculation: W_down = K * Wt_ms^(5/3)
# Where K = 0.042194 (derived from previous discussions)
K_DOWNDRAFT_STRENGTH = 0.042194

# Global epsilon for floating point comparisons
EPSILON = 1e-9

# NEW CONSTANT: Maximum distance the glider will search for thermals
MAX_SEARCH_DISTANCE_METERS = 162000.0  # 100 miles (approx 162 km)

# --- AMENDED CODE: New constant for the glidepath corridor width ---
GLIDEPATH_CORRIDOR_WIDTH_METERS = 2000.0


# --- Scenario Parameters (Moved to Global Scope for Easy Configuration) ---
# These parameters define the single simulation scenario.
SCENARIO_Z_CBL = 2500.0  # Convective Boundary Layer (CBL) height in meters
SCENARIO_GLIDE_RATIO = 40  # Glider's glide ratio (e.g., 40:1)
SCENARIO_MC_SNIFF = 2  # Pilot's Macready setting for sniffing in m/s
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.2  # Average number of thermals per square kilometer (Poisson lambda)
SCENARIO_LAMBDA_STRENGTH = 3  # Mean strength of thermals (Poisson lambda, clamped 1-10 m/s)


# --- Helper function for calculating sniffing radius ---
def calculate_sniffing_radius(Wt_ms_ambient, MC_for_sniffing_ms, thermal_type="NORMAL"):
    """
    Calculates the sniffing radius based on ambient thermal strength (Wt_ms_ambient)
    and the pilot's Macready setting for sniffing (MC_for_sniffing_ms).

    Args:
        Wt_ms_ambient (float): Ambient Thermal Strength in m/s.
        MC_for_sniffing_ms (float): Macready speed in m/s, used for sniffing radius calculation.
        thermal_type (str): "NORMAL" or "NARROW" (affects C_thermal constant).

    Returns:
        float: The calculated sniffing radius in meters.
    """
    C_thermal = 0.033 if thermal_type == "NORMAL" else 0.10

    Wt_knots_for_sniff = Wt_ms_ambient / KNOT_TO_MS
    MC_knots_for_sniff = MC_for_sniffing_ms / KNOT_TO_MS
    y_MC_knots_for_sniff = Wt_knots_for_sniff - MC_knots_for_sniff

    # R_sniffing_feet is the radius where the Macready speed equals the ambient Wt
    # Handle cases where y_MC_knots_for_sniff / C_thermal is non-positive
    if y_MC_knots_for_sniff / C_thermal > 0:
        R_sniffing_feet = 100 * ((y_MC_knots_for_sniff / C_thermal) ** (1 / 3))
    else:
        R_sniffing_feet = 0

    D_sniffing_meters = 2 * (R_sniffing_feet * FT_TO_M)
    sniffing_radius_meters = D_sniffing_meters / 2

    return sniffing_radius_meters


# --- AMENDED: NEW Helper function to calculate distance from a point to a line segment ---
def distance_from_point_to_line_segment(point, line_start, line_end):
    """
    Calculates the minimum distance from a point to a line segment and
    returns the closest point on the line segment.

    Args:
        point (tuple): (px, py) coordinates of the point.
        line_start (tuple): (x1, y1) coordinates of the line segment's start.
        line_end (tuple): (x2, y2) coordinates of the line segment's end.

    Returns:
        tuple: (distance, closest_point) where distance is the shortest distance
               and closest_point is a tuple (cx, cy) of the coordinates.
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

    # Calculate the projection of the point onto the line
    t = ((px - x1) * dx + (py - y1) * dy) / line_segment_length_sq
    t = max(0, min(1, t))  # Clamp t to the range [0, 1]

    # Find the closest point on the line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Calculate the distance
    distance = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    return distance, (closest_x, closest_y)


def generate_poisson_updraft_thermals(sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength):
    """
    Generates only updraft thermal positions and properties using a Poisson distribution.
    Downdraft rings are derived from these updrafts later.

    Args:
        sim_area_side_meters (float): The side length of the square simulation area in meters.
        lambda_thermals_per_sq_km (float): The average number of thermals per square kilometer.
        lambda_strength (float): The mean (lambda) for the Poisson distribution of thermal strength magnitude.

    Returns:
        list: A list of dictionaries, each representing an updraft thermal with its properties.
              [{'center': (x, y), 'updraft_radius': r, 'updraft_strength': s}, ...]
    """
    sim_area_sq_km = (sim_area_side_meters / 1000) ** 2
    expected_num_thermals = lambda_thermals_per_sq_km * sim_area_sq_km
    num_thermals = np.random.poisson(expected_num_thermals)

    updraft_thermals = []
    for _ in range(num_thermals):
        # Generate random position within the simulation area (centered at 0,0)
        center_x = random.uniform(-sim_area_side_meters / 2, sim_area_side_meters / 2)
        center_y = random.uniform(-sim_area_side_meters / 2, sim_area_side_meters / 2)

        # Generate strength magnitude (Poisson distributed, clamped to 1-10)
        updraft_strength_magnitude = 0
        while updraft_strength_magnitude == 0:  # Re-roll if 0 is generated, ensure min strength of 1
            updraft_strength_magnitude = np.random.poisson(lambda_strength)
        updraft_strength_magnitude = min(10, updraft_strength_magnitude)  # Cap at 10 m/s

        # Calculate updraft radius where strength goes to zero based on formula
        # Wt_ms = C * R_up^3 => R_up = (Wt_ms / C)^(1/3)
        updraft_radius = (updraft_strength_magnitude / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)

        updraft_thermals.append({
            'center': (center_x, center_y),
            'updraft_radius': updraft_radius,
            'updraft_strength': updraft_strength_magnitude
        })
    return updraft_thermals


def draw_poisson_thermals_and_glide_path_with_intercept_check(
        z_cbl_meters, glide_ratio, mc_for_sniffing_ms,
        lambda_thermals_per_sq_km, lambda_strength,
        fig_width=12, fig_height=12
):
    """
    Draws a single visualization of the thermal grid (Poisson distributed),
    with updrafts and encircling downdraft rings, glide path, and intercepts.
    The glide path search is limited to MAX_SEARCH_DISTANCE_METERS.
    """
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.set_aspect('equal')

    # --- Calculations for Glide Path Length ---
    available_glide_height = z_cbl_meters - 500  # Glider needs to start landing at 500 AGL
    if available_glide_height <= 0:
        print(
            f"Warning: Z = {z_cbl_meters}m results in non-positive available glide height ({available_glide_height}m). Cannot draw meaningful glide path.")
        plt.title(f"Cannot draw glide path for Z={z_cbl_meters}m (Available Height <= 0m AGL)")
        plt.show()
        return

    glide_path_horizontal_length_meters = available_glide_height * glide_ratio

    # NEW: Limit the effective glide path length for search and plotting
    effective_glide_path_length = min(glide_path_horizontal_length_meters, MAX_SEARCH_DISTANCE_METERS)

    # --- Sniffing radius calculation uses lambda_strength as proxy for ambient Wt for sniffing calc
    ambient_wt_for_sniff_calc = lambda_strength
    sniffing_radius_meters_base = calculate_sniffing_radius(
        ambient_wt_for_sniff_calc, mc_for_sniffing_ms
    )
    if sniffing_radius_meters_base <= 0:
        print("Warning: Calculated Macready sniffing radius is non-positive. Setting to 1m for visualization.")
        sniffing_radius_meters_base = 1.0

    # The intercept radius is the sum of the base sniffing radius and half the corridor width
    intercept_radius = sniffing_radius_meters_base + (GLIDEPATH_CORRIDOR_WIDTH_METERS / 2)


    # Max possible updraft radius (if Wt_ms=10)
    max_updraft_radius_possible = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)  # Approx 255.4m
    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS  # 600m

    # Simulation area side should cover the effective glide path plus max thermal/sniffing radius on both sides
    sim_area_side_meters = (
                                       effective_glide_path_length + max_thermal_system_radius * 2 + intercept_radius * 2) * 1.1  # Add 10% padding

    # --- Generate Updraft Thermals (Poisson Distribution) ---
    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    # --- Plotting the Glide Path Line (using effective length) ---
    line_angle_radians = random.uniform(0, 2 * math.pi)
    line_start_x, line_start_y = 0, 0
    line_end_x = line_start_x + effective_glide_path_length * math.cos(line_angle_radians)
    line_end_y = line_start_y + effective_glide_path_length * math.sin(line_angle_radians)
    # AMENDED CODE: Store line bearing for later use
    line_bearing_degrees = math.degrees(math.atan2(line_end_y - line_start_y, line_end_x - line_start_x))
    if line_bearing_degrees < 0:
        line_bearing_degrees += 360

    # --- AMENDED CODE: Plotting the glidepath corridor as a semi-transparent rectangle ---
    half_width = GLIDEPATH_CORRIDOR_WIDTH_METERS / 2
    path_dx = line_end_x - line_start_x
    path_dy = line_end_y - line_start_y
    path_len = math.sqrt(path_dx**2 + path_dy**2)

    if path_len > 0:
        # Calculate the perpendicular vector
        perp_dx = -path_dy / path_len
        perp_dy = path_dx / path_len

        # Define the four corners of the rectangle
        corner1 = (line_start_x - half_width * perp_dx, line_start_y - half_width * perp_dy)
        corner2 = (line_start_x + half_width * perp_dx, line_start_y + half_width * perp_dy)
        corner3 = (line_end_x + half_width * perp_dx, line_end_y + half_width * perp_dy)
        corner4 = (line_end_x - half_width * perp_dx, line_end_y - half_width * perp_dy)

        glide_path_corridor = patches.Polygon([corner1, corner2, corner3, corner4],
                                                closed=True, color='blue', alpha=0.1, label='Glide Path Corridor')
        ax.add_patch(glide_path_corridor)

    # Plot the central line of the glide path (zero-width) for reference
    ax.plot(
        [line_start_x, line_end_x],
        [line_start_y, line_end_y],
        color='blue',
        linewidth=2,
        label=f'Glide Path ({glide_ratio}:1)'
    )
    ax.legend()


    # --- Plot Thermals (Updrafts and Encircling Downdrafts) ---
    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']
        updraft_radius = thermal_info['updraft_radius']
        updraft_strength = thermal_info['updraft_strength']

        # Plot Updraft (red circle)
        updraft_circle = patches.Circle(
            updraft_center,
            updraft_radius,
            facecolor='red',
            alpha=0.6,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(updraft_circle)

        # Plot Encircling Downdraft (green annulus)
        # Downdraft inner radius is updraft_radius, outer is fixed_thermal_system_outer_radius
        downdraft_inner_radius = updraft_radius
        downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

        # Only draw if the downdraft ring is valid (outer radius > inner radius)
        if downdraft_outer_radius > downdraft_inner_radius:
            downdraft_annulus = patches.Circle(
                updraft_center,
                downdraft_outer_radius,
                facecolor='green',
                alpha=0.05,  # Very transparent to show underlying red
                edgecolor='green',
                linewidth=0.5,
                fill=True,
                hatch='/'  # Add hatching for better visual distinction
            )
            ax.add_patch(downdraft_annulus)

    # --- Perform Intersection Checks ---
    updraft_intercepts_count = 0
    downdraft_encounters_count = 0
    # Changed to store a list of dicts to also include the thermal's position
    red_initial_intercept_data = []
    green_initial_intercept_distances_meters = []  # For downdraft encounters

    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']
        updraft_radius = thermal_info['updraft_radius']

        distance_to_glidepath_line, closest_point = distance_from_point_to_line_segment(updraft_center, (line_start_x, line_start_y), (line_end_x, line_end_y))

        # The intercept is a success if the distance is within the thermal's sniffing radius + half the corridor width
        if distance_to_glidepath_line <= intercept_radius:
            updraft_intercepts_count += 1
            # Plot the sniffing circle (transparent purple outline), which now includes the corridor width
            sniffing_circle_patch = patches.Circle(
                updraft_center, intercept_radius, color='purple', fill=False, alpha=0.1, linestyle='--',
                linewidth=0.5
            )
            ax.add_patch(sniffing_circle_patch)

            # AMENDED CODE: Calculate distance and bearing from line origin (0,0) to thermal center
            intercept_distance_from_origin = math.sqrt(updraft_center[0]**2 + updraft_center[1]**2)
            intercept_bearing_degrees = math.degrees(math.atan2(updraft_center[1], updraft_center[0]))
            if intercept_bearing_degrees < 0:
                intercept_bearing_degrees += 360

            # NEW AMENDMENT: Calculate the angle from the line bearing to the updraft bearing
            angle_from_line = intercept_bearing_degrees - line_bearing_degrees
            # Normalize angle to be in the range (-180, 180] for clarity
            if angle_from_line > 180:
                angle_from_line -= 360
            elif angle_from_line <= -180:
                angle_from_line += 360

            red_initial_intercept_data.append({
                'distance': intercept_distance_from_origin,
                'bearing': intercept_bearing_degrees,
                'angle_from_line': angle_from_line
            })

        # Check for Downdraft Annulus Encounter
        downdraft_inner_radius = updraft_radius
        downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

        # A downdraft encounter happens if the distance to the glidepath is within the downdraft's outer radius
        # but outside the updraft's inner radius.
        if downdraft_outer_radius > downdraft_inner_radius: # Only check if annulus is valid
            distance_to_downdraft_annulus_outer, closest_point_downdraft = distance_from_point_to_line_segment(updraft_center, (line_start_x, line_start_y), (line_end_x, line_end_y))

            # An encounter with the downdraft annulus occurs if the closest point of the glide path is within the
            # downdraft annulus.
            downdraft_encounter_check_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS + (GLIDEPATH_CORRIDOR_WIDTH_METERS / 2)
            downdraft_inner_check_radius = updraft_radius + (GLIDEPATH_CORRIDOR_WIDTH_METERS / 2)

            if distance_to_downdraft_annulus_outer <= downdraft_encounter_check_radius and distance_to_downdraft_annulus_outer > downdraft_inner_check_radius:
                downdraft_encounters_count += 1
                green_initial_intercept_distances_meters.append(distance_to_downdraft_annulus_outer)
                # Plot the downdraft intercept marker (e.g., 'o' in green)
                # This marker will be plotted at the closest point on the line segment
                closest_point = closest_point_downdraft
                ax.plot(closest_point[0], closest_point[1], 'o', color='green', markersize=8, markeredgecolor='black', linewidth=1.0)


    # Sort the updraft intercepts by distance for clear output
    red_initial_intercept_data.sort(key=lambda x: x['distance'])
    green_initial_intercept_distances_meters.sort()

    # --- Construct the footer text for the plot ---
    red_dist_str = "None"
    if red_initial_intercept_data:
        # AMENDED CODE: Updated text to reflect the bearing is to the thermal center and include relative angle
        red_dist_str = ", ".join([f"{d['distance']:.0f}m ({d['bearing']:.0f}° from origin, {d['angle_from_line']:.0f}° from line)" for d in red_initial_intercept_data])
    green_dist_str = "None"
    if green_initial_intercept_distances_meters:
        green_dist_str = ", ".join([f"{d:.0f}m" for d in green_initial_intercept_distances_meters])

    # NEW AMENDMENT: Add line bearing to the footer
    footer_text = (
        f"Z={z_cbl_meters}m, Glide Path: {glide_ratio}:1, Length={glide_path_horizontal_length_meters / 1000:.1f}km\n"
        f"Search Limit: {MAX_SEARCH_DISTANCE_METERS / 1000:.0f}km, Line Bearing: {line_bearing_degrees:.2f}°\n"
        f"Thermal Density: {lambda_thermals_per_sq_km}/km², Avg Strength: {lambda_strength} (1-10m/s)\n"
        f"Sniffing Radius (Base): {sniffing_radius_meters_base:.0f}m (MC={mc_for_sniffing_ms}m/s)\n"
        f"Glide Path Corridor Width: {GLIDEPATH_CORRIDOR_WIDTH_METERS}m\n"
        f"Updraft Intercept Distances (to thermal center) & Bearings: {red_dist_str}\n"
        f"Downdraft Encounter Distances: {green_dist_str}"
    )

    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=9, color='gray')

    print(f"Total updrafts generated: {len(updraft_thermals_info)}")
    print(f"\nIntercepts/Encounters with glide path (sniffing radius for updrafts):")
    print(f"  - Updraft Sniffing Intercepts: {updraft_intercepts_count}")
    print(f"  - Downdraft Annulus Encounters: {downdraft_encounters_count}")

    if red_initial_intercept_data:
        # AMENDED CODE: Updated text to reflect the bearing is to the thermal center and include relative angle
        print(f"\nInitial Intercept Data for Updrafts (from origin to thermal center) with Line Bearing: {line_bearing_degrees:.2f}°:")
        for i, data in enumerate(red_initial_intercept_data):
            print(f"  Intercept {i + 1}: {data['distance']:.2f} m, Bearing: {data['bearing']:.2f}° (Relative to line: {data['angle_from_line']:.2f}°)")
    else:
        print("\nNo initial intercepts for Updrafts.")

    if green_initial_intercept_distances_meters:
        print("\nInitial Encounter Distances for Downdraft Annuli (meters):")
        for i, dist in enumerate(green_initial_intercept_distances_meters):
            print(f"  Encounter {i + 1}: {dist:.2f} m")
    else:
        print("\nNo initial encounters for Downdraft Annuli.")

    # --- Adjust Plot Limits to fit everything ---
    # The plot limits should be based on the simulation area side, centered at (0,0)
    # This ensures the entire generated thermal field is visible and aligned with the border.
    plot_limit_extent = sim_area_side_meters / 2
    plot_padding_factor = 0.05  # Add 5% padding to the limits for better visual spacing

    ax.set_xlim(-(plot_limit_extent * (1 + plot_padding_factor)), plot_limit_extent * (1 + plot_padding_factor))
    ax.set_ylim(-(plot_limit_extent * (1 + plot_padding_factor)), plot_limit_extent * (1 + plot_padding_factor))

    plt.show()


def simulate_intercept_experiment_poisson(
        z_cbl_meters, glide_ratio, mc_for_sniffing_ms,
        lambda_thermals_per_sq_km, lambda_strength
):
    """
    Performs a single Monte Carlo experiment with Poisson-distributed updraft thermals
    to check for an intercept with an updraft (red) thermal's sniffing radius.
    Downdraft rings are implicitly modeled but do not affect success/failure here.
    The glide path search is limited to MAX_SEARCH_DISTANCE_METERS.
    """
    # --- Glide Path Length Calculation ---
    available_glide_height = z_cbl_meters - 500  # Glider starts landing at 500 AGL
    if available_glide_height <= 0:
        return False  # Cannot glide if starting below or at landing height

    glide_path_horizontal_length_meters = available_glide_height * glide_ratio

    if glide_path_horizontal_length_meters <= 0:
        return False

    # NEW: Limit the effective glide path length for search
    effective_glide_path_length = min(glide_path_horizontal_length_meters, MAX_SEARCH_DISTANCE_METERS)

    # --- Sniffing Radius Calculation (based on ambient Wt and pilot's MC) ---
    # Use lambda_strength as a proxy for ambient Wt for sniffing calc
    ambient_wt_for_sniff_calc = lambda_strength
    sniffing_radius_meters_base = calculate_sniffing_radius(
        ambient_wt_for_sniff_calc, mc_for_sniffing_ms
    )
    if sniffing_radius_meters_base <= 0:
        return False  # No intercept possible if sniffing radius is non-positive

    # The intercept radius is the sum of the base sniffing radius and half the corridor width
    intercept_radius = sniffing_radius_meters_base + (GLIDEPATH_CORRIDOR_WIDTH_METERS / 2)


    # --- Determine Simulation Area Side Length for Poisson Process ---
    # Max possible updraft radius (if Wt_ms=10)
    max_updraft_radius_possible = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)  # Approx 255.4m
    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS  # 600m

    # Simulation area side should cover the effective glide path plus max thermal/sniffing radius on both sides
    sim_area_side_meters = (
                                       effective_glide_path_length + max_thermal_system_radius * 2 + intercept_radius * 2) * 1.1  # Add 10% padding

    # --- Generate Updraft Thermals (Poisson Distribution) ---
    updraft_thermals = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    # --- Glide Path Line Calculation (new random path each time for each trial, using effective length) ---
    line_angle_radians = random.uniform(0, 2 * math.pi)
    line_start_x, line_start_y = 0, 0  # Glide path starts at origin of simulation area
    line_end_x = line_start_x + effective_glide_path_length * math.cos(line_angle_radians)
    line_end_y = line_start_y + effective_glide_path_length * math.sin(line_angle_radians)

    # --- Check Intersections for Updraft Sniffing ---
    for thermal_info in updraft_thermals:
        updraft_center = thermal_info['center']
        # --- AMENDED CODE: Check if the distance from the thermal center to the line is within the new, combined radius ---
        distance_to_glidepath_line, closest_point = distance_from_point_to_line_segment(updraft_center, (line_start_x, line_start_y), (line_end_x, line_end_y))

        if distance_to_glidepath_line <= intercept_radius:
            return True  # Found an intercept with an updraft's sniffing radius, trial is a success

    return False  # No intercept with an updraft's sniffing radius found in this trial


# --- Main execution block ---
if __name__ == '__main__':
    print("Choose an option:")
    print("1. Generate a single plot (visualize Poisson-distributed thermals with encircling downdrafts)")
    print("2. Run Monte Carlo simulation (compute probability for a single scenario and export CSV)")

    choice = input("Enter 1 or 2: ")

    # The scenario parameters are now defined globally at the top of the script.
    # We reference them directly here.

    if choice == '1':
        print("\n--- Generating Single Plot with Poisson Thermals (Updrafts with Encircling Downdrafts) ---")
        draw_poisson_thermals_and_glide_path_with_intercept_check(
            z_cbl_meters=SCENARIO_Z_CBL,
            glide_ratio=SCENARIO_GLIDE_RATIO,
            mc_for_sniffing_ms=SCENARIO_MC_SNIFF,
            lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            lambda_strength=SCENARIO_LAMBDA_STRENGTH
        )

    elif choice == '2':
        num_simulations = 100000 # Number of trials for the Monte Carlo simulation

        print(f"\n--- Running Monte Carlo Simulation for a Single Scenario ({num_simulations} trials) ---")
        print(f"Scenario Parameters:")
        print(f"  Z (CBL Height): {SCENARIO_Z_CBL} m")
        print(f"  Glide Ratio: {SCENARIO_GLIDE_RATIO}:1")
        print(f"  Pilot MC Sniff: {SCENARIO_MC_SNIFF} m/s")
        print(f"  Thermal Density (Lambda): {SCENARIO_LAMBDA_THERMALS_PER_SQ_KM} thermals/km²")
        print(f"  Thermal Strength Mean (Lambda): {SCENARIO_LAMBDA_STRENGTH} m/s (clamped 1-10 m/s)")
        print(f"  Glidepath Corridor Width: {GLIDEPATH_CORRIDOR_WIDTH_METERS} m")
        print("-" * 50)

        intercept_count = 0
        tqdm_desc = "Running Monte Carlo Trials"
        for _ in tqdm(range(num_simulations), desc=tqdm_desc):
            if simulate_intercept_experiment_poisson(
                    z_cbl_meters=SCENARIO_Z_CBL,
                    glide_ratio=SCENARIO_GLIDE_RATIO,
                    mc_for_sniffing_ms=SCENARIO_MC_SNIFF,
                    lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
                    lambda_strength=SCENARIO_LAMBDA_STRENGTH
            ):
                intercept_count += 1

        probability = intercept_count / num_simulations

        # Calculate the sniffing radius for display in results
        calculated_sniffing_radius = calculate_sniffing_radius(
            SCENARIO_LAMBDA_STRENGTH, SCENARIO_MC_SNIFF
        )
        # Calculate the glide path length for display in results
        available_glide_height = SCENARIO_Z_CBL - 500
        current_glide_path_length_meters = available_glide_height * SCENARIO_GLIDE_RATIO

        # Ensure the reported glide path length in results reflects the search limit
        reported_glide_path_length = min(current_glide_path_length_meters, MAX_SEARCH_DISTANCE_METERS)

        # Calculate the effective sniffing radius for reporting
        reported_effective_sniffing_radius = calculated_sniffing_radius + (GLIDEPATH_CORRIDOR_WIDTH_METERS / 2)

        all_results = [{
            'Z (m)': SCENARIO_Z_CBL,
            'Wt_Ambient (m/s)': SCENARIO_LAMBDA_STRENGTH,  # Using lambda_strength as proxy for ambient Wt
            'MC_Sniff (m/s)': SCENARIO_MC_SNIFF,
            'Sniffing Radius (Base)(m)': calculated_sniffing_radius,
            'Glide Path Corridor Width (m)': GLIDEPATH_CORRIDOR_WIDTH_METERS,
            'Sniffing Radius (Effective)(m)': reported_effective_sniffing_radius,
            'Glide Path Length (m)': reported_glide_path_length,  # Use reported length
            'Thermal Density (per km^2)': SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            'Thermal Strength Lambda': SCENARIO_LAMBDA_STRENGTH,
            'Probability': probability
        }]

        print("\n" + "=" * 120)
        print("\n--- Monte Carlo Simulation Results for Single Scenario ---")
        headers = [
            'Z (m)', 'Wt_Ambient (m/s)', 'MC_Sniff (m/s)', 'Sniffing Radius (Base)(m)',
            'Glide Path Corridor Width (m)', 'Sniffing Radius (Effective)(m)',
            'Glide Path Length (m)', 'Thermal Density (per km^2)', 'Thermal Strength Lambda',
            'Probability'
        ]

        # Print headers
        print(
            f"{headers[0]:<8} | {headers[1]:<18} | {headers[2]:<15} | {headers[3]:<28} | {headers[4]:<30} | {headers[5]:<30} |"
            f"{headers[6]:<23} | {headers[7]:<25} | {headers[8]:<25} | {headers[9]:<15}"
        )
        print("-" * 300)

        for row in all_results:
            print(
                f"{row['Z (m)']:<8} | {row['Wt_Ambient (m/s)']:<18.1f} | {row['MC_Sniff (m/s)']:<15.1f} | {row['Sniffing Radius (Base)(m)']:<28.2f} | "
                f"{row['Glide Path Corridor Width (m)']:<30.2f} | {row['Sniffing Radius (Effective)(m)']:<30.2f} | "
                f"{row['Glide Path Length (m)']:<23.2f} | {row['Thermal Density (per km^2)']:<25.2f} | {row['Thermal Strength Lambda']:<25.1f} | {row['Probability']:<15.4f}"
            )

        # --- Export results to CSV file ---
        csv_filename = "thermal_intercept_simulation_results_poisson_dist_encircling_amended_corrected.csv"
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