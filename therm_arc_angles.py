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
# - Flight Path: The glider's path is a straight line of a specified length,
#   originating from (0,0) and randomly oriented for each trial.
# - Interception Logic: An intercept is considered successful if a thermal's center
#   is located within a specified angular arc from the glide path's origin, or
#   if its sniffing radius overlaps with any of the arc's boundaries. The arc
#   is defined as +/- 15 degrees (total 30 degrees) relative to the glider's bearing.
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
# - SEARCH_ARC_ANGLE_DEGREES: The total angle of the search arc, centered on the glider's flight path.
#   A value of 30 degrees corresponds to +/- 15 degrees from the glider's bearing.
# - MAX_SEARCH_DISTANCE_METERS: The maximum distance the glider will search for thermals. This
#   simulates a practical limit to a cross-country flight.
# -------------------

# Comprehensive Monte Carlo simulation designed to calculate the probability of a glider intercepting an updraft thermal.
# This version uses a Poisson distribution for spatial thermal placement and for thermal strengths.
# All generated thermals are updrafts, with derived downdraft rings encircling them.
# IMPORTANT: The LENGTH of the glider's search path for thermals is a variable on line 35 in km.
# SIMILARLY: The LAMBDA variable sets the AMBIENT Thermal Strength
# AND: SCENARIO_MC_SNIFF sets the diameter of the search from the thermal centre, effectively the
# Macready setting and sensitivity for sniffing thermals

#########
# This is the BASIC MC sim on which more complex scripts in this repo are built.
# It can be used to calculate prob of intercepting a thermal with chosen parameter values
# Specifically, and for example, LAMBDA 4.0, LENGTH 10miles, DENSITY 0.3, yields prob 0.1, as per Cochrane 1999
########

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

# NEW CONSTANT: Search arc angle in degrees
SEARCH_ARC_ANGLE_DEGREES = 20.0

# --- Scenario Parameters (Moved to Global Scope for Easy Configuration) ---
# These parameters define the single simulation scenario.
SCENARIO_Z_CBL = 2500.0  # Convective Boundary Layer (CBL) height in meters
SCENARIO_GLIDE_RATIO = 40  # Glider's glide ratio (e.g., 40:1)
SCENARIO_MC_SNIFF = 2  # Pilot's Macready setting for sniffing in m/s
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.1  # Average number of thermals per square kilometer (Poisson lambda)
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


# --- Helper function to calculate distance from a point to a line segment ---
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

    # Max possible updraft radius (if Wt_ms=10)
    max_updraft_radius_possible = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)  # Approx 255.4m
    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS  # 600m

    # Simulation area side should cover the effective glide path plus max thermal/sniffing radius on both sides
    sim_area_side_meters = (
                                   effective_glide_path_length + max_thermal_system_radius * 2 + sniffing_radius_meters_base * 2) * 1.1  # Add 10% padding

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

    # Plot the central line of the glide path
    ax.plot(
        [line_start_x, line_end_x],
        [line_start_y, line_end_y],
        color='blue',
        linewidth=2,
        label=f'Glide Path ({glide_ratio}:1)'
    )
    ax.legend()

    # --- NEW AMENDMENT: Plotting the search arc
    arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2
    start_angle_rad = line_angle_radians - math.radians(arc_half_angle_degrees)
    end_angle_rad = line_angle_radians + math.radians(arc_half_angle_degrees)

    # Create a wedge patch for the search arc
    wedge_patch = patches.Wedge(
        (0, 0),  # Center
        effective_glide_path_length,  # Radius
        math.degrees(start_angle_rad),  # Start angle
        math.degrees(end_angle_rad),  # End angle
        width=effective_glide_path_length,  # Make it a full circle slice
        alpha=0.1,
        color='cyan',
        label=f'Search Arc (+/- {arc_half_angle_degrees}°)'
    )
    ax.add_patch(wedge_patch)

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

    # --- Perform Intersection Checks with the ARC boundaries ---
    updraft_intercepts_count = 0
    red_initial_intercept_data = []

    # Get the endpoints of the straight lines forming the arc boundaries
    arc_boundary_start = (line_start_x, line_start_y)
    arc_boundary_end_left_x = effective_glide_path_length * math.cos(
        line_angle_radians - math.radians(arc_half_angle_degrees))
    arc_boundary_end_left_y = effective_glide_path_length * math.sin(
        line_angle_radians - math.radians(arc_half_angle_degrees))
    arc_boundary_end_right_x = effective_glide_path_length * math.cos(
        line_angle_radians + math.radians(arc_half_angle_degrees))
    arc_boundary_end_right_y = effective_glide_path_length * math.sin(
        line_angle_radians + math.radians(arc_half_angle_degrees))

    arc_boundary_end_left = (arc_boundary_end_left_x, arc_boundary_end_left_y)
    arc_boundary_end_right = (arc_boundary_end_right_x, arc_boundary_end_right_y)

    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']
        updraft_radius = thermal_info['updraft_radius']

        intercept_distance_from_origin = math.sqrt(updraft_center[0] ** 2 + updraft_center[1] ** 2)
        intercept_bearing_degrees = math.degrees(math.atan2(updraft_center[1], updraft_center[0]))
        if intercept_bearing_degrees < 0:
            intercept_bearing_degrees += 360

        angle_from_line = intercept_bearing_degrees - line_bearing_degrees
        if angle_from_line > 180:
            angle_from_line -= 360
        elif angle_from_line <= -180:
            angle_from_line += 360

        # NEW CORRECTED LOGIC: Check if the thermal is inside the arc or within sniffing distance of its boundaries
        is_intercept = False

        # 1. Check if the thermal's center is directly within the arc
        if (abs(angle_from_line) <= arc_half_angle_degrees and
                intercept_distance_from_origin <= effective_glide_path_length):
            is_intercept = True

        # 2. Check for "sniffing distance" from the straight-line boundaries
        dist_to_left_line, _ = distance_from_point_to_line_segment(updraft_center, arc_boundary_start,
                                                                   arc_boundary_end_left)
        dist_to_right_line, _ = distance_from_point_to_line_segment(updraft_center, arc_boundary_start,
                                                                    arc_boundary_end_right)

        if dist_to_left_line <= sniffing_radius_meters_base or dist_to_right_line <= sniffing_radius_meters_base:
            is_intercept = True

        # 3. Check for "sniffing distance" from the curved boundary
        distance_to_curved_boundary = abs(intercept_distance_from_origin - effective_glide_path_length)
        if distance_to_curved_boundary <= sniffing_radius_meters_base and abs(
                angle_from_line) <= arc_half_angle_degrees:
            is_intercept = True

        if is_intercept:
            updraft_intercepts_count += 1
            # Plot the sniffing circle (transparent purple outline)
            sniffing_circle_patch = patches.Circle(
                updraft_center, sniffing_radius_meters_base, color='purple', fill=False, alpha=0.1, linestyle='--',
                linewidth=0.5
            )
            ax.add_patch(sniffing_circle_patch)

            # Plot an intercept marker at the thermal's center
            # REVERTING THE CHANGE: The user wanted a standard 'x', not a red 'X'.
            ax.plot(updraft_center[0], updraft_center[1], 'x', color='black', markersize=10,
                    markeredgecolor='black', linewidth=1.5)

            red_initial_intercept_data.append({
                'distance': intercept_distance_from_origin,
                'bearing': intercept_bearing_degrees,
                'angle_from_line': angle_from_line
            })

    # Sort the updraft intercepts by distance for clear output
    red_initial_intercept_data.sort(key=lambda x: x['distance'])

    # --- Construct the footer text for the plot ---
    red_dist_str = "None"
    if red_initial_intercept_data:
        red_dist_str = ", ".join(
            [f"{d['distance']:.0f}m ({d['bearing']:.0f}° from origin, {d['angle_from_line']:.0f}° from line)" for d in
             red_initial_intercept_data])

    footer_text = (
        f"Z={z_cbl_meters}m, Glide Path: {glide_ratio}:1, Length={glide_path_horizontal_length_meters / 1000:.1f}km\n"
        f"Search Limit: {MAX_SEARCH_DISTANCE_METERS / 1000:.0f}km, Line Bearing: {line_bearing_degrees:.2f}°\n"
        f"Thermal Density: {lambda_thermals_per_sq_km}/km², Avg Strength: {lambda_strength} (1-10m/s)\n"
        f"Sniffing Radius (Base): {sniffing_radius_meters_base:.0f}m (MC={mc_for_sniffing_ms}m/s)\n"
        f"Search Arc Angle: +/- {arc_half_angle_degrees}° (Total {SEARCH_ARC_ANGLE_DEGREES}°)\n"
        f"Updraft Intercept Distances (to thermal center) & Bearings: {red_dist_str}"
    )

    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=9, color='gray')

    print(f"Total updrafts generated: {len(updraft_thermals_info)}")
    print(f"\nIntercepts/Encounters with search arc:")
    print(f"  - Updraft Intercepts: {updraft_intercepts_count}")

    if red_initial_intercept_data:
        print(
            f"\nInitial Intercept Data for Updrafts (from origin to thermal center) with Line Bearing: {line_bearing_degrees:.2f}°:")
        for i, data in enumerate(red_initial_intercept_data):
            print(
                f"  Intercept {i + 1}: {data['distance']:.2f} m, Bearing: {data['bearing']:.2f}° (Relative to line: {data['angle_from_line']:.2f}°)")
    else:
        print("\nNo initial intercepts for Updrafts.")

    # --- Adjust Plot Limits to fit everything ---
    # The plot limits should be based on the simulation area side, centered at (0,0)
    # This ensures the entire generated thermal field is visible and aligned with the border.
    plot_limit_extent = sim_area_side_meters / 2
    plot_padding_factor = 0.05  # Add 5% padding to the limits for better visual spacing

    # --- UPDATED CODE: Ensure the thermal grid fills the entire plot by using a fixed 10km padding
    # to extend the axes beyond the glide path and the original sim area.
    # The user-requested change to extend the axes 10km beyond the glidepath is already implemented.
    # Now, we ensure the thermal generation area matches these extended plot limits.
    plot_limit = max(effective_glide_path_length, plot_limit_extent) + 10000

    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    plt.show()


def simulate_intercept_experiment_poisson(
        z_cbl_meters, glide_ratio, mc_for_sniffing_ms,
        lambda_thermals_per_sq_km, lambda_strength
):
    """
    Performs a single Monte Carlo experiment with Poisson-distributed updraft thermals
    to check for an intercept with an updraft (red) thermal's sniffing radius.
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

    # --- Determine Simulation Area Side Length for Poisson Process ---
    # Max possible updraft radius (if Wt_ms=10)
    max_updraft_radius_possible = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)  # Approx 255.4m
    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS  # 600m

    # Simulation area side should cover the effective glide path plus max thermal/sniffing radius on both sides
    sim_area_side_meters = (
                                   effective_glide_path_length + max_thermal_system_radius * 2 + sniffing_radius_meters_base * 2) * 1.1  # Add 10% padding

    # --- Generate Updraft Thermals (Poisson Distribution) ---
    updraft_thermals = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    # --- Glide Path Line Calculation (new random path each time for each trial, using effective length) ---
    line_angle_radians = random.uniform(0, 2 * math.pi)
    line_start_x, line_start_y = 0, 0  # Glide path starts at origin of simulation area
    line_end_x = line_start_x + effective_glide_path_length * math.cos(line_angle_radians)
    line_end_y = line_start_y + effective_glide_path_length * math.sin(line_angle_radians)

    # Store line bearing
    line_bearing_degrees = math.degrees(math.atan2(line_end_y - line_start_y, line_end_x - line_start_x))
    if line_bearing_degrees < 0:
        line_bearing_degrees += 360

    arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2
    # Get the endpoints of the straight lines forming the arc boundaries
    arc_boundary_start = (line_start_x, line_start_y)
    arc_boundary_end_left_x = effective_glide_path_length * math.cos(
        line_angle_radians - math.radians(arc_half_angle_degrees))
    arc_boundary_end_left_y = effective_glide_path_length * math.sin(
        line_angle_radians - math.radians(arc_half_angle_degrees))
    arc_boundary_end_right_x = effective_glide_path_length * math.cos(
        line_angle_radians + math.radians(arc_half_angle_degrees))
    arc_boundary_end_right_y = effective_glide_path_length * math.sin(
        line_angle_radians + math.radians(arc_half_angle_degrees))

    arc_boundary_end_left = (arc_boundary_end_left_x, arc_boundary_end_left_y)
    arc_boundary_end_right = (arc_boundary_end_right_x, arc_boundary_end_right_y)

    # --- Check Intersections with the ARC boundaries ---
    for thermal_info in updraft_thermals:
        updraft_center = thermal_info['center']

        intercept_distance_from_origin = math.sqrt(updraft_center[0] ** 2 + updraft_center[1] ** 2)
        intercept_bearing_degrees = math.degrees(math.atan2(updraft_center[1], updraft_center[0]))
        if intercept_bearing_degrees < 0:
            intercept_bearing_degrees += 360

        angle_from_line = intercept_bearing_degrees - line_bearing_degrees
        if angle_from_line > 180:
            angle_from_line -= 360
        elif angle_from_line <= -180:
            angle_from_line += 360

        # Check if the thermal is inside the arc or within sniffing distance of its boundaries
        # 1. Check if the thermal's center is directly within the arc
        if (abs(angle_from_line) <= arc_half_angle_degrees and
                intercept_distance_from_origin <= effective_glide_path_length):
            return True

        # 2. Check for "sniffing distance" from the straight-line boundaries
        dist_to_left_line, _ = distance_from_point_to_line_segment(updraft_center, arc_boundary_start,
                                                                   arc_boundary_end_left)
        dist_to_right_line, _ = distance_from_point_to_line_segment(updraft_center, arc_boundary_start,
                                                                    arc_boundary_end_right)

        if dist_to_left_line <= sniffing_radius_meters_base or dist_to_right_line <= sniffing_radius_meters_base:
            return True

        # 3. Check for "sniffing distance" from the curved boundary
        distance_to_curved_boundary = abs(intercept_distance_from_origin - effective_glide_path_length)
        if distance_to_curved_boundary <= sniffing_radius_meters_base and abs(
                angle_from_line) <= arc_half_angle_degrees:
            return True

    return False  # No intercept found


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
        num_simulations = 100000  # Number of trials for the Monte Carlo simulation

        print(f"\n--- Running Monte Carlo Simulation for a Single Scenario ({num_simulations} trials) ---")
        print(f"Scenario Parameters:")
        print(f"  Z (CBL Height): {SCENARIO_Z_CBL} m")
        print(f"  Glide Ratio: {SCENARIO_GLIDE_RATIO}:1")
        print(f"  Pilot MC Sniff: {SCENARIO_MC_SNIFF} m/s")
        print(f"  Thermal Density (Lambda): {SCENARIO_LAMBDA_THERMALS_PER_SQ_KM} thermals/km²")
        print(f"  Thermal Strength Mean (Lambda): {SCENARIO_LAMBDA_STRENGTH} m/s (clamped 1-10 m/s)")
        print(f"  Search Arc Angle: +/- {SEARCH_ARC_ANGLE_DEGREES / 2}° (Total {SEARCH_ARC_ANGLE_DEGREES}°)")
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

        all_results = [{
            'Z (m)': SCENARIO_Z_CBL,
            'Wt_Ambient (m/s)': SCENARIO_LAMBDA_STRENGTH,  # Using lambda_strength as proxy for ambient Wt
            'MC_Sniff (m/s)': SCENARIO_MC_SNIFF,
            'Sniffing Radius (Base)(m)': calculated_sniffing_radius,
            'Search Arc Angle (deg)': SEARCH_ARC_ANGLE_DEGREES,
            'Glide Path Length (m)': reported_glide_path_length,  # Use reported length
            'Thermal Density (per km^2)': SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            'Thermal Strength Lambda': SCENARIO_LAMBDA_STRENGTH,
            'Probability': probability
        }]

        print("\n" + "=" * 120)
        print("\n--- Monte Carlo Simulation Results for Single Scenario ---")
        headers = [
            'Z (m)', 'Wt_Ambient (m/s)', 'MC_Sniff (m/s)', 'Sniffing Radius (Base)(m)',
            'Search Arc Angle (deg)',
            'Glide Path Length (m)', 'Thermal Density (per km^2)', 'Thermal Strength Lambda',
            'Probability'
        ]

        # Print headers
        print(
            f"{headers[0]:<8} | {headers[1]:<18} | {headers[2]:<15} | {headers[3]:<28} | {headers[4]:<23} |"
            f"{headers[5]:<23} | {headers[6]:<25} | {headers[7]:<25} | {headers[8]:<15}"
        )
        print("-" * 300)

        for row in all_results:
            print(
                f"{row['Z (m)']:<8} | {row['Wt_Ambient (m/s)']:<18.1f} | {row['MC_Sniff (m/s)']:<15.1f} | {row['Sniffing Radius (Base)(m)']:<28.2f} | "
                f"{row['Search Arc Angle (deg)']:<23.2f} | "
                f"{row['Glide Path Length (m)']:<23.2f} | {row['Thermal Density (per km^2)']:<25.2f} | {row['Thermal Strength Lambda']:<25.1f} | {row['Probability']:<15.4f}"
            )

        # --- Export results to CSV file ---
        csv_filename = "thermal_intercept_simulation_results_poisson_dist_arc_search_amended_corrected_v2.csv"
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