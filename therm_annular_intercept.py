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
MAX_SEARCH_DISTANCE_METERS = 1620.0  # 10 kilometers

# --- Scenario Parameters (Moved to Global Scope for Easy Configuration) ---
# These parameters define the single simulation scenario.
SCENARIO_Z_CBL = 2500.0  # Convective Boundary Layer (CBL) height in meters
SCENARIO_GLIDE_RATIO = 40  # Glider's glide ratio (e.g., 40:1)
SCENARIO_MC_SNIFF = 4  # Pilot's Macready setting for sniffing in m/s
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.2  # Average number of thermals per square kilometer (Poisson lambda)
SCENARIO_LAMBDA_STRENGTH = 8  # Mean strength of thermals (Poisson lambda, clamped 1-10 m/s)


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


# --- Helper function for circle-line segment intersection check ---
def check_circle_line_segment_intersection(circle_center, radius, line_start, line_end):
    """
    Checks if a circle intersects with a line segment.

    Args:
        circle_center (tuple): (fx, fy) coordinates of the circle's center.
        radius (float): Radius of the circle.
        line_start (tuple): (x1, y1) coordinates of the line segment's start.
        line_end (tuple): (x2, y2) coordinates of the line segment's end.

    Returns:
        bool: True if the circle intersects the line segment, False otherwise.
        list: A list of intersection points (x, y) if any, otherwise empty.
    """
    fx, fy = circle_center
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1

    # Coefficients for the quadratic equation At^2 + Bt + C = 0
    A = dx ** 2 + dy ** 2
    B = 2 * (dx * (x1 - fx) + dy * (y1 - fy))
    C = (x1 - fx) ** 2 + (y1 - fy) ** 2 - radius ** 2

    discriminant = B ** 2 - 4 * A * C

    if discriminant < 0:
        return False, []  # No real solutions, so no intersection

    # Handle cases where A is very small (line is a point or very short segment)
    if A < EPSILON:  # If line segment is effectively a point
        distance_sq_to_center = (x1 - fx) ** 2 + (y1 - fy) ** 2
        return distance_sq_to_center <= radius ** 2, [(x1, y1)] if distance_sq_to_center <= radius ** 2 else []

    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)

    intersection_points = []
    # Using the global EPSILON for floating point comparisons
    epsilon = EPSILON  # Local alias for clarity

    # Check if intersection points lie on the segment (0 <= t <= 1)
    if -epsilon <= t1 <= 1 + epsilon:
        ix1 = x1 + t1 * dx
        iy1 = y1 + t1 * dy
        intersection_points.append((ix1, iy1))

    # Add t2 only if it's distinct from t1 and on the segment
    if -epsilon <= t2 <= 1 + epsilon and abs(t1 - t2) > epsilon:
        ix2 = x1 + t2 * dx
        iy2 = y1 + t2 * dy
        intersection_points.append((ix2, iy2))

    return len(intersection_points) > 0, intersection_points


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

    # --- Determine Simulation Area Side Length ---
    # Sniffing radius calculation uses lambda_strength as proxy for ambient Wt for sniffing calc
    ambient_wt_for_sniff_calc = lambda_strength
    sniffing_radius_meters = calculate_sniffing_radius(
        ambient_wt_for_sniff_calc, mc_for_sniffing_ms
    )
    if sniffing_radius_meters <= 0:
        print("Warning: Calculated Macready sniffing radius is non-positive. Setting to 1m for visualization.")
        sniffing_radius_meters = 1.0

    # Max possible updraft radius (if Wt_ms=10)
    max_updraft_radius_possible = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)  # Approx 255.4m
    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS  # 600m

    # Simulation area side should cover the effective glide path plus max thermal/sniffing radius on both sides
    sim_area_side_meters = (
                                       effective_glide_path_length + max_thermal_system_radius * 2 + sniffing_radius_meters * 2) * 1.1  # Add 10% padding

    # --- Generate Updraft Thermals (Poisson Distribution) ---
    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    # --- Plotting the Glide Path Line (using effective length) ---
    line_angle_radians = random.uniform(0, 2 * math.pi)
    line_start_x, line_start_y = 0, 0
    line_end_x = line_start_x + effective_glide_path_length * math.cos(line_angle_radians)
    line_end_y = line_start_y + effective_glide_path_length * math.sin(line_angle_radians)

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

            # Mask out the inner updraft area from the downdraft annulus for cleaner visualization
            mask_circle = patches.Circle(
                updraft_center,
                downdraft_inner_radius,
                facecolor='white',  # Match background or make fully transparent
                alpha=1.0,
                edgecolor='none',
                zorder=2  # Ensure it's above the downdraft annulus
            )
            # ax.add_patch(mask_circle) # This doesn't work well for masking in matplotlib directly for patches.
            # A better way would be to use PathPatch or a custom collection.
            # For simplicity, we'll rely on the alpha of the downdraft.
            # The red circle will be drawn over it.

    # --- Perform Intersection Checks ---
    updraft_intercepts_count = 0
    downdraft_encounters_count = 0
    red_initial_intercept_distances_meters = []
    green_initial_intercept_distances_meters = []  # For downdraft encounters

    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']
        updraft_radius = thermal_info['updraft_radius']
        updraft_strength = thermal_info['updraft_strength']

        # Check for Updraft Sniffing Intercept
        intersects_sniffing, sniff_intersection_pts = check_circle_line_segment_intersection(
            updraft_center, sniffing_radius_meters, (line_start_x, line_start_y), (line_end_x, line_end_y)
        )

        if intersects_sniffing:
            updraft_intercepts_count += 1
            # Plot the sniffing circle (transparent purple outline)
            sniffing_circle_patch = patches.Circle(
                updraft_center, sniffing_radius_meters, color='purple', fill=False, alpha=0.1, linestyle='--',
                linewidth=0.5
            )
            ax.add_patch(sniffing_circle_patch)

            # Find the closest intercept point for the updraft
            closest_pt_to_origin = None
            min_dist_sq = float('inf')
            for pt in sniff_intersection_pts:
                current_dist_sq = (pt[0] - line_start_x) ** 2 + (pt[1] - line_start_y) ** 2
                if current_dist_sq < min_dist_sq:
                    min_dist_sq = current_dist_sq
                    closest_pt_to_origin = pt

            if closest_pt_to_origin:
                ax.plot(closest_pt_to_origin[0], closest_pt_to_origin[1], 'X', color='red', markersize=10,
                        markeredgecolor='black', linewidth=1.5)
                red_initial_intercept_distances_meters.append(math.sqrt(min_dist_sq))

        # Check for Downdraft Annulus Encounter
        downdraft_inner_radius = updraft_radius
        downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

        if downdraft_outer_radius > downdraft_inner_radius:  # Only check if annulus is valid
            intersects_outer_ring, outer_pts = check_circle_line_segment_intersection(
                updraft_center, downdraft_outer_radius, (line_start_x, line_start_y), (line_end_x, line_end_y)
            )
            intersects_inner_core, inner_pts = check_circle_line_segment_intersection(
                updraft_center, downdraft_inner_radius, (line_start_x, line_start_y), (line_end_x, line_end_y)
            )

            # An encounter with the downdraft annulus occurs if the line intersects the outer boundary
            # AND (it does not intersect the inner boundary OR the intersection point on the outer boundary
            # is outside the inner boundary). This logic is simplified for visualization.
            # The most robust check for annulus intersection is complex.
            # For plotting, we'll mark if the glide path is within the annulus.
            downdraft_encountered_in_plot = False
            if intersects_outer_ring:
                for pt in outer_pts:
                    dist_from_center = math.sqrt((pt[0] - updraft_center[0]) ** 2 + (pt[1] - updraft_center[1]) ** 2)
                    if dist_from_center > downdraft_inner_radius + EPSILON:  # Check if point is truly in annulus
                        downdraft_encountered_in_plot = True
                        break
                # Also check if line endpoints are within the annulus
                dist_start_from_center = math.sqrt(
                    (line_start_x - updraft_center[0]) ** 2 + (line_start_y - updraft_center[1]) ** 2)
                dist_end_from_center = math.sqrt(
                    (line_end_x - updraft_center[0]) ** 2 + (line_end_y - updraft_center[1]) ** 2)
                if (downdraft_inner_radius <= dist_start_from_center <= downdraft_outer_radius) or \
                        (downdraft_inner_radius <= dist_end_from_center <= downdraft_outer_radius):
                    downdraft_encountered_in_plot = True

            if downdraft_encountered_in_plot:
                downdraft_encounters_count += 1
                # Plot the downdraft intercept marker (e.g., 'o' in green)
                # Find the closest point on the glide path that is within the annulus
                closest_pt_on_path_in_annulus = None
                min_dist_sq_to_path_start = float('inf')

                # Iterate through points on the line segment (simplified for visual)
                num_segments = 100
                for i in range(num_segments + 1):
                    t = i / num_segments
                    px = line_start_x + t * (line_end_x - line_start_x)
                    py = line_start_y + t * (line_end_y - line_start_y)
                    dist_from_center = math.sqrt((px - updraft_center[0]) ** 2 + (py - updraft_center[1]) ** 2)

                    if downdraft_inner_radius <= dist_from_center <= downdraft_outer_radius:
                        current_dist_sq = (px - line_start_x) ** 2 + (py - line_start_y) ** 2
                        if current_dist_sq < min_dist_sq_to_path_start:
                            min_dist_sq_to_path_start = current_dist_sq
                            closest_pt_on_path_in_annulus = (px, py)

                if closest_pt_on_path_in_annulus:
                    ax.plot(closest_pt_on_path_in_annulus[0], closest_pt_on_path_in_annulus[1], 'o', color='green',
                            markersize=8, markeredgecolor='black', linewidth=1.0)
                    green_initial_intercept_distances_meters.append(math.sqrt(min_dist_sq_to_path_start))

    red_initial_intercept_distances_meters.sort()
    green_initial_intercept_distances_meters.sort()

    # --- Construct the footer text for the plot ---
    red_dist_str = "None"
    if red_initial_intercept_distances_meters:
        red_dist_str = ", ".join([f"{d:.0f}m" for d in red_initial_intercept_distances_meters])
    green_dist_str = "None"
    if green_initial_intercept_distances_meters:
        green_dist_str = ", ".join([f"{d:.0f}m" for d in green_initial_intercept_distances_meters])

    # Updated footer text to reflect the search limit
    footer_text = (
        f"Z={z_cbl_meters}m, Glide Path: {glide_ratio}:1, Length={glide_path_horizontal_length_meters / 1000:.1f}km\n"
        f"Search Limit: {MAX_SEARCH_DISTANCE_METERS / 1000:.0f}km\n"  # Added search limit
        f"Thermal Density: {lambda_thermals_per_sq_km}/km², Avg Strength: {lambda_strength} (1-10m/s)\n"
        f"Sniffing Radius: {sniffing_radius_meters:.0f}m (MC={mc_for_sniffing_ms}m/s)\n"
        f"Updraft Intercept Distances: {red_dist_str}\n"
        f"Downdraft Encounter Distances: {green_dist_str}"
    )

    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=9, color='gray')

    print(f"Total updrafts generated: {len(updraft_thermals_info)}")
    print(f"\nIntercepts/Encounters with glide path (sniffing radius for updrafts):")
    print(f"  - Updraft Sniffing Intercepts: {updraft_intercepts_count}")
    print(f"  - Downdraft Annulus Encounters: {downdraft_encounters_count}")

    if red_initial_intercept_distances_meters:
        print("\nInitial Intercept Distances for Updrafts (meters):")
        for i, dist in enumerate(red_initial_intercept_distances_meters):
            print(f"  Intercept {i + 1}: {dist:.2f} m")
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

    Args:
        z_cbl_meters (float): The convective boundary layer height (Z) for this simulation.
        glide_ratio (int): The fixed glide ratio of the glider.
        mc_for_sniffing_ms (float): The Macready speed used to calculate the sniffing radius.
        lambda_thermals_per_sq_km (float): The average number of thermals per square kilometer.
        lambda_strength (float): The mean (lambda) for the Poisson distribution of thermal strength magnitude.

    Returns:
        bool: True if at least one red thermal's sniffing radius is intercepted, False otherwise.
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
    sniffing_radius_meters = calculate_sniffing_radius(
        ambient_wt_for_sniff_calc, mc_for_sniffing_ms
    )
    if sniffing_radius_meters <= 0:
        return False  # No intercept possible if sniffing radius is non-positive

    # --- Determine Simulation Area Side Length for Poisson Process ---
    # Max possible updraft radius (if Wt_ms=10)
    max_updraft_radius_possible = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)  # Approx 255.4m
    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS  # 600m

    # Simulation area side should cover the effective glide path plus max thermal/sniffing radius on both sides
    sim_area_side_meters = (
                                       effective_glide_path_length + max_thermal_system_radius * 2 + sniffing_radius_meters * 2) * 1.1  # Add 10% padding

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
        # Check for Updraft Sniffing Intercept
        intersects_sniffing, _ = check_circle_line_segment_intersection(
            updraft_center, sniffing_radius_meters,
            (line_start_x, line_start_y), (line_end_x, line_end_y)
        )
        if intersects_sniffing:
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
        num_simulations = 100 # Number of trials for the Monte Carlo simulation

        print(f"\n--- Running Monte Carlo Simulation for a Single Scenario ({num_simulations} trials) ---")
        print(f"Scenario Parameters:")
        print(f"  Z (CBL Height): {SCENARIO_Z_CBL} m")
        print(f"  Glide Ratio: {SCENARIO_GLIDE_RATIO}:1")
        print(f"  Pilot MC Sniff: {SCENARIO_MC_SNIFF} m/s")
        print(f"  Thermal Density (Lambda): {SCENARIO_LAMBDA_THERMALS_PER_SQ_KM} thermals/km²")
        print(f"  Thermal Strength Mean (Lambda): {SCENARIO_LAMBDA_STRENGTH} m/s (clamped 1-10 m/s)")
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
            'Sniffing Radius (m)': calculated_sniffing_radius,
            'Glide Path Length (m)': reported_glide_path_length,  # Use reported length
            'Thermal Density (per km^2)': SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            'Thermal Strength Lambda': SCENARIO_LAMBDA_STRENGTH,
            'Probability': probability
        }]

        print("\n" + "=" * 120)
        print("\n--- Monte Carlo Simulation Results for Single Scenario ---")
        headers = [
            'Z (m)', 'Wt_Ambient (m/s)', 'MC_Sniff (m/s)', 'Sniffing Radius (m)',
            'Glide Path Length (m)', 'Thermal Density (per km^2)', 'Thermal Strength Lambda',
            'Probability'
        ]
        print(
            f"{headers[0]:<8} | {headers[1]:<18} | {headers[2]:<15} | {headers[3]:<22} | {headers[4]:<23} | "
            f"{headers[5]:<25} | {headers[6]:<25} | {headers[7]:<15}"
        )
        print("-" * 200)  # Increased separator length

        for row in all_results:
            print(
                f"{row['Z (m)']:<8} | {row['Wt_Ambient (m/s)']:<18.1f} | {row['MC_Sniff (m/s)']:<15.1f} | {row['Sniffing Radius (m)']:<22.2f} | {row['Glide Path Length (m)']:<23.2f} | "
                f"{row['Thermal Density (per km^2)']:<25.2f} | {row['Thermal Strength Lambda']:<25.1f} | {row['Probability']:<15.4f}"
            )

        # --- Export results to CSV file ---
        csv_filename = "thermal_intercept_simulation_results_poisson_dist_encircling.csv"
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

