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
MAX_SEARCH_DISTANCE_METERS = 162000.0  # 162 kilometers (100 miles)

# NEW CONSTANT: The width of the glider's search corridor
GLIDE_PATH_WIDTH_METERS = 2000.0  # 5 km wide path

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


# --- Helper function for circle-rectangle intersection check ---
def check_circle_rectangle_intersection(circle_center, radius, rect_center, rect_width, rect_height, rect_angle_radians):
    """
    Checks if a circle intersects with a rotated rectangle.

    Args:
        circle_center (tuple): (cx, cy) coordinates of the circle's center.
        radius (float): Radius of the circle.
        rect_center (tuple): (rx, ry) coordinates of the rectangle's center.
        rect_width (float): Width of the rectangle (along its local x-axis).
        rect_height (float): Height of the rectangle (along its local y-axis).
        rect_angle_radians (float): Rotation of the rectangle in radians.

    Returns:
        bool: True if the circle intersects the rectangle, False otherwise.
    """
    cx, cy = circle_center
    rx, ry = rect_center
    angle = -rect_angle_radians  # Rotate the circle's center, not the rectangle

    # Translate and rotate the circle center to the rectangle's local coordinate system
    dx = cx - rx
    dy = cy - ry
    rotated_cx = dx * math.cos(angle) - dy * math.sin(angle)
    rotated_cy = dx * math.sin(angle) + dy * math.cos(angle)

    # Find the closest point on the rectangle to the rotated circle center
    closest_x = np.clip(rotated_cx, -rect_width / 2, rect_width / 2)
    closest_y = np.clip(rotated_cy, -rect_height / 2, rect_height / 2)

    # Calculate the distance squared from the rotated circle center to this closest point
    distance_x = rotated_cx - closest_x
    distance_y = rotated_cy - closest_y
    distance_squared = distance_x**2 + distance_y**2

    return distance_squared < radius**2


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
        glide_path_width_meters,
        fig_width=12, fig_height=12
):
    """
    Draws a single visualization of the thermal grid (Poisson distributed),
    with updrafts and encircling downdraft rings, glide path, and intercepts.
    The glide path search is now a rectangle with a specified width.
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
    ambient_wt_for_sniff_calc = lambda_strength
    sniffing_radius_meters = calculate_sniffing_radius(
        ambient_wt_for_sniff_calc, mc_for_sniffing_ms
    )
    if sniffing_radius_meters <= 0:
        print("Warning: Calculated Macready sniffing radius is non-positive. Setting to 1m for visualization.")
        sniffing_radius_meters = 1.0

    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS
    sim_area_side_meters = (
                                       effective_glide_path_length + max_thermal_system_radius * 2 + sniffing_radius_meters * 2 + glide_path_width_meters) * 1.1

    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    # --- Plotting the Glide Path Rectangle as a Polygon ---
    path_angle_radians = random.uniform(0, 2 * math.pi)
    rect_center_x = (effective_glide_path_length / 2) * math.cos(path_angle_radians)
    rect_center_y = (effective_glide_path_length / 2) * math.sin(path_angle_radians)

    # Calculate corner points of the rotated rectangle
    half_width = effective_glide_path_length / 2
    half_height = glide_path_width_meters / 2
    cos_angle = math.cos(path_angle_radians)
    sin_angle = math.sin(path_angle_radians)

    # Local coordinates of corners relative to center (0,0)
    p1 = (half_width, half_height)
    p2 = (-half_width, half_height)
    p3 = (-half_width, -half_height)
    p4 = (half_width, -half_height)

    # Rotate and translate corners
    corners = []
    for x, y in [p1, p2, p3, p4]:
        rotated_x = x * cos_angle - y * sin_angle
        rotated_y = x * sin_angle + y * cos_angle
        corners.append((rotated_x + rect_center_x, rotated_y + rect_center_y))

    glide_path_polygon = patches.Polygon(corners, color='blue', alpha=0.2, zorder=1)
    ax.add_patch(glide_path_polygon)

    # --- Plot Thermals (Updrafts and Encircling Downdrafts) ---
    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']
        updraft_radius = thermal_info['updraft_radius']

        updraft_circle = patches.Circle(
            updraft_center,
            updraft_radius,
            facecolor='red',
            alpha=0.6,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(updraft_circle)

        downdraft_inner_radius = updraft_radius
        downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

        if downdraft_outer_radius > downdraft_inner_radius:
            downdraft_annulus = patches.Circle(
                updraft_center,
                downdraft_outer_radius,
                facecolor='green',
                alpha=0.05,
                edgecolor='green',
                linewidth=0.5,
                fill=True,
                hatch='/'
            )
            ax.add_patch(downdraft_annulus)

    # --- Perform Intersection Checks ---
    updraft_intercepts_count = 0
    downdraft_encounters_count = 0

    rect_center = (rect_center_x, rect_center_y)
    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']

        intersects_sniffing = check_circle_rectangle_intersection(
            updraft_center, sniffing_radius_meters,
            rect_center, effective_glide_path_length, glide_path_width_meters, path_angle_radians
        )

        if intersects_sniffing:
            updraft_intercepts_count += 1
            sniffing_circle_patch = patches.Circle(
                updraft_center, sniffing_radius_meters, color='purple', fill=False, alpha=0.1, linestyle='--',
                linewidth=0.5
            )
            ax.add_patch(sniffing_circle_patch)

            ax.plot(updraft_center[0], updraft_center[1], 'X', color='red', markersize=10,
                    markeredgecolor='black', linewidth=1.5)

        downdraft_inner_radius = thermal_info['updraft_radius']
        downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

        if downdraft_outer_radius > downdraft_inner_radius:
            intersects_downdraft = check_circle_rectangle_intersection(
                updraft_center, downdraft_outer_radius,
                rect_center, effective_glide_path_length, glide_path_width_meters, path_angle_radians
            )
            if intersects_downdraft:
                is_in_updraft_core = check_circle_rectangle_intersection(
                    updraft_center, downdraft_inner_radius,
                    rect_center, effective_glide_path_length, glide_path_width_meters, path_angle_radians
                )
                if not is_in_updraft_core:
                    downdraft_encounters_count += 1
                    ax.plot(updraft_center[0], updraft_center[1], 'o', color='green',
                            markersize=8, markeredgecolor='black', linewidth=1.0)


    # --- Construct the footer text for the plot ---
    footer_text = (
        f"Z={z_cbl_meters}m, Glide Path: {glide_ratio}:1, Length={glide_path_horizontal_length_meters / 1000:.1f}km\n"
        f"Search Corridor Width: {glide_path_width_meters / 1000:.1f}km\n"
        f"Search Limit: {MAX_SEARCH_DISTANCE_METERS / 1000:.0f}km\n"
        f"Thermal Density: {lambda_thermals_per_sq_km}/km², Avg Strength: {lambda_strength} (1-10m/s)\n"
        f"Sniffing Radius: {sniffing_radius_meters:.0f}m (MC={mc_for_sniffing_ms}m/s)\n"
        f"Updraft Intercepts (within corridor): {updraft_intercepts_count}\n"
        f"Downdraft Encounters (within corridor): {downdraft_encounters_count}"
    )

    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=9, color='gray')

    plot_limit_extent = sim_area_side_meters / 2
    plot_padding_factor = 0.05
    ax.set_xlim(-(plot_limit_extent * (1 + plot_padding_factor)), plot_limit_extent * (1 + plot_padding_factor))
    ax.set_ylim(-(plot_limit_extent * (1 + plot_padding_factor)), plot_limit_extent * (1 + plot_padding_factor))

    plt.show()


def simulate_intercept_experiment_poisson(
        z_cbl_meters, glide_ratio, mc_for_sniffing_ms,
        lambda_thermals_per_sq_km, lambda_strength,
        glide_path_width_meters
):
    """
    Performs a single Monte Carlo experiment with Poisson-distributed updraft thermals
    to check for an intercept with an updraft's sniffing radius within a wide rectangular path.

    Args:
        z_cbl_meters (float): The convective boundary layer height (Z) for this simulation.
        glide_ratio (int): The fixed glide ratio of the glider.
        mc_for_sniffing_ms (float): The Macready speed used to calculate the sniffing radius.
        lambda_thermals_per_sq_km (float): The average number of thermals per square kilometer.
        lambda_strength (float): The mean (lambda) for the Poisson distribution of thermal strength magnitude.
        glide_path_width_meters (float): The width of the rectangular glide path.

    Returns:
        bool: True if at least one thermal's sniffing radius intersects the path, False otherwise.
    """
    available_glide_height = z_cbl_meters - 500
    if available_glide_height <= 0:
        return False

    glide_path_horizontal_length_meters = available_glide_height * glide_ratio
    effective_glide_path_length = min(glide_path_horizontal_length_meters, MAX_SEARCH_DISTANCE_METERS)

    if effective_glide_path_length <= 0:
        return False

    ambient_wt_for_sniff_calc = lambda_strength
    sniffing_radius_meters = calculate_sniffing_radius(
        ambient_wt_for_sniff_calc, mc_for_sniffing_ms
    )
    if sniffing_radius_meters <= 0:
        return False

    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS
    sim_area_side_meters = (
                                       effective_glide_path_length + max_thermal_system_radius * 2 + sniffing_radius_meters * 2 + glide_path_width_meters) * 1.1

    updraft_thermals = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    path_angle_radians = random.uniform(0, 2 * math.pi)
    rect_center_x = (effective_glide_path_length / 2) * math.cos(path_angle_radians)
    rect_center_y = (effective_glide_path_length / 2) * math.sin(path_angle_radians)
    rect_center = (rect_center_x, rect_center_y)

    for thermal_info in updraft_thermals:
        updraft_center = thermal_info['center']
        intersects_sniffing = check_circle_rectangle_intersection(
            updraft_center, sniffing_radius_meters,
            rect_center, effective_glide_path_length, glide_path_width_meters, path_angle_radians
        )
        if intersects_sniffing:
            return True

    return False


# --- Main execution block ---
if __name__ == '__main__':
    print("Choose an option:")
    print("1. Generate a single plot (visualize Poisson-distributed thermals with encircling downdrafts)")
    print("2. Run Monte Carlo simulation (compute probability for a single scenario and export CSV)")

    choice = input("Enter 1 or 2: ")

    if choice == '1':
        print("\n--- Generating Single Plot with Poisson Thermals (Updrafts with Encircling Downdrafts) ---")
        draw_poisson_thermals_and_glide_path_with_intercept_check(
            z_cbl_meters=SCENARIO_Z_CBL,
            glide_ratio=SCENARIO_GLIDE_RATIO,
            mc_for_sniffing_ms=SCENARIO_MC_SNIFF,
            lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            lambda_strength=SCENARIO_LAMBDA_STRENGTH,
            glide_path_width_meters=GLIDE_PATH_WIDTH_METERS
        )

    elif choice == '2':
        num_simulations = 1000

        print(f"\n--- Running Monte Carlo Simulation for a Single Scenario ({num_simulations} trials) ---")
        print(f"Scenario Parameters:")
        print(f"  Z (CBL Height): {SCENARIO_Z_CBL} m")
        print(f"  Glide Ratio: {SCENARIO_GLIDE_RATIO}:1")
        print(f"  Glide Path Width: {GLIDE_PATH_WIDTH_METERS / 1000} km")
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
                    lambda_strength=SCENARIO_LAMBDA_STRENGTH,
                    glide_path_width_meters=GLIDE_PATH_WIDTH_METERS
            ):
                intercept_count += 1

        probability = intercept_count / num_simulations

        calculated_sniffing_radius = calculate_sniffing_radius(
            SCENARIO_LAMBDA_STRENGTH, SCENARIO_MC_SNIFF
        )
        available_glide_height = SCENARIO_Z_CBL - 500
        current_glide_path_length_meters = available_glide_height * SCENARIO_GLIDE_RATIO
        reported_glide_path_length = min(current_glide_path_length_meters, MAX_SEARCH_DISTANCE_METERS)

        all_results = [{
            'Z (m)': SCENARIO_Z_CBL,
            'Wt_Ambient (m/s)': SCENARIO_LAMBDA_STRENGTH,
            'MC_Sniff (m/s)': SCENARIO_MC_SNIFF,
            'Sniffing Radius (m)': calculated_sniffing_radius,
            'Glide Path Length (m)': reported_glide_path_length,
            'Glide Path Width (m)': GLIDE_PATH_WIDTH_METERS,
            'Thermal Density (per km^2)': SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            'Thermal Strength Lambda': SCENARIO_LAMBDA_STRENGTH,
            'Probability': probability
        }]

        print("\n" + "=" * 120)
        print("\n--- Monte Carlo Simulation Results for Single Scenario ---")
        headers = [
            'Z (m)', 'Wt_Ambient (m/s)', 'MC_Sniff (m/s)', 'Sniffing Radius (m)',
            'Glide Path Length (m)', 'Glide Path Width (m)', 'Thermal Density (per km^2)',
            'Thermal Strength Lambda', 'Probability'
        ]
        print(
            f"{headers[0]:<8} | {headers[1]:<18} | {headers[2]:<15} | {headers[3]:<22} | {headers[4]:<23} | "
            f"{headers[5]:<22} | {headers[6]:<25} | {headers[7]:<25} | {headers[8]:<15}"
        )
        print("-" * 200)

        for row in all_results:
            print(
                f"{row['Z (m)']:<8} | {row['Wt_Ambient (m/s)']:<18.1f} | {row['MC_Sniff (m/s)']:<15.1f} | {row['Sniffing Radius (m)']:<22.2f} | "
                f"{row['Glide Path Length (m)']:<23.2f} | {row['Glide Path Width (m)']:<22.2f} | "
                f"{row['Thermal Density (per km^2)']:<25.2f} | {row['Thermal Strength Lambda']:<25.1f} | {row['Probability']:<15.4f}"
            )

        csv_filename = "thermal_intercept_simulation_results_poisson_dist_encircling_with_width.csv"
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