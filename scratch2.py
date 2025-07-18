# comprehensive Monte Carlo simulation designed to calculate the probability of a glider intercepting an updraft thermal
# systematically explores how varying thermal height (Z), ambient thermal strength (Wt), and the pilot's Macready setting
# for sniffing (MC_Sniff) affect this probability

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
# from tqdm import tqdm  # Removed tqdm as requested
import csv  # For CSV export
import seaborn as sns  # For heatmaps


# --- Helper function for calculating thermal radii ---
def calculate_thermal_parameters(Wt_ms, MC_for_thermal_char_ms, thermal_type):
    """
    Calculates various thermal diameters based on input parameters (Wt, MC, thermal type).
    Note: MC_for_thermal_char_ms is used to characterize the thermal's MC core diameter,
    which then feeds into the sniffing radius calculation if used.

    Args:
        Wt_ms (float): Ambient Thermal Strength in m/s.
        MC_for_thermal_char_ms (float): Macready speed in m/s, used for thermal characterization.
        thermal_type (str): "NORMAL" or "NARROW".

    Returns:
        dict: A dictionary containing calculated diameters and input parameters.
    """
    KNOT_TO_MS = 0.514444  # 1 knot = 0.514444 m/s
    FT_TO_M = 0.3048  # 1 foot = 0.3048 m

    if thermal_type == "NORMAL":
        C_thermal = 0.033
    elif thermal_type == "NARROW":
        C_thermal = 0.10
    else:
        raise ValueError("Invalid thermal_type. Must be 'NORMAL' or 'NARROW'.")

    # Updraft (MC Core) Diameter calculation
    Wt_knots = Wt_ms / KNOT_TO_MS
    MC_knots_for_char = MC_for_thermal_char_ms / KNOT_TO_MS  # Use this specific MC for thermal char
    y_MC_knots = Wt_knots - MC_knots_for_char
    try:
        # If y_MC_knots is non-positive, R_MC_feet is 0
        R_MC_feet = 100 * ((y_MC_knots / C_thermal) ** (1 / 3)) if y_MC_knots / C_thermal > 0 else 0
    except (ValueError, ZeroDivisionError):
        R_MC_feet = 0  # Default to 0 if calculation fails or leads to invalid power argument
    D_up_mc_meters = 2 * (R_MC_feet * FT_TO_M)

    # Downdraft Diameter (Fixed as per previous discussions)
    D_down_meters = 1200.0

    return {
        'D_up_mc_meters': D_up_mc_meters,
        'D_down_meters': D_down_meters,
        'Wt_ms': Wt_ms,
        'MC_for_thermal_char_ms': MC_for_thermal_char_ms,  # Return the MC used for characterization
        'thermal_type': thermal_type,
        'glide_ratio': 40  # Fixed glide ratio for thermal params
    }


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
    if A < 1e-9:
        distance_sq_to_center = (x1 - fx) ** 2 + (y1 - fy) ** 2
        return distance_sq_to_center <= radius ** 2, [(x1, y1)] if distance_sq_to_center <= radius ** 2 else []

    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)

    intersection_points = []
    epsilon = 1e-9  # For floating point comparisons

    # Check if intersection points lie on the segment (0 <= t <= 1)
    if -epsilon <= t1 <= 1 + epsilon:
        ix1 = x1 + t1 * dx
        iy1 = y1 + t1 * dy
        intersection_points.append((ix1, iy1))

    # Add t2 only if it's distinct from t1 and on the segment
    if -epsilon <= t2 <= 1 + epsilon and abs(t1 - t2) > epsilon:
        ix2 = x1 + t2 * dx
        iy2 = y2 + t2 * dy
        intersection_points.append((ix2, iy2))

    return len(intersection_points) > 0, intersection_points


def draw_thermals_and_glide_path_with_intercept_check(size=1.0, fig_width=12, fig_height=12):
    """
    Draws a single visualization of the thermal grid, glide path, and intercepts.
    This function is primarily for visualization purposes, showing a single instance of the simulation.
    The glide path length is now calculated as (Z - 500) * glide_ratio.
    Sniffing radius is derived from the MC_ms used for thermal characterization.
    """
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.set_aspect('equal')

    # --- Parameters for this single plot visualization (example values) ---
    global_Wt_ms = 3.0
    # For visualization, we need an MC_ms to characterize the *thermal's* MC core diameter
    # and this will also be the MC that defines the sniffing radius.
    global_MC_ms_for_sniffing = 1.5
    global_thermal_type = "NORMAL"
    global_z_cbl_meters = 2000.0  # Example Z for visualization
    global_glide_ratio = 40

    params = calculate_thermal_parameters(
        global_Wt_ms, global_MC_ms_for_sniffing, global_thermal_type  # Use this MC for thermal char
    )
    D_up_mc_meters = params['D_up_mc_meters']
    D_down_meters = params['D_down_meters']
    Wt_ms = params['Wt_ms']
    MC_ms = params['MC_for_thermal_char_ms']  # This is the MC that determined the core diameter
    thermal_type = params['thermal_type']
    glide_ratio = params['glide_ratio']

    # --- Set sniffing_radius_meters based on calculated Macready core diameter ---
    sniffing_radius_meters = D_up_mc_meters / 2
    if sniffing_radius_meters <= 0:
        print("Warning: Calculated Macready sniffing radius is non-positive. Setting to 1m for visualization.")
        sniffing_radius_meters = 1.0

    # --- Calculations for Plotting Scale ---
    # L_cell_meters_reference is the conceptual average spacing between thermals,
    # often related to CBL height.
    L_cell_meters_reference = 1.5 * global_z_cbl_meters

    thermal_spacing_plot_units = size * math.sqrt(3)
    meters_to_plot_units = thermal_spacing_plot_units / L_cell_meters_reference

    updraft_radius_plot_units = (D_up_mc_meters / 2) * meters_to_plot_units
    downdraft_radius_plot_units = (D_down_meters / 2) * meters_to_plot_units
    sniffing_radius_plot_units = sniffing_radius_meters * meters_to_plot_units

    # --- Glide Path Line Calculation (NEW LOGIC for visualization) ---
    available_glide_height = global_z_cbl_meters - 500  # Glider needs to start landing at 500 AGL
    if available_glide_height <= 0:
        print(
            f"Warning: Z = {global_z_cbl_meters}m results in non-positive available glide height ({available_glide_height}m). Cannot draw meaningful glide path.")
        plt.title(f"Cannot draw glide path for Z={global_z_cbl_meters}m (Available Height <= 0m AGL)")
        plt.show()
        return

    glide_path_horizontal_length_meters = available_glide_height * glide_ratio
    glide_path_horizontal_length_plot_units = glide_path_horizontal_length_meters * meters_to_plot_units

    line_angle_radians = random.uniform(0, 2 * math.pi)
    line_start_x, line_start_y = 0, 0
    line_end_x = line_start_x + glide_path_horizontal_length_plot_units * math.cos(line_angle_radians)
    line_end_y = line_start_y + glide_path_horizontal_length_plot_units * math.sin(line_angle_radians)

    line_origin = (0.0, 0.0)

    print("\n--- Thermal Parameters and Calculated Diameters (for this plot) ---")
    print(f"Ambient Thermal Strength (Wt): {Wt_ms} m/s")
    print(f"Macready Speed (MC): {MC_ms} m/s")
    print(f"Thermal Type: {thermal_type}")
    print(f"--------------------------------------------------")
    print(f"Updraft (MC Core) Diameter: {D_up_mc_meters:.2f} m")
    print(f"Downdraft Diameter (Fixed): {D_down_meters:.2f} m")
    print(f"Sniffing Radius (from MC Core Radius): {sniffing_radius_meters:.2f}m")
    print(f"Thermal Height (Z): {global_z_cbl_meters} m")
    print(f"Available Glide Height: {available_glide_height} m (Z - 500m AGL)")
    print(f"Calculated Glide Path Length: {glide_path_horizontal_length_meters:.2f} m")
    print("--------------------------------------------------\n")

    ax.grid(True, linestyle=':', alpha=0.6)

    # --- NEW: Exponential Thermal Distribution Logic for Visualization ---
    # Parameters for exponential spacing (choose an appropriate average for visualization)
    # This d_x is the *average physical distance* between thermals along the X-axis in meters.
    # It influences the density of thermals in the simulation.
    d_x_for_thermals_meters = L_cell_meters_reference / 3  # Example: 1/3 of the reference cell size

    # Define a simulation extent for thermals (a square area around the glide path origin)
    # This needs to be large enough to cover the glide path and sniffing radii
    simulation_extent_meters = glide_path_horizontal_length_meters + (sniffing_radius_meters * 4)  # Buffer for thermals

    # Generate X-positions (in meters) with exponential spacing
    thermal_x_meters = [0]
    current_x_meter_pos = 0
    # Estimate how many thermals we might need to cover the extent
    # This is an approximation; we'll break if we go past the extent
    while current_x_meter_pos < simulation_extent_meters + d_x_for_thermals_meters * 2:  # Add buffer
        dx = np.random.exponential(scale=d_x_for_thermals_meters)
        current_x_meter_pos += dx
        thermal_x_meters.append(current_x_meter_pos)

    # Generate Y-positions (in meters) randomly within the extent
    # We want a roughly square area around the glide path origin (0,0)
    # So Y should span from -extent/2 to +extent/2
    thermal_y_meters = np.random.uniform(-simulation_extent_meters / 2, simulation_extent_meters / 2,
                                         len(thermal_x_meters))

    all_plotted_thermal_info = []

    for i in range(len(thermal_x_meters)):
        # Convert thermal positions from meters to plot units
        # We also need to center the simulation around the origin for the plot
        plotted_center_x = (thermal_x_meters[i] - simulation_extent_meters / 2) * meters_to_plot_units
        plotted_center_y = thermal_y_meters[i] * meters_to_plot_units

        if random.random() < 0.5:  # 50% chance for downdraft (green)
            thermal_color = 'green'
            thermal_radius = downdraft_radius_plot_units
        else:  # Updraft (red)
            thermal_color = 'red'
            thermal_radius = updraft_radius_plot_units

        circle = patches.Circle(
            (plotted_center_x, plotted_center_y),
            thermal_radius,
            facecolor=thermal_color,
            alpha=0.6,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(circle)
        all_plotted_thermal_info.append({
            'center': (plotted_center_x, plotted_center_y),
            'plotted_radius': thermal_radius,
            'type': thermal_color
        })

    # --- Plotting the Glide Path Line ---
    # Make sure glide path is also centered if thermals are
    ax.plot(
        [line_start_x - simulation_extent_meters / 2 * meters_to_plot_units,
         line_end_x - simulation_extent_meters / 2 * meters_to_plot_units],
        [line_start_y, line_end_y],  # Y is not shifted if thermals generated around 0 for y
        color='blue',
        linewidth=2,
        label=f'Glide Path ({glide_ratio}:1)'
    )
    ax.legend()

    # --- Perform Intersection Checks with Sniffing Radius and collect distances ---
    intersecting_updrafts_count = 0
    intersecting_downdrafts_count = 0
    red_initial_intercept_distances_meters = []
    green_initial_intercept_distances_meters = []

    # Adjust line segment for intersection check if thermal centers were shifted for plotting
    shifted_line_start_x = line_start_x - simulation_extent_meters / 2 * meters_to_plot_units
    shifted_line_end_x = line_end_x - simulation_extent_meters / 2 * meters_to_plot_units
    shifted_line_start = (shifted_line_start_x, line_start_y)
    shifted_line_end = (shifted_line_end_x, line_end_y)

    if sniffing_radius_meters > 0:
        for thermal_info in all_plotted_thermal_info:
            center = thermal_info['center']

            intersects, intersection_pts = check_circle_line_segment_intersection(
                center, sniffing_radius_plot_units, shifted_line_start, shifted_line_end
            )

            if intersects:
                # Increment counts based on thermal type
                if thermal_info['type'] == 'red':
                    intersecting_updrafts_count += 1
                else:  # 'green'
                    intersecting_downdrafts_count += 1

                # Plot the sniffing circle (transparent purple outline)
                sniffing_circle_patch = patches.Circle(
                    center, sniffing_radius_plot_units, color='purple', fill=False, alpha=0.1, linestyle='--',
                    linewidth=0.5
                )
                ax.add_patch(sniffing_circle_patch)

                # Find the intersection point closest to the line origin (0,0) -- for shifted line
                closest_pt_to_origin = None
                min_dist_sq = float('inf')

                # Calculate the distance from the *start of the shifted glide path*
                # The line origin for distance calculation should be the shifted line_start
                line_calc_origin = shifted_line_start

                for pt in intersection_pts:
                    current_dist_sq = (pt[0] - line_calc_origin[0]) ** 2 + (pt[1] - line_calc_origin[1]) ** 2
                    if current_dist_sq < min_dist_sq:
                        min_dist_sq = current_dist_sq
                        closest_pt_to_origin = pt

                # Plot the initial intercept point for this thermal with its corresponding color
                if closest_pt_to_origin:
                    marker_color = thermal_info['type']  # Will be 'red' or 'green'
                    ax.plot(closest_pt_to_origin[0], closest_pt_to_origin[1], 'X', color=marker_color, markersize=10,
                            markeredgecolor='black', linewidth=1.5)

                    # Convert this single initial intercept distance to meters
                    dist_plot_units = math.sqrt(min_dist_sq)
                    dist_meters = dist_plot_units / meters_to_plot_units

                    if thermal_info['type'] == 'red':
                        red_initial_intercept_distances_meters.append(dist_meters)
                    else:
                        green_initial_intercept_distances_meters.append(dist_meters)

    red_initial_intercept_distances_meters.sort()
    green_initial_intercept_distances_meters.sort()

    # --- Construct the footer text for the plot ---
    red_dist_str = "None"
    if red_initial_intercept_distances_meters:
        red_dist_str = ", ".join([f"{d:.0f}m" for d in red_initial_intercept_distances_meters])

    footer_text = (
        f"Wt={Wt_ms}m/s, Updraft (MC={MC_ms}m/s, Dia={D_up_mc_meters:.0f}m), "
        f"Downdrafts (Dia={D_down_meters:.0f}m)\n"
        f"Glide Path: {glide_ratio}:1, Avail. H={available_glide_height}m, Length={glide_path_horizontal_length_meters / 1000:.1f}km "
        f"(Angle={math.degrees(line_angle_radians):.0f}°)\n"
        f"Sniffing Radius: {sniffing_radius_meters:.0f}m (MC Core Radius), "
        f"Red Intercept Distances: {red_dist_str}\n"
        f"Thermal Spacing: Exponential (Avg dx={d_x_for_thermals_meters:.0f}m)"
    )

    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=9, color='gray')

    print(f"Total thermals (red/green circles) generated: {len(all_plotted_thermal_info)}")
    print(f"\nCircles intersecting the glide path with sniffing radius {sniffing_radius_meters:.2f}m:")
    print(f"  - Red (Updraft) Intercepts: {intersecting_updrafts_count}")
    print(f"  - Green (Downdraft) Intercepts: {intersecting_downdrafts_count}")

    if red_initial_intercept_distances_meters:
        print("\nInitial Intercept Distances for RED Thermals (meters):")
        for i, dist in enumerate(red_initial_intercept_distances_meters):
            print(f"  Intercept {i + 1}: {dist:.2f} m")
    else:
        print("\nNo initial intercepts for RED Thermals.")

    if green_initial_intercept_distances_meters:
        print("\nInitial Intercept Distances for GREEN Thermals (meters):")
        for i, dist in enumerate(green_initial_intercept_distances_meters):
            print(f"  Intercept {i + 1}: {dist:.2f} m")
    else:
        print("\nNo initial intercepts for GREEN Thermals.")

    # --- Adjust Plot Limits to fit everything ---
    # The plot limits should be based on the simulation_extent_meters, centered around 0
    max_plot_extent_units = (simulation_extent_meters / 2) * meters_to_plot_units
    padding = max(updraft_radius_plot_units, downdraft_radius_plot_units, sniffing_radius_plot_units) * 2.0

    ax.set_xlim(-max_plot_extent_units - padding, max_plot_extent_units + padding)
    ax.set_ylim(-max_plot_extent_units - padding, max_plot_extent_units + padding)

    plt.show()


def simulate_intercept_experiment(current_wt_ms, current_z_cbl_meters, MC_for_sniffing_ms, fixed_glide_ratio,
                                  d_x_for_thermals_meters):
    """
    Performs a single Monte Carlo experiment to check for an intercept with an updraft thermal.
    The sniffing radius is calculated based on MC_for_sniffing_ms.
    The thermal grid is now generated using exponential spacing for x-coordinates and random y-coordinates.

    Args:
        current_wt_ms (float): The ambient thermal strength (Wt) for this simulation.
        current_z_cbl_meters (float): The convective boundary layer height (Z) for this simulation.
        MC_for_sniffing_ms (float): The Macready speed used to calculate the sniffing radius.
        fixed_glide_ratio (int): The fixed glide ratio of the glider.
        d_x_for_thermals_meters (float): Average distance between thermals along the X-axis for exponential spacing.

    Returns:
        bool: True if at least one red thermal is intercepted, False otherwise.
    """
    # Fixed parameters (from original code, not changing per trial)
    thermal_type = "NORMAL"
    size = 1.0  # Base spacing unit for grid (used in meters_to_plot_units conversion, but less relevant for exponential grid now)

    # Calculate thermal parameters (D_up_mc_meters here describes the size of the *thermals* themselves
    # based on the general ambient Wt, NOT the MC used for sniffing)
    nominal_MC_for_thermal_size_ms = 0.5  # A fixed, low MC to define a "base" updraft core size

    params_for_thermal_size = calculate_thermal_parameters(
        current_wt_ms, nominal_MC_for_thermal_size_ms, thermal_type
    )
    D_up_mc_meters_for_thermal_size = params_for_thermal_size['D_up_mc_meters']
    D_down_meters = params_for_thermal_size['D_down_meters']

    # Calculate the *sniffing radius* based on the iterated MC_for_sniffing_ms
    KNOT_TO_MS = 0.514444
    FT_TO_M = 0.3048
    C_thermal = 0.033 if thermal_type == "NORMAL" else 0.10

    Wt_knots_for_sniff = current_wt / KNOT_TO_MS
    MC_knots_for_sniff = MC_for_sniffing_ms / KNOT_TO_MS
    y_MC_knots_for_sniff = Wt_knots_for_sniff - MC_knots_for_sniff

    R_sniffing_feet = 100 * (
                (y_MC_knots_for_sniff / C_thermal) ** (1 / 3)) if y_MC_knots_for_sniff / C_thermal > 0 else 0
    D_sniffing_meters = 2 * (R_sniffing_feet * FT_TO_M)
    sniffing_radius_meters = D_sniffing_meters / 2

    if sniffing_radius_meters <= 0:
        return False  # No intercept possible if sniffing radius is non-positive

    # --- Glide Path Length Calculation (still dynamic based on Z) ---
    available_glide_height = current_z_cbl_meters - 500
    if available_glide_height <= 0:
        return False

    glide_path_horizontal_length_meters = available_glide_height * fixed_glide_ratio

    if glide_path_horizontal_length_meters <= 0:
        return False

    # Constants for plotting scale (re-used for simulation's internal geometry)
    # L_cell_meters_reference is still used for the general scaling of plot_units,
    # but the actual thermal density is now driven by d_x_for_thermals_meters
    L_cell_meters_reference = 1.5 * current_z_cbl_meters
    meters_to_plot_units = (size * math.sqrt(3)) / L_cell_meters_reference

    # Updraft radius for simulating the *physical* red thermals
    updraft_radius_plot_units = (D_up_mc_meters_for_thermal_size / 2) * meters_to_plot_units
    # downdraft_radius_plot_units = (D_down_meters / 2) * meters_to_plot_units # Not strictly needed if only checking updrafts

    # Sniffing radius for the actual intersection check
    sniffing_radius_plot_units = sniffing_radius_meters * meters_to_plot_units

    glide_path_horizontal_length_plot_units = glide_path_horizontal_length_meters * meters_to_plot_units

    # Glide Path Line Calculation (new random path each time for each trial)
    # Start the glide path from a central point (0,0) in plot units
    line_angle_radians = random.uniform(0, 2 * math.pi)
    line_start_x_plot_units, line_start_y_plot_units = 0, 0  # Origin of the glide path in plot units
    line_end_x_plot_units = line_start_x_plot_units + glide_path_horizontal_length_plot_units * math.cos(
        line_angle_radians)
    line_end_y_plot_units = line_start_y_plot_units + glide_path_horizontal_length_plot_units * math.sin(
        line_angle_radians)

    # --- NEW: Exponential Thermal Distribution Logic for Simulation ---
    # Determine the required simulation area in meters to cover the glide path plus buffer
    # The buffer should account for thermal sizes and sniffing radius
    min_required_thermal_area_extent_meters = glide_path_horizontal_length_meters + (sniffing_radius_meters * 4) + (
                D_up_mc_meters_for_thermal_size * 2)

    # Generate thermal X-positions (in meters) using exponential spacing
    # We need enough thermals to potentially cover the entire required extent
    thermal_x_meters = []
    current_x_meter_pos = 0
    # Generate thermals over a slightly larger extent than strictly required to ensure coverage
    # The maximum x position of thermals should go a bit beyond the max possible x of the glide path + buffer
    # Add a safety margin to the generation loop condition
    max_x_to_generate = min_required_thermal_area_extent_meters + d_x_for_thermals_meters * 5  # A larger buffer for generation

    # Generate at least a certain number of thermals to ensure good density
    num_thermals_to_generate_approx = int(
        min_required_thermal_area_extent_meters / d_x_for_thermals_meters) * 2 + 10  # heuristic

    while len(thermal_x_meters) < num_thermals_to_generate_approx and current_x_meter_pos < max_x_to_generate:
        dx = np.random.exponential(scale=d_x_for_thermals_meters)
        current_x_meter_pos += dx
        thermal_x_meters.append(current_x_meter_pos)

    # Handle cases where not enough thermals were generated
    if not thermal_x_meters:
        return False  # No thermals generated, no intercept possible

    # Generate Y-positions (in meters) randomly for each x_position
    # These y-positions should span an area around the glide path.
    # The glide path origin (0,0) will correspond to the center of the simulated thermal area.
    # So Y range should be from -extent/2 to +extent/2 meters.
    thermal_y_meters = np.random.uniform(-min_required_thermal_area_extent_meters / 2,
                                         min_required_thermal_area_extent_meters / 2,
                                         len(thermal_x_meters))

    # Shift all thermal x-positions so that the 'center' of the generated thermal field
    # (i.e., half of the max generated x_meters) aligns with the glide path's origin (0,0) in plot units.
    # This ensures the glide path is effectively "in the middle" of the generated thermals.
    x_offset_for_centering_meters = max(thermal_x_meters) / 2

    # Generate Thermals and Check Intersections immediately
    for i in range(len(thermal_x_meters)):
        plotted_center_x = (thermal_x_meters[i] - x_offset_for_centering_meters) * meters_to_plot_units
        plotted_center_y = thermal_y_meters[i] * meters_to_plot_units

        # Randomly assign as updraft (red) or downdraft (green)
        is_updraft = random.random() >= 0.5

        if is_updraft:
            # Only check against updrafts (red thermals) for the success condition
            intersects, _ = check_circle_line_segment_intersection(
                (plotted_center_x, plotted_center_y), sniffing_radius_plot_units,
                (line_start_x_plot_units, line_start_y_plot_units), (line_end_x_plot_units, line_end_y_plot_units)
            )

            if intersects:
                return True  # Found an intercept with a red thermal, simulation trial is a success

    return False  # No intercept with a red thermal found in this trial


# --- Main execution block ---
if __name__ == '__main__':
    # Automatically run option 2 as requested by the user
    # print("Choose an option:")
    # print("1. Generate a single plot (visualize with MC-derived sniffing radius)")
    # print("2. Run Monte Carlo simulation (compute probability table and export CSV)")

    # choice = input("Enter 1 or 2: ")

    # if choice == '1':
    #     draw_thermals_and_glide_path_with_intercept_check(size=1.0)

    # elif choice == '2': # This block will now run automatically

    # Fixed parameters for the simulation
    thermal_type_fixed = "NORMAL"
    glide_ratio_fixed = 40
    num_simulations = 1000

    # Iteration ranges for Z, Wt, and MC_Sniffing
    z_values = [1500, 2000, 2500, 3000]
    Wt_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    mc_sniffing_values = np.arange(0.0, 5.5, 0.5)

    # NEW: Average thermal spacing for the exponential distribution
    d_x_for_thermals_meters_sim = 1000  # meters, average distance between thermals along X-axis

    all_results = []

    print(f"\n--- Running Comprehensive Monte Carlo Simulations ({num_simulations} trials per scenario) ---")
    print(
        f"Thermal Grid Generation: Exponential spacing (avg dx={d_x_for_thermals_meters_sim:.0f}m) along X, random Y.")
    print(f"Fixed Glide Ratio: {glide_ratio_fixed}:1")

    total_scenarios = len(z_values) * len(Wt_values) * len(mc_sniffing_values)
    current_scenario_counter = 0

    for current_z_cbl in z_values:
        for current_wt in Wt_values:
            for current_mc_sniff in mc_sniffing_values:
                current_scenario_counter += 1

                available_glide_height = current_z_cbl - 500
                current_glide_path_length_meters = available_glide_height * glide_ratio_fixed

                # Calculate the specific MC sniffing radius for this Wt and current_mc_sniff
                KNOT_TO_MS = 0.514444
                FT_TO_M = 0.3048
                C_thermal = 0.033 if thermal_type_fixed == "NORMAL" else 0.10

                Wt_knots_for_sniff = current_wt / KNOT_TO_MS
                MC_knots_for_sniff = current_mc_sniff / KNOT_TO_MS
                y_MC_knots_for_sniff = Wt_knots_for_sniff - MC_knots_for_sniff

                R_sniffing_feet = 100 * ((y_MC_knots_for_sniff / C_thermal) ** (
                            1 / 3)) if y_MC_knots_for_sniff / C_thermal > 0 else 0
                calculated_sniffing_radius = 2 * (R_sniffing_feet * FT_TO_M) / 2

                probability = 0.0

                if available_glide_height <= 0:
                    print(
                        f"    Scenario {current_scenario_counter}/{total_scenarios}: Z={current_z_cbl}m, Wt={current_wt:.1f}m/s, MC_Sniff={current_mc_sniff:.1f}m/s: No available glide height (Prob: {probability:.4f})")
                elif calculated_sniffing_radius <= 0:
                    print(
                        f"    Scenario {current_scenario_counter}/{total_scenarios}: Z={current_z_cbl}m, Wt={current_wt:.1f}m/s, MC_Sniff={current_mc_sniff:.1f}m/s: Sniffing radius <= 0 (Prob: {probability:.4f})")
                else:
                    intercept_count = 0
                    print(f"    Scenario {current_scenario_counter}/{total_scenarios}: "
                          f"Z={current_z_cbl}m, Wt={current_wt:.1f}m/s, MC_Sniff={current_mc_sniff:.1f}m/s, R_sniff={calculated_sniffing_radius:.0f}m")
                    for _ in range(num_simulations):
                        # Pass the new d_x_for_thermals_meters_sim to the simulation function
                        if simulate_intercept_experiment(current_wt, current_z_cbl, current_mc_sniff, glide_ratio_fixed,
                                                         d_x_for_thermals_meters_sim):
                            intercept_count += 1

                    probability = intercept_count / num_simulations

                all_results.append({
                    'Z (m)': current_z_cbl,
                    'Wt (m/s)': current_wt,
                    'MC_Sniff (m/s)': current_mc_sniff,
                    'Sniffing Radius (m)': calculated_sniffing_radius,
                    'Glide Path Length (m)': current_glide_path_length_meters,
                    'Thermal Spacing Avg_dx (m)': d_x_for_thermals_meters_sim,  # New column for average spacing
                    'Probability': probability
                })

    print("\n" + "=" * 140)
    print("\n--- All Simulations Complete ---")

    # --- Print Results Table ---
    print("\n--- Probability of Intercepting a Red Thermal (Varying Sniffing MC & Exponential Spacing) ---")
    headers = ['Z (m)', 'Wt (m/s)', 'MC_Sniff (m/s)', 'Sniffing Radius (m)', 'Glide Path Length (m)',
               'Thermal Spacing Avg_dx (m)', 'Probability']
    print(
        f"{headers[0]:<8} | {headers[1]:<10} | {headers[2]:<15} | {headers[3]:<22} | {headers[4]:<23} | {headers[5]:<26} | {headers[6]:<15}")
    print("-" * 140)

    for row in all_results:
        print(
            f"{row['Z (m)']:<8} | {row['Wt (m/s)']:<10.1f} | {row['MC_Sniff (m/s)']:<15.1f} | {row['Sniffing Radius (m)']:<22.2f} | {row['Glide Path Length (m)']:<23.2f} | {row['Thermal Spacing Avg_dx (m)']:<26.0f} | {row['Probability']:<15.4f}")

    # --- Export results to CSV file ---
    csv_filename = "thermal_intercept_simulation_results_exponential_grid.csv"
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

    # --- Generate Probability Maps (Heatmaps) ---
    print("\n--- Generating Probability Maps ---")

    # Create a mapping for Wt and MC_Sniff values to array indices
    wt_unique = sorted(list(set([res['Wt (m/s)'] for res in all_results])))
    mc_sniff_unique = sorted(list(set([res['MC_Sniff (m/s)'] for res in all_results])))

    wt_to_idx = {wt: i for i, wt in enumerate(wt_unique)}
    mc_sniff_to_idx = {mc: i for i, mc in enumerate(mc_sniff_unique)}

    for z_val in sorted(list(set([res['Z (m)'] for res in all_results]))):
        # Initialize a 2D array for probabilities for the current Z value
        prob_matrix = np.full((len(wt_unique), len(mc_sniff_unique)), np.nan)

        for res in all_results:
            if res['Z (m)'] == z_val:
                wt_idx = wt_to_idx[res['Wt (m/s)']]
                mc_sniff_idx = mc_sniff_to_idx[res['MC_Sniff (m/s)']]
                prob_matrix[wt_idx, mc_sniff_idx] = res['Probability']

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            prob_matrix,
            annot=True,  # Show probability values on the heatmap
            fmt=".3f",  # Format annotation to 3 decimal places
            cmap="viridis",  # Color map
            cbar_kws={'label': 'Probability of Intercept'},
            xticklabels=[f"{val:.1f}" for val in mc_sniff_unique],
            yticklabels=[f"{val:.1f}" for val in wt_unique],
            linewidths=.5,  # Add lines between cells
            linecolor='black'
        )

        plt.title(f'Probability Map (Z={z_val}m, Thermal Spacing Avg dx={d_x_for_thermals_meters_sim:.0f}m)')
        plt.xlabel('MC Sniff (m/s)')
        plt.ylabel('Ambient Thermal Strength Wt (m/s)')

        # Ensure proper display of tick labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        map_filename = f"probability_map_Z_{z_val}m.png"
        plt.savefig(map_filename, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
        print(f"Generated probability map: '{map_filename}'")

    print("\n--- All Probability Maps Generated ---")