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
# - Interception Logic: A thermal is considered a potential intercept if its center is within the
#   search arc OR if its sniffing radius overlaps with the search arc's boundary lines.
# - **FIXED:** The bug causing an infinite loop has been resolved. Intercepted thermals are now
#   removed from the list of available thermals, forcing the glider to always move forward.
# - **NEW:** The initial line orientation is now randomized by generating a random
#   end point at a fixed distance from the origin.
# - **FIXED:** The final step is now labeled 'Final Glide' with distance and angle relative to the initial glide path.
# - Visualization: The search arc is now visually drawn on the plot to demonstrate
#   the search area for each path segment, and the specific intercepted thermal
#   is highlighted with a unique marker.
# - Data Output: The output for the single plot mode is now simplified to only
#   show the distance from the origin and the relative bearing, without redundant labels.
# - Modularity: The script is designed with a clear separation of concerns,
#   using functions for thermal generation, intersection checks, and visualization,
#   making it easy to understand and modify parameters.
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

# --- Scenario Parameters ---
SCENARIO_Z_CBL = 2500.0
SCENARIO_GLIDE_RATIO = 40
SCENARIO_MC_SNIFF = 2
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.1
SCENARIO_LAMBDA_STRENGTH = 3
SEARCH_ARC_ANGLE_DEGREES = 30.0
MAX_SEARCH_DISTANCE_METERS = 100000.0  # 100 kilometers for a dynamic path simulation
RANDOM_END_POINT_DISTANCE = math.hypot(40000, 40000)


# --- Helper functions (from previous scripts, unchanged) ---
def calculate_sniffing_radius(Wt_ms_ambient, MC_for_sniffing_ms, thermal_type="NORMAL"):
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


# --- Main Dynamic Simulation Function for Visualization ---
def simulate_dynamic_glide_path_and_draw(
        z_cbl_meters, glide_ratio, mc_for_sniffing_ms,
        lambda_thermals_per_sq_km, lambda_strength,
        end_point, fig_width=12, fig_height=12
):
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.set_aspect('equal')

    available_glide_height = z_cbl_meters - 500
    if available_glide_height <= 0:
        print("Error: CBL height too low for a meaningful glide.")
        return

    effective_glide_path_length = MAX_SEARCH_DISTANCE_METERS
    ambient_wt_for_sniff_calc = lambda_strength
    sniffing_radius_meters_base = calculate_sniffing_radius(ambient_wt_for_sniff_calc, mc_for_sniffing_ms)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS
    sim_area_side_meters = max(
        math.hypot(end_point[0], end_point[1]) * 1.5,
        effective_glide_path_length + max_thermal_system_radius * 2 + sniffing_radius_meters_base * 2
    )

    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

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
    path_segments = []

    ax.plot(end_point[0], end_point[1], 's', color='black', markersize=10, label='End Point')
    ax.plot(current_pos[0], current_pos[1], 'o', color='blue', markersize=10, label='Start Point')

    start_point = current_pos
    bearing_to_end_initial = math.degrees(math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0]))
    if bearing_to_end_initial < 0:
        bearing_to_end_initial += 360

    print(f"Starting simulation from (0,0) to random end point {end_point}")
    print(f"Initial Bearing from origin to end point: {bearing_to_end_initial:.2f}°")

    step = 0
    thermals_to_remove = []  # Keep track of thermals to remove
    while math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]) > EPSILON:
        step += 1
        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        if distance_to_end <= 0:
            break

        bearing_to_end_radians = math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0])
        bearing_to_end_degrees = math.degrees(bearing_to_end_radians)
        if bearing_to_end_degrees < 0:
            bearing_to_end_degrees += 360

        arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2
        segment_length = min(distance_to_end, effective_glide_path_length)

        arc_start_angle = bearing_to_end_degrees - arc_half_angle_degrees
        arc_end_angle = bearing_to_end_degrees + arc_half_angle_degrees
        wedge = patches.Wedge(
            path_start, segment_length,
            arc_start_angle, arc_end_angle,
            facecolor='gray', alpha=0.1, edgecolor='none'
        )
        ax.add_patch(wedge)

        path_end = (path_start[0] + segment_length * math.cos(bearing_to_end_radians),
                    path_start[1] + segment_length * math.sin(bearing_to_end_radians))

        nearest_thermal = None
        min_dist_to_thermal = float('inf')

        for thermal in updraft_thermals_info:
            thermal_center = thermal['center']

            dist_to_thermal = math.hypot(thermal_center[0] - path_start[0], thermal_center[1] - path_start[1])
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

            arc_line_upper_end = (path_start[0] + segment_length * math.cos(math.radians(arc_end_angle)),
                                  path_start[1] + segment_length * math.sin(math.radians(arc_end_angle)))
            arc_line_lower_end = (path_start[0] + segment_length * math.cos(math.radians(arc_start_angle)),
                                  path_start[1] + segment_length * math.sin(math.radians(arc_start_angle)))

            dist_to_upper_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_upper_end)
            dist_to_lower_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_lower_end)

            is_near_arc_edge = (dist_to_upper_line <= sniffing_radius_meters_base) or (
                        dist_to_lower_line <= sniffing_radius_meters_base)

            if (is_in_arc or is_near_arc_edge) and dist_to_thermal <= segment_length:
                if dist_to_thermal < min_dist_to_thermal:
                    min_dist_to_thermal = dist_to_thermal
                    nearest_thermal = thermal

        if nearest_thermal:
            thermal_center = nearest_thermal['center']
            path_segments.append((path_start, thermal_center))
            current_pos = thermal_center

            # Remove the intercepted thermal so it's not found again
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

            print(f"\n{step}: {thermal_dist:.2f} m, {relative_bearing:.2f}°")

        else:
            path_segments.append((path_start, end_point))
            current_pos = end_point

            # Calculate distance and bearing for the final glide relative to the initial path
            final_glide_distance = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])
            final_glide_bearing = math.degrees(math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0]))
            if final_glide_bearing < 0:
                final_glide_bearing += 360

            final_glide_relative_bearing = final_glide_bearing - bearing_to_end_initial
            if final_glide_relative_bearing > 180:
                final_glide_relative_bearing -= 360
            elif final_glide_relative_bearing <= -180:
                final_glide_relative_bearing += 360

            print(
                f"\nFinal Glide: {final_glide_distance:.2f} m, {final_glide_relative_bearing:.2f}° to end point {end_point}.")

    # --- Plot the final path ---
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

    path_coords_x.append(end_point[0])
    path_coords_y.append(end_point[1])

    ax.plot(path_coords_x, path_coords_y, color='blue', linewidth=2, label='Glider Path')
    ax.plot([], [], 'x', color='red', markersize=8, markeredgecolor='black', linewidth=1, label='Intercepted Thermal')

    all_x = path_coords_x + [t['center'][0] for t in updraft_thermals_info]
    all_y = path_coords_y + [t['center'][1] for t in updraft_thermals_info]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    padding = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y)) * 0.1
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_xlabel('East-West (m)')
    ax.set_ylabel('North-South (m)')
    ax.set_title(f"Dynamic Glider Path Simulation with Thermal Intercepts")
    ax.legend()
    plt.show()


# --- Main Dynamic Simulation Function for Monte Carlo ---
def simulate_intercept_experiment_dynamic(
        z_cbl_meters, glide_ratio, mc_for_sniffing_ms,
        lambda_thermals_per_sq_km, lambda_strength,
        end_point
):
    available_glide_height = z_cbl_meters - 500
    if available_glide_height <= 0:
        return False

    effective_glide_path_length = MAX_SEARCH_DISTANCE_METERS

    ambient_wt_for_sniff_calc = lambda_strength
    sniffing_radius_meters_base = calculate_sniffing_radius(
        ambient_wt_for_sniff_calc, mc_for_sniffing_ms
    )
    if sniffing_radius_meters_base <= 0:
        return False

    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS
    sim_area_side_meters = max(
        math.hypot(end_point[0], end_point[1]) * 1.5,
        effective_glide_path_length + max_thermal_system_radius * 2 + sniffing_radius_meters_base * 2
    )

    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    current_pos = (0, 0)
    has_intercepted_thermal = False

    while math.hypot(end_point[0] - current_pos[0], end_point[1] - current_pos[1]) > EPSILON:
        path_start = current_pos
        distance_to_end = math.hypot(end_point[0] - path_start[0], end_point[1] - path_start[1])

        if distance_to_end <= 0:
            break

        bearing_to_end_radians = math.atan2(end_point[1] - path_start[1], end_point[0] - path_start[0])
        bearing_to_end_degrees = math.degrees(bearing_to_end_radians)
        if bearing_to_end_degrees < 0:
            bearing_to_end_degrees += 360

        arc_half_angle_degrees = SEARCH_ARC_ANGLE_DEGREES / 2
        segment_length = min(distance_to_end, effective_glide_path_length)

        arc_start_angle = bearing_to_end_degrees - arc_half_angle_degrees
        arc_end_angle = bearing_to_end_degrees + arc_half_angle_degrees

        path_end = (path_start[0] + segment_length * math.cos(bearing_to_end_radians),
                    path_start[1] + segment_length * math.sin(bearing_to_end_radians))

        nearest_thermal = None
        min_dist_to_thermal = float('inf')

        for thermal in updraft_thermals_info:
            thermal_center = thermal['center']
            dist_to_thermal = math.hypot(thermal_center[0] - path_start[0], thermal_center[1] - path_start[1])
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

            arc_line_upper_end = (path_start[0] + segment_length * math.cos(math.radians(arc_end_angle)),
                                  path_start[1] + segment_length * math.sin(math.radians(arc_end_angle)))
            arc_line_lower_end = (path_start[0] + segment_length * math.cos(math.radians(arc_start_angle)),
                                  path_start[1] + segment_length * math.sin(math.radians(arc_start_angle)))

            dist_to_upper_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_upper_end)
            dist_to_lower_line, _ = distance_from_point_to_line_segment(thermal_center, path_start, arc_line_lower_end)

            is_near_arc_edge = (dist_to_upper_line <= sniffing_radius_meters_base) or (
                        dist_to_lower_line <= sniffing_radius_meters_base)

            if (is_in_arc or is_near_arc_edge) and dist_to_thermal <= segment_length:
                if dist_to_thermal < min_dist_to_thermal:
                    min_dist_to_thermal = dist_to_thermal
                    nearest_thermal = thermal

        if nearest_thermal:
            thermal_center = nearest_thermal['center']
            current_pos = thermal_center
            updraft_thermals_info.remove(nearest_thermal)
            has_intercepted_thermal = True
        else:
            current_pos = end_point

    return has_intercepted_thermal


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
            glide_ratio=SCENARIO_GLIDE_RATIO,
            mc_for_sniffing_ms=SCENARIO_MC_SNIFF,
            lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            lambda_strength=SCENARIO_LAMBDA_STRENGTH,
            end_point=random_end_point
        )
    elif choice == '2':
        num_simulations = 100000
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
            random_angle = random.uniform(0, 360)
            end_point_x = RANDOM_END_POINT_DISTANCE * math.cos(math.radians(random_angle))
            end_point_y = RANDOM_END_POINT_DISTANCE * math.sin(math.radians(random_angle))
            random_end_point = (end_point_x, end_point_y)
            if simulate_intercept_experiment_dynamic(
                    z_cbl_meters=SCENARIO_Z_CBL,
                    glide_ratio=SCENARIO_GLIDE_RATIO,
                    mc_for_sniffing_ms=SCENARIO_MC_SNIFF,
                    lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
                    lambda_strength=SCENARIO_LAMBDA_STRENGTH,
                    end_point=random_end_point
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
            'Sniffing Radius (Base)(m)': calculated_sniffing_radius,
            'Search Arc Angle (deg)': SEARCH_ARC_ANGLE_DEGREES,
            'Glide Path Length (m)': reported_glide_path_length,
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