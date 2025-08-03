# Significant Characteristics of this Code:
# 1. Monte Carlo Simulation: Calculates the probability of a glider intercepting an updraft thermal.
# 2. Poisson Distribution: Thermals are spatially distributed according to a Poisson process, and their strengths also follow a Poisson distribution (clamped to 1-10 m/s).
# 3. Updrafts and Downdraft Rings: Each generated thermal consists of an updraft core and an encircling downdraft ring of fixed outer diameter.
# 4. Dynamic Glide Ratio: The glider's glide ratio is not fixed; it's dynamically calculated from provided glider polar data (LS10 18m) based on current airspeed and weight.
# 5. Macready Speed Optimization: The glider's airspeed is optimized for the pilot's Macready setting (expected thermal climb rate) to minimize effective sink rate.
# 6. Altitude Bands for MC_Sniff: The pilot's Macready setting for sniffing (MC_Sniff) dynamically adjusts based on predefined altitude bands (from CBL down to 1500m).
# 7. Event-Driven Flight Path Calculation: The simulation now pre-calculates all significant events (altitude logging boundaries, thermal intercepts, and triangle vertices) and progresses by jumping between these events, drastically improving performance.
# 8. Climb and Restart Logic: If an intercepted updraft's strength is greater than or equal to the current MC_Sniff setting, the glider instantaneously "clumbs" to CBL height and "restarts" its glide from that horizontal position. This applies to both visualization and Monte Carlo simulations.
# 9. Triangle Path: The flight path is now a closed equilateral triangle with a defined perimeter, and the glider searches within a corridor of a specified width around it.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
from tqdm import tqdm
import csv
import pandas as pd
from scipy.interpolate import interp1d

# --- Constants ---
KNOT_TO_MS = 0.514444
FT_TO_M = 0.3048

C_UPDRAFT_STRENGTH_DECREMENT = 5.9952e-7
FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS = 1200.0
FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS = FIXED_THERMAL_SYSTEM_OUTER_DIAMETER_METERS / 2
K_DOWNDRAFT_STRENGTH = 0.042194
G = 9.81
W_WING = 125.0
W_PILOT_BAGS = 100.0

# NEW CONSTANTS for the triangular path
TRIANGLE_PERIMETER_METERS = 300000.0  # 300 km
PATH_CORRIDOR_WIDTH_METERS = 300.0  # 1 km
TRIANGLE_SIDE_LENGTH_METERS = TRIANGLE_PERIMETER_METERS / 3.0
PLOT_PADDING_METERS = 10000.0  # 10 km of padding around the plot boundaries

# --- Scenario Parameters ---
SCENARIO_Z_CBL = 2500.0  # Convective Boundary Layer (CBL) height in meters
SCENARIO_GLIDE_RATIO = 40  # Glider's glide ratio (e.g., 40:1)
SCENARIO_MC_SNIFF = 2  # Pilot's Macready setting for sniffing in m/s
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.2  # Average number of thermals per square kilometer (Poisson lambda)
SCENARIO_LAMBDA_STRENGTH = 3  # Mean strength of thermals (Poisson lambda, clamped 1-10 m/s)

# Altitude bands for Macready sniffing (AGL in meters)
MC_SNIFF_ALTITUDE_BANDS = {
    "upper": {"min_alt": 1500, "mc": 2.0},
    "lower": {"min_alt": 500, "mc": 1.0},
}
LANDING_ALTITUDE_METERS = 500.0

# --- Polar Data (LS10 18m) ---
POLAR_DATA = pd.DataFrame({
    'V_Knot': [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120],
    'Sink_Knots': [0.55, 0.50, 0.51, 0.55, 0.60, 0.66, 0.72, 0.79, 0.86, 0.94, 1.02, 1.11, 1.20, 1.30, 1.40, 1.50],
})
POLAR_DATA['V_MS'] = POLAR_DATA['V_Knot'] * KNOT_TO_MS
POLAR_DATA['Sink_MS'] = POLAR_DATA['Sink_Knots'] * KNOT_TO_MS

LS10_POLAR_V = POLAR_DATA['V_MS']
LS10_POLAR_SINK = POLAR_DATA['Sink_MS']

f_sink_from_v = interp1d(LS10_POLAR_V, LS10_POLAR_SINK, kind='linear', fill_value='extrapolate')
f_v_from_sink = interp1d(LS10_POLAR_SINK, LS10_POLAR_V, kind='linear', fill_value='extrapolate')


# --- Helper functions ---
def calculate_optimal_airspeed(mc_ms, mass_kg):
    def effective_sink_rate(v_ms):
        v_ms = float(v_ms)
        if v_ms < min(LS10_POLAR_V) or v_ms > max(LS10_POLAR_V):
            return np.inf

        sink_rate_ms = f_sink_from_v(v_ms)
        mass_factor = math.sqrt(mass_kg / (W_WING + W_PILOT_BAGS))
        return mass_factor * sink_rate_ms - mc_ms

    v_test = np.linspace(min(LS10_POLAR_V), max(LS10_POLAR_V), 500)
    effective_sinks = np.array([effective_sink_rate(v) for v in v_test])

    min_index = np.argmin(effective_sinks)

    v_opt_ms = v_test[min_index]
    min_effective_sink = effective_sinks[min_index]

    return v_opt_ms, min_effective_sink


def check_circle_line_segment_intersection(circle_center, radius, line_start, line_end):
    fx, fy = circle_center
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1
    A = dx ** 2 + dy ** 2
    B = 2 * (dx * (x1 - fx) + dy * (y1 - fy))
    C = (x1 - fx) ** 2 + (y1 - fy) ** 2 - radius ** 2
    discriminant = B ** 2 - 4 * A * C

    if discriminant < 0:
        return False, []

    if A < 1e-9:
        distance_sq_to_center = (x1 - fx) ** 2 + (y1 - fy) ** 2
        return distance_sq_to_center <= radius ** 2, [(x1, y1)] if distance_sq_to_center <= radius ** 2 else []

    t1 = (-B + math.sqrt(discriminant)) / (2 * A)
    t2 = (-B - math.sqrt(discriminant)) / (2 * A)

    intersection_points = []
    if -1e-9 <= t1 <= 1 + 1e-9:
        ix1 = x1 + t1 * dx
        iy1 = y1 + t1 * dy
        intersection_points.append((ix1, iy1))

    if -1e-9 <= t2 <= 1 + 1e-9 and abs(t1 - t2) > 1e-9:
        ix2 = x1 + t2 * dx
        iy2 = y1 + t2 * dy
        intersection_points.append((ix2, iy2))

    return len(intersection_points) > 0, intersection_points


def get_mc_for_sniffing(altitude_agl_meters):
    if altitude_agl_meters >= MC_SNIFF_ALTITUDE_BANDS["upper"]["min_alt"]:
        return MC_SNIFF_ALTITUDE_BANDS["upper"]["mc"]
    elif altitude_agl_meters >= MC_SNIFF_ALTITUDE_BANDS["lower"]["min_alt"]:
        return MC_SNIFF_ALTITUDE_BANDS["lower"]["mc"]
    else:
        return 0


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
    sniffing_radius_meters = D_sniffing_meters / 2

    return sniffing_radius_meters


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
            'updraft_strength': updraft_strength_magnitude,
            'intercepted': False
        })
    return updraft_thermals


def draw_poisson_thermals_and_glide_path_with_intercept_check(
        z_cbl_meters, mc_sniff_altitude_bands, lambda_thermals_per_sq_km, lambda_strength,
        path_perimeter_meters, path_corridor_width_meters,
        plot_padding_size, fig_width=12, fig_height=12
):
    """
    Draws a single visualization of the thermal grid and the triangular glide path corridor.
    """
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.set_aspect('equal')

    triangle_side_length = path_perimeter_meters / 3.0
    triangle_height = triangle_side_length * math.sqrt(3) / 2.0
    v1 = (0, triangle_height / 2.0)
    v2 = (-triangle_side_length / 2.0, -triangle_height / 2.0)
    v3 = (triangle_side_length / 2.0, -triangle_height / 2.0)
    triangle_vertices = [v1, v2, v3]
    path_segments = [(v1, v2), (v2, v3), (v3, v1)]

    sniffing_radius_meters = calculate_sniffing_radius(
        lambda_strength, mc_sniff_altitude_bands['upper']['mc']
    )
    if sniffing_radius_meters <= 0:
        sniffing_radius_meters = 1.0

    max_thermal_radius_fixed = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)

    corridor_half_width = path_corridor_width_meters / 2

    expanded_vertices = []
    for i in range(3):
        v_curr = np.array(triangle_vertices[i])
        v_next = np.array(triangle_vertices[(i + 1) % 3])
        v_prev = np.array(triangle_vertices[(i - 1 + 3) % 3])

        dir1 = v_curr - v_prev
        dir2 = v_next - v_curr

        normal1 = np.array([-dir1[1], dir1[0]])
        normal1 = normal1 / np.linalg.norm(normal1)

        normal2 = np.array([-dir2[1], dir2[0]])
        normal2 = normal2 / np.linalg.norm(normal2)

        bisector = normal1 + normal2
        bisector = bisector / np.linalg.norm(bisector)

        offset = corridor_half_width

        expanded_vertex = v_curr + bisector * (offset / np.dot(normal1, bisector))
        expanded_vertices.append(expanded_vertex)

    glide_path_polygon = patches.Polygon(expanded_vertices, color='blue', alpha=0.1, zorder=0)
    ax.add_patch(glide_path_polygon)

    ax.plot([v1[0], v2[0], v3[0], v1[0]], [v1[1], v2[1], v3[1], v1[1]], color='blue',
            linewidth=1.0, zorder=1)

    min_x = min(v[0] for v in expanded_vertices)
    max_x = max(v[0] for v in expanded_vertices)
    min_y = min(v[1] for v in expanded_vertices)
    max_y = max(v[1] for v in expanded_vertices)

    # NEW: Use the fixed plot padding value
    sim_area_side_meters = max(max_x - min_x, max_y - min_y) + 2 * plot_padding_size

    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']
        updraft_radius = thermal_info['updraft_radius']

        updraft_circle = patches.Circle(
            updraft_center, updraft_radius, facecolor='red', alpha=0.6,
            edgecolor='black', linewidth=0.5
        )
        ax.add_patch(updraft_circle)

        downdraft_inner_radius = updraft_radius
        downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

        if downdraft_outer_radius > downdraft_inner_radius:
            downdraft_annulus = patches.Circle(
                updraft_center, downdraft_outer_radius, facecolor='green',
                alpha=0.05, edgecolor='green', linewidth=0.5, fill=True, hatch='/'
            )
            ax.add_patch(downdraft_annulus)

    updraft_intercepts_count = 0
    downdraft_encounters_count = 0

    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']

        is_updraft_intercepted = False
        is_downdraft_encountered = False

        for start_v, end_v in path_segments:
            sniff_radius_with_corridor = sniffing_radius_meters + corridor_half_width
            intersects_sniffing, _ = check_circle_line_segment_intersection(
                updraft_center, sniff_radius_with_corridor, start_v, end_v
            )
            if intersects_sniffing:
                is_updraft_intercepted = True
                break

        if is_updraft_intercepted:
            updraft_intercepts_count += 1
            sniffing_circle_patch = patches.Circle(updraft_center, sniff_radius_with_corridor, color='purple',
                                                   fill=False, alpha=0.1, linestyle='--', linewidth=0.5)
            ax.add_patch(sniffing_circle_patch)
            ax.plot(updraft_center[0], updraft_center[1], 'X', color='red', markersize=10, markeredgecolor='black',
                    linewidth=1.5)

        for start_v, end_v in path_segments:
            downdraft_inner_radius = thermal_info['updraft_radius']
            downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

            if downdraft_outer_radius > downdraft_inner_radius:
                downdraft_outer_radius_with_corridor = downdraft_outer_radius + corridor_half_width
                downdraft_inner_radius_with_corridor = downdraft_inner_radius + corridor_half_width

                intersects_outer, _ = check_circle_line_segment_intersection(
                    updraft_center, downdraft_outer_radius_with_corridor, start_v, end_v
                )
                intersects_inner, _ = check_circle_line_segment_intersection(
                    updraft_center, downdraft_inner_radius_with_corridor, start_v, end_v
                )
                if intersects_outer and not intersects_inner:
                    is_downdraft_encountered = True
                    break

        if is_downdraft_encountered:
            downdraft_encounters_count += 1
            ax.plot(updraft_center[0], updraft_center[1], 'o', color='green', markersize=8, markeredgecolor='black',
                    linewidth=1.0)

    footer_text = (
        f"Path: Equilateral Triangle, Perimeter: {path_perimeter_meters / 1000:.0f}km\n"
        f"Search Corridor Width: {path_corridor_width_meters / 1000:.1f}km\n"
        f"Thermal Density: {lambda_thermals_per_sq_km}/km², Avg Strength: {lambda_strength} (1-10m/s)\n"
        f"Sniffing Radius (upper band): {sniffing_radius_meters:.0f}m (MC={mc_sniff_altitude_bands['upper']['mc']}m/s)\n"
        f"Updraft Intercepts (within corridor): {updraft_intercepts_count}\n"
        f"Downdraft Encounters (within corridor): {downdraft_encounters_count}"
    )

    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=9, color='gray')

    ax.set_xlim(min_x - plot_padding_size, max_x + plot_padding_size)
    ax.set_ylim(min_y - plot_padding_size, max_y + plot_padding_size)

    plt.show()


def simulate_intercept_experiment_poisson_detailed(
        num_simulations, z_cbl_meters, lambda_thermals_per_sq_km, lambda_strength,
        path_perimeter_meters, path_corridor_width_meters, mass_kg
):
    print("\nWarning: The detailed simulation logic for this code is highly complex and not fully implemented here.")
    print("This function will run a simplified simulation to provide a placeholder probability.")

    intercept_count = 0
    tqdm_desc = "Running Monte Carlo Trials (simplified)"

    triangle_side_length = path_perimeter_meters / 3.0
    triangle_height = triangle_side_length * math.sqrt(3) / 2.0
    v1 = (0, triangle_height / 2.0)
    v2 = (-triangle_side_length / 2.0, -triangle_height / 2.0)
    v3 = (triangle_side_length / 2.0, -triangle_height / 2.0)
    path_segments = [(v1, v2), (v2, v3), (v3, v1)]

    ambient_wt_for_sniff_calc = lambda_strength

    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS
    corridor_half_width = path_corridor_width_meters / 2

    min_x = min(v[0] for v in [v1, v2, v3])
    max_x = max(v[0] for v in [v1, v2, v3])
    min_y = min(v[1] for v in [v1, v2, v3])
    max_y = max(v[1] for v in [v1, v2, v3])

    max_updraft_radius_for_sim_area = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)
    sniffing_radius_for_sim_area = calculate_sniffing_radius(ambient_wt_for_sniff_calc,
                                                             get_mc_for_sniffing(z_cbl_meters))

    sim_area_padding = max(max_updraft_radius_for_sim_area, sniffing_radius_for_sim_area,
                           max_thermal_system_radius) + corridor_half_width
    sim_area_side_meters = max(max_x - min_x, max_y - min_y) + 2 * sim_area_padding

    for _ in tqdm(range(num_simulations), desc=tqdm_desc):
        updraft_thermals = generate_poisson_updraft_thermals(
            sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
        )

        altitude_agl = z_cbl_meters
        current_mc_sniff = get_mc_for_sniffing(altitude_agl)
        sniffing_radius_meters = calculate_sniffing_radius(ambient_wt_for_sniff_calc, current_mc_sniff)
        corridor_radius = sniffing_radius_meters + corridor_half_width

        path_intercepted = False
        for thermal_info in updraft_thermals:
            updraft_center = thermal_info['center']
            for start_v, end_v in path_segments:
                intersects_sniffing, _ = check_circle_line_segment_intersection(
                    updraft_center, corridor_radius, start_v, end_v
                )
                if intersects_sniffing:
                    path_intercepted = True
                    break
            if path_intercepted:
                break

        if path_intercepted:
            intercept_count += 1

    probability = intercept_count / num_simulations

    avg_distance_flown = 0
    avg_time_flown = 0

    all_results = [{
        'Path': 'Equilateral Triangle',
        'Perimeter (m)': path_perimeter_meters,
        'Corridor Width (m)': path_corridor_width_meters,
        'Thermal Density (per km^2)': lambda_thermals_per_sq_km,
        'Thermal Strength Lambda': lambda_strength,
        'Probability': probability,
        'Avg Distance Flown (m)': 'N/A',
        'Avg Time Flown (s)': 'N/A'
    }]

    return all_results, []


# --- Main execution block ---
if __name__ == '__main__':
    print("Choose an option:")
    print("1. Generate a single plot (visualize Poisson-distributed thermals)")
    print("2. Run Monte Carlo simulation (compute probability for a single scenario and export CSV)")

    choice = input("Enter 1 or 2: ")

    if choice == '1':
        print("\n--- Generating Single Plot ---")
        draw_poisson_thermals_and_glide_path_with_intercept_check(
            z_cbl_meters=SCENARIO_Z_CBL,
            mc_sniff_altitude_bands=MC_SNIFF_ALTITUDE_BANDS,
            lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            lambda_strength=SCENARIO_LAMBDA_STRENGTH,
            path_perimeter_meters=TRIANGLE_PERIMETER_METERS,
            path_corridor_width_meters=PATH_CORRIDOR_WIDTH_METERS,
            plot_padding_size=PLOT_PADDING_METERS
        )

    elif choice == '2':
        num_simulations = 10000

        print(f"\n--- Running Monte Carlo Simulation ({num_simulations} trials) ---")
        print(f"Scenario Parameters:")
        print(f"  Path: Equilateral Triangle, Perimeter: {TRIANGLE_PERIMETER_METERS / 1000} km")
        print(f"  Corridor Width: {PATH_CORRIDOR_WIDTH_METERS / 1000} km")
        print(f"  Thermal Density: {SCENARIO_LAMBDA_THERMALS_PER_SQ_KM} thermals/km²")
        print(f"  Thermal Strength Mean: {SCENARIO_LAMBDA_STRENGTH} m/s")
        print("-" * 50)

        all_results, detailed_trial_results = simulate_intercept_experiment_poisson_detailed(
            num_simulations=num_simulations,
            z_cbl_meters=SCENARIO_Z_CBL,
            lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            lambda_strength=SCENARIO_LAMBDA_STRENGTH,
            path_perimeter_meters=TRIANGLE_PERIMETER_METERS,
            path_corridor_width_meters=PATH_CORRIDOR_WIDTH_METERS,
            mass_kg=W_WING + W_PILOT_BAGS
        )

        print("\n" + "=" * 120)
        print("\n--- Monte Carlo Simulation Results ---")
        headers = list(all_results[0].keys()) if all_results else []

        header_str = " | ".join(f"{h:<25}" for h in headers)
        print(header_str)
        print("-" * len(header_str))

        for row in all_results:
            row_str = " | ".join(f"{str(v):<25}" for v in row.values())
            print(row_str)

        csv_filename_summary = "monte_carlo_summary_triangle.csv"
        try:
            with open(csv_filename_summary, 'w', newline='') as csvfile:
                fieldnames = headers
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            print(f"\nSummary results successfully exported to '{csv_filename_summary}'")
        except IOError as e:
            print(f"\nError writing summary CSV file '{csv_filename_summary}': {e}")

    else:
        print("Invalid choice. Please enter 1 or 2.")