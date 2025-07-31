# Significant Characteristics of this Code:
# 1. Monte Carlo Simulation: Calculates the probability of a glider intercepting an updraft thermal.
# 2. Poisson Distribution: Thermals are spatially distributed according to a Poisson process, and their strengths also follow a Poisson distribution (clamped to 1-10 m/s).
# 3. Updrafts and Downdraft Rings: Each generated thermal consists of an updraft core and an encircling downdraft ring of fixed outer diameter.
# 4. Dynamic Glide Ratio: The glider's glide ratio is not fixed; it's dynamically calculated from provided glider polar data (LS10 18m) based on current airspeed and weight.
# 5. Macready Speed Optimization: The glider's airspeed is optimized for the pilot's Macready setting (expected thermal climb rate) to minimize effective sink rate.
# 6. Altitude Bands for MC_Sniff: The pilot's Macready setting for sniffing (MC_Sniff) dynamically adjusts based on predefined altitude bands (from CBL down to 1500m).
# 7. Event-Driven Flight Path Calculation: The simulation now pre-calculates all significant events (altitude logging boundaries, thermal intercepts, and triangle vertices) and progresses by jumping between these events, drastically improving performance.
# 8. Climb and Restart Logic: If an intercepted updraft's strength is greater than or equal to the current MC_Sniff setting, the glider instantaneously "clumbs" to CBL height and "restarts" its glide from that horizontal position. This applies to both visualization and Monte Carlo.
# 9. Fixed Triangular Search Path: The glider's search path is now a fixed equilateral triangle of 300km total perimeter (100km per side), with a random initial orientation.
# 10. Detailed CSV Logging: A single flight simulation (Option 1) exports a detailed log to 'single_flight_log.csv', including altitude (integer), airspeed (knots, integer), sink rate (m/s, 2 decimals), and distance flown (km, 3 decimals) at specific altitude band boundaries.
# 11. Visual Plotting: Option 1 generates a plot showing thermal locations, downdraft rings, the glider's dynamic glide path (blue), and markers for updraft intercepts (red 'X'), downdraft encounters (green 'o'), and climb events (orange 'o').
# 12. No Sniffing Circles on Plot: The visual plotting of the large, transparent sniffing circles has been removed for a cleaner visualization.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
from tqdm import tqdm  # For progress bar
import csv  # For CSV export
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar  # For finding optimal airspeed for Macready

# --- Constants ---
KNOT_TO_MS = 0.514444  # 1 knot = 0.514444 m/s
MS_TO_KNOT = 1 / KNOT_TO_MS  # m/s to knots
FT_TO_M = 0.3048  # 1 foot = 0.3048 m
KMH_TO_MS = 1000 / 3600  # 1 km/h = 1000m / 3600s = 0.277778 m/s

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

# Maximum distance the glider will search for thermals
# Now fixed to the perimeter of the equilateral triangle
MAX_SEARCH_DISTANCE_METERS = 300000.0  # 300 kilometers (perimeter of triangle)
TRIANGLE_SIDE_LENGTH_METERS = MAX_SEARCH_DISTANCE_METERS / 3  # 100 kilometers per side

# Altitude step for dynamic glide path simulation (will be calculated dynamically)
# This is now the 'band height' rather than a fixed step for every log entry.
ALTITUDE_STEP_METERS = 10.0  # Default value, will be overridden by band calculation

# Number of height bands for MC_Sniff adjustment (default 3)
NUMBER_OF_HEIGHT_BANDS = 3

# Number of Monte Carlo simulation trials
NUM_SIMULATIONS_TRIALS = 100  # Changed default

# --- Glider Polar Data (Extracted from LS10 18m polar.pdf) ---
# Airspeed in km/h, Sink Speed in m/s
# LS10 18m 400kg (Light Blue Curve)
AIRSPEED_KMH_400KG = np.array([75, 90, 105, 120, 140, 160, 180, 200, 220])
SINK_MS_400KG = np.array([0.45, 0.47, 0.52, 0.65, 0.9, 1.25, 1.7, 2.25, 2.9])

# LS10 18m 600kg (Dark Blue Curve)
AIRSPEED_KMH_600KG = np.array([90, 100, 120, 140, 160, 180, 200, 220, 240])
SINK_MS_600KG = np.array([0.55, 0.57, 0.68, 0.9, 1.2, 1.6, 2.1, 2.7, 3.4])

# Convert airspeeds to m/s for interpolation
AIRSPEED_MS_400KG = AIRSPEED_KMH_400KG * KMH_TO_MS
AIRSPEED_MS_600KG = AIRSPEED_KMH_600KG * KMH_TO_MS

# Create interpolation functions for sink rate
# 'linear' interpolation is simple and generally sufficient for this
sink_rate_interp_400kg = interp1d(AIRSPEED_MS_400KG, SINK_MS_400KG, kind='linear', fill_value="extrapolate")
sink_rate_interp_600kg = interp1d(AIRSPEED_MS_600KG, SINK_MS_600KG, kind='linear', fill_value="extrapolate")

# --- Scenario Parameters (Moved to Global Scope for Easy Configuration) ---
# These parameters define the single simulation scenario.
SCENARIO_Z_CBL = 2500.0  # Convective Boundary Layer (CBL) height in meters
SCENARIO_GLIDER_WEIGHT_KG = 400  # Changed default
# SCENARIO_MC_SNIFF is now dynamically determined based on altitude bands
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.2  # Changed default
SCENARIO_LAMBDA_STRENGTH = 3.0  # Changed default
SCENARIO_MC_SNIFF_TOP_MANUAL = 2.0  # Changed default
SCENARIO_MC_SNIFF_BOTTOM_MANUAL = .5  # NEW: Manually adjustable MC_Sniff for the bottom band (1500m)


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


# --- Helper function to get sink rate from polar data ---
def get_sink_rate_from_polar(airspeed_ms, glider_weight_kg):
    """
    Returns the sink rate (m/s) for a given airspeed (m/s) and glider weight (kg)
    using the interpolated polar data. Handles out-of-bounds airspeeds.
    """
    if glider_weight_kg == 400:
        interp_func = sink_rate_interp_400kg
        min_airspeed_ms = AIRSPEED_MS_400KG.min()
        max_airspeed_ms = AIRSPEED_MS_400KG.max()
    elif glider_weight_kg == 600:
        interp_func = sink_rate_interp_600kg
        min_airspeed_ms = AIRSPEED_MS_600KG.min()
        max_airspeed_ms = AIRSPEED_MS_600KG.max()
    else:
        # Fallback for unsupported weight, or raise an error
        print(f"Warning: Unsupported glider weight {glider_weight_kg}kg. Using 400kg polar.")
        interp_func = sink_rate_interp_400kg
        min_airspeed_ms = AIRSPEED_MS_400KG.min()
        max_airspeed_ms = AIRSPEED_MS_400KG.max()

    if airspeed_ms < min_airspeed_ms or airspeed_ms > max_airspeed_ms:
        # Extrapolate, but warn or return a high sink rate for extreme values
        # For simplicity, we'll let interp1d handle extrapolation with fill_value="extrapolate"
        # but we can add a penalty for flying outside the polar's valid range.
        # Here, returning a very high sink rate for speeds outside the practical range.
        if airspeed_ms < min_airspeed_ms * 0.9 or airspeed_ms > max_airspeed_ms * 1.1:  # 10% buffer
            return 10.0  # Very high sink rate to discourage flying too slow/fast

    return interp_func(airspeed_ms).item()  # .item() converts numpy array to scalar


# --- Helper function to find airspeed for a given Macready setting ---
def get_airspeed_for_macready(mc_setting_ms, expected_thermal_climb_ms, glider_weight_kg):
    """
    Finds the optimal airspeed (m/s) for a given Macready setting (expected thermal climb rate)
    and glider weight, by minimizing the effective sink rate.

    Args:
        mc_setting_ms (float): The pilot's Macready setting in m/s (expected climb rate in next thermal).
        expected_thermal_climb_ms (float): The expected climb rate in the next thermal (often same as mc_setting_ms).
        glider_weight_kg (int): The weight of the glider (400 or 600 kg).

    Returns:
        float: The optimal airspeed in m/s.
    """

    # Objective function to minimize: effective sink rate (sink_rate + expected_thermal_climb) / airspeed
    # We want to find the airspeed that minimizes this ratio.
    def objective_function(airspeed_ms):
        if airspeed_ms <= 0:  # Avoid division by zero or negative airspeeds
            return np.inf
        sink_rate = get_sink_rate_from_polar(airspeed_ms, glider_weight_kg)
        return (sink_rate + expected_thermal_climb_ms) / airspeed_ms

    # Define the bounds for airspeed search (based on polar data)
    if glider_weight_kg == 400:
        bounds = (AIRSPEED_MS_400KG.min(), AIRSPEED_MS_400KG.max())
    elif glider_weight_kg == 600:
        bounds = (AIRSPEED_MS_600KG.min(), AIRSPEED_MS_600KG.max())
    else:
        bounds = (AIRSPEED_MS_400KG.min(), AIRSPEED_MS_400KG.max())  # Default if weight is unsupported

    # Use scalar minimization (e.g., Brent method)
    res = minimize_scalar(objective_function, bounds=bounds, method='bounded')

    if res.success:
        return res.x
    else:
        # Fallback if optimization fails, e.g., return best glide speed or a default
        print(f"Warning: Optimization for Macready airspeed failed. Returning default (e.g., 25 m/s).")
        return 25.0  # A reasonable default airspeed


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


def get_mc_sniff_for_altitude(current_altitude, z_cbl, manual_mc_sniff_top, manual_mc_sniff_bottom,
                              altitude_step_meters, number_of_bands):
    """
    Determines the MC_Sniff setting based on the current altitude band,
    using a manually set top MC_Sniff and interpolating down to a specified value at 1500m.

    Args:
        current_altitude (float): The current altitude of the glider in meters.
        z_cbl (float): Convective Boundary Layer (CBL) height in meters.
        manual_mc_sniff_top (float): The manually set MC_Sniff value for the top band.
        manual_mc_sniff_bottom (float): The manually set MC_Sniff value for the bottom band (at 1500m).
        altitude_step_meters (float): The calculated altitude step for each band (not directly used for interpolation, but for context).
        number_of_bands (int): The number of height bands (not directly used for interpolation, but for context).

    Returns:
        float: The MC_Sniff setting for the current altitude.
    """
    # Handle altitudes below 1500m
    if current_altitude <= 1500:
        return 0.0  # Glider focuses on staying airborne

    # Define the MC_Sniff values at the top (CBL) and bottom (1500m) of the relevant altitude range
    mc_sniff_at_cbl = manual_mc_sniff_top
    mc_sniff_at_1500m = manual_mc_sniff_bottom  # Now uses the new parameter

    # Ensure there's a valid range for interpolation
    if z_cbl - 1500 < EPSILON:  # If CBL is at or near 1500m
        return mc_sniff_at_cbl  # Return the top value if there's no range to interpolate

    # Define altitude points and corresponding MC_Sniff values for linear interpolation
    alt_points = np.array([1500.0, z_cbl])  # Order from low to high altitude
    mc_sniff_values = np.array([mc_sniff_at_1500m, mc_sniff_at_cbl])  # Corresponding MC_Sniff values

    # Create an interpolation function
    # bounds_error=False allows extrapolation, fill_value="extrapolate" handles values outside the range
    interp_mc_sniff = interp1d(alt_points, mc_sniff_values, kind='linear', fill_value="extrapolate", bounds_error=False)

    # Get the interpolated value
    sniff_value = interp_mc_sniff(current_altitude).item()

    # Clip the value to ensure it's non-negative and doesn't exceed the manually set top value
    sniff_value = max(0.0, min(manual_mc_sniff_top, sniff_value))

    return sniff_value


def draw_poisson_thermals_and_glide_path_with_intercept_check(
        z_cbl_meters, glider_weight_kg, lambda_thermals_per_sq_km, lambda_strength, manual_mc_sniff_top,
        manual_mc_sniff_bottom,
        fig_width=12, fig_height=12
):
    """
    Draws a single visualization of the thermal grid (Poisson distributed),
    with updrafts and encircling downdraft rings, glide path, and intercepts.
    The glide path search is limited to MAX_SEARCH_DISTANCE_METERS.
    The glide path is now dynamically simulated using polar data, including climb/restart.
    Also exports a CSV log of the flight path.
    """
    print("--- Starting draw_poisson_thermals_and_glide_path_with_intercept_check function ---")  # Debug print
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.set_aspect('equal')

    # Calculate dynamic ALTITUDE_STEP_METERS (this is the height of each band from CBL to 1500m)
    if z_cbl_meters <= 1500:
        print("Warning: CBL height is at or below 1500m. No altitude bands for MC_Sniff will be applied above 1500m.")
        calculated_altitude_step_meters = 0  # No steps if no range
    else:
        calculated_altitude_step_meters = round((z_cbl_meters - 1500) / NUMBER_OF_HEIGHT_BANDS, -2)
        if calculated_altitude_step_meters <= 0:  # Ensure a positive step if CBL > 1500
            calculated_altitude_step_meters = 100.0  # Fallback to 100m step if calculation yields 0 or less

    # Use the calculated step for the simulation loop
    global ALTITUDE_STEP_METERS  # Make sure to update the global constant for consistency
    ALTITUDE_STEP_METERS = calculated_altitude_step_meters

    # Determine specific altitudes for logging based on bands
    log_altitudes_raw = []
    if z_cbl_meters > 1500:
        # Add top of CBL
        log_altitudes_raw.append(z_cbl_meters)
        # Add band boundaries down to 1500m
        for i in range(1, NUMBER_OF_HEIGHT_BANDS + 1):
            band_boundary_altitude = z_cbl_meters - i * ALTITUDE_STEP_METERS
            if band_boundary_altitude >= 1500:
                log_altitudes_raw.append(band_boundary_altitude)
            else:  # If a calculated boundary falls below 1500, just add 1500 and break
                log_altitudes_raw.append(1500.0)
                break
    # Ensure 1500m is included if not already (important for the 1500-500m band start)
    if 1500.0 not in log_altitudes_raw:
        log_altitudes_raw.append(1500.0)
    # Add the final landing altitude
    log_altitudes_raw.append(500.0)

    # Convert to integers, remove duplicates, and sort in descending order
    log_altitudes = sorted(list(set([round(alt) for alt in log_altitudes_raw])), reverse=True)

    # Ensure the starting altitude is the first one, and 500m is the last, handling edge cases
    if log_altitudes[0] != round(z_cbl_meters):
        log_altitudes.insert(0, round(z_cbl_meters))
    if log_altitudes[-1] != 500:
        log_altitudes.append(500)
    log_altitudes = sorted(list(set(log_altitudes)), reverse=True)  # Re-sort and remove duplicates again

    # --- Calculations for Glide Path ---
    current_altitude = float(z_cbl_meters)  # Start at exact CBL
    current_x, current_y = 0.0, 0.0  # Starting point for the entire flight path
    total_horizontal_distance_covered = 0.0

    # Use a list of (x, y) coordinates with None for breaks for plotting
    plot_path_coords = [(current_x, current_y)]
    flight_log_data = []  # To store data for CSV export

    # Log initial state at CBL
    flight_log_data.append({
        'Altitude (m)': round(current_altitude),
        'Airspeed (knots)': round(get_airspeed_for_macready(
            get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, manual_mc_sniff_top, manual_mc_sniff_bottom,
                                      ALTITUDE_STEP_METERS, NUMBER_OF_HEIGHT_BANDS),
            get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, manual_mc_sniff_top, manual_mc_sniff_bottom,
                                      ALTITUDE_STEP_METERS, NUMBER_OF_HEIGHT_BANDS), glider_weight_kg) * MS_TO_KNOT),
        'Sink Rate (m/s)': round(get_sink_rate_from_polar(get_airspeed_for_macready(
            get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, manual_mc_sniff_top, manual_mc_sniff_bottom,
                                      ALTITUDE_STEP_METERS, NUMBER_OF_HEIGHT_BANDS),
            get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, manual_mc_sniff_top, manual_mc_sniff_bottom,
                                      ALTITUDE_STEP_METERS, NUMBER_OF_HEIGHT_BANDS), glider_weight_kg),
                                                          glider_weight_kg), 2),
        'Distance Flown (km)': round(total_horizontal_distance_covered / 1000, 3)
    })

    # Initial MC_Sniff for sniffing radius calculation (will be updated in loop)
    initial_mc_sniff_for_radius = get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, manual_mc_sniff_top,
                                                            manual_mc_sniff_bottom, ALTITUDE_STEP_METERS,
                                                            NUMBER_OF_HEIGHT_BANDS)
    sniffing_radius_meters = calculate_sniffing_radius(
        lambda_strength, initial_mc_sniff_for_radius  # Use lambda_strength as proxy for ambient Wt
    )
    if sniffing_radius_meters <= 0:
        print("Warning: Calculated Macready sniffing radius is non-positive. Setting to 1m for visualization.")
        sniffing_radius_meters = 1.0

    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS  # 600m

    # Determine simulation area side length based on MAX_SEARCH_DISTANCE_METERS
    # Ensure the simulation area is large enough to contain the entire triangular path plus buffer
    sim_area_side_meters = (
                                   MAX_SEARCH_DISTANCE_METERS + max_thermal_system_radius * 2 + sniffing_radius_meters * 2) * 1.1  # Add 10% padding

    # --- Generate Updraft Thermals (Poisson Distribution) ---
    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    # --- Triangular Path Setup ---
    initial_path_angle_radians = random.uniform(0, 2 * math.pi)
    # Angles for each side of the equilateral triangle
    segment_angles = [
        initial_path_angle_radians,
        initial_path_angle_radians + (2 * math.pi / 3),  # +120 degrees
        initial_path_angle_radians + (4 * math.pi / 3)  # +240 degrees
    ]
    current_segment_idx = 0
    distance_into_current_segment = 0.0  # Distance flown along the current 100km side

    # --- Event-Driven Simulation Loop ---
    # Lists to store distances for footer text
    red_non_climb_intercept_distances_meters = []
    orange_climb_start_distances_meters = []
    green_downdraft_encounter_distances_meters = []

    while current_altitude > 500 and total_horizontal_distance_covered < MAX_SEARCH_DISTANCE_METERS:
        mc_sniff_at_current_alt = get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, manual_mc_sniff_top,
                                                            manual_mc_sniff_bottom, ALTITUDE_STEP_METERS,
                                                            NUMBER_OF_HEIGHT_BANDS)
        airspeed_for_macready = get_airspeed_for_macready(mc_sniff_at_current_alt, mc_sniff_at_current_alt,
                                                          glider_weight_kg)
        base_sink_rate_ms = get_sink_rate_from_polar(airspeed_for_macready, glider_weight_kg)

        current_path_angle = segment_angles[current_segment_idx % 3]

        # Determine the maximum horizontal distance we can travel in this step
        # This is either to the end of the current triangle segment, or the overall max search distance.
        remaining_in_current_triangle_side = TRIANGLE_SIDE_LENGTH_METERS - distance_into_current_segment
        remaining_total_search_distance = MAX_SEARCH_DISTANCE_METERS - total_horizontal_distance_covered

        max_horizontal_dist_for_event_check = min(remaining_in_current_triangle_side, remaining_total_search_distance)

        if max_horizontal_dist_for_event_check < EPSILON:
            # If we are at the end of a segment or max distance, adjust and continue or break
            if remaining_in_current_triangle_side < EPSILON and total_horizontal_distance_covered < MAX_SEARCH_DISTANCE_METERS:
                current_segment_idx += 1
                distance_into_current_segment = 0.0
                continue  # Restart loop to get new angle and max_horizontal_dist_for_event_check
            else:
                break  # Reached end of total search distance or below 500m

        # Define the end point of the current potential segment for intersection checks
        potential_end_x_for_intercepts = current_x + max_horizontal_dist_for_event_check * math.cos(current_path_angle)
        potential_end_y_for_intercepts = current_y + max_horizontal_dist_for_event_check * math.sin(current_path_angle)

        # Collect all potential events in the current glide segment
        events = []  # Stores (horizontal_distance_from_current_pos, event_type, data)

        # 1. Altitude Logging Events
        for target_alt in log_altitudes:
            if target_alt < current_altitude:  # Only consider altitudes below current
                altitude_to_descend = current_altitude - target_alt
                if base_sink_rate_ms > EPSILON:
                    time_to_descend = altitude_to_descend / base_sink_rate_ms
                    horizontal_dist_to_alt_event = airspeed_for_macready * time_to_descend
                    if horizontal_dist_to_alt_event <= max_horizontal_dist_for_event_check + EPSILON:
                        events.append((horizontal_dist_to_alt_event, 'log_altitude', target_alt))

        # 2. Thermal Intercept Events (Updraft Sniffing and Downdraft Annulus)
        for thermal_info in updraft_thermals_info:
            intersects_sniffing, sniff_intersection_pts = check_circle_line_segment_intersection(
                thermal_info['center'], sniffing_radius_meters, (current_x, current_y),
                (potential_end_x_for_intercepts, potential_end_y_for_intercepts)
            )
            if intersects_sniffing:
                for pt in sniff_intersection_pts:
                    dist_from_current_pos = math.sqrt((pt[0] - current_x) ** 2 + (pt[1] - current_y) ** 2)
                    if dist_from_current_pos <= max_horizontal_dist_for_event_check + EPSILON:
                        # Calculate updraft strength at this specific intercept point
                        dist_to_thermal_center_at_pt = math.sqrt(
                            (pt[0] - thermal_info['center'][0]) ** 2 + (pt[1] - thermal_info['center'][1]) ** 2)
                        updraft_strength_at_pt = thermal_info['updraft_strength'] * (
                                    1 - (dist_to_thermal_center_at_pt / thermal_info['updraft_radius']) ** 3)
                        events.append((dist_from_current_pos, 'updraft_intercept',
                                       {'thermal_info': thermal_info, 'intercept_point': pt,
                                        'updraft_strength_at_pt': updraft_strength_at_pt}))

            # Check for downdraft annulus encounters
            downdraft_inner_radius = thermal_info['updraft_radius']
            downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS
            if downdraft_outer_radius > downdraft_inner_radius:
                intersects_downdraft, downdraft_intersection_pts = check_circle_line_segment_intersection(
                    thermal_info['center'], downdraft_outer_radius, (current_x, current_y),
                    (potential_end_x_for_intercepts, potential_end_y_for_intercepts)
                )
                if intersects_downdraft:
                    for pt in downdraft_intersection_pts:
                        dist_from_current_pos = math.sqrt((pt[0] - current_x) ** 2 + (pt[1] - current_y) ** 2)
                        # Ensure the point is actually within the annulus, not just the outer circle
                        dist_to_thermal_center_at_pt = math.sqrt(
                            (pt[0] - thermal_info['center'][0]) ** 2 + (pt[1] - thermal_info['center'][1]) ** 2)
                        if downdraft_inner_radius <= dist_to_thermal_center_at_pt <= downdraft_outer_radius:
                            if dist_from_current_pos <= max_horizontal_dist_for_event_check + EPSILON:
                                events.append((dist_from_current_pos, 'downdraft_encounter',
                                               {'thermal_info': thermal_info, 'encounter_point': pt}))

        # 3. Vertex Transition Event
        if remaining_in_current_triangle_side > EPSILON:  # Only add if not already at end of segment
            events.append((remaining_in_current_triangle_side, 'vertex_transition', None))

        # 4. Max Distance Event (overall search limit)
        if remaining_total_search_distance > EPSILON:  # Only add if not already at max distance
            events.append((remaining_total_search_distance, 'max_distance', None))

        # Sort events by horizontal distance from current position
        events.sort(key=lambda x: x[0])

        # Process the very next event
        event_processed_in_this_iteration = False
        for event_dist_from_current, event_type, event_data in events:
            if event_dist_from_current < EPSILON:  # Skip events at current position unless they are the *only* event
                continue

            horizontal_dist_this_step = event_dist_from_current

            # Ensure we don't overshoot the overall max search distance
            if total_horizontal_distance_covered + horizontal_dist_this_step > MAX_SEARCH_DISTANCE_METERS + EPSILON:
                horizontal_dist_this_step = MAX_SEARCH_DISTANCE_METERS - total_horizontal_distance_covered
                if horizontal_dist_this_step < EPSILON:  # Already at or past max distance
                    break  # Exit event processing and outer loop

            time_taken_this_step = horizontal_dist_this_step / airspeed_for_macready if airspeed_for_macready > EPSILON else 0.0
            altitude_change_this_step = base_sink_rate_ms * time_taken_this_step

            new_x = current_x + horizontal_dist_this_step * math.cos(current_path_angle)
            new_y = current_y + horizontal_dist_this_step * math.sin(current_path_angle)
            new_altitude = current_altitude - altitude_change_this_step

            # Update global state for this step
            current_x, current_y = new_x, new_y
            current_altitude = new_altitude
            total_horizontal_distance_covered += horizontal_dist_this_step
            distance_into_current_segment += horizontal_dist_this_step

            plot_path_coords.append((current_x, current_y))  # Add the new point to the path

            event_processed_in_this_iteration = True

            # Handle the specific event type
            if event_type == 'log_altitude':
                current_altitude = event_data  # Ensure exact altitude for logging
                flight_log_data.append({
                    'Altitude (m)': round(current_altitude),
                    'Airspeed (knots)': round(airspeed_for_macready * MS_TO_KNOT),
                    'Sink Rate (m/s)': round(base_sink_rate_ms, 2),
                    'Distance Flown (km)': round(total_horizontal_distance_covered / 1000, 3)
                })
            elif event_type == 'updraft_intercept':
                updraft_strength_at_pt = event_data['updraft_strength_at_pt']
                thermal_info = event_data['thermal_info']

                if updraft_strength_at_pt >= mc_sniff_at_current_alt and updraft_strength_at_pt > EPSILON:
                    # CLIMB AND RESTART
                    ax.plot(current_x, current_y, 'o', color='orange', markersize=8, markeredgecolor='black',
                            label='Climb Start' if not orange_climb_start_distances_meters else "")
                    orange_climb_start_distances_meters.append(
                        total_horizontal_distance_covered)  # Log distance for climb start

                    flight_log_data.append({
                        'Altitude (m)': round(z_cbl_meters),
                        'Airspeed (knots)': round(airspeed_for_macready * MS_TO_KNOT),
                        'Sink Rate (m/s)': round(-updraft_strength_at_pt, 2),  # Negative for climb
                        'Distance Flown (km)': round(total_horizontal_distance_covered / 1000, 3)
                    })
                    current_altitude = z_cbl_meters  # Reset altitude
                    # current_x, current_y remain the same. total_horizontal_distance_covered and
                    # distance_into_current_segment also remain as they were at the point of climb.
                    # Crucially, break from this inner event loop and restart the outer while loop
                    # to re-evaluate events from the new (reset) altitude.
                    plot_path_coords.append(None)  # Break the line after a climb
                    plot_path_coords.append((current_x, current_y))  # Start new segment from current (x,y)
                    break  # Break from for event_dist, event_type in events loop
                else:
                    # Not a climb-worthy updraft, still log as an intercept for plotting
                    ax.plot(current_x, current_y, 'X', color='red', markersize=10, markeredgecolor='black',
                            linewidth=1.5)
                    red_non_climb_intercept_distances_meters.append(
                        total_horizontal_distance_covered)  # Log distance for non-climb intercept
            elif event_type == 'downdraft_encounter':
                ax.plot(current_x, current_y, 'o', color='green', markersize=8, markeredgecolor='black', linewidth=1.0)
                green_downdraft_encounter_distances_meters.append(
                    total_horizontal_distance_covered)  # Log distance for downdraft encounter
            elif event_type == 'vertex_transition':
                # Ensure we are exactly at the vertex by setting distance_into_current_segment to 0
                # and incrementing segment index. This handles floating point inaccuracies.
                current_segment_idx += 1
                distance_into_current_segment = 0.0
                plot_path_coords.append(None)  # Break the line at the vertex
                plot_path_coords.append((current_x, current_y))  # Start new segment from vertex
            elif event_type == 'max_distance':
                break  # Reached max search distance, terminate simulation

            # If we processed an event and haven't broken (e.g., due to climb or max_distance),
            # then we should stop processing further events for this iteration and re-evaluate
            # the state for the next step.
            break  # Process only the first relevant event per outer loop iteration

        if not event_processed_in_this_iteration:  # If no events were processed (e.g., only past events remained)
            break  # Exit to prevent infinite loop

        # If loop broke due to climb, continue outer while loop
        if current_altitude == z_cbl_meters and event_type == 'updraft_intercept':
            continue  # Continue outer while loop

    # --- Plotting the Path Segments ---
    # Extract x and y coordinates from the plot_path_coords list, handling None for breaks
    path_xs = [p[0] if p is not None else None for p in plot_path_coords]
    path_ys = [p[1] if p is not None else None for p in plot_path_coords]

    ax.plot(
        path_xs,
        path_ys,
        color='blue',
        linewidth=2,
        label=f'Dynamic Glide Path (Weight={glider_weight_kg}kg)'
    )
    ax.legend()
    if not plot_path_coords:  # Check if any points were added
        print("No glide path segments generated. Check Z_CBL or other parameters.")
        plt.title("No Glide Path Generated")

    # --- Plot Thermals (Updrafts and Encircling Downdraftso) ---
    for thermal_info in updraft_thermals_info:
        updraft_center = thermal_info['center']
        updraft_radius = thermal_info['updraft_radius']

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
        downdraft_inner_radius = updraft_radius
        downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

        if downdraft_outer_radius > downdraft_inner_radius:
            downdraft_annulus = patches.Circle(
                updraft_center,
                downdraft_outer_radius,
                facecolor='green',
                alpha=0.2,  # Increased alpha for better visibility
                edgecolor='green',
                linewidth=0.5,
                fill=True,
                hatch='/'
            )
            ax.add_patch(downdraft_annulus)

    # Sort distances for consistent display
    red_non_climb_intercept_distances_meters.sort()
    orange_climb_start_distances_meters.sort()
    green_downdraft_encounter_distances_meters.sort()

    # --- Construct the footer text for the plot ---
    # Updated to show counts instead of distances
    footer_text = (
        f"Z={z_cbl_meters}m, Glider Weight={glider_weight_kg}kg\n"
        f"Search Limit: {MAX_SEARCH_DISTANCE_METERS / 1000:.0f}km (Triangular Path), Actual Glide Distance: {total_horizontal_distance_covered / 1000:.1f}km\n"
        f"Altitude Step (Band): {ALTITUDE_STEP_METERS}m, Bands: {NUMBER_OF_HEIGHT_BANDS}\n"
        f"Pilot MC Sniff (Manual Top): {manual_mc_sniff_top:.1f} m/s (Dynamic)\n"
        f"Pilot MC Sniff (Manual Bottom): {manual_mc_sniff_bottom:.1f} m/s (Dynamic)\n"  # Added new line for bottom MC_Sniff
        f"Thermal Density: {lambda_thermals_per_sq_km}/km², Avg Strength: {lambda_strength} (1-10m/s)\n"
        f"Non-Climb Updraft Intercepts: {len(red_non_climb_intercept_distances_meters)}\n"  # Changed to count
        f"Climb Start Intercepts: {len(orange_climb_start_distances_meters)}\n"  # Changed to count
        f"Downdraft Encounter Intercepts: {len(green_downdraft_encounter_distances_meters)}"  # Changed to count
    )

    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=9, color='gray')

    print(f"Total updrafts generated: {len(updraft_thermals_info)}")
    print(f"\nIntercepts/Encounters with glide path:")
    print(f"  - Non-Climb Updraft Intercepts: {len(red_non_climb_intercept_distances_meters)}")
    print(f"  - Climb Start Intercepts: {len(orange_climb_start_distances_meters)}")
    print(f"  - Downdraft Annulus Encounters: {len(green_downdraft_encounter_distances_meters)}")

    if red_non_climb_intercept_distances_meters:
        print("\nNon-Climb Updraft Intercept Distances (meters):")
        for i, dist in enumerate(red_non_climb_intercept_distances_meters):
            print(f"  Intercept {i + 1}: {dist:.2f} m")
    else:
        print("\nNo non-climb updraft intercepts.")

    if orange_climb_start_distances_meters:
        print("\nClimb Start Intercept Distances (meters):")
        for i, dist in enumerate(orange_climb_start_distances_meters):
            print(f"  Climb Start {i + 1}: {dist:.2f} m")
    else:
        print("\nNo climb start intercepts.")

    if green_downdraft_encounter_distances_meters:
        print("\nInitial Encounter Distances for Downdraft Annuli (meters):")
        for i, dist in enumerate(green_downdraft_encounter_distances_meters):
            print(f"  Encounter {i + 1}: {dist:.2f} m")
    else:
        print("\nNo initial encounters for Downdraft Annuli.")

    # --- Adjust Plot Limits to fit everything ---
    # Calculate the bounding box of the triangle
    # Start at (0,0)
    vertex_coords = [(0.0, 0.0)]
    current_x_vertex, current_y_vertex = 0.0, 0.0
    for i in range(3):
        angle = segment_angles[i]
        current_x_vertex += TRIANGLE_SIDE_LENGTH_METERS * math.cos(angle)
        current_y_vertex += TRIANGLE_SIDE_LENGTH_METERS * math.sin(angle)
        vertex_coords.append((current_x_vertex, current_y_vertex))

    all_x_coords = [p[0] for p in vertex_coords]
    all_y_coords = [p[1] for p in vertex_coords]

    min_x = min(all_x_coords)
    max_x = max(all_x_coords)
    min_y = min(all_y_coords)
    max_y = max(all_y_coords)

    # Add a fixed padding (e.g., 10% of the triangle's side length)
    plot_padding_meters = TRIANGLE_SIDE_LENGTH_METERS * 0.10  # 10% of 100km = 10km

    ax.set_xlim(min_x - plot_padding_meters, max_x + plot_padding_meters)
    ax.set_ylim(min_y - plot_padding_meters, max_y + plot_padding_meters)

    # Print for debugging
    print(f"Calculated sim_area_side_meters: {sim_area_side_meters:.2f} m")
    print(f"Set plot xlim: {ax.get_xlim()}")
    print(f"Set plot ylim: {ax.get_ylim()}")

    # --- Export flight log to CSV ---
    csv_filename = "single_flight_log.csv"
    print(f"Attempting to write {len(flight_log_data)} entries to '{csv_filename}'...")  # Debug print
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            # Updated fieldnames for desired output
            fieldnames = ['Altitude (m)', 'Airspeed (knots)', 'Sink Rate (m/s)', 'Distance Flown (km)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in flight_log_data:
                writer.writerow(row)
        print(f"\nDetailed single flight log successfully exported to '{csv_filename}'")
    except IOError as e:
        print(f"\nError writing to CSV file '{csv_filename}': {e}")

    plt.show()  # Moved plt.show() after CSV export


def simulate_intercept_experiment_poisson(
        z_cbl_meters, glider_weight_kg, lambda_thermals_per_sq_km, lambda_strength, manual_mc_sniff_top,
        manual_mc_sniff_bottom
):
    """
    Performs a single Monte Carlo experiment with Poisson-distributed updraft thermals
    to check for the probability of completing the entire triangular path without landing.

    Args:
        z_cbl_meters (float): The convective Boundary Layer height (Z) for this simulation.
        glider_weight_kg (int): The weight of the glider (400 or 600 kg).
        lambda_thermals_per_sq_km (float): The average number of thermals per square kilometer.
        lambda_strength (float): The mean (lambda) for the Poisson distribution of thermal strength magnitude.
        manual_mc_sniff_top (float): The manually set MC_Sniff value for the top band.
        manual_mc_sniff_bottom (float): The manually set MC_Sniff value for the bottom band (at 1500m).

    Returns:
        tuple: (total_horizontal_distance_covered, list of climb_intercept_distances)
               The total horizontal_distance_covered in this trial, and a list of distances
               where climb-worthy updrafts were intercepted.
    """
    # Calculate dynamic ALTITUDE_STEP_METERS for this simulation
    if z_cbl_meters <= 1500:
        calculated_altitude_step_meters = 0
    else:
        calculated_altitude_step_meters = round((z_cbl_meters - 1500) / NUMBER_OF_HEIGHT_BANDS, -2)
        if calculated_altitude_step_meters <= 0:
            calculated_altitude_step_meters = 100.0

    # --- Glide Path Setup ---
    current_altitude = z_cbl_meters
    current_x, current_y = 0.0, 0.0  # Glide path starts at origin of simulation area
    total_horizontal_distance_covered = 0.0
    climb_intercept_distances = []  # List to store distances of climb updraft intercepts

    if current_altitude <= 500:  # Glider starts below or at landing height
        return 0.0, []  # Returns 0 distance and empty list if it can't even start

    # Initial MC_Sniff for sniffing radius calculation (will be updated in loop)
    initial_mc_sniff_for_radius = get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, manual_mc_sniff_top,
                                                            manual_mc_sniff_bottom,
                                                            calculated_altitude_step_meters, NUMBER_OF_HEIGHT_BANDS)
    sniffing_radius_meters = calculate_sniffing_radius(
        lambda_strength, initial_mc_sniff_for_radius
    )
    if sniffing_radius_meters <= 0:
        # If sniffing radius is non-positive, no thermals can be intercepted, so triangle cannot be completed
        return 0.0, []  # Returns 0 distance and empty list if no sniffing is possible

    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

    # Simulation area side should cover the effective glide path plus max thermal/sniffing radius on both sides
    sim_area_side_meters = (
                                   MAX_SEARCH_DISTANCE_METERS + max_thermal_system_radius * 2 + sniffing_radius_meters * 2) * 1.1  # Add 10% padding

    # --- Generate Updraft Thermals (Poisson Distribution) ---
    updraft_thermals = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    # --- Triangular Path Setup for Monte Carlo ---
    initial_path_angle_radians = random.uniform(0, 2 * math.pi)
    segment_angles = [
        initial_path_angle_radians,
        initial_path_angle_radians + (2 * math.pi / 3),
        initial_path_angle_radians + (4 * math.pi / 3)
    ]
    current_segment_idx = 0
    distance_into_current_segment = 0.0

    # --- Dynamic Glide Path Simulation Loop for Completion Check ---
    while current_altitude > 500 and total_horizontal_distance_covered < MAX_SEARCH_DISTANCE_METERS:
        mc_sniff_at_current_alt = get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, manual_mc_sniff_top,
                                                            manual_mc_sniff_bottom, calculated_altitude_step_meters,
                                                            NUMBER_OF_HEIGHT_BANDS)
        airspeed_for_macready = get_airspeed_for_macready(mc_sniff_at_current_alt, mc_sniff_at_current_alt,
                                                          glider_weight_kg)
        base_sink_rate_ms = get_sink_rate_from_polar(airspeed_for_macready, glider_weight_kg)

        current_path_angle = segment_angles[current_segment_idx % 3]

        remaining_in_current_triangle_side = TRIANGLE_SIDE_LENGTH_METERS - distance_into_current_segment
        remaining_total_search_distance = MAX_SEARCH_DISTANCE_METERS - total_horizontal_distance_covered

        max_horizontal_dist_for_event_check = min(remaining_in_current_triangle_side, remaining_total_search_distance)

        if max_horizontal_dist_for_event_check < EPSILON:
            if remaining_in_current_triangle_side < EPSILON and total_horizontal_distance_covered < MAX_SEARCH_DISTANCE_METERS:
                current_segment_idx += 1
                distance_into_current_segment = 0.0
                continue
            else:
                break  # Reached end of total search distance or below 500m

        potential_end_x_for_intercepts = current_x + max_horizontal_dist_for_event_check * math.cos(current_path_angle)
        potential_end_y_for_intercepts = current_y + max_horizontal_dist_for_event_check * math.sin(current_path_angle)

        # Collect all potential events in the current glide segment
        events = []  # Stores (horizontal_distance_from_current_pos, event_type, data)

        # 1. Altitude Logging Event (only to 500m for Monte Carlo termination)
        altitude_to_descend_to_500m = current_altitude - 500.0
        if base_sink_rate_ms > EPSILON:
            time_to_descend_to_500m = altitude_to_descend_to_500m / base_sink_rate_ms
            horizontal_dist_to_500m_event = airspeed_for_macready * time_to_descend_to_500m
            if horizontal_dist_to_500m_event <= max_horizontal_dist_for_event_check + EPSILON:
                events.append((horizontal_dist_to_500m_event, 'land_at_500m', None))

        # 2. Thermal Intercept Events (Updraft Sniffing)
        for thermal_info in updraft_thermals:
            intersects_sniffing, sniff_intersection_pts = check_circle_line_segment_intersection(
                thermal_info['center'], sniffing_radius_meters, (current_x, current_y),
                (potential_end_x_for_intercepts, potential_end_y_for_intercepts)
            )
            if intersects_sniffing:
                for pt in sniff_intersection_pts:
                    dist_from_current_pos = math.sqrt((pt[0] - current_x) ** 2 + (pt[1] - current_y) ** 2)
                    if dist_from_current_pos <= max_horizontal_dist_for_event_check + EPSILON:
                        dist_to_thermal_center_at_pt = math.sqrt(
                            (pt[0] - thermal_info['center'][0]) ** 2 + (pt[1] - thermal_info['center'][1]) ** 2)
                        updraft_strength_at_pt = thermal_info['updraft_strength'] * (
                                    1 - (dist_to_thermal_center_at_pt / thermal_info['updraft_radius']) ** 3)

                        if updraft_strength_at_pt >= mc_sniff_at_current_alt and updraft_strength_at_pt > EPSILON:
                            events.append((dist_from_current_pos, 'climb_worthy_intercept', updraft_strength_at_pt))

        # 3. Vertex Transition Event
        if remaining_in_current_triangle_side > EPSILON:
            events.append((remaining_in_current_triangle_side, 'vertex_transition', None))

        # 4. Max Distance Event (overall search limit)
        if remaining_total_search_distance > EPSILON:
            events.append((remaining_total_search_distance, 'max_distance', None))

        # Sort events by horizontal distance from current position
        events.sort(key=lambda x: x[0])

        # Process the very next event
        event_processed_in_this_iteration = False
        for event_dist_from_current, event_type, event_data in events:
            if event_dist_from_current < EPSILON:
                continue

            horizontal_dist_this_step = event_dist_from_current

            if total_horizontal_distance_covered + horizontal_dist_this_step > MAX_SEARCH_DISTANCE_METERS + EPSILON:
                horizontal_dist_this_step = MAX_SEARCH_DISTANCE_METERS - total_horizontal_distance_covered
                if horizontal_dist_this_step < EPSILON:
                    break

            time_taken_this_step = horizontal_dist_this_step / airspeed_for_macready if airspeed_for_macready > EPSILON else 0.0
            altitude_change_this_step = base_sink_rate_ms * time_taken_this_step

            current_x += horizontal_dist_this_step * math.cos(current_path_angle)
            current_y += horizontal_dist_this_step * math.sin(current_path_angle)
            current_altitude -= altitude_change_this_step
            total_horizontal_distance_covered += horizontal_dist_this_step
            distance_into_current_segment += horizontal_dist_this_step

            event_processed_in_this_iteration = True

            if event_type == 'climb_worthy_intercept':
                climb_intercept_distances.append(total_horizontal_distance_covered)  # Record the distance
                current_altitude = z_cbl_meters  # Instantaneous climb
                break  # Break from inner event loop to restart outer while loop
            elif event_type == 'land_at_500m':
                current_altitude = 500.0  # Ensure exact 500m
                break  # Terminate trial, glider landed
            elif event_type == 'vertex_transition':
                current_segment_idx += 1
                distance_into_current_segment = 0.0
            elif event_type == 'max_distance':
                break  # Terminate trial, max distance reached

            break  # Process only the first relevant event per outer loop iteration

        if not event_processed_in_this_iteration:
            break

        if current_altitude <= 500 or total_horizontal_distance_covered >= MAX_SEARCH_DISTANCE_METERS:
            break

    return total_horizontal_distance_covered, climb_intercept_distances


# --- Main execution block ---
if __name__ == '__main__':
    print("Choose an option:")
    print(
        "1. Generate a single plot (visualize Poisson-distributed thermals with encircling downdrafts and flight log)")
    print("2. Run Monte Carlo simulation (compute probability for a single scenario and export CSV)")

    choice = input("Enter 1 or 2: ")

    # The scenario parameters are now defined globally at the top of the script.
    # We reference them directly here.

    if choice == '1':
        print("\n--- Generating Single Plot with Poisson Thermals (Updrafts with Encircling Downdrafts) ---")
        draw_poisson_thermals_and_glide_path_with_intercept_check(
            z_cbl_meters=SCENARIO_Z_CBL,
            glider_weight_kg=SCENARIO_GLIDER_WEIGHT_KG,
            lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            lambda_strength=SCENARIO_LAMBDA_STRENGTH,
            manual_mc_sniff_top=SCENARIO_MC_SNIFF_TOP_MANUAL,
            manual_mc_sniff_bottom=SCENARIO_MC_SNIFF_BOTTOM_MANUAL  # Pass the new parameter
        )

    elif choice == '2':
        # Use the globally defined NUM_SIMULATIONS_TRIALS
        num_simulations = NUM_SIMULATIONS_TRIALS

        print(f"\n--- Running Monte Carlo Simulation for a Single Scenario ({num_simulations} trials) ---")
        print(f"Scenario Parameters:")
        print(f"  Z (CBL Height): {SCENARIO_Z_CBL} m")
        print(f"  Glider Weight: {SCENARIO_GLIDER_WEIGHT_KG} kg")
        print(f"  Thermal Density (Lambda): {SCENARIO_LAMBDA_THERMALS_PER_SQ_KM} thermals/km²")
        print(f"  Thermal Strength Mean (Lambda): {SCENARIO_LAMBDA_STRENGTH} m/s (clamped 1-10 m/s)")
        print(f"  Pilot MC Sniff (Manual Top): {SCENARIO_MC_SNIFF_TOP_MANUAL} m/s")
        print(
            f"  Pilot MC Sniff (Manual Bottom): {SCENARIO_MC_SNIFF_BOTTOM_MANUAL} m/s")  # New print for bottom MC_Sniff
        print("-" * 50)

        triangle_completion_count = 0
        non_completed_distances_flown = []  # Changed to store only non-completed distances
        detailed_trial_results = []

        # Calculate dynamic ALTITUDE_STEP_METERS for Monte Carlo
        if SCENARIO_Z_CBL <= 1500:
            mc_altitude_step_meters = 0
        else:
            mc_altitude_step_meters = round((SCENARIO_Z_CBL - 1500) / NUMBER_OF_HEIGHT_BANDS, -2)
            if mc_altitude_step_meters <= 0:
                mc_altitude_step_meters = 100.0

        tqdm_desc = "Running Monte Carlo Trials"
        for i in tqdm(range(num_simulations), desc=tqdm_desc):
            actual_distance_flown, climb_intercepts_for_trial = simulate_intercept_experiment_poisson(
                z_cbl_meters=SCENARIO_Z_CBL,
                glider_weight_kg=SCENARIO_GLIDER_WEIGHT_KG,
                lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
                lambda_strength=SCENARIO_LAMBDA_STRENGTH,
                manual_mc_sniff_top=SCENARIO_MC_SNIFF_TOP_MANUAL,
                manual_mc_sniff_bottom=SCENARIO_MC_SNIFF_BOTTOM_MANUAL
            )

            if actual_distance_flown >= MAX_SEARCH_DISTANCE_METERS - EPSILON:
                triangle_completion_count += 1
            else:
                non_completed_distances_flown.append(actual_distance_flown)  # Only add if not completed

            detailed_trial_results.append({
                'Trial': i + 1,
                'Distance Flown (m)': actual_distance_flown,
                'Climb Intercept Distances (m)': ", ".join(
                    [f"{d:.2f}" for d in climb_intercepts_for_trial]) if climb_intercepts_for_trial else "N/A"
            })

        probability_of_completion = triangle_completion_count / num_simulations

        average_distance_flown = "N/A"
        if non_completed_distances_flown:  # Calculate average only for non-completed flights
            average_distance_flown = np.mean(non_completed_distances_flown)

        # Calculate the sniffing radius for display in results (using top band MC_Sniff as representative)
        representative_mc_sniff = get_mc_sniff_for_altitude(SCENARIO_Z_CBL, SCENARIO_Z_CBL,
                                                            SCENARIO_MC_SNIFF_TOP_MANUAL,
                                                            SCENARIO_MC_SNIFF_BOTTOM_MANUAL,
                                                            mc_altitude_step_meters, NUMBER_OF_HEIGHT_BANDS)
        calculated_sniffing_radius = calculate_sniffing_radius(
            SCENARIO_LAMBDA_STRENGTH, representative_mc_sniff
        )
        # The reported glide path length is now the MAX_SEARCH_DISTANCE_METERS
        reported_glide_path_length = MAX_SEARCH_DISTANCE_METERS

        all_results = [{
            'Z (m)': SCENARIO_Z_CBL,
            'Glider Weight (kg)': SCENARIO_GLIDER_WEIGHT_KG,
            'Wt_Ambient (m/s)': SCENARIO_LAMBDA_STRENGTH,  # Using lambda_strength as proxy for ambient Wt
            'MC_Sniff (m/s) (Top Band)': representative_mc_sniff,  # Show representative MC_Sniff
            'Sniffing Radius (m)': calculated_sniffing_radius,
            'Max Glide Path Length (m)': reported_glide_path_length,  # Now represents the search limit
            'Thermal Density (per km^2)': SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
            'Thermal Strength Lambda': SCENARIO_LAMBDA_STRENGTH,
            'Probability of Completing Triangle': probability_of_completion,
            'Avg Distance Flown (m)': average_distance_flown
        }]

        print("\n" + "=" * 120)
        print("\n--- Monte Carlo Simulation Results for Single Scenario ---")
        headers = [
            'Z (m)', 'Glider Weight (kg)', 'Wt_Ambient (m/s)', 'MC_Sniff (m/s) (Top Band)', 'Sniffing Radius (m)',
            'Max Glide Path Length (m)', 'Thermal Density (per km^2)', 'Thermal Strength Lambda',
            'Probability of Completing Triangle', 'Avg Distance Flown (m)'
        ]
        print(
            f"{headers[0]:<8} | {headers[1]:<18} | {headers[2]:<18} | {headers[3]:<25} | {headers[4]:<22} | "
            f"{headers[5]:<25} | {headers[6]:<25} | {headers[7]:<25} | {headers[8]:<35} | {headers[9]:<25}"
        )
        print("-" * 280)

        for row in all_results:
            avg_dist_str = f"{row['Avg Distance Flown (m)']:.2f}" if isinstance(row['Avg Distance Flown (m)'],
                                                                                float) else str(
                row['Avg Distance Flown (m)'])
            print(
                f"{row['Z (m)']:<8} | {row['Glider Weight (kg)']:<18} | {row['Wt_Ambient (m/s)']:<18.1f} | {row['MC_Sniff (m/s) (Top Band)']:<25.1f} | {row['Sniffing Radius (m)']:<22.2f} | "
                f"{row['Max Glide Path Length (m)']:<25.2f} | {row['Thermal Density (per km^2)']:<25.2f} | {row['Thermal Strength Lambda']:<25.1f} | {row['Probability of Completing Triangle']:<35.4f} | "
                f"{avg_dist_str:<25}"
            )

        # --- Export summary results to CSV file ---
        csv_filename_summary = "thermal_intercept_simulation_results_poisson_dist_encircling_dynamic.csv"
        try:
            with open(csv_filename_summary, 'w', newline='') as csvfile:
                fieldnames = headers
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in all_results:
                    row_for_csv = row.copy()
                    if not isinstance(row_for_csv['Avg Distance Flown (m)'], float):
                        row_for_csv['Avg Distance Flown (m)'] = str(row_for_csv['Avg Distance Flown (m)'])
                    writer.writerow(row_for_csv)
            print(f"\nSummary results successfully exported to '{csv_filename_summary}'")
        except IOError as e:
            print(f"\nError writing summary CSV file '{csv_filename_summary}': {e}")

        # --- Export detailed per-trial results to a new CSV file ---
        csv_filename_detailed = "monte_carlo_detailed_log.csv"
        detailed_headers = ['Trial', 'Distance Flown (m)', 'Climb Intercept Distances (m)']
        try:
            with open(csv_filename_detailed, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=detailed_headers)
                writer.writeheader()
                writer.writerows(detailed_trial_results)
            print(f"Detailed per-trial log successfully exported to '{csv_filename_detailed}'")
        except IOError as e:
            print(f"Error writing detailed CSV file '{csv_filename_detailed}': {e}")

    else:
        print("Invalid choice. Please enter 1 or 2.")
