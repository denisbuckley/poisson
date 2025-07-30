# Significant Characteristics of this Code:
# 1. Monte Carlo Simulation: Calculates the probability of a glider intercepting an updraft thermal.
# 2. Poisson Distribution: Thermals are spatially distributed according to a Poisson process, and their strengths also follow a Poisson distribution (clamped to 1-10 m/s).
# 3. Updrafts and Downdraft Rings: Each generated thermal consists of an updraft core and an encircling downdraft ring of fixed outer diameter.
# 4. Dynamic Glide Ratio: The glider's glide ratio is not fixed; it's dynamically calculated from provided glider polar data (LS10 18m) based on current airspeed and weight.
# 5. Macready Speed Optimization: The glider's airspeed is optimized for the pilot's Macready setting (expected thermal climb rate) to minimize effective sink rate.
# 6. Altitude Bands for MC_Sniff: The pilot's Macready setting for sniffing (MC_Sniff) dynamically adjusts based on predefined altitude bands (from CBL down to 1500m).
# 7. Fixed Glide Path Calculation: Unlike previous versions, the downdraft strength is *not* explicitly included in the granular flight path calculation for performance reasons. Downdraft *encounters* are still logged and plotted.
# 8. Search Limit: The glider's search for thermals is limited to a maximum horizontal distance (e.g., 100km).
# 9. Detailed CSV Logging: A single flight simulation (Option 1) exports a detailed log to 'single_flight_log.csv', including altitude (integer), airspeed (knots, integer), sink rate (m/s, 2 decimals), and distance flown (km, 3 decimals) at specific altitude band boundaries.
# 10. Visual Plotting: Option 1 generates a plot showing thermal locations, downdraft rings, the glider's glide path, and markers for updraft intercepts and downdraft encounters.
# 11. No Sniffing Circles on Plot: The visual plotting of the large, transparent sniffing circles has been removed for a cleaner visualization.

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
MAX_SEARCH_DISTANCE_METERS = 100000.0  # 100 kilometers

# Altitude step for dynamic glide path simulation (will be calculated dynamically)
# This is now the 'band height' rather than a fixed step for every log entry.
ALTITUDE_STEP_METERS = 10.0  # Default value, will be overridden by band calculation

# Number of height bands for MC_Sniff adjustment (default 3)
NUMBER_OF_HEIGHT_BANDS = 3

# Number of Monte Carlo simulation trials
NUM_SIMULATIONS_TRIALS = 10000  # Default number of iterations

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
SCENARIO_GLIDER_WEIGHT_KG = 400  # Glider weight for polar lookup: 400 or 600 kg
# SCENARIO_MC_SNIFF is now dynamically determined based on altitude bands
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.5  # Average number of thermals per square kilometer (Poisson lambda)
SCENARIO_LAMBDA_STRENGTH = 3.0  # Mean strength of thermals (Poisson lambda, clamped 1-10 m/s)


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


def get_mc_sniff_for_altitude(current_altitude, z_cbl, lambda_strength, altitude_step_meters, number_of_bands):
    """
    Determines the MC_Sniff setting based on the current altitude band.

    Args:
        current_altitude (float): The current altitude of the glider in meters.
        z_cbl (float): Convective Boundary Layer (CBL) height in meters.
        lambda_strength (float): Mean strength of thermals (Poisson lambda).
        altitude_step_meters (float): The calculated altitude step for each band.
        number_of_bands (int): The number of height bands.

    Returns:
        float: The MC_Sniff setting for the current altitude.
    """
    # Define MC_Sniff values for the bands
    mc_sniff_top = max(0.0, lambda_strength - 0.5)  # Ensure it's not negative
    mc_sniff_lowest_band_above_1500 = 1.0
    mc_sniff_middle = (mc_sniff_top + mc_sniff_lowest_band_above_1500) / 2

    # Altitude ranges for the bands (from CBL down to 1500m)
    # Band 1 (Top): (z_cbl - altitude_step_meters, z_cbl]
    # Band 2 (Middle): (z_cbl - 2*altitude_step_meters, z_cbl - altitude_step_meters]
    # Band 3 (Lowest): [1500, z_cbl - 2*altitude_step_meters]

    # Handle altitudes below 1500m
    if current_altitude <= 1500:
        return 0.0  # Glider focuses on staying airborne

    # Calculate band boundaries (descending)
    band_boundaries = [z_cbl]
    for i in range(1, number_of_bands + 1):
        boundary = z_cbl - i * altitude_step_meters
        # Ensure the lowest boundary doesn't go below 1500 if CBL is very low
        band_boundaries.append(max(1500.0, boundary))

    # Sort in descending order to easily check bands from top down
    band_boundaries.sort(reverse=True)  # Ensure descending order

    # Determine which band the current altitude falls into
    # Check top band
    if current_altitude > band_boundaries[
        1]:  # current_altitude is between band_boundaries[0] (CBL) and band_boundaries[1]
        return mc_sniff_top

    # Check middle band (if applicable)
    if number_of_bands >= 2 and current_altitude > band_boundaries[
        2]:  # current_altitude is between band_boundaries[1] and band_boundaries[2]
        return mc_sniff_middle

    # Check lowest band (above 1500m, if applicable)
    if number_of_bands >= 3 and current_altitude >= 1500:  # current_altitude is between band_boundaries[2] and 1500
        return mc_sniff_lowest_band_above_1500

    # Fallback, should not be reached if logic is correct and current_altitude > 1500
    return 0.0


def draw_poisson_thermals_and_glide_path_with_intercept_check(
        z_cbl_meters, glider_weight_kg, lambda_thermals_per_sq_km, lambda_strength,
        fig_width=12, fig_height=12
):
    """
    Draws a single visualization of the thermal grid (Poisson distributed),
    with updrafts and encircling downdraft rings, glide path, and intercepts.
    The glide path search is limited to MAX_SEARCH_DISTANCE_METERS.
    The glide path is now dynamically simulated using polar data.
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
    current_altitude = float(log_altitudes[0])  # Start at exact CBL
    line_start_x, line_start_y = 0, 0
    current_x, current_y = line_start_x, line_start_y
    total_horizontal_distance_covered = 0

    glide_path_segments = []  # To store points for plotting the dynamic path
    flight_log_data = []  # To store data for CSV export

    # Initial MC_Sniff for sniffing radius calculation (will be updated in loop)
    initial_mc_sniff = get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, lambda_strength, ALTITUDE_STEP_METERS,
                                                 NUMBER_OF_HEIGHT_BANDS)
    sniffing_radius_meters = calculate_sniffing_radius(
        lambda_strength, initial_mc_sniff  # Use lambda_strength as proxy for ambient Wt
    )
    if sniffing_radius_meters <= 0:
        print("Warning: Calculated Macready sniffing radius is non-positive. Setting to 1m for visualization.")
        sniffing_radius_meters = 1.0

    # Max possible updraft radius (if Wt_ms=10)
    max_updraft_radius_possible = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)  # Approx 255.4m
    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS  # 600m

    # Determine simulation area side length based on MAX_SEARCH_DISTANCE_METERS
    # This should be the extent of the plot
    sim_area_side_meters = (
                                       MAX_SEARCH_DISTANCE_METERS + max_thermal_system_radius * 2 + sniffing_radius_meters * 2) * 1.1  # Add 10% padding

    # --- Generate Updraft Thermals (Poisson Distribution) ---
    updraft_thermals_info = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    # The random angle for the glide path is fixed for this entire plot's path for consistency
    plot_line_angle_radians = random.uniform(0, 2 * math.pi)

    # Log initial state
    current_mc_sniff = get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, lambda_strength, ALTITUDE_STEP_METERS,
                                                 NUMBER_OF_HEIGHT_BANDS)
    current_airspeed_ms = get_airspeed_for_macready(current_mc_sniff, current_mc_sniff, glider_weight_kg)
    current_sink_rate_ms = get_sink_rate_from_polar(current_airspeed_ms, glider_weight_kg)

    flight_log_data.append({
        'Altitude (m)': round(current_altitude),
        'Airspeed (knots)': round(current_airspeed_ms * MS_TO_KNOT),
        'Sink Rate (m/s)': round(current_sink_rate_ms, 2),
        'Distance Flown (km)': round(total_horizontal_distance_covered / 1000, 3)
    })

    # Iterate through target altitudes for logging
    for i in range(1, len(log_altitudes)):
        prev_altitude_for_calc = current_altitude  # Altitude at the start of this segment for calculations
        target_altitude_for_log = float(log_altitudes[i])  # The next altitude to log

        # Stop if we've gone below 500m or exceeded max search distance
        if prev_altitude_for_calc <= 500 or total_horizontal_distance_covered >= MAX_SEARCH_DISTANCE_METERS:
            break

        # Calculate the altitude change for this segment
        altitude_to_descend = prev_altitude_for_calc - target_altitude_for_log

        if altitude_to_descend <= EPSILON:  # Already at or below target altitude, or target is higher (due to rounding/logic)
            current_altitude = target_altitude_for_log  # Just update altitude to target
            continue

        # Determine MC_Sniff for the *starting* altitude of this segment
        current_mc_sniff = get_mc_sniff_for_altitude(prev_altitude_for_calc, z_cbl_meters, lambda_strength,
                                                     ALTITUDE_STEP_METERS, NUMBER_OF_HEIGHT_BANDS)

        # Determine airspeed and sink rate based on this MC_Sniff
        airspeed_for_macready = get_airspeed_for_macready(current_mc_sniff, current_mc_sniff, glider_weight_kg)
        current_sink_rate_ms = get_sink_rate_from_polar(airspeed_for_macready, glider_weight_kg)

        if current_sink_rate_ms <= EPSILON:  # Ensure sink rate is positive to avoid division by zero
            current_sink_rate_ms = 0.1  # Small positive value to allow progress

        time_step_seconds = altitude_to_descend / current_sink_rate_ms
        horizontal_dist_step = airspeed_for_macready * time_step_seconds

        total_horizontal_distance_covered += horizontal_dist_step

        next_x = current_x + horizontal_dist_step * math.cos(plot_line_angle_radians)
        next_y = current_y + horizontal_dist_step * math.sin(plot_line_angle_radians)

        glide_path_segments.append(((current_x, current_y), (next_x, next_y)))

        current_x, current_y = next_x, next_y
        current_altitude = target_altitude_for_log  # Update current altitude to the target for the next iteration

        # Log data for this step
        flight_log_data.append({
            'Altitude (m)': round(current_altitude),
            'Airspeed (knots)': round(airspeed_for_macready * MS_TO_KNOT),
            'Sink Rate (m/s)': round(current_sink_rate_ms, 2),
            'Distance Flown (km)': round(total_horizontal_distance_covered / 1000, 3)
        })

        if total_horizontal_distance_covered >= MAX_SEARCH_DISTANCE_METERS:
            break

    # --- Plotting the Dynamic Glide Path ---
    if glide_path_segments:
        # Combine segments for plotting
        path_xs = [seg[0][0] for seg in glide_path_segments] + [glide_path_segments[-1][1][0]]
        path_ys = [seg[0][1] for seg in glide_path_segments] + [glide_path_segments[-1][1][1]]

        ax.plot(
            path_xs,
            path_ys,
            color='blue',
            linewidth=2,
            label=f'Dynamic Glide Path (Weight={glider_weight_kg}kg)'
        )
        ax.legend()
    else:
        print("No glide path segments generated. Check Z_CBL or other parameters.")
        plt.title("No Glide Path Generated")

    # --- Plot Thermals (Updrafts and Encircling Downdrafts) ---
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

    # --- Perform Intersection Checks ---
    updraft_intercepts_count = 0
    downdraft_encounters_count = 0
    red_initial_intercept_distances_meters = []
    green_initial_intercept_distances_meters = []

    # Iterate through each small segment of the glide path
    for segment_start, segment_end in glide_path_segments:
        for thermal_info in updraft_thermals_info:
            updraft_center = thermal_info['center']

            # Check for Updraft Sniffing Intercept
            intersects_sniffing, sniff_intersection_pts = check_circle_line_segment_intersection(
                updraft_center, sniffing_radius_meters, segment_start, segment_end
            )

            if intersects_sniffing:
                updraft_intercepts_count += 1
                # Plot the sniffing circle (transparent purple outline)
                sniffing_circle_patch = patches.Circle(
                    updraft_center, sniffing_radius_meters, color='purple', fill=False, alpha=0.1, linestyle='--',
                    linewidth=0.5
                )
                ax.add_patch(sniffing_circle_patch)

                # Find the closest intercept point for the updraft relative to the *start of the total glide path*
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

                # IMPORTANT: If an updraft sniffing intercept occurs, we consider the task complete for this visualization.
                # This prevents over-counting and ensures the "initial" intercept is truly initial.
                # If you want to show ALL intercepts, remove this break.
                # For a single plot, this is fine. For Monte Carlo, we return True immediately.
                # break # Break from inner thermal loop, but continue segment loop to plot all downdrafts

            # Check for Downdraft Annulus Encounter
            downdraft_inner_radius = thermal_info['updraft_radius']
            downdraft_outer_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

            if downdraft_outer_radius > downdraft_inner_radius:
                downdraft_encountered_in_segment = False
                # Check if any part of the segment is within the annulus
                # This is a simplified check for visualization purposes
                segment_length = math.sqrt(
                    (segment_end[0] - segment_start[0]) ** 2 + (segment_end[1] - segment_start[1]) ** 2)
                if segment_length > EPSILON:
                    for t_check in np.linspace(0, 1, 5):  # Check a few points along the segment
                        px = segment_start[0] + t_check * (segment_end[0] - segment_start[0])
                        py = segment_start[1] + t_check * (segment_end[1] - segment_start[1])
                        dist_from_center = math.sqrt((px - updraft_center[0]) ** 2 + (py - updraft_center[1]) ** 2)
                        if downdraft_inner_radius <= dist_from_center <= downdraft_outer_radius:
                            downdraft_encountered_in_segment = True
                            break

                if downdraft_encountered_in_segment:
                    downdraft_encounters_count += 1
                    # Plot the downdraft intercept marker (e.g., 'o' in green)
                    # Find the closest point on the glide path that is within the annulus
                    closest_pt_on_path_in_annulus = None
                    min_dist_sq_to_path_start = float('inf')

                    # Iterate through points on the line segment (simplified for visual)
                    num_points_check = 10  # Check 10 points along the segment
                    for i in range(num_points_check + 1):
                        t = i / num_points_check
                        px = segment_start[0] + t * (segment_end[0] - segment_start[0])
                        py = segment_start[1] + t * (segment_end[1] - segment_start[1])
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

    footer_text = (
        f"Z={z_cbl_meters}m, Glider Weight={glider_weight_kg}kg\n"
        f"Search Limit: {MAX_SEARCH_DISTANCE_METERS / 1000:.0f}km, Actual Glide Distance: {total_horizontal_distance_covered / 1000:.1f}km\n"
        f"Altitude Step: {ALTITUDE_STEP_METERS}m, Bands: {NUMBER_OF_HEIGHT_BANDS}\n"  # Added altitude step and bands
        f"Pilot MC Sniff (Dynamic)\n"  # Changed to dynamic
        f"Thermal Density: {lambda_thermals_per_sq_km}/km², Avg Strength: {lambda_strength} (1-10m/s)\n"
        f"Sniffing Radius (at avg Wt): {sniffing_radius_meters:.0f}m\n"
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
        z_cbl_meters, glider_weight_kg, lambda_thermals_per_sq_km, lambda_strength
):
    """
    Performs a single Monte Carlo experiment with Poisson-distributed updraft thermals
    to check for an intercept with an updraft (red) thermal's sniffing radius.
    Downdraft rings are implicitly modeled but do not affect success/failure here.
    The glide path search is limited to MAX_SEARCH_DISTANCE_METERS.
    The glide path is now dynamically simulated using polar data.

    Args:
        z_cbl_meters (float): The convective Boundary Layer height (Z) for this simulation.
        glider_weight_kg (int): The weight of the glider (400 or 600 kg).
        lambda_thermals_per_sq_km (float): The average number of thermals per square kilometer.
        lambda_strength (float): The mean (lambda) for the Poisson distribution of thermal strength magnitude.

    Returns:
        float: The horizontal distance covered before intercept, or float('inf') if no intercept.
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
    line_start_x, line_start_y = 0, 0  # Glide path starts at origin of simulation area
    current_x, current_y = line_start_x, line_start_y
    total_horizontal_distance_covered = 0

    if current_altitude <= 500:  # Glider starts below or at landing height
        return float('inf')

    # Initial MC_Sniff for sniffing radius calculation (will be updated in loop)
    initial_mc_sniff = get_mc_sniff_for_altitude(current_altitude, z_cbl_meters, lambda_strength,
                                                 calculated_altitude_step_meters, NUMBER_OF_HEIGHT_BANDS)
    sniffing_radius_meters = calculate_sniffing_radius(
        lambda_strength, initial_mc_sniff
    )
    if sniffing_radius_meters <= 0:
        return float('inf')  # No intercept possible if sniffing radius is non-positive

    # Max possible updraft radius (if Wt_ms=10)
    max_updraft_radius_possible = (10 / C_UPDRAFT_STRENGTH_DECREMENT) ** (1 / 3)
    # Max radius of any thermal system (updraft + downdraft ring)
    max_thermal_system_radius = FIXED_THERMAL_SYSTEM_OUTER_RADIUS_METERS

    # Simulation area side should cover the effective glide path plus max thermal/sniffing radius on both sides
    sim_area_side_meters = (
                                       MAX_SEARCH_DISTANCE_METERS + max_thermal_system_radius * 2 + sniffing_radius_meters * 2) * 1.1  # Add 10% padding

    # --- Generate Updraft Thermals (Poisson Distribution) ---
    updraft_thermals = generate_poisson_updraft_thermals(
        sim_area_side_meters, lambda_thermals_per_sq_km, lambda_strength
    )

    # --- Dynamic Glide Path Simulation Loop for Intercept Check ---
    # The random angle for the glide path is fixed for this entire trial's path
    trial_line_angle_radians = random.uniform(0, 2 * math.pi)

    # Determine specific altitudes for logging based on bands for the simulation loop
    log_altitudes_raw_sim = []
    if z_cbl_meters > 1500:
        log_altitudes_raw_sim.append(z_cbl_meters)
        for i in range(1, NUMBER_OF_HEIGHT_BANDS + 1):
            band_boundary_altitude = z_cbl_meters - i * calculated_altitude_step_meters
            if band_boundary_altitude >= 1500:
                log_altitudes_raw_sim.append(band_boundary_altitude)
            else:
                log_altitudes_raw_sim.append(1500.0)
                break
    if 1500.0 not in log_altitudes_raw_sim:
        log_altitudes_raw_sim.append(1500.0)
    log_altitudes_raw_sim.append(500.0)
    sim_altitudes = sorted(list(set([round(alt) for alt in log_altitudes_raw_sim])), reverse=True)
    if sim_altitudes[0] != round(z_cbl_meters):
        sim_altitudes.insert(0, round(z_cbl_meters))
    if sim_altitudes[-1] != 500:
        sim_altitudes.append(500)
    sim_altitudes = sorted(list(set(sim_altitudes)), reverse=True)

    # Use a loop over the specific altitudes for simulation
    for i in range(1, len(sim_altitudes)):
        prev_altitude_for_calc = current_altitude
        target_altitude_for_sim = float(sim_altitudes[i])

        if prev_altitude_for_calc <= 500 or total_horizontal_distance_covered >= MAX_SEARCH_DISTANCE_METERS:
            break

        altitude_to_descend = prev_altitude_for_calc - target_altitude_for_sim

        if altitude_to_descend <= EPSILON:
            current_altitude = target_altitude_for_sim
            continue

        current_mc_sniff = get_mc_sniff_for_altitude(prev_altitude_for_calc, z_cbl_meters, lambda_strength,
                                                     calculated_altitude_step_meters, NUMBER_OF_HEIGHT_BANDS)
        airspeed_for_macready = get_airspeed_for_macready(current_mc_sniff, current_mc_sniff, glider_weight_kg)
        current_sink_rate_ms = get_sink_rate_from_polar(airspeed_for_macready, glider_weight_kg)

        if current_sink_rate_ms <= EPSILON:
            current_sink_rate_ms = 0.1

        time_step_seconds = altitude_to_descend / current_sink_rate_ms
        horizontal_dist_step = airspeed_for_macready * time_step_seconds

        # Check intersections for this segment
        segment_start = (current_x, current_y)
        next_x = current_x + horizontal_dist_step * math.cos(trial_line_angle_radians)
        next_y = current_y + horizontal_dist_step * math.sin(trial_line_angle_radians)
        segment_end = (next_x, next_y)

        for thermal_info in updraft_thermals:
            updraft_center = thermal_info['center']
            intersects_sniffing, _ = check_circle_line_segment_intersection(
                updraft_center, sniffing_radius_meters,
                segment_start, segment_end
            )
            if intersects_sniffing:
                # Return the total distance covered up to this point
                return total_horizontal_distance_covered + math.sqrt(
                    (next_x - current_x) ** 2 + (next_y - current_y) ** 2) / 2

        # Update position and altitude for next step
        current_x, current_y = next_x, next_y
        current_altitude = target_altitude_for_sim
        total_horizontal_distance_covered += horizontal_dist_step

    return float('inf')  # No intercept with an updraft's sniffing radius found within search limit


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
            lambda_strength=SCENARIO_LAMBDA_STRENGTH
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
        print("-" * 50)

        intercept_count = 0
        intercept_distances = []  # Collect distances for successful intercepts

        # Calculate dynamic ALTITUDE_STEP_METERS for Monte Carlo
        if SCENARIO_Z_CBL <= 1500:
            mc_altitude_step_meters = 0
        else:
            mc_altitude_step_meters = round((z_cbl_meters - 1500) / NUMBER_OF_HEIGHT_BANDS, -2)
            if mc_altitude_step_meters <= 0:
                mc_altitude_step_meters = 100.0

        tqdm_desc = "Running Monte Carlo Trials"
        for _ in tqdm(range(num_simulations), desc=tqdm_desc):
            distance_at_intercept = simulate_intercept_experiment_poisson(
                z_cbl_meters=SCENARIO_Z_CBL,
                glider_weight_kg=SCENARIO_GLIDER_WEIGHT_KG,
                lambda_thermals_per_sq_km=SCENARIO_LAMBDA_THERMALS_PER_SQ_KM,
                lambda_strength=SCENARIO_LAMBDA_STRENGTH
            )
            if distance_at_intercept != float('inf'):
                intercept_count += 1
                intercept_distances.append(distance_at_intercept)

        probability = intercept_count / num_simulations

        average_intercept_distance = "N/A"
        if intercept_distances:
            average_intercept_distance = np.mean(intercept_distances)

        # Calculate the sniffing radius for display in results (using top band MC_Sniff as representative)
        representative_mc_sniff = get_mc_sniff_for_altitude(SCENARIO_Z_CBL, SCENARIO_Z_CBL, SCENARIO_LAMBDA_STRENGTH,
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
            'Probability': probability,
            'Avg Intercept Dist (m)': average_intercept_distance
        }]

        print("\n" + "=" * 120)
        print("\n--- Monte Carlo Simulation Results for Single Scenario ---")
        headers = [
            'Z (m)', 'Glider Weight (kg)', 'Wt_Ambient (m/s)', 'MC_Sniff (m/s) (Top Band)', 'Sniffing Radius (m)',
            'Max Glide Path Length (m)', 'Thermal Density (per km^2)', 'Thermal Strength Lambda',
            'Probability', 'Avg Intercept Dist (m)'
        ]
        print(
            f"{headers[0]:<8} | {headers[1]:<18} | {headers[2]:<18} | {headers[3]:<25} | {headers[4]:<22} | "
            f"{headers[5]:<25} | {headers[6]:<25} | {headers[7]:<25} | {headers[8]:<15} | {headers[9]:<22}"
        )
        print("-" * 280)  # Adjusted separator length

        for row in all_results:
            avg_dist_str = f"{row['Avg Intercept Dist (m)']:.2f}" if isinstance(row['Avg Intercept Dist (m)'],
                                                                                float) else str(
                row['Avg Intercept Dist (m)'])
            print(
                f"{row['Z (m)']:<8} | {row['Glider Weight (kg)']:<18} | {row['Wt_Ambient (m/s)']:<18.1f} | {row['MC_Sniff (m/s) (Top Band)']:<25.1f} | {row['Sniffing Radius (m)']:<22.2f} | "
                f"{row['Max Glide Path Length (m)']:<25.2f} | {row['Thermal Density (per km^2)']:<25.2f} | {row['Thermal Strength Lambda']:<25.1f} | {row['Probability']:<15.4f} | "
                f"{avg_dist_str:<22}"
            )

        # --- Export results to CSV file ---
        csv_filename = "thermal_intercept_simulation_results_poisson_dist_encircling_dynamic.csv"
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = headers
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in all_results:
                    # Convert average_intercept_distance to string for CSV if it's N/A
                    row_for_csv = row.copy()
                    if not isinstance(row_for_csv['Avg Intercept Dist (m)'], float):
                        row_for_csv['Avg Intercept Dist (m)'] = str(row_for_csv['Avg Intercept Dist (m)'])
                    writer.writerow(row_for_csv)
            print(f"\nResults successfully exported to '{csv_filename}'")
        except IOError as e:
            print(f"\nError writing to CSV file '{csv_filename}': {e}")

    else:
        print("Invalid choice. Please enter 1 or 2.")
