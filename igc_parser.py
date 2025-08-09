#
# This script reads an IGC file, parses the flight data, and plots the flight path.
# It uses a simple heuristic to identify potential thermals based on rapid altitude
# changes, marking them with a red 'x'.
#
# Required libraries: igc-parser, matplotlib
# Install them with: pip install igc-parser matplotlib
#

import igc_parser
import matplotlib.pyplot as plt


def plot_igc_with_thermals(filepath):
    """
    Reads an IGC file, identifies potential thermals based on altitude gain,
    and plots the flight path with thermals marked on a map.

    Args:
        filepath (str): The path to the IGC file.
    """
    try:
        # --- 1. Read and parse the IGC file ---
        # The igc_parser library handles the complex file format.
        with open(filepath, 'r') as file:
            igc_data = file.read()

        flight = igc_parser.parse_igc(igc_data)

        # --- 2. Extract key data points ---
        # We need the latitude, longitude, and altitude for each recorded point.
        latitudes = [rec.latitude for rec in flight.b_records]
        longitudes = [rec.longitude for rec in flight.b_records]
        altitudes = [rec.gps_altitude for rec in flight.b_records]

        # --- 3. Identify potential thermals (simple heuristic) ---
        # A thermal is a column of rising air. We can approximate this by
        # looking for points where the altitude increases significantly
        # in a short period (between consecutive recordings).
        thermals_lat = []
        thermals_lon = []
        altitude_change_threshold = 20  # meters, adjust as needed

        print("Analyzing flight data for thermals...")
        for i in range(1, len(altitudes)):
            altitude_gain = altitudes[i] - altitudes[i - 1]
            if altitude_gain > altitude_change_threshold:
                # If a significant altitude gain is detected, mark this point as a thermal.
                thermals_lat.append(latitudes[i])
                thermals_lon.append(longitudes[i])

        print(f"Found {len(thermals_lat)} potential thermals.")

        # --- 4. Plot the flight path and thermals on a map ---
        plt.figure(figsize=(10, 8))

        # Plot the main flight path as a blue line.
        plt.plot(longitudes, latitudes, color='blue', label='Flight Path')

        # Plot the identified thermals as red 'x' marks.
        plt.scatter(thermals_lon, thermals_lat, c='red', marker='x', s=100, label='Potential Thermals')

        # --- 5. Add labels, title, and a legend for clarity ---
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('IGC Flight Path with Thermals')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Example Usage ---
# Replace 'path/to/your/file.igc' with the actual path to your file.
# Make sure to uncomment the line below to run the function.
# plot_igc_with_thermals('path/to/your/file.igc')

