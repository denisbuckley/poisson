import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# --- Parameters ---
grid_size_x = 10  # Width of the grid
grid_size_y = 10  # Height of the grid

# Average distance between thermals (from your formula, in grid units)
# Choose a value that makes sense for your grid.
# If d is small (e.g., 1), the grid will be very dense with thermals.
# If d is large (e.g., 5), there will be fewer thermals.
d = 2.5 # Example: Average 2.5 grid units between thermals

# --- Calculate parameters for a 2D Poisson Point Process ---

# We interpret d as the characteristic spacing.
# Density (lambda) = number of thermals per unit area.
# If d is the average linear spacing, a simple interpretation for density
# is that 1 thermal occupies an average area of d*d.
# So, lambda = 1 / d^2
density_lambda = 1 / (d**2)

# Calculate the expected number of thermals in the 10x10 grid
expected_num_thermals = (grid_size_x * grid_size_y) * density_lambda

# --- Generate Thermal Locations ---

# The actual number of thermals will follow a Poisson distribution
# with the calculated expected_num_thermals as its mean.
num_thermals = poisson.rvs(expected_num_thermals)

# Generate random (x, y) coordinates for each thermal
# These coordinates are uniformly distributed within the grid boundaries.
thermal_x = np.random.uniform(0, grid_size_x, num_thermals)
thermal_y = np.random.uniform(0, grid_size_y, num_thermals)

# --- Plotting ---
plt.figure(figsize=(8, 8))
plt.scatter(thermal_x, thermal_y, s=100, color='red', marker='o', label='Thermals')
plt.xlim(0, grid_size_x)
plt.ylim(0, grid_size_y)
plt.xticks(np.arange(0, grid_size_x + 1, 1)) # Set x-axis ticks at integer intervals
plt.yticks(np.arange(0, grid_size_y + 1, 1)) # Set y-axis ticks at integer intervals
plt.grid(True, linestyle='--', alpha=0.7)
plt.title(f'Map of Thermals on a {grid_size_x}x{grid_size_y} Grid\n'
          f'(Average 1D spacing d = {d} units, Expected Thermals = {expected_num_thermals:.1f}, Actual = {num_thermals})')
plt.xlabel('X-coordinate (grid units)')
plt.ylabel('Y-coordinate (grid units)')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box') # Ensure grid cells are square
plt.show()

print(f"Generated {num_thermals} thermals on the {grid_size_x}x{grid_size_y} grid.")
print(f"Using d = {d} units, which implies an average density of {density_lambda:.3f} thermals per square unit.")
print(f"Expected number of thermals on this grid: {expected_num_thermals:.1f}")