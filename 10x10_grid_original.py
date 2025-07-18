import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
d_x = 10  # Average distance between thermals along the X-axis for exponential spacing
grid_size = 100 # Define the conceptual grid extent (e.g., 100 units x 100 units)
num_thermals = 100 # Approximate number of thermals to simulate on the grid

# --- Generate Thermal Locations on a 2D Grid ---

# 1. Generate X-coordinates using exponential distribution for spacing
x_positions = [0] # Start the first thermal at position 0
current_x = 0
while len(x_positions) < num_thermals:
    # Generate a random distance 'dx' from the exponential distribution
    # np.random.exponential(scale=d_x) samples from 1/d_x * exp(-x/d_x)
    dx = np.random.exponential(scale=d_x)
    current_x += dx
    x_positions.append(current_x)

# Trim x_positions to only include up to num_thermals if more were generated
x_positions = x_positions[:num_thermals]

# 2. Generate random Y-coordinates within the grid size
y_positions = np.random.uniform(0, grid_size, len(x_positions))

# Adjust x_positions to fit within the grid_size for better visualization
# If the sum of dx goes beyond grid_size, we rescale them proportionally
max_x_simulated = max(x_positions)
if max_x_simulated > grid_size:
    x_positions_scaled = [pos * (grid_size / max_x_simulated) for pos in x_positions]
else:
    x_positions_scaled = x_positions

# --- Plotting ---
plt.figure(figsize=(10, 10))
plt.scatter(x_positions_scaled, y_positions, s=50, alpha=0.7, color='red', label='Simulated Thermals')

# Add grid lines for visual context
plt.xticks(np.arange(0, grid_size + 1, grid_size / 10))
plt.yticks(np.arange(0, grid_size + 1, grid_size / 10))
plt.grid(True, linestyle=':', alpha=0.6)

plt.title(f'Simulated Thermals on a Grid (X-axis Spacing Exponential, d={d_x})')
plt.xlabel('X-coordinate (units)')
plt.ylabel('Y-coordinate (units)')
plt.xlim(0, grid_size)
plt.ylim(0, grid_size)
plt.gca().set_aspect('equal', adjustable='box') # Ensure the grid aspect ratio is square
plt.legend()
plt.savefig('thermals_on_grid_exponential_x.png')

plt.show()
print("Plot generated and saved as 'thermals_on_grid_exponential_x.png'")