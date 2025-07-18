import numpy as np
import matplotlib.pyplot as plt

# Choose a value for 'd' (average distance)
# You can change this value to represent your specific average distance
d = 500  # Example: 500 feet or 500 meters

# Define a range for 'x' (distance between adjacent thermals)
# We plot from 0 up to 5 times the average distance to show the decay
x = np.linspace(0, 5 * d, 500)

# Calculate f(x) using the given formula for the exponential distribution
f_x = (1 / d) * np.exp(-x / d)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, f_x, label=f'f(x) = (1/{d}) * exp(-x/{d})')

# Add labels and title for clarity
plt.title(f'Probability Density of Distance Between Thermals (Average Distance d = {d})')
plt.xlabel(f'Distance Between Thermals, x (units of d)')
plt.ylabel('Probability Density, f(x)')
plt.grid(True)
plt.legend()

# Save the plot to a file
# You can change the filename if needed
plt.savefig('thermal_distance_pdf.png')

print("Plot generated and saved as 'thermal_distance_pdf.png'")