This generates a simulated field of thermals and draws a line representing a sailplane glidepath, and returns intercepts by the glider with thermals.
The field is a 2D Poisson distribution spatially, with thermal strength also a random Poisson distribution.
Thermals are notionally 800m in diameter, surrounded by a downdraft annulus of 1200m outer diameter. The downdraft strength is a function of updraft strength, recycling all air in the cell. These are shown in option 1 below as a green annulus surrounding a red dot, with glidepath as blue line.
Down = 0.8 * Up − 1.22792448 in m/s
The effective updraft diameter is according to Speight (2015:40) y=0.033 (x/100)^3, where y is the decrement in thermal strength and x is the distance from the thermal centre.
As Speight's formula is in knots/feet, for m/s the formula is y = 5.9952 * 10^-7 * 10^3
If this calc decrements up strength to zero, that is the horizontal extent of the updraft.
thermal_sim_poisson_segmented.py has 2 run options:
1. Generate a single plot (visualize Poisson-distributed thermals with encircling downdrafts and flight log)
2. Run Monte Carlo simulation (compute probability for a single scenario and export CSV)

The script has variable parameters: defaults are
NUM_SIMULATIONS_TRIALS = 10000, 
SCENARIO_Z_CBL = 2500m (height of convective boundary layer), 
SCENARIO_GLIDER_WEIGHT_KG = 400/600 (LS10st polar incorporated in sim), 
SCENARIO_LAMBDA_THERMALS_PER_SQ_KM = 0.5, 
SCENARIO_LAMBDA_STRENGTH = 3.0, 
SCENARIO_MC_SNIFF_TOP_MANUAL = 3.0 (sets diameter of updraft interception, smaller number increases chances of interception), 
NUMBER_OF_HEIGHT_BANDS = 3 (script adjusts Macready setting/speed as glider descends with no intercept),
The script stops/climbs to CBL when intercept and resets the glidepath
Each iteration generates a fresh glidepath and thermal grid
