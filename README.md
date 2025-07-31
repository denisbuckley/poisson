This generates a simulated field of thermals and draws a line representing a sailplane glidepath, and returns intercepts by the glider with thermals.
The field is a 2D Poisson distribution spatially, with thermal strength also a random Poisson distribution.
Thermals are notionally 800m in diameter, surrounded by a downdraft annulus of 1200m outer diameter. The downdraft strength is a function of updraft strength, recycling all air in the cell.
Down = 0.8 * Up − 1.22792448 in m/s
The effective updraft diameter is according to Speight (2015:40) y=0.033 (x/100)^3, where y is the decrement in thermal strength and x is the distance from the thermal centre.
As Speight's formula is in knots/feet, for m/s the formula is y = 5.9952 * 10^-7 * 10^3
The scri
