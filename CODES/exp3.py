import numpy as np
from scipy import optimize, interpolate

# Experiment 3: Utilizing basic SciPy Functions
# Aim: To implement optimization and interpolation utilities available in SciPy.

print("--- 1. Optimization with SciPy ---")
# Define an objective function we want to minimize: f(x) = (x-3)^2 + 2x
def objective_function(x):
    return (x - 3)**2 + 2*x

# We ask SciPy to find the minimum of this function, starting its search at x=10
initial_guess = 10.0
result = optimize.minimize(objective_function, initial_guess, method='BFGS')

print(f"Starting guess at: {initial_guess}")
print(f"Optimal Minimum found at X = {result.x[0]:.4f}")
print(f"Lowest possible Y value achieved: {result.fun:.4f}")
print(f"Number of iterations performed: {result.nit}")

print("\n--- 2. Interpolation with SciPy ---")
# Let's say we only collected 5 specific data points
x_known = np.linspace(0, 5, 5)  # [0, 1.25, 2.5, 3.75, 5]
y_known = np.array([0, 1.5, 3.2, 2.1, 0.5])

# SciPy can generate an interpolation function to estimate points between these known values
linear_interp_func = interpolate.interp1d(x_known, y_known, kind='linear')
cubic_interp_func = interpolate.interp1d(x_known, y_known, kind='cubic')

# Now we can query a value we never actually measured directly (like x=2.0)
query_x = 2.0
print(f"Measured X array positions: {x_known}")
print(f"Estimating Y at X={query_x} via Linear Interpolation: {linear_interp_func(query_x):.4f}")
print(f"Estimating Y at X={query_x} via Cubic Interpolation: {cubic_interp_func(query_x):.4f}")
