import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Experiment 7: Visualizing Activation Functions
# Aim: To calculate specific neural network activation functions using NumPy and visualize them using Pandas and Matplotlib.

print("--- Data Pipeline for Activation Functions ---")

# 1. Prepare an array of numbers from -10 to 10
x_inputs = np.linspace(-10, 10, 200)

# Calculate the various mathematical activation formulas
y_sigmoid = 1 / (1 + np.exp(-x_inputs))
y_tanh = np.tanh(x_inputs)
y_relu = np.maximum(0, x_inputs)
y_leaky_relu = np.where(x_inputs > 0, x_inputs, x_inputs * 0.1) 

# 2. Package all the values neatly into a Pandas DataFrame
dataset = {
    'Input': x_inputs,
    'Sigmoid Output': y_sigmoid,
    'Tanh Output': y_tanh,
    'ReLU Output': y_relu,
    'Leaky ReLU Output': y_leaky_relu
}
df = pd.DataFrame(dataset)

print("Generated DataFrame for plotting:")
print(df.head())

# 3. Create a 4x4 visual grid using Matplotlib
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot Sigmoid (forces values between 0 and 1)
axs[0, 0].plot(df['Input'], df['Sigmoid Output'], color='blue', linewidth=2.5)
axs[0, 0].set_title("Sigmoid Function (0 to 1)")
axs[0, 0].grid(True, linestyle='--', alpha=0.6)

# Plot Tanh (forces values between -1 and 1)
axs[0, 1].plot(df['Input'], df['Tanh Output'], color='red', linewidth=2.5)
axs[0, 1].set_title("Tanh Function (-1 to 1)")
axs[0, 1].grid(True, linestyle='--', alpha=0.6)

# Plot ReLU (turns all negative values to zero)
axs[1, 0].plot(df['Input'], df['ReLU Output'], color='green', linewidth=2.5)
axs[1, 0].set_title("ReLU Function (> 0)")
axs[1, 0].grid(True, linestyle='--', alpha=0.6)

# Plot Leaky ReLU (leaves a tiny gentle slope for negative numbers)
axs[1, 1].plot(df['Input'], df['Leaky ReLU Output'], color='orange', linewidth=2.5)
axs[1, 1].set_title("Leaky ReLU Function (0.1 Threshold Slope)")
axs[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
