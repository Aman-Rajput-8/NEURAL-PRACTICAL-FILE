import matplotlib.pyplot as plt
import numpy as np

# Experiment 4: Introduction to Matplotlib Visualizations
# Aim: To learn how to construct basic plots, subplots, and labels in python.

# 1. Prepare some mock data arrays
x = np.linspace(0, 10, 100)
y1 = x * 2           # Linear relationship
y2 = x ** 2          # Quadratic relationship

print("--- Generating Matplotlib Visualizations ---")
print("Check your taskbar for the popped-up window to view the graph!")

# 2. Build a figure with two side-by-side subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot 1: Standard Line Plots ---
ax1.plot(x, y1, label='y = 2x', color='blue', linestyle='--')
ax1.plot(x, y2, label='y = x^2', color='red', linewidth=2)

ax1.set_title('Basic Mathematical Functions')
ax1.set_xlabel('X Axis (Inputs)')
ax1.set_ylabel('Y Axis (Outputs)')
ax1.legend()
ax1.grid(True)

# --- Plot 2: Scatter Plot ---
# Generating fake data for two distinct clusters
cluster_1_x = np.random.normal(3, 1, 50)
cluster_1_y = np.random.normal(5, 1, 50)

cluster_2_x = np.random.normal(7, 1, 50)
cluster_2_y = np.random.normal(2, 1, 50)

ax2.scatter(cluster_1_x, cluster_1_y, color='purple', marker='o', label='Cluster A')
ax2.scatter(cluster_2_x, cluster_2_y, color='orange', marker='x', label='Cluster B')

ax2.set_title('Random Scatter Plot')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.legend()

# 3. Add a main title over both subplots and display
plt.suptitle('Experiment 4: Matplotlib Demo', fontsize=16)
plt.tight_layout()
plt.show()
