import numpy as np

# Experiment 1: Introduction to Python & Numerical Computing
# Aim: To familiarize students with NumPy array operations and mathematical functions.

print("--- 1. Array Creation Functions ---")
array_1d = np.array([1, 2, 3, 4, 5])                # Convert a list to a 1D array
zeros_array = np.zeros((2, 3))                      # Create a 2x3 matrix filled with zeros
ones_array = np.ones((2, 3))                        # Create a 2x3 matrix filled with ones
random_array = np.random.rand(2, 2)                 # Create a 2x2 matrix with random values
sequence = np.arange(0, 10, 2)                      # Sequence from 0 to 10 with a step of 2
spaced = np.linspace(0, 1, 5)                       # 5 evenly spaced numbers from 0 to 1

print("1D Array:", array_1d)
print("Arange Sequence:", sequence)
print("Linspace:", spaced)

print("\n--- 2. Shape and Array Manipulation ---")
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("Original 2D Array Shape:", array_2d.shape)
reshaped = np.reshape(array_1d, (5, 1))             # Reshape the 1D array into a 5x1 column
print("Reshaped 1D to (5,1):\n", reshaped)
transposed = np.transpose(array_2d)                 # Swap rows and columns
print("Transposed 2D Array:\n", transposed)

print("\n--- 3. Mathematical Operations ---")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Matrix Addition:\n", np.add(A, B))           # Element-wise addition
print("Matrix Subtraction:\n", np.subtract(A, B))   # Element-wise subtraction
print("Matrix Multiplication (Element-wise):\n", np.multiply(A, B))
print("Dot Product (True matrix multiplication):\n", np.dot(A, B))

print("\n--- 4. Statistical & Aggregation Functions ---")
data = np.array([15, 20, 35, 40, 50])
print("Max value:", np.max(data))
print("Min value:", np.min(data))
print("Mean (Average):", np.mean(data))
print("Sum of elements:", np.sum(data))
print("Standard Deviation:", np.std(data).round(2))
