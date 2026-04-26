import numpy as np

# Experiment 5: Perceptron Learning Algorithm
# Aim: To build a single-layer perceptron from scratch and teach it the AND logic gate.

print("--- Perceptron Learning Algorithm (AND Gate) ---")

# 1. Provide the input data and the expected answers
X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

y = np.array([0, 0, 0, 1])  # Target outputs for an AND gate

# 2. Setup standard perceptron variables
w = np.zeros(2)       # Start with 0 for both weights
b = 0                 # Start with 0 for bias
lr = 0.1              # Learning rate controls the size of updates

epochs = 10           # How many times we pass over the full dataset

print(f"Initial Setup -> Weights: {w}, Bias: {b}, Learning Rate: {lr}")
print("Starting training loop...\n")

# 3. Core Training Loop
for epoch in range(epochs):
    errors_in_epoch = 0 
    for i in range(len(X)):
        # Calculate the raw linear output: (X * W) + b
        linear_output = np.dot(X[i], w) + b
        
        # Apply the step activation function
        y_pred = 1 if linear_output >= 0 else 0
        
        # Check if our prediction was wrong
        error = y[i] - y_pred
        
        # If there's an error, adjust the weights and bias
        if error != 0:
            errors_in_epoch += 1
            w += lr * error * X[i] 
            b += lr * error        
            
    print(f"Epoch {epoch+1}/{epochs} | Errors detected: {errors_in_epoch}")

# 4. Show the final results
print("\n--- Training Finished ---")
print("Final Weights:", w)
print("Final Bias:", b)

print("\nVerifying the model predictions on the original inputs:")
for i in range(len(X)):
    final_output = 1 if np.dot(X[i], w) + b >= 0 else 0
    print(f"Input: {X[i]} -> Expected: {y[i]} | Model Guessed: {final_output}")
