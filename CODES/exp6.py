import numpy as np

# Experiment 6: Limitation of Perceptron – XOR Problem
# Aim: To demonstrate why a single-layer perceptron fails to solve the XOR logic problem.

print("--- Perceptron Limit Testing (XOR Gate) ---")

# 1. Define the input dataset (X) and expected targets (y)
# The XOR gate outputs 1 only if exactly ONE input is 1. (0,1 or 1,0).
X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

y = np.array([0, 1, 1, 0])  # Target outputs for an XOR gate

# 2. Setup the exact same Perceptron format as Experiment 5
w = np.zeros(2)       
b = 0                 
lr = 0.1              

# We will run this for several epochs to prove it never stabilizes
epochs = 15           

print(f"Beginning Setup -> Weights: {w}, Bias: {b}, Learning Rate: {lr}")
print("Training Perceptron on the non-linear XOR dataset...\n")

# 3. Training Loop
for epoch in range(epochs):
    errors_in_epoch = 0 
    for i in range(len(X)):
        # Calculate raw linear output
        linear_output = np.dot(X[i], w) + b
        
        # Apply step activation function
        y_pred = 1 if linear_output >= 0 else 0
        
        error = y[i] - y_pred
        
        # Adjust weights and bias if the prediction is wrong
        if error != 0:
            errors_in_epoch += 1
            w += lr * error * X[i] 
            b += lr * error        
            
    # Print the error count to show the algorithm is infinitely struggling
    print(f"Epoch {epoch+1:02d}/{epochs} | Errors detected: {errors_in_epoch}")

print("\n--- Why Did It Fail? ---")
print("Conclusion: The perceptron fundamentally fails to solve XOR.")
print("Reasoning: A single layer algorithm can only draw a SINGLE straight line across the data points.")
print("It is physically impossible to draw one straight line that separates the XOR points cleanly.")
