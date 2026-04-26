import numpy as np

# Experiment 8: Comparing Optimizers (Gradient Descent vs. Adam Optimizer)
# Aim: To manually build and compare classical Gradient Descent against the advanced Adam Optimizer.

# 1. Define a simple U-shaped curve (Loss function: L(w) = w^2)
def compute_gradient(w):
    # The derivative of w^2 is explicitly 2w
    return 2 * w

print("--- Testing Optimization Algorithms ---")
print("Target: Find the absolute minimum of the U-shaped curve L(w) = w^2 (Minimum is exactly at w = 0)")

# 2. Standard Gradient Descent Implementation
w_gd = 100.0  # Starting point
lr_gd = 0.1   # Standard learning rate
epochs = 15

print("\n--- 1. Running Standard Gradient Descent ---")
print("[Details] Starts at w = 100.0. Uses fixed step sizes based ONLY on the current gradient slope.")
for epoch in range(1, epochs + 1):
    grad = compute_gradient(w_gd)
    
    # Calculate exact distance to jump
    step_size = lr_gd * grad
    w_gd = w_gd - step_size
    
    print(f"GD Epoch {epoch:02d} | Gradient Slope: {grad:7.2f} | Step Jump Size: {step_size:6.2f} | New Weight: {w_gd:.4f}")

# 3. Adam Optimizer Implementation
w_adam = 100.0 
lr_adam = 0.5 

# Adam Specific Momentum Parameters
beta1 = 0.9    # Tracks past momentum
beta2 = 0.999  # Tracks past gradient squared (variance)
epsilon = 1e-8 
m = 0          
v = 0          

print("\n--- 2. Running Adam Optimizer ---")
print("[Details] Starts at w = 100.0. Adapts step size completely by tracking 'Momentum' and varying slope sizes.")
for epoch in range(1, epochs + 1):
    grad = compute_gradient(w_adam)
    
    # Track the moving average of gradients (Momentum)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    # Correct the bias so early steps aren't artificially tiny
    m_hat = m / (1 - beta1 ** epoch)
    v_hat = v / (1 - beta2 ** epoch)
    
    # Determine the mathematically optimal jump size
    step_size = lr_adam * m_hat / (np.sqrt(v_hat) + epsilon)
    w_adam = w_adam - step_size
    
    print(f"Adam Epoch {epoch:02d} | Slope: {grad:7.2f} | Momentum Built: {m_hat:6.2f} | Step Jump Size: {step_size:5.2f} | New Weight: {w_adam:.4f}")
