import numpy as np
import matplotlib.pyplot as plt

# Experiment 11: Hopfield Network Implementation
# Aim: To build a discrete Hopfield Network to memorize and recall visual patterns corrupted by noise.

def create_pattern():
    """Generates a simple 5x5 'T' shape using +1s and -1s."""
    pattern = np.ones((5, 5)) * -1
    pattern[0, :] = 1  # Top horizontal bar of T
    pattern[:, 2] = 1  # Center vertical axis
    return pattern.flatten()

def introduce_noise(pattern, noise_level=0.3):
    """Randomly flips a given percentage of bits inside the binary array."""
    corrupted = pattern.copy()
    num_to_flip = int(len(pattern) * noise_level)
    indices = np.random.choice(len(pattern), num_to_flip, replace=False)
    corrupted[indices] *= -1
    return corrupted

def train_hopfield(patterns):
    """Derives a weight matrix for the network using the standard Hebbian learning rule."""
    n_features = len(patterns[0])
    weights = np.zeros((n_features, n_features))
    for p in patterns:
        weights += np.outer(p, p)
    
    # We enforce self-connections to equal 0 on the matrix diagonal
    np.fill_diagonal(weights, 0)
    return weights / len(patterns)

def retrieve_hopfield(weights, corrupted_pattern, steps=5):
    """Feeds the corrupted image back into the network recurrently to retrieve the saved memory."""
    retrieved = corrupted_pattern.copy()
    for step in range(steps):
        retrieved = np.sign(np.dot(weights, retrieved))
        retrieved[retrieved == 0] = 1
    return retrieved

if __name__ == "__main__":
    print("\n--- Discrete Hopfield Network Memory Simulation ---")
    
    # 1. Start by fetching the original Target Pattern memory
    original_pattern = create_pattern()
    print("\n1. Generating Perfect Target Pattern (5x5 'T' shape):")
    print("   (Displaying 1s as '■' and -1s as '□')")
    visual_pattern = np.where(original_pattern.reshape(5,5) == 1, '■', '□')
    for row in visual_pattern:
        print("   " + " ".join(row))

    # 2. Calculate the brain logic (Weight Matrix) to memorize the pattern permanently
    weight_matrix = train_hopfield([original_pattern])
    print(f"\n2. Training Hopfield Network via Hebbian Rule...")
    print(f"   Generated Weight Matrix (Shape: {weight_matrix.shape}):")
    print(f"   Max Weight: {np.max(weight_matrix)}")
    
    # 3. Simulate environmental noise by corrupting the memory 
    noisy_input = introduce_noise(original_pattern, noise_level=0.3)
    print("\n3. Injecting 30% Random Noise Corruption into Target Pattern...")
    visual_noisy = np.where(noisy_input.reshape(5,5) == 1, '■', '□')
    for row in visual_noisy:
        print("   " + " ".join(row))

    # 4. Use the trained Weight Matrix to search and stabilize the distorted memory
    print("\n4. Triggering Retrieval Recovery search iteratively...")
    retrieved_memory = retrieve_hopfield(weight_matrix, noisy_input, steps=3)
    
    print("\n5. Restored Memory Pattern retrieved securely back from Noise:")
    visual_retrieved = np.where(retrieved_memory.reshape(5,5) == 1, '■', '□')
    for row in visual_retrieved:
        print("   " + " ".join(row))

    print("\nPattern physics match! Launching Matplotlib graphics dashboard...")

    # Display the final comparison GUI visually
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(original_pattern.reshape(5,5), cmap='gray_r')
    axs[0].set_title("Original Pattern")
    axs[0].axis('off')

    axs[1].imshow(noisy_input.reshape(5,5), cmap='gray_r')
    axs[1].set_title("Noisy Input")
    axs[1].axis('off')

    axs[2].imshow(retrieved_memory.reshape(5,5), cmap='gray_r')
    axs[2].set_title("Retrieved Pattern")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
