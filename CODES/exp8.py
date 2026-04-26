import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np

# Suppress standard Scikit-Learn loop warnings to keep the terminal output clean
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Experiment 9: Neural Network using Scikit-Learn

# 1. Prepare Data
X, y = make_classification(n_samples=500, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build Model
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=10, warm_start=True, random_state=42)

# ==========================================
# Terminal GUI Layout
# ==========================================

print("\n" + "═"*45)
print(" 🧠 Scikit-Learn MLP Training ")
print("═"*45 + "\n")

# 3. Training Loop with clean progress tracking
for epoch in range(1, 11): 
    mlp.fit(X_train, y_train)
    progress_bar = "█" * epoch + "░" * (10 - epoch)
    print(f"  Epoch {epoch:02d}/10  │ {progress_bar} │  Loss: {mlp.loss_:.4f}")

# 4. Predict & Evaluate
predictions = mlp.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "═"*45)
print(" 🎯 Final Results ")
print("═"*45 + "\n")

print(f"  Accuracy Score : {accuracy * 100:.1f}%\n")

print("  🔍 Label Comparison (First 8 Samples):")
print(f"     Expected : {y_test[:8].tolist()}")
print(f"     Guessed  : {predictions[:8].tolist()}")
print("\n" + "═"*45 + "\n")
