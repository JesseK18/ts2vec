import numpy as np
import matplotlib.pyplot as plt

# Generate a simple synthetic time series (sinusoidal wave)
t = np.linspace(0, 4 * np.pi, 100)  # Time points
X = np.sin(t)  # Original time series

# Simulate the gradient of the loss function
gradient = np.cos(t)  # Assume gradient is the cosine of the time points

# Apply FGSM perturbation
epsilon = 0.1  # Small perturbation factor
eta = epsilon * np.sign(gradient)  # Compute the adversarial perturbation
X_adv = X + eta  # Create adversarial example

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t, X, label="Original Time Series (X)", color="blue")
plt.plot(t, X_adv, label="Perturbed Time Series (X + η)", color="red", linestyle="dashed")
plt.plot(t, eta, label="Perturbation (η)", color="green", linestyle="dotted")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Applying FGSM to a Time Series")
plt.show()
