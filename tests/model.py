import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# Evidence (your data)
# ---------------------------------------

b_r = (50, 1)
b_h = (50, 1)
b_f = (50,1)

pos = np.array([b_r[0], b_h[0], b_f[0]])
neg = np.array([b_r[1], b_h[1], b_f[1]])


# Prior (uniform Beta)

alpha = 1 + pos
beta = 1 + neg


# Weights (must sum to 1)

w = np.array([0.33, 0.33, 0.34]) 


# Monte Carlo sampling

N = 50000

theta_samples = np.random.beta(alpha[:, None], beta[:, None], size=(3, N))


# Aggregate into trust

T_samples = np.dot(w, theta_samples)


# Summary

mean_T = np.mean(T_samples)
ci_lower, ci_upper = np.quantile(T_samples, [0.025, 0.975])

plt.figure()

# Histogram
plt.hist(T_samples, bins=100, density=True, alpha=0.6)

# Mean line
plt.axvline(mean_T, linestyle="--", label=f"Mean = {mean_T:.3f}")

# CI lines
plt.axvline(ci_lower, linestyle=":", label=f"2.5% = {ci_lower:.3f}")
plt.axvline(ci_upper, linestyle=":", label=f"97.5% = {ci_upper:.3f}")

plt.title("Trust Distribution with Mean and 95% CI")
plt.xlabel("Trust (T)")
plt.ylabel("Density")
plt.legend()

plt.show()