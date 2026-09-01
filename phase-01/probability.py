# Normal/Gaussian Distribution
import numpy as np
from scipy import stats

normal = stats.norm(loc = 0, scale = 1)
sample = normal.rvs(size = 1000)

print(f"Sample mean: {np.mean(sample):.4f}")
print(f"Sample standard deviation: {np.std(sample):.4f}")
print(f"PDF at x = 1.96 : {normal.cdf(1.96):.4f}")


# SoftMax

from scipy.special import softmax, log_softmax

logits = np.array([2.0, 1.0, 0.1])

softmax_probs = softmax(logits)
log_softmax_probs = log_softmax(logits)

print(f"Softmax probabilities: {softmax_probs}")
print(f"Log Softmax probabilities: {log_softmax_probs}")