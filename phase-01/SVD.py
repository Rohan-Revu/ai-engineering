import numpy as np

np.random.seed(42)
A = np.random.randn(5, 4)


U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("NumPy singular values:", np.round(S_np, 4))

A_reconstructed = U_np @ np.diag(S_np) @ Vt_np.T
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")