import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")

# Dot product
print(f"a . b = {np.dot(a, b)}")

# Magnitude
print(f"|a| = {np.linalg.norm(a)}")

# Normalization
normalized_a = a / np.linalg.norm(a)
print(f"Normalized a = {normalized_a}")

# Cosine similarity
cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Cosine similarity between a and b = {cosine_similarity}")


# Projection of a onto b
projection = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of a onto b = {projection}")

# Gram-Schmidt process
def gram_schmidt(vectors):
    orthonormal_vectors = []

    for v in vectors:
        v = v.copy()

        for u in orthonormal_vectors:
            v = v - np.dot(v, u) * u

        v = v / np.linalg.norm(v)

        orthonormal_vectors.append(v)

    return orthonormal_vectors

GS = gram_schmidt([a, b])

print("Gram-Schmidt:")
print(np.column_stack(GS))


# QR decomposition where Q is orthonormal and R is upper triangular
Q, R = np.linalg.qr(np.column_stack([a, b]))

print("QR:")
print(Q)


#rank of a Matrix
matrix = np.array([[1, 2], [3, 4], [5, 6]])
rank = np.linalg.matrix_rank(matrix)
print(f"Rank of the matrix:\n{matrix}\n is {rank}")