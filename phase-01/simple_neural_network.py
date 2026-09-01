import random

class Matrix:
    def __init__(self, rows):
        self.rows = rows
        self.num_rows = len(rows)
        self.num_cols = len(rows[0])

    def __matmul__(self, vector):
        result = []

        # Go through each row of the matrix
        for i in range(self.num_rows):

            total = 0

            # Multiply each matrix value
            # with the corresponding vector value
            for j in range(self.num_cols):
                total += self.rows[i][j] * vector.components[j]

            result.append(total)

        return Vector(result)

class Vector:
    def __init__(self, components):
        self.components = list(components)

    def __repr__(self):
        return f"Vector({self.components})"



random.seed(42)
weights = []

for i in range(2):

    row = []

    for j in range(3):
        value = random.gauss(0, 0.1)
        row.append(value)

    weights.append(row)

weights = Matrix(weights)
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")