class Vector:
    def __init__(self, components):
        self.components = list(components)

    def __repr__(self):
        return f"Vector({self.components})"


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


rotation_90 = Matrix([
    [0, -1],
    [1,  0]
])

point = Vector([3, 1])

rotated = rotation_90 @ point

print("Original:", point)
print("Rotated:", rotated)