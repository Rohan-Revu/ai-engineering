import torch

# Created on CPU by default
t = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6]],
    dtype=torch.float32
)

print(t.shape)
print(t.stride())
print(t.is_contiguous())

print(t.reshape(3, 2))
print(t.unsqueeze(0))
print(t.transpose(0, 1))

# Transpose creates a non-contiguous view,
# contiguous() creates a contiguous copy
print(t.transpose(0, 1).contiguous())

# Example tensors for einsum
A = torch.tensor(
    [[1., 2., 3.],
     [4., 5., 6.]]
)

B = torch.tensor(
    [[1., 2.],
     [3., 4.],
     [5., 6.]]
)

print(torch.einsum("ik,kj->ij", A, B))