import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)

w1 = torch.tensor(0.5, requires_grad=True)
w2 = torch.tensor(-0.5, requires_grad=True)

b = torch.tensor(2.0, requires_grad=True)

y = w1 * x1 + w2 * x2 + b

z = torch.relu(y)

z.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")
print(f"PyTorch dy/dx2 = {x2.grad.item()}")
print(f"PyTorch dy/dw1 = {w1.grad.item()}")
print(f"PyTorch dy/dw2 = {w2.grad.item()}")
print(f"PyTorch dy/db = {b.grad.item()}")
print(f"y = {y.item():.4f}, z = {z.item():.4f}")