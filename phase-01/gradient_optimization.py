import torch

model = torch.nn.Linear(784, 10) 

print(model.weight.shape)
print(model.bias.shape)


x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# forward pass
output = model(x)

# compute loss
loss = loss_fn(output, y)

# backward pass
optimizer.zero_grad()
loss.backward()

# print gradients
print("Loss:", loss.item())
print("Weight gradient shape:", model.weight.grad.shape)
print("Bias gradient shape:", model.bias.grad.shape)

# update parameters
optimizer.step()


# run the forward pass again to see the effect of the update
output = model(x)

loss = loss_fn(output, y)

optimizer.zero_grad()

loss.backward()

print("Loss:", loss.item())
print("Weight gradient shape:", model.weight.grad.shape)
print("Bias gradient shape:", model.bias.grad.shape)


