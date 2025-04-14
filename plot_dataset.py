# %%
from torch_dataset import get_train_data
import matplotlib.pyplot as plt
import numpy as np
import torch

# Get training and test data
input_train, target_train, input_test, target_test = get_train_data(3, shuffle=True)

# Split input into x and y components (shape: (N, n²/2))
half_index = input_train.shape[1] // 2
x_component_train = input_train[:, :half_index]
y_component_train = input_train[:, half_index:]

# Plot the training input points
plt.figure(figsize=(6, 6))

for i in range(len(x_component_train)):
    x_vals = x_component_train[i].cpu().tolist()
    y_vals = y_component_train[i].cpu().tolist()


    plt.scatter(x_vals, y_vals, s=30, label='Input Points' if i == 0 else "")

plt.legend()
plt.title("Input Points in 2D Space")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.show()

# %%
