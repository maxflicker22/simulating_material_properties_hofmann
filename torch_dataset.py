# %%
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from generate_test_data import return_reciprocal_lattice_points 


# Define 2D lattice vectors as NumPy arrays
a1_0_np = np.array([1, 0], dtype=np.float32)
a2_0_np = np.array([0, 1], dtype=np.float32)  # square lattice

a1_1_np = np.array([1, 0], dtype=np.float32)
a2_1_np = np.array([0, 2], dtype=np.float32)  # rectangular

a1_2_np = np.array([1, 0], dtype=np.float32)
a2_2_np = np.array([0.5, np.sqrt(3)/2], dtype=np.float32)  # hexagonal

# Example input shape: 2 x n²
n = 10 # Change this to your actual grid size

# Generate reciprocal lattice points
lattice_points_0 = return_reciprocal_lattice_points(a1_0_np, a2_0_np, n)
lattice_points_1 = return_reciprocal_lattice_points(a1_1_np, a2_1_np, n)
lattice_points_2 = return_reciprocal_lattice_points(a1_2_np, a2_2_np, n)

# Convert to torch tensors
input1 = torch.tensor(lattice_points_0.tolist(), dtype=torch.float32)  # Example input
input2 = torch.tensor(lattice_points_1.tolist(), dtype=torch.float32)  # Example input
input3 = torch.tensor(lattice_points_2.tolist(), dtype=torch.float32)  # Example input

# Check the shape of the inputs
assert input1.shape == (2, (2*n + 1) ** 2), "Input1 shape mismatch: expected (2, (2n+1)²), got {}".format(input1.shape)
assert input2.shape == (2, (2*n + 1) ** 2), "Input2 shape mismatch: expected (2, (2n+1)²), got {}".format(input2.shape)
assert input3.shape == (2, (2*n + 1) ** 2), "Input3 shape mismatch: expected (2, (2n+1)²), got {}".format(input3.shape)


print("input1 shape:", input1.shape)  # Should be (2, n²)   
print("input1:", input1)  # Print the input tensor


# Convert NumPy arrays to torch tensors
a1_0 = torch.tensor(a1_0_np.tolist(), dtype=torch.float32).unsqueeze(1)  # shape: (2, 1)
a2_0 = torch.tensor(a2_0_np.tolist(), dtype=torch.float32).unsqueeze(1)

a1_1 = torch.tensor(a1_1_np.tolist(), dtype=torch.float32).unsqueeze(1)
a2_1 = torch.tensor(a2_1_np.tolist(), dtype=torch.float32).unsqueeze(1)

a1_2 = torch.tensor(a1_2_np.tolist(), dtype=torch.float32).unsqueeze(1)
a2_2 = torch.tensor(a2_2_np.tolist(), dtype=torch.float32).unsqueeze(1)



# Combine two 2x1 vectors into 2x2 matrices
output1 = torch.cat([a1_0, a2_0], dim=1)  # shape: (2, 2)
output2 = torch.cat([a1_1, a2_1], dim=1)
output3 = torch.cat([a1_2, a2_2], dim=1)
# Check the shape of the vectors
assert output1.shape == (2, 2), "Output1 shape mismatch: expected (2, 2), got {}".format(output1.shape)
assert output2.shape == (2, 2), "Output2 shape mismatch: expected (2, 2), got {}".format(output2.shape)
assert output3.shape == (2, 2), "Output3 shape mismatch: expected (2, 2), got {}".format(output3.shape)
print("output1 shape:", output1.shape)  # Should be (2, 2)



# Prepare lists of inputs and outputs
# Transpose input and output tensors to match the expected shape
# expected shape = (x1,y1,x2,y2,....) instead of (x1,x2,y1,y2,...)
inputs = [input1.T, input2.T, input3.T]
outputs = [output1.T, output2.T, output3.T]

# Transpose inputs and outputs to consistent format
# We'll store inputs as flat 1D vectors: 2 x n² → flattened (2*n²,)
# We'll store outputs as flattened 2x2 → (4,)
X = torch.stack([inp.flatten() for inp in inputs])    # shape: (3, 2*n²)
Y = torch.stack([out.flatten() for out in outputs])   # shape: (3, 4)

# Create a custom dataset
class VectorDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



# get Test Data
def get_test_data(batch_size, shuffle=True):
    """
    Returns a DataLoader for the test data.
    """
    # Create dataset and dataloader
    dataset = VectorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    for inputs, targets in dataloader:
        #print("inputs shape:", inputs.shape)
        #print("targets shape:", targets.shape)
        return inputs, targets

# %%
