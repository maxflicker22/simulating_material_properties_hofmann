# %%
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from generate_test_data import return_reciprocal_lattice_points 


# Define 2D lattice vectors as NumPy arrays

# tp - Tetragonal Primitive (Square lattice)
a1_tp = np.array([1, 0], dtype=np.float32)
a2_tp = np.array([0, 1], dtype=np.float32)

# op - Orthorhombic Primitive (Rectangular)
a1_op = np.array([1, 0], dtype=np.float32)
a2_op = np.array([0, 2], dtype=np.float32)

# oc - Orthorhombic Centered (Centered Rectangular)
a1_oc = np.array([0.5,  0.5], dtype=np.float32)
a2_oc = np.array([0.5, -0.5], dtype=np.float32)

# hp - Hexagonal Primitive
a1_hp = np.array([1, 0], dtype=np.float32)
a2_hp = np.array([0.5, np.sqrt(3)/2], dtype=np.float32)

# mp - Monoclinic Primitive (Oblique, 70° angle)
a1_mp = np.array([1, 0], dtype=np.float32)
a2_mp = np.array([0.3420, 0.9397], dtype=np.float32)  # cos(70°), sin(70°)

# Example input shape: (2 * n + 1)²
n = 50 # Change this to your actual grid size

# Generate reciprocal lattice points
lattice_points_0 = return_reciprocal_lattice_points(a1_tp, a2_tp, n)
lattice_points_1 = return_reciprocal_lattice_points(a1_op, a2_op, n)
lattice_points_2 = return_reciprocal_lattice_points(a1_oc, a2_oc, n)
lattice_points_3 = return_reciprocal_lattice_points(a1_hp, a2_hp, n)
lattice_points_4 = return_reciprocal_lattice_points(a1_mp, a2_mp, n)

# Convert to torch tensors
input1 = torch.tensor(lattice_points_0.tolist(), dtype=torch.float32)  # Example input
input2 = torch.tensor(lattice_points_1.tolist(), dtype=torch.float32)  # Example input
input3 = torch.tensor(lattice_points_2.tolist(), dtype=torch.float32)  # Example input
input4 = torch.tensor(lattice_points_3.tolist(), dtype=torch.float32)  # Example input
input5 = torch.tensor(lattice_points_4.tolist(), dtype=torch.float32)  # Example input

# Check the shape of the inputs
assert input1.shape == (2, (2*n + 1) ** 2), "Input1 shape mismatch: expected (2, (2n+1)²), got {}".format(input1.shape)
assert input2.shape == (2, (2*n + 1) ** 2), "Input2 shape mismatch: expected (2, (2n+1)²), got {}".format(input2.shape)
assert input3.shape == (2, (2*n + 1) ** 2), "Input3 shape mismatch: expected (2, (2n+1)²), got {}".format(input3.shape)
assert input4.shape == (2, (2*n + 1) ** 2), "Input2 shape mismatch: expected (2, (2n+1)²), got {}".format(input2.shape)
assert input5.shape == (2, (2*n + 1) ** 2), "Input3 shape mismatch: expected (2, (2n+1)²), got {}".format(input3.shape)


#print("input1 shape:", input1.shape)  # Should be (2, n²)   
#print("input1:", input1)  # Print the input tensor


# Convert NumPy arrays to torch tensors
a1_tp = torch.tensor(a1_tp.tolist(), dtype=torch.float32).unsqueeze(1)  # shape: (2, 1)
a2_tp = torch.tensor(a2_tp.tolist(), dtype=torch.float32).unsqueeze(1)

a1_op = torch.tensor(a1_op.tolist(), dtype=torch.float32).unsqueeze(1)
a2_op = torch.tensor(a2_op.tolist(), dtype=torch.float32).unsqueeze(1)

a1_oc = torch.tensor(a1_oc.tolist(), dtype=torch.float32).unsqueeze(1)
a2_oc = torch.tensor(a2_oc.tolist(), dtype=torch.float32).unsqueeze(1)

a1_hp = torch.tensor(a1_hp.tolist(), dtype=torch.float32).unsqueeze(1)
a2_hp = torch.tensor(a2_hp.tolist(), dtype=torch.float32).unsqueeze(1)

a1_mp = torch.tensor(a1_mp.tolist(), dtype=torch.float32).unsqueeze(1)
a2_mp = torch.tensor(a2_mp.tolist(), dtype=torch.float32).unsqueeze(1)


# Create related output classification Vector
output1 = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32).unsqueeze(0)  # shape: (1, 5)
output2 = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32).unsqueeze(0)  # shape: (1, 5)
output3 = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32).unsqueeze(0)  # shape: (1, 5)
output4 = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32).unsqueeze(0)  # shape: (1, 5)
output5 = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32).unsqueeze(0)  # shape: (1, 5)

# Check the shape of the vectors
assert output1.shape == (1, 5), "Output1 shape mismatch: expected (1, 4), got {}".format(output1.shape)
assert output2.shape == (1, 5), "Output1 shape mismatch: expected (1, 4), got {}".format(output1.shape)
assert output3.shape == (1, 5), "Output1 shape mismatch: expected (1, 4), got {}".format(output1.shape)
assert output4.shape == (1, 5), "Output1 shape mismatch: expected (1, 4), got {}".format(output1.shape)
assert output5.shape == (1, 5), "Output1 shape mismatch: expected (1, 4), got {}".format(output1.shape)
#print("output1 shape:", output1.shape)  # Should be (2, 2)



# Prepare lists of inputs and outputs
# Transpose input and output tensors to match the expected shape
# expected shape = (x1,y1,x2,y2,....) instead of (x1,x2,y1,y2,...)
inputs = [input1, input2, input3, input4, input5] # Currently not Transposed!!
outputs = [output1, output2, output3, output4, output5]


# Number of splits per input
num_splits = 64

X_list = []
Y_list = []

for inp, out in zip(inputs, outputs):
    # inp: shape (2, total_points), e.g., (2, (2*n+1)²)
    # out: shape (1, 5)
    #print("inp shape:", inp.shape)  # Should be (2, total_points)
    splits = torch.chunk(inp, num_splits, dim=1) # List of 8 tensors, each (2, total_points/8)
    
    # Extract min length of the certain dimension
    min_len = 2 * min(split.shape[1] for split in splits) # 2 * Because of x and y components
    
    for split in splits:
        #print("splits shape:", split.shape)  # Should be (2, total_points/8)
        #print("split flatten", split.flatten().shape)  # Should be (2 * total_points/8,)
        flat = split.flatten()  # Flatten the split tensor
        X_list.append(flat[:min_len])  # Shape: (2 * sub_length,)
        Y_list.append(out.squeeze(0))   # Shape: (5,)


# Stack into final tensors
#print("X_list length:", len(X_list))  # Should be 5*num_splits = 40
X = torch.stack(X_list)  # Shape: (5*num_splits, input_dim)
Y = torch.stack(Y_list)  # Shape: (5*num_splits, 5)

#print("X shape:", X.shape)
#print("Y shape:", Y.shape)

perm = torch.randperm(X.size(0))  # generate shuffled indices
X_shuffled = X[perm]
Y_shuffled = Y[perm]

#print("X_shuffled shape:", X_shuffled.shape)
#print("Y_shuffled shape:", Y_shuffled.shape)

# Create a custom dataset
class VectorDataset(Dataset):
    def __init__(self, X, Y, train_ratio=0.8, seed=42):
        assert X.shape[0] % 2 == 0, "X must have even number of samples (x + y blocks)"
        assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"

        self.N = X.shape[0] // 2  # Anzahl x/y-Paare
        
        # Split indices into train/test
        split_idx = int(train_ratio * self.N)

        # Randomly shuffle only the first half (x-blocks)
        indices = torch.arange(self.N)

        # Split indices into train/test
        split_idx = int(train_ratio * self.N)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        # Build full index sets (x + corresponding y)
        train_idx_full = torch.cat([train_idx, train_idx + self.N])
        test_idx_full = torch.cat([test_idx, test_idx + self.N])

        # Assign splits
        self.X_train = X[train_idx_full]
        self.Y_train = Y[train_idx_full]

        self.X_test = X[test_idx_full]
        self.Y_test = Y[test_idx_full]

        #print("X_train", self.X_train)
        #print("Y_train", self.Y_train)
        #print("X_test", self.X_test)
        #print("Y_test", self.Y_test)

        # For training via DataLoader:
        self.X = self.X_train
        self.Y = self.Y_train

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_test_data(self):
        return self.X_test, self.Y_test



# get Train Data
def get_train_data(batch_size, shuffle=True):
    """
    Returns a DataLoader for the train data.
    """
    # Create dataset and dataloader
    dataset = VectorDataset(X_shuffled, Y_shuffled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Test data
    x_test, y_test = dataset.get_test_data()

    for inputs, targets in dataloader:
        #print("inputs shape:", inputs.shape)
        #print("targets shape:", targets.shape)
        return inputs, targets, x_test, y_test

# %%
