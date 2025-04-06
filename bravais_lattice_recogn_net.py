# %%
import torch
import torch.nn as nn
from torch_dataset import get_test_data


class BravaisLatticeRecognitionNet(nn.Module):
    def __init__(self, num_points):
        super(BravaisLatticeRecognitionNet, self).__init__(),

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Output: [x1, y1, x2, y2]
        )

    def forward(self, x):
        # x shape: (batch_size, num_points, 2)
        #x = x.view(x.size(0), -1)  # flatten to (batch_size, input_size)
        return self.model(x)  # linear output, shape: (batch_size, 4)



# %%
# Generate test data
input_test_data, target_test_data = get_test_data(batch_size=32, shuffle=True)
print("Input test data shape:", input_test_data.shape[1])  # Should be (2, 2*n²)
print("Input test data:", input_test_data)
print("Target test data shape:", target_test_data.shape)  # Should be (2, 4)
print("Target test data:", target_test_data)


# %%
# Test the model
num_points = input_test_data.shape[1] # Number of points in the lattice
model = BravaisLatticeRecognitionNet(num_points)
output_test_data = model(input_test_data)
print("Output test data shape:", output_test_data.shape)  # Should be (2, 4)
print("Output test data:", output_test_data)
# %%





"""
input = torch.randn(32, num_points, 2)  # Batch of 32 samples
target = torch.randn(32, 4)  # Target output for the model

# Forward pass  
output = model(input)
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)
print("Output shape:", output.shape)    
print("Loss:", loss.item()) 
# Backward pass 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("Updated model parameters after one step.")
 
"""
# %%
