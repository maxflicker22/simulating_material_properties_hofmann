# %%
import comet_ml
COMET_API_KEY = "YpvtAhxzAypP59UQx6Wxey7vk"
COMET_PROJECT_NAME = "bravais-lattice-recognition_5_lattice_vectors"
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_dataset import get_train_data
import os
import matplotlib.pyplot as plt
import numpy as np

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())


#assert torch.cuda.is_available(), "Please enable GPU from runtime settings"
assert COMET_API_KEY != "", "Please insert your Comet API Key"



class BravaisLatticeRecognitionNet(nn.Module):
    def __init__(self, num_points):
        super(BravaisLatticeRecognitionNet, self).__init__(),

        self.model = nn.Sequential(
            nn.Linear(num_points, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Output: [x1, y1, x2, y2]
        )

  
    def forward(self, x):
        # x shape: (batch_size, num_points, 2)
        #x = x.view(x.size(0), -1)  # flatten to (batch_size, input_size)
        out = self.model(x)        # shape: (batch_size, 4)
        #norm = torch.norm(out, dim=1, keepdim=True) + 1e-8  # avoid division by zero
        out_normalized = out / 1
        return out_normalized
    
### Hyperparameter setting and optimization ###

# Model parameters:
params = dict(
  num_training_iterations = 1500,  # Increase this to train longer
  batch_size = 5,  # Experiment between 1 and 64
  learning_rate = 5e-4,  # Experiment between 1e-5 and 1e-1
  #hidden_size = 1024,  # Experiment between 1 and 2048
  shuffle = True,  # Shuffle the data
)
    
# Training Method
def train_step(x, y):
    # Set model to training mode
    model.train()
    
    # Zero gradients for every step
    optimizer.zero_grad()

    # Forward pass
    y_hat = model(x)

    ## Log Confusion Matrix
    #experiment.log_confusion_matrix(
    #    y_true=y,
    #    y_pred=y_hat,
    #    title="Confusion Matrix",
    #    step=iter
    #)

    # Compute loss
    loss = loss_fn(y, y_hat)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    return loss.item(), y_hat

### Create a Comet experiment to track our training run ###

def create_experiment():
  # end any prior experiments
  if 'experiment' in locals():
    experiment.end()

  # initiate the comet experiment for tracking
  experiment = comet_ml.Experiment(
                  api_key=COMET_API_KEY,
                  project_name=COMET_PROJECT_NAME)
  # log our hyperparameters, defined above, to the experiment
  for param, value in params.items():
    experiment.log_parameter(param, value)


  
  experiment.flush()

  return experiment

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)


# Get Device for GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_train_data, target_train_data = get_train_data(params["batch_size"], shuffle=True)  # returns torch tensors
num_points = input_train_data.shape[1]
print("num_points:", num_points)

### Define optimizer and training operation ###

# Model, loss, optimizer

model = BravaisLatticeRecognitionNet(num_points)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

# Move the model to the GPU
model.to(device)


##################
# Begin training!#
##################

history = []
experiment = create_experiment()

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
for iter in tqdm(range(params["num_training_iterations"])):

    # Grab a batch and propagate it through the network
    x_batch, y_batch = get_train_data(params["batch_size"], shuffle=params["shuffle"]) # returns torch tensors

    #x_batch = torch.tensor(x_batch, dtype=torch.long).to(device)
    #y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

    # Take a train step
    loss, y_pred = train_step(x_batch, y_batch)

    # Log the loss to the Comet interface
    experiment.log_metric("loss", loss, step=iter)

   

    # Update the progress bar and visualize within notebook
    history.append(loss)

    # Save model checkpoint
    if iter % 100 == 0:
        torch.save(model.state_dict(), checkpoint_prefix)

# Save the final trained model
torch.save(model.state_dict(), checkpoint_prefix)
experiment.flush()

experiment.end()







### Predict Crystal Lattice ###


output_fin = model(input_train_data)
print("Final output shape:", output_fin.shape)  # Should be (batch_size, 4)
print("Final output:", output_fin)
print("Target test data:", target_train_data)

"""
# Plot input and output points
plt.figure(figsize=(6, 6))

for i in range(len(input_train_data)):
    # Convert PyTorch tensors to NumPy arrays
    input_points = input_train_data[i].detach().cpu().numpy().reshape(-1, 2)
    output_points = output_fin[i].detach().cpu().numpy().reshape(-1, 2)

    # Plot input points (only label once)
    plt.scatter(input_points[:, 0], input_points[:, 1], s=30, label='Input Points' if i == 0 else "")

    # Plot output points (only label once)
    plt.scatter(output_points[:, 0], output_points[:, 1], s=30, label='Output Points' if i == 0 else "")

plt.legend()
plt.title("Input vs. Output Points")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.show()

"""