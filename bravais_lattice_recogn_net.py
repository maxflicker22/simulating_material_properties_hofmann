# %%
import comet_ml
COMET_API_KEY = "YpvtAhxzAypP59UQx6Wxey7vk"
COMET_PROJECT_NAME = "bravais-lattice-recognition_5_lattice_vectors_classification"
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, num_points, vocab_size=5, hidden_dim=128, num_layers=3):
        super(BravaisLatticeRecognitionNet, self).__init__(),

        self.hidden_layers = nn.ModuleList()

        self.input_dim = num_points

        for i in range(num_layers):
           in_dim = self.input_dim if i == 0 else hidden_dim
           self.hidden_layers.append(nn.Linear(in_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)  # Output layer for 4 classes
        self.activation = nn.ReLU()



    def forward(self, x):
      for layer in self.hidden_layers:
         x = self.activation(layer(x))  # Apply activation function after each layer

      logits = self.output_layer(x)  # shape: (batch_size, vocab_size)
      return logits
    

### Load training data ###

batch_size = 16

input_train_data, target_train_data, input_test_data, target_test_data = get_train_data(batch_size, shuffle=True)  # returns torch tensors
num_points = input_train_data.shape[1]



### Hyperparameter setting and optimization ###

# Model parameters:
params = dict(
  num_training_iterations = 3000,  # Increase this to train longer
  batch_size = batch_size,  # Experiment between 1 and 64
  learning_rate = 5e-5,  # Experiment between 1e-5 and 1e-1
  hidden_dim = 512,  # Experiment between 1 and 2048
  shuffle = True,  # Shuffle the data
  num_layers = 16,  # Number of hidden layers
  vocab_size = 5,  # Number of classes
  num_points = num_points,  # Number of points in the input
  activation = "ReLU",  # Activation function
)
    
# Training Method
def train_step(x, y):
    # Set model to training mode
    model.train() # Random Zeros
    
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
    loss = loss_fn(y_hat, y)
    
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



### Define optimizer and training operation ###

# Model, loss, optimizer

model = BravaisLatticeRecognitionNet(params["num_points"], params["vocab_size"], params["hidden_dim"], params["num_layers"])
print("Model structure:", model)
loss_fn = nn.CrossEntropyLoss()
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
    x_batch, y_batch, _, _  = get_train_data(params["batch_size"], shuffle=params["shuffle"]) # returns torch tensors

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

# Predict the class of a batch of input data
def predict_one_hot(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)  # raw output
        probs = torch.softmax(logits, dim=1)  # convert to probabilities
        predicted_indices = probs.argmax(dim=1)  # get the class with max prob

        # Convert indices to one-hot vectors
        one_hot = F.one_hot(predicted_indices, num_classes=probs.size(1)).float()

        return one_hot  # shape: (batch_size, num_classes)



predicted = predict_one_hot(model, input_test_data)


accuracy = (predicted.argmax(dim=1)  == target_test_data.argmax(dim=1)).float().mean()
print("Accuracy:", (predicted.argmax(dim=1)  == target_test_data.argmax(dim=1)).float())  # Should be close to 1.0
percent_right = torch.sum(accuracy) # Relative error
print("Final output shape:", predicted.shape)  # Should be (batch_size, 5)
print("Final output:", predicted)
print("Final input:", target_test_data)
print("Prozent erraten:", percent_right)

