#%%
import comet_ml
COMET_API_KEY = "YpvtAhxzAypP59UQx6Wxey7vk"
COMET_PROJECT_NAME = "bravais-lattice-recognition_5_lattice_vectors_classification"
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_generator import get_train_data, get_label_name, BRAVAIS_TYPES
import os
import matplotlib.pyplot as plt
import numpy as np
from generate_test_data import generate_lattice_points


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
    


### Hyperparameter setting and optimization ###

BATCH_SIZE = 10
N_POINTS = 100
SAMPLE_PER_TYPE = 10
NUM_ITERATIONS = 3000
LEARNING_RATE = 5e-2
HIDDEN_DIM = 256
NUM_LAYERS = 1
VOCAB_SIZE = 5
ACTIVATION_FUNC = "ReLU"

# Model parameters:
params = dict(
  num_training_iterations = NUM_ITERATIONS,  # Increase this to train longer
  batch_size = BATCH_SIZE,  # Experiment between 1 and 64
  learning_rate = LEARNING_RATE,  # Experiment between 1e-5 and 1e-1
  hidden_dim = HIDDEN_DIM,  # Experiment between 1 and 2048
  num_layers = NUM_LAYERS,  # Number of hidden layers
  vocab_size = VOCAB_SIZE,  # Number of classes
  num_points = N_POINTS,  # Number of points in the input
  activation = ACTIVATION_FUNC,  # Activation function
)



### Load training data ###


train_loader, test_loader, input_dim = get_train_data(batch_size=BATCH_SIZE, n_points=N_POINTS, samples_per_type=SAMPLE_PER_TYPE)


# Beispielbatch laden
for x, y in train_loader:
    print("Length of x:", len(x))
    print("shape of x:", x.shape)  # shape: (batch_size, 2 * n²)
    input_train_vec = x
    label_train_vec = y  

# Beispielbatch laden
for x, y in test_loader:
    print("Length of x:", len(x))
    print("shape of x:", x.shape)  # shape: (batch_size, 2 * n²)
    input_test_vec = x  # shape: (2 * n²,)
    label_test_vec = y  # shape: (5,)




    
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

model = BravaisLatticeRecognitionNet(
    num_points=input_dim,  # ← das ist die tatsächliche Eingabegröße!
    vocab_size=params["vocab_size"],
    hidden_dim=params["hidden_dim"],
    num_layers=params["num_layers"]
)
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
    #x_batch, y_batch, _, _  = get_train_data(params["batch_size"], shuffle=params["shuffle"]) # returns torch tensors
    train_loader, _, _ = get_train_data(batch_size=BATCH_SIZE, n_points=N_POINTS, samples_per_type=SAMPLE_PER_TYPE)
    for x_batch, y_batch in train_loader:
       
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



predicted = predict_one_hot(model, input_test_vec)


accuracy = (predicted.argmax(dim=1)  == label_test_vec.argmax(dim=1)).float().mean()
print("Accuracy:", (predicted.argmax(dim=1)  == label_test_vec.argmax(dim=1)).float())  # Should be close to 1.0
percent_right = torch.sum(accuracy) # Relative error
print("Final output shape:", predicted.shape)  # Should be (batch_size, 5)
print("Final output:", predicted)
print("Final input:", label_test_vec)
print("Prozent erraten:", percent_right)




# %%
