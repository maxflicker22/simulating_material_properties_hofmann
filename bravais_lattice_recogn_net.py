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
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Check that we are using a GPU, if not switch runtimes
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())

assert COMET_API_KEY != "", "Please insert your Comet API Key"

class BravaisLatticeRecognitionNet(nn.Module):
    def __init__(self, num_points, vocab_size=5, hidden_dim=128, num_layers=3):
        super(BravaisLatticeRecognitionNet, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.input_dim = num_points
        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            self.hidden_layers.append(nn.Linear(in_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

### Hyperparameter setting and optimization ###
BATCH_SIZE = 4
N_POINTS = 320
NUM_ITERATIONS = 2500
LEARNING_RATE = 5e-2
HIDDEN_DIM = 256
NUM_LAYERS = 1
VOCAB_SIZE = 5
ACTIVATION_FUNC = "ReLU"

params = dict(
  num_training_iterations = NUM_ITERATIONS,
  batch_size = BATCH_SIZE,
  learning_rate = LEARNING_RATE,
  hidden_dim = HIDDEN_DIM,
  num_layers = NUM_LAYERS,
  vocab_size = VOCAB_SIZE,
  num_points = N_POINTS,
  activation = ACTIVATION_FUNC,
)

train_loader, test_loader, input_dim = get_train_data(batch_size=BATCH_SIZE, n_points=N_POINTS)
print("Input dimension:", input_dim)
model = BravaisLatticeRecognitionNet(
    num_points=input_dim,
    vocab_size=params["vocab_size"],
    hidden_dim=params["hidden_dim"],
    num_layers=params["num_layers"]
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

### Create a Comet experiment to track our training run ###
def create_experiment():
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME
    )
    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()
    return experiment

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

### Training Loop ###
history = []
experiment = create_experiment()

if hasattr(tqdm, '_instances'): tqdm._instances.clear()
for iter in tqdm(range(params["num_training_iterations"])):
    train_loader, _, _ = get_train_data(batch_size=BATCH_SIZE, n_points=N_POINTS)
    for x_batch, y_batch in train_loader:
        model.train()
        optimizer.zero_grad()
        y_hat = model(x_batch)
        loss = loss_fn(y_hat, y_batch)
        loss.backward()
        optimizer.step()
    experiment.log_metric("loss", loss.item(), step=iter)
    history.append(loss.item())
    if iter % 100 == 0:
        torch.save(model.state_dict(), checkpoint_prefix)

torch.save(model.state_dict(), checkpoint_prefix)
experiment.flush()
experiment.end()

### Predict Crystal Lattice ###
def predict_one_hot(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        predicted_indices = probs.argmax(dim=1)
        one_hot = F.one_hot(predicted_indices, num_classes=probs.size(1)).float()
        return one_hot

### Evaluate on Test Data ###
y_true = []
y_pred = []

train_loader, test_loader, input_dim = get_train_data(batch_size=1000, n_points=N_POINTS)
#%%
for x_batch, y_batch in test_loader:
    preds = predict_one_hot(model, x_batch)
    print("Predictions shape:", preds.shape)
    y_true.extend(y_batch.argmax(dim=1).cpu().tolist())
    y_pred.extend(preds.argmax(dim=1).cpu().tolist())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(BRAVAIS_TYPES.keys()), yticklabels=list(BRAVAIS_TYPES.keys()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
#plt.show()

# Accuracy

accuracy = cm.diagonal().sum() / cm.sum()
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy:.4f}", file=open("accuracy.txt", "w"))

# Logge Bild zu Comet
experiment = create_experiment()
experiment.log_image("confusion_matrix.png")

## Logge Confusion Matrix als Tabelle direkt in Comet
#experiment.log_confusion_matrix(
#    y_true=y_true,
#    y_pred=y_pred,
#    labels=list(BRAVAIS_TYPES.keys()),
#    title="Confusion Matrix"
#)

experiment.end()

# %%
