### Hyperparameter configuration ###

# Training settings
BATCH_SIZE = 6
NUM_ITERATIONS = 500
LEARNING_RATE = 5e-3

# Model architecture
HIDDEN_DIM = 312
NUM_LAYERS = 1
ACTIVATION_FUNC = "ReLU"

# Data settings
IMAGE_DIM = 128
VOCAB_SIZE = 5  # number of Bravais lattice types
K_MAX_PEAKS = 32  # number of peaks to extract

# Other useful constants
DEVICE = "cuda"  # or "cpu"
SEED = 42
