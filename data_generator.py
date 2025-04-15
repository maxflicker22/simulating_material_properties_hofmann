import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from generate_test_data import reciprocal_lattice_2d, generate_lattice_points

# --- Bravais Gitterdefinitionen ---
BRAVAIS_TYPES = {
    "tp": (np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32)),
    "op": (np.array([1, 0], dtype=np.float32), np.array([0, 2], dtype=np.float32)),
    "oc": (np.array([0.5, 0.5], dtype=np.float32), np.array([0.5, -0.5], dtype=np.float32)),
    "hp": (np.array([1, 0], dtype=np.float32), np.array([0.5, np.sqrt(3)/2], dtype=np.float32)),
    "mp": (np.array([1, 0], dtype=np.float32), np.array([0.3420, 0.9397], dtype=np.float32))
}

# --- One-Hot-Encoding ---
TYPE_TO_ONEHOT = {
    name: torch.nn.functional.one_hot(torch.tensor(i), num_classes=5).float()
    for i, name in enumerate(BRAVAIS_TYPES.keys())
}
ONEHOT_TO_NAME = {v.argmax().item(): k for k, v in TYPE_TO_ONEHOT.items()}

def get_label_name(one_hot_tensor):
    return ONEHOT_TO_NAME[one_hot_tensor.argmax().item()]

# --- Normalisierung auf [-1, 1] ---
def normalize_to_unit_box(tensor):
    # Konvertiere falls nötig
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    x = tensor[0]
    y = tensor[1]
    max_val = torch.max(x.abs().max(), y.abs().max()) + 1e-8
    return tensor / max_val


# --- PyTorch Dataset ---
class BravaisLatticeDataset(Dataset):
    def __init__(self, n_points=100, samples_per_type=50, seed=42):
        np.random.seed(seed)
        self.data = []
        self.labels = []
        # Berechne n aus gewünschter Punktzahl: (2n+1)^2 = n_points
        n = int((np.sqrt(n_points) - 1) // 2)

      

        for name, (a1, a2) in BRAVAIS_TYPES.items():
            b1, b2 = reciprocal_lattice_2d(a1, a2)
            points = generate_lattice_points(b1, b2, n)  # shape: (2, (2n+1)^2)
            points = normalize_to_unit_box(points)

            for _ in range(samples_per_type):
                self.data.append(torch.tensor(points.flatten(), dtype=torch.float32))
                self.labels.append(TYPE_TO_ONEHOT[name])

        self.data = torch.stack(self.data)
        self.labels = torch.stack(self.labels)

        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- DataLoader-Funktion ---
def get_train_data(batch_size=16, test_split=0.2, **kwargs):
    dataset = BravaisLatticeDataset(**kwargs)
    N = len(dataset)
    split = int(N * (1 - test_split))
    # Split in Trainings- und Testdaten
    # random wählt zufällige element für test und training aus
    # mit seed bleibt der test set immer gleich
    generator = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(dataset, [split, N - split], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, dataset[0][0].shape[0]
