import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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

def reciprocal_lattice_2d(a1, a2):
    a1_3d = np.array([a1[0], a1[1], 0])
    a2_3d = np.array([a2[0], a2[1], 0])
    area = np.cross(a1_3d, a2_3d)[2]
    b1 = 2 * np.pi * np.cross([0, 0, 1], a2_3d)[:2] / area
    b2 = 2 * np.pi * np.cross(a1_3d, [0, 0, 1])[:2] / area
    return b1.astype(np.float32), b2.astype(np.float32)

def generate_lattice_points(b1, b2, N):
    lattice_points = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            point = i * b1 + j * b2
            lattice_points.append(point)
    return np.array(lattice_points).T.astype(np.float32)  # shape: (2, (2N+1)^2)



# --- Normalisierung auf [-1, 1] ---
def normalize_to_unit_box(tensor):
    # Konvertiere falls nötig
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    x = tensor[0]
    y = tensor[1]
    x = x - x.mean()
    y = y - y.mean()
    max_val = torch.max(x.abs().max(), y.abs().max()) + 1e-8
    return torch.stack([x / max_val, y / max_val])

# --- Caluclate reciprocal lattice ---
def reciprocal_lattice_2d(a1, a2):
    a1_3d = np.array([a1[0], a1[1], 0])
    a2_3d = np.array([a2[0], a2[1], 0])
    area = np.cross(a1_3d, a2_3d)[2]
    b1 = 2 * np.pi * np.cross([0, 0, 1], a2_3d)[:2] / area
    b2 = 2 * np.pi * np.cross(a1_3d, [0, 0, 1])[:2] / area
    return b1.astype(np.float32), b2.astype(np.float32)

# --- Generierung der Gitterpunkte ---
def generate_lattice_points(b1, b2, N):
    lattice_points = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            point = i * b1 + j * b2
            lattice_points.append(point)
    return np.array(lattice_points).T.astype(np.float32)  # shape: (2, (2N+1)^2)

# --- Transform infinite sharp peaks to 2D Gaussian for leed image generation ---
def gaussian_2d(x, y, means, sigma_x=0.01, sigma_y=0.01):
    X, Y = np.meshgrid(x, y)
    means = np.array(means.tolist())
    print("means:", means)
    Z = np.zeros(X.shape)
    for i in range(means.shape[0]):
        a = 1 / (2 * np.pi * sigma_x * sigma_y)
        x0 = means[i, 0]
        y0 = means[i, 1]
        b = -((X - x0) ** 2 / (2 * sigma_x ** 2) + (Y - y0) ** 2 / (2 * sigma_y ** 2))

        Z += a * np.exp(b)
        
    return Z

    


# --- PyTorch Dataset ---
class BravaisLatticeDataset(Dataset):
    def __init__(self, n_points=64, seed=42, num_pixel=128, sigma_x=0.01, sigma_y=0.01):
        np.random.seed(seed)
        self.data = []
        self.labels = []
        # Berechne n aus gewünschter Punktzahl: (2n+1)^2 = n_points
        n = int((np.sqrt(n_points) - 1) // 2)

        for name, (a1, a2) in BRAVAIS_TYPES.items():
            b1, b2 = reciprocal_lattice_2d(a1, a2)
            points = generate_lattice_points(b1, b2, n)  # shape: (2, (2n+1)^2)
            points = normalize_to_unit_box(points)

            x = np.linspace(torch.min(points[0, :]).tolist(), torch.max(points[0, :]).tolist(), num_pixel)
            y = np.linspace(torch.min(points[1, :]).tolist(), torch.max(points[1, :]).tolist(), num_pixel)

            Z = gaussian_2d(x, y, points.T, sigma_x, sigma_y)
            Z = torch.tensor(Z).flatten()
            Z = Z / torch.max(Z)  # Normalisierung auf [0, 1]

            self.data.append(Z)
            self.labels.append(TYPE_TO_ONEHOT[name])

        self.labels = torch.stack(self.labels)
        self.data = torch.stack(self.data)


        # Gaussian peaks
        #for i in range(points.shape[1]):
            # Gaussian peaks
                  # Make Gaussian peaks instead of sharp peaks
       
        

        #perm = torch.randperm(len(self.data))
        #self.data = self.data[perm]
        #self.labels = self.labels[perm]

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
    print("train_set:", len(train_set))
    print("test_set:", len(test_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, dataset[0][0].shape[0]

def generate_test_data_and_store(filename="dataset.pt"):
    # Beispielaufruf
    print("Generating datasets...")
    d1 = BravaisLatticeDataset(n_points=32, num_pixel=128, sigma_x=0.01, sigma_y=0.01)
    d2 = BravaisLatticeDataset(n_points=96, num_pixel=128, sigma_x=0.01, sigma_y=0.01)
    d3 = BravaisLatticeDataset(n_points=128, num_pixel=128, sigma_x=0.02, sigma_y=0.02)
    d4 = BravaisLatticeDataset(n_points=64, num_pixel=128, sigma_x=0.02, sigma_y=0.02)
    d5 = BravaisLatticeDataset(n_points=48, num_pixel=128, sigma_x=0.01, sigma_y=0.01)
    d6 = BravaisLatticeDataset(n_points=64, num_pixel=128, sigma_x=0.015, sigma_y=0.015)
    d7 = BravaisLatticeDataset(n_points=128, num_pixel=128, sigma_x=0.03, sigma_y=0.03)
    d8 = BravaisLatticeDataset(n_points=256, num_pixel=128, sigma_x=0.015, sigma_y=0.015)
    print("Combining...")
    data = torch.cat([d1.data, d2.data, d3.data, d4.data, d5.data, d6.data, d7.data, d8.data], dim=0)
    labels = torch.cat([d1.labels, d2.labels, d3.labels, d4.labels, d5.labels, d6.labels, d7.labels, d8.labels], dim=0)

    print("Shuffling...")
    perm = torch.randperm(len(data))
    data = data[perm]
    labels = labels[perm]

    print("Saving...")
    torch.save({"images": data, "labels": labels}, filename)

    # Speichern der Daten
    torch.save(data, "data.pt")
    torch.save(labels, "labels.pt")
    print("Done.")


    # --- DataLoader from saved dataset ---
def get_train_data_from_file(filepath="dataset.pt", batch_size=16, test_split=0.2):
    # Load saved tensors
    data_dict = torch.load(filepath)
    images = data_dict["images"]
    labels = data_dict["labels"]

    dataset = torch.utils.data.TensorDataset(images, labels)
    N = len(dataset)
    split = int(N * (1 - test_split))

    # Reproducible random split
    generator = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(dataset, [split, N - split], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Loaded dataset from '{filepath}'")
    print("train_set:", len(train_set))
    print("test_set:", len(test_set))

    return train_loader, test_loader, images.shape[1]  # assuming images have shape [N^2, D]

def plot_dataset(filename="dataset.pt", num_samples=5):
    # Lade Daten
    train_loader, test_loader, input_dim = get_train_data_from_file("dataset.pt", batch_size=num_samples)
    # Beispielbatch laden
  
    for x, y in train_loader:
        for i in range(num_samples):
            input_vec = x[i]  # shape: (n²,)
            label_vec = y[i]  # shape: (5,)

            # Zurück in 2D Punkte aufteilen
            n = int(((input_vec.shape[0]) ** 0.5))
            xy_mesh_vals = input_vec.view(n,n).tolist()
            xy_mesh_vals = np.array(xy_mesh_vals)

            # Reziprokes Gitter plotten
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(xy_mesh_vals, extent=[0, 128, 0, 128], cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Intensity')
            plt.title(f"Reciprocal Lattice of {get_label_name(label_vec)}")
            plt.xlabel("kx [Pixel]")
            plt.ylabel("ky [Pixel]")
            #plt.grid(True)
            #plt.axis("equal")
            #plt.ylim(range(xy_mesh_vals.shape[0]-1, -1, -1))  # y-Achse umkehren
            #plt.xlim(0, xy_mesh_vals.shape[1]-1)  # x-Achse anpassen
            # Reales Gitter plotten (aus a1, a2)
            name = get_label_name(label_vec)
            a1, a2 = BRAVAIS_TYPES[name]
            real_points = generate_lattice_points(a1, a2, 4)

            plt.subplot(1, 2, 2)
            plt.scatter(real_points[0], real_points[1], s=30, color='orange')
            plt.title(f"Real Lattice\nClass: {name}")
            plt.xlabel("x [a]")
            plt.ylabel("y [a]")
            plt.grid(True)
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

        break  # Nur ein Beispiel anzeigen


#generate_test_data_and_store("dataset.pt")
#plot_dataset("dataset.pt", num_samples=10)





