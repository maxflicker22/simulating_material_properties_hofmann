# train.py
#%%
from data_generator import get_train_data, get_label_name, BRAVAIS_TYPES
import matplotlib.pyplot as plt
import torch
import numpy as np
from generate_test_data import generate_lattice_points

# Lade Daten
train_loader, test_loader = get_train_data(batch_size=1, n_points=100, samples_per_type=100)

# Beispielbatch laden
for x, y in train_loader:
    print("Length of x:", len(x))
    print("shape of x:", x.shape)  # shape: (batch_size, 2 * n²)
    input_vec = x[0]  # shape: (2 * n²,)
    label_vec = y[0]  # shape: (5,)

    # Zurück in 2D Punkte aufteilen
    n = int((input_vec.shape[0] // 2) ** 0.5)
    x_vals = input_vec[:n*n].reshape(n, n).flatten().tolist()
    y_vals = input_vec[n*n:].reshape(n, n).flatten().tolist()

    # Reziprokes Gitter plotten
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x_vals, y_vals, s=30)
    plt.title(f"Reciprocal Lattice\nClass: {get_label_name(label_vec)}")
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.grid(True)
    plt.axis("equal")

    # Reales Gitter plotten (aus a1, a2)
    name = get_label_name(label_vec)
    a1, a2 = BRAVAIS_TYPES[name]
    real_points = generate_lattice_points(a1, a2, n)

    plt.subplot(1, 2, 2)
    plt.scatter(real_points[0], real_points[1], s=30, color='orange')
    plt.title(f"Real Lattice\nClass: {name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")

    plt.tight_layout()
    plt.show()

    break  # Nur ein Beispiel anzeigen
