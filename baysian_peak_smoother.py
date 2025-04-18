from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
from data_generator import get_train_data_from_file

train_data, test_data, input_dim = get_train_data_from_file("dataset.pt", batch_size=1)

for x, y in train_data:
    
    input_vec = x[0]  # shape: (n²,)
    label_vec = y[0]  # shape: (5,)

    # Zurück in 2D Punkte aufteilen
    n = int(((input_vec.shape[0]) ** 0.5))
    xy_mesh_vals = input_vec.view(n,n).tolist()
    xy_mesh_vals = np.array(xy_mesh_vals)

    # Nimm die linke untere Ecke (1/4 in Fläche, also 1/8 in jeder Achse)
    # Because of Periodicity, we can only use the left lower corner
    # sub_n = n // 2
    sub_n = n // 1
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    X_sub = X[:sub_n, :sub_n]
    Y_sub = Y[:sub_n, :sub_n]
    Z_sub = xy_mesh_vals[:sub_n, :sub_n]

    X_sub_flat = np.vstack([X_sub.ravel(), Y_sub.ravel()]).T
    Z_sub_flat = xy_mesh_vals[:sub_n, :sub_n].ravel()   

    epsilon = 1e-4  # Toleranzschwelle für "nahe bei Null"
    # 2. Maske erstellen: nur Werte behalten, wo Z_sub_flat nicht ~0 ist
    mask = np.abs(Z_sub_flat) > epsilon

    # 3. Gefilterte Arrays
    X_filtered = X_sub_flat[mask]
    Z_filtered = Z_sub_flat[mask]
    
    plt.figure(figsize=(6,5))
    plt.tricontourf(X_filtered[:, 0], X_filtered[:, 1], Z_filtered, levels=100, cmap='viridis')
    plt.colorbar(label='Z value')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolated Z values at filtered points")
    plt.axis('equal')
    plt.show()



                         # shape (N,)
    """
    # X_flat = np.vstack([X.ravel(), Y.ravel()]).T
    X_flat = np.vstack([X.ravel(), Y.ravel()]).T
    Z_flat = xy_mesh_vals.ravel()  # shape (N,)


    kernel = 1.0 * ExpSineSquared(length_scale=0.2, periodicity=0.5) + WhiteKernel(noise_level=0.5)
    print("Length of x:", len(x))
    print("shape of x:", x.shape)  # shape: (batch_size, 2 * n²)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True)
    gp.fit(X_sub_flat, Z_sub_flat)
    print("Length of x:", len(x))
    print("nach fit of x:", x.shape)  # shape: (batch_size, 2 * n²)
    Z_mean, Z_std = gp.predict(X_flat, return_std=True)
    Z_mean = Z_mean.reshape(xy_mesh_vals.shape)

    Z_std = Z_std.reshape(xy_mesh_vals.shape)

    print("Z_mean shape:", Z_mean.shape)
    plt.imshow(Z_mean, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    plt.title("Gaussian-gefiltertes Bild")
    plt.colorbar()
    plt.show()
    """
    

