from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max
from data_generator import get_train_data_from_file
import numpy as np
import matplotlib.pyplot as plt


def detect_peaks(image, min_distance=3, threshold_abs=0.05):
    coordinates = peak_local_max(
        image,
        min_distance=min_distance,    # Abstand zw. Peaks (in Pixeln)
        threshold_abs=threshold_abs,  # Min. Intensität
    )
    return coordinates  # array of shape (N_peaks, 2): [row, col]
def pixel_to_xy(coords, image_shape, x_range=(-1, 1), y_range=(-1, 1)):
    rows, cols = image_shape
    x_vals = np.linspace(*x_range, cols)
    y_vals = np.linspace(*y_range, rows)
    return np.array([
        [x_vals[c], y_vals[r]] for r, c in coords
    ])

train_dataset, test_dataset, input_dim = get_train_data_from_file("dataset.pt", batch_size=1)

n = int(((input_dim) ** 0.5))
for x, y in train_dataset:
    input_vec = x[0]  # shape: (n²,)
    label_vec = y[0]  # shape: (5,)
    # Zurück in 2D Punkte aufteilen
    n = int(((input_vec.shape[0]) ** 0.5))
    xy_mesh_vals = input_vec.view(n,n).tolist()
    leed_image = np.array(xy_mesh_vals)

# Gegeben: leed_image = 128x128 numpy array
    coords = detect_peaks(leed_image, min_distance=4, threshold_abs=0.05)
    xy_coords = pixel_to_xy(coords, leed_image.shape)
    print("xy_coords:", xy_coords)
    print("xy_coords.shape:", xy_coords.shape)  # shape: (N_peaks, 2)








fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(leed_image, cmap='viridis', origin='lower')
axs[0].set_title("Original LEED pattern")
axs[1].scatter(xy_coords[:, 0], xy_coords[:, 1], color='red', s=20)
axs[1].set_title("Reconstructed Peaks")
plt.show()


