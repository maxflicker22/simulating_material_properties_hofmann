from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max
from data_generator import get_train_data_from_file
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from config import (
    BATCH_SIZE, NUM_ITERATIONS, LEARNING_RATE,
    HIDDEN_DIM, NUM_LAYERS, VOCAB_SIZE, IMAGE_DIM, K_MAX_PEAKS, DEVICE
)
### Mal nur kopiert!! hier anpssen und einbinden

def get_fixed_number_of_peaks(image, k=K_MAX_PEAKS, min_distance=4, threshold_abs=0.5):
    from skimage.feature import peak_local_max

    coords = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold_abs)

    # Distance to image center (in pixel space)
    distance_from_origin = np.sqrt(
        (coords[:, 0] - (IMAGE_DIM / 2))**2 + (coords[:, 1] - (IMAGE_DIM / 2))**2
    )

    # Sort by distance to center (ascending)
    idx_sorted = np.argsort(distance_from_origin)
    coords = coords[idx_sorted]

    # Limit to top-k if needed
    if k is not None and len(coords) > k:
        coords = coords[:k]

    # Convert to normalized (x, y) in [-1, 1]
    peaks_xy = pixel_to_xy(coords, image.shape)

    # Pad with fake values far outside the image domain (e.g., x=0, y=0)
    if len(peaks_xy) < k:
        num_missing = k - len(peaks_xy)
        far_value = 0.  # well outside of [-1, 1]
        pad = np.full((num_missing, 2), far_value)
        peaks_xy = np.vstack([peaks_xy, pad])
    else:
        peaks_xy = peaks_xy[:k]

    return peaks_xy  # shape: (k, 2)



def detect_peaks(image, min_distance=4, threshold_abs=0.05, k_max=None):
    coords = peak_local_max(
        image,
        min_distance=min_distance,
        threshold_abs=threshold_abs
    )
    if k_max is not None and len(coords) > k_max:
        intensities = image[coords[:, 0], coords[:, 1]]
        top_idx = np.argsort(-intensities)[:k_max]
        coords = coords[top_idx]
    return coords

def pixel_to_xy(coords, image_shape, x_range=(-1, 1), y_range=(-1, 1), sort_by_distance=True):
    rows, cols = image_shape
    x_vals = np.linspace(*x_range, cols)
    y_vals = np.linspace(*y_range, rows)

    # Convert to real (x, y) values
    xy_coords = np.array([
        [x_vals[c], y_vals[r]] for r, c in coords
    ])

    if sort_by_distance:
        # Compute distance to origin (0, 0) in normalized space
        distances = np.linalg.norm(xy_coords, axis=1)
        idx_sorted = np.argsort(distances)
        xy_coords = xy_coords[idx_sorted]

    return xy_coords  # shape: (N, 2)


def generate_leed_pattern(peaks, amplitudes=None, grid_size=IMAGE_DIM, sigma=0.05, x_range=(-1, 1), y_range=(-1, 1)):
    x = np.linspace(*x_range, grid_size)
    y = np.linspace(*y_range, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    if amplitudes is None:
        amplitudes = np.ones(len(peaks))

    for (x0, y0), A in zip(peaks, amplitudes):
        Z += A * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    return Z

