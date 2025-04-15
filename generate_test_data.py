# generate_test_data.py
import numpy as np

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
