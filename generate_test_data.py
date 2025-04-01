# %%
import numpy as np
import matplotlib.pyplot as plt

# ChatGPT frage: how to plot a 2d bravais lattice with python

# ChatGPT frage: my output are 2 Vectors which have each a x and y component. i wanna predict the x and y components of those two vectors. so summarized i have 4 outputs. 
# The input are N points where each point has x and y components so 2 times N Inputs


# Define lattice vectors (example: oblique lattice)
a1 = np.array([[1, 0], [1,0], [1,0]])
a2 = np.array([[0, 1], [0, 2], [0.5, np.sqrt(3)/2]])  # 60° angle

# Grid size (how many unit cells in each direction)
#N = 10


# Generate lattice Points of specific lattic Vectors and Integer Range
def generate_lattice_points(a1, a2, N):
    lattice_points = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            point = i * a1 + j * a2
            lattice_points.append(point)
    return np.array(lattice_points)


# Function for converting into reciprocal lattice
def reciprocal_lattice_2d(a1, a2):
    # Convert to 3D (add z = 0) for cross product handling
    a1_3d = np.array([a1[0], a1[1], 0])
    a2_3d = np.array([a2[0], a2[1], 0])
    
    # Compute area of unit cell (scalar triple product)
    area = np.cross(a1_3d, a2_3d)[2]
    
    # Reciprocal lattice vectors (back to 2D)
    b1 = 2 * np.pi * np.cross([0, 0, 1], a2_3d)[:2] / area
    b2 = 2 * np.pi * np.cross(a1_3d, [0, 0, 1])[:2] / area
    
    return b1, b2

# b1, b2 = np.zeros_like(a1), np.zeros_like(a2)

# for i in range(len(a1)):
#    b1[i], b2[i] = reciprocal_lattice_2d(a1[i], a2[i])

# Function to generate reciprocal lattice points and return them
def return_reciprocal_lattice_points(a1, a2, N):
    b1, b2 = reciprocal_lattice_2d(a1, a2)
    lattice_points = generate_lattice_points(b1, b2, N)
    return lattice_points.T.astype(np.float32)  # Now shape: (2, N*N)


# Generate Test Plot of the Lattice
# Plot
"""
plt.figure(figsize=(6, 6))
for i in range(len(a1)):
    lattice_points = generate_lattice_points(a1[i, :], a2[i, :], N)
    #plt.scatter(lattice_points[:,0], lattice_points[:,1], s=30)
    #print(lattice_points)
plt.axis('equal')
plt.title("2D Bravais Lattice")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
# %%
plt.figure(figsize=(6, 6))
for i in range(len(b1)):
    lattice_points = generate_lattice_points(b1[2, :], b2[2, :], N)
    plt.scatter(lattice_points[:,0], lattice_points[:,1], s=30)
    print(lattice_points.shape)
    
plt.axis('equal')
plt.title("2D Bravais Lattice Reciprocal")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
"""
# %%
