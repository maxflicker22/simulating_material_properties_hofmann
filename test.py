import numpy as np
import matplotlib.pyplot as plt

# Einstellungen
point_positions = np.array([-1.0, 1.0])  # Positionen der "Atome"
amplitudes = np.ones_like(point_positions)  # gleiche Gewichtung
G_values = [-3*np.pi, -np.pi, 0, np.pi, 3*np.pi]  # Beispielhafte G-Werte

# Plot Setup
figs = []
for G in G_values:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.linspace(-5, 5, 1000)
    wave = np.exp(1j * G * x)

    # Reelle Teil der Welle
    ax1.plot(x, np.real(wave), label="Re[$e^{iGx}$]")
    ax1.plot(point_positions, np.real(amplitudes * np.exp(1j * G * point_positions)), 'ro', label="Re Beiträge")
    ax1.plot(point_positions, np.imag(amplitudes * np.exp(1j * G * point_positions)), 'go', label="Im Beiträge")
    ax1.set_title(f"Welle und Punkte (G = {G:.2f})")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend()

    # Vektor-Summen-Plot
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.set_title("Vektorsumme der Beiträge")

    origin = np.array([0, 0])
    sum_vector = np.array([0.0, 0.0])
    for c in amplitudes * np.exp(1j * G * point_positions):
        vec = np.array([np.real(c), np.imag(c)])
        ax2.arrow(*origin, *vec, head_width=0.05, color='gray', alpha=0.6)
        sum_vector += vec
        origin = origin + vec

    # Gesamtsumme
    ax2.arrow(0, 0, *sum_vector, head_width=0.1, color='blue', label='Summe')
    ax2.plot(sum_vector[0], sum_vector[1], 'bo')
    ax2.legend()

    figs.append(fig)

plt.show()
#plt.close('all')
figs
