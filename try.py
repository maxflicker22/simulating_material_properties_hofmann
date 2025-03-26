
import numpy as np
import matplotlib.pyplot as plt  # Added import for Matplotlib


# %%

# Example plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label="sin(x)")
plt.title("Example Plot")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()

# %%
