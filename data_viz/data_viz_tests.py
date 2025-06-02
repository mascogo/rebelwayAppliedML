import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0,5, 11)
y = x ** 2

print (x)
fig, axes = plt.subplots(1,2, figsize=(10, 4))

axes[0].plot(x, x**2, x**3, lw=2)
axes[0].grid(True)

axes[1].plot(x, x**2, x**3, lw=2)
axes[1].grid(color='r', alpha=0.5, linestyle='dashed', linewidth=0.5)

plt.show()