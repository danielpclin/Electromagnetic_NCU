# Requires matplotlib and scipy, run using python 3

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

r0 = 1/np.sqrt(4*np.pi*scipy.constants.epsilon_0)


def electric_field(Q, ox, oy, x, y):
    x_vector = x - ox
    y_vector = y - oy
    distance = np.hypot(x_vector, y_vector)
    det = 4*np.pi*scipy.constants.epsilon_0*(distance**3)
    u, v = (Q*x_vector/det, Q*y_vector/det) if det != 0 else (0, 0)
    return u, v


normalize_x = np.linspace(-4, 4, 17)
normalize_y = np.linspace(-4, 4, 17)
NX, NY = np.meshgrid(normalize_x, normalize_y)
x = np.linspace(-4*r0, 4*r0, 17)
y = np.linspace(-4*r0, 4*r0, 17)
X, Y = np.meshgrid(x, y)

vectorize_electric_field = np.vectorize(electric_field, excluded=['q', 'rx', 'ry'])
U, V = vectorize_electric_field(1, 0, 0, X, Y)
plt.axis('equal')
q = plt.quiver(NX, NY, U, V, units='xy', scale=1,)
plt.xlabel(r"$\frac{x}{R_0}$")
plt.ylabel(r"$\frac{y}{R_0}$")
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.plot(0, 0, color='red', marker='o', fillstyle='none')
plt.grid()
plt.show()
