# Requires matplotlib, run using python 3

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

r0 = 1


def current_density_of_washer(x, y):
    r = np.hypot(x, y)
    return r0/r


normalize_r = np.linspace(1, 2.5, 100)
theta = np.linspace(0, 0.5*np.pi, 100)
r = np.linspace(1, 2.5, 100)
NR, NT = np.meshgrid(normalize_r, theta)
R, T = np.meshgrid(r, theta)
NX = NR * np.cos(NT)
NY = NR * np.sin(NT)
X = R * np.cos(T)
Y = R * np.sin(T)
vectorize_current_density_of_washer = np.vectorize(current_density_of_washer)
Z = vectorize_current_density_of_washer(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.xlabel(r"$\frac{x}{r0}$")
plt.ylabel(r"$\frac{y}{r0}$")
ax.view_init(90, -90)
plt.title("Current Density with surface")
surface = ax.plot_surface(NX, NY, Z, cmap='rainbow')
plt.colorbar(surface)

plt.figure()
ax2 = plt.gca()
clev = np.arange(Z.min(), Z.max(), .001)
c = ax2.contourf(NX, NY, Z, clev, cmap='rainbow')
plt.colorbar(c)
plt.axis('equal')
plt.title("Current Density with contour")
plt.grid()
plt.xlabel(r"$\frac{x}{r0}$")
plt.ylabel(r"$\frac{y}{r0}$")

plt.show()
