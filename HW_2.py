# Requires matplotlib and scipy, run using python 3

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

r0 = 1/np.sqrt(4*np.pi*scipy.constants.epsilon_0)
d = r0
pd = 1


def electric_field(Q, ox, oz, x, z):
    x_vector = x - ox
    z_vector = z - oz
    distance = np.hypot(x_vector, z_vector)
    det = 4*np.pi*scipy.constants.epsilon_0*(distance**3)
    u, v = (Q*x_vector/det, Q*z_vector/det) if det != 0 else (0, 0)
    return u, v


def approx_dipole_electric_field(Q, ox, oz, x, z):
    x_vector = x - ox
    z_vector = z - oz
    distance = np.hypot(x_vector, z_vector)
    det = 4*np.pi*scipy.constants.epsilon_0*(distance**3)
    u, v = (Q*d*(3*z_vector*x_vector/distance**2)/det, Q*d*(3*z_vector*z_vector/distance**2-1)/det) if det != 0 else (0, 0)
    return u, v


def potential_field(Q, ox, oz, x, z):
    x_vector = x - ox
    z_vector = z - oz
    distance = np.hypot(x_vector, z_vector)
    det = 4*np.pi*scipy.constants.epsilon_0*distance
    return Q/det if det != 0 else np.sign(Q)*np.Inf


def approx_dipole_potential_field(Q, ox, oz, x, z):
    x_vector = x - ox
    z_vector = z - oz
    distance = np.hypot(x_vector, z_vector)
    det = 4*np.pi*scipy.constants.epsilon_0*(distance**3)
    return Q*pd*z_vector/det if det != 0 else 0


normalize_x = np.linspace(-4, 4, 17)
normalize_y = np.linspace(-4, 4, 17)
NX, NY = np.meshgrid(normalize_x, normalize_y)
x = np.linspace(-4*d, 4*d, 17)
y = np.linspace(-4*d, 4*d, 17)
X, Y = np.meshgrid(x, y)
vectorize_electric_field = np.vectorize(electric_field, excluded=['q', 'rx', 'ry'])
vectorize_approx_dipole_electric_field = np.vectorize(approx_dipole_electric_field, excluded=['q', 'rx', 'ry'])
vectorize_potential_field = np.vectorize(potential_field, excluded=['q', 'rx', 'ry'])
vectorize_approx_dipole_potential_field = np.vectorize(approx_dipole_potential_field, excluded=['q', 'rx', 'ry'])
U, V = np.add(vectorize_electric_field(1, 0, 0.5*d, X, Y), vectorize_electric_field(-1, 0, -0.5*d, X, Y))
U_approx, V_approx = vectorize_approx_dipole_electric_field(1, 0, 0, X, Y)

px = np.linspace(-4*pd, 4*pd, 101)
py = np.linspace(-4*pd, 4*pd, 101)
pX, pY = np.meshgrid(px, py)
normalize_px = np.linspace(-4, 4, 101)
normalize_py = np.linspace(-4, 4, 101)
pNX, pNY = np.meshgrid(normalize_px, normalize_py)
potential = np.add(vectorize_potential_field(4*np.pi*scipy.constants.epsilon_0, 0, 0.5*pd, pX, pY), vectorize_potential_field(-4*np.pi*scipy.constants.epsilon_0, 0, -0.5*pd, pX, pY))
potential_approx = vectorize_approx_dipole_potential_field(4*np.pi*scipy.constants.epsilon_0, 0, 0, pX, pY)
plt.figure()
plt.axis('equal')
q = plt.quiver(NX, NY, U, V, units='xy', scale=1, width=0.04, minlength=0.5,)
plt.xlabel(r"$\frac{x}{d}$")
plt.ylabel(r"$\frac{z}{d}$")
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.plot(0, 0.5, color='red', marker='o', fillstyle='none')
plt.plot(0, -0.5, color='blue', marker='o', fillstyle='none')
plt.grid()

plt.figure()
plt.axis('equal')
q1 = plt.quiver(NX, NY, U, V, units='xy', scale=1, width=0.04, minlength=0.5,)
q2 = plt.quiver(NX, NY, U_approx, V_approx, units='xy', scale=1, color='g', width=0.04, minlength=0.5,)
plt.xlabel(r"$\frac{x}{d}$")
plt.ylabel(r"$\frac{z}{d}$")
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.plot(0, 0.5, color='red', marker='o', fillstyle='none')
plt.plot(0, -0.5, color='blue', marker='o', fillstyle='none')
plt.grid()

plt.figure()
plt.axis('equal')
level = np.concatenate((np.linspace(-4, -1, 4), np.linspace(-0.5, 0.5, 21), np.linspace(1, 4, 4)))
c2 = plt.contour(pNX, pNY, potential_approx, level, colors='green', linestyles='dashed', linewidths=0.5)
c1 = plt.contour(pNX, pNY, potential, level, colors='black', linestyles='solid', linewidths=0.5)
plt.xlabel(r"$\frac{x}{d}$")
plt.ylabel(r"$\frac{z}{d}$")
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.plot(0, 0.5, color='red', marker='o', fillstyle='none')
plt.plot(0, -0.5, color='blue', marker='o', fillstyle='none')
plt.grid()

plt.show()

