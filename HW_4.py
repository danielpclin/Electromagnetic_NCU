# Requires matplotlib and scipy, run using python 3

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors

plt.cla()
plt.clf()
plt.close()

I0 = const.pi/(2*2**(1/2)*const.mu_0)
W = 1
MU0I0_4PI = const.mu_0 * I0/(4*const.pi)


def b1z(x, y, z):
    return MU0I0_4PI * (-(x - W / 2) / ((x - W / 2) ** 2 + z ** 2)) * (
                ((W / 2 - y) / ((W / 2 - y) ** 2 + (x - W / 2) ** 2 + z ** 2) ** 0.5) - (
                    (-W / 2 - y) / ((-W / 2 - y) ** 2 + (x - W / 2) ** 2 + z ** 2) ** 0.5))


def b1x(x, y, z):
    return MU0I0_4PI * (z / ((x - W / 2) ** 2 + z ** 2)) * (
                ((W / 2 - y) / ((W / 2 - y) ** 2 + (x - W / 2) ** 2 + z ** 2) ** 0.5) - (
                    (-W / 2 - y) / ((-W / 2 - y) ** 2 + (x - W / 2) ** 2 + z ** 2) ** 0.5))


def b3z(x, y, z):
    return MU0I0_4PI * (-(x + W / 2) / ((x + W / 2) ** 2 + z ** 2)) * (
                ((-W / 2 - y) / ((-W / 2 - y) ** 2 + (x + W / 2) ** 2 + z ** 2) ** 0.5) - (
                    (W / 2 - y) / ((W / 2 - y) ** 2 + (x + W / 2) ** 2 + z ** 2) ** 0.5))


def b3x(x, y, z):
    return MU0I0_4PI * (z / ((x + W / 2) ** 2 + z ** 2)) * (
                ((-W / 2 - y) / ((-W / 2 - y) ** 2 + (x + W / 2) ** 2 + z ** 2) ** 0.5) - (
                    (W / 2 - y) / ((W / 2 - y) ** 2 + (x + W / 2) ** 2 + z ** 2) ** 0.5))


def b2z(x, y, z):
    return MU0I0_4PI * ((y - W / 2) / ((y - W / 2) ** 2 + z ** 2)) * (
                ((-W / 2 - x) / ((-W / 2 - x) ** 2 + (y - W / 2) ** 2 + z ** 2) ** 0.5) - (
                    (W / 2 - x) / ((W / 2 - x) ** 2 + (y - W / 2) ** 2 + z ** 2) ** 0.5))


def b2y(x, y, z):
    return MU0I0_4PI * (-z / ((y - W / 2) ** 2 + z ** 2)) * (
                ((-W / 2 - x) / ((-W / 2 - x) ** 2 + (y - W / 2) ** 2 + z ** 2) ** 0.5) - (
                    (W / 2 - x) / ((W / 2 - x) ** 2 + (y - W / 2) ** 2 + z ** 2) ** 0.5))


def b4z(x, y, z):
    return MU0I0_4PI * ((y + W / 2) / ((y + W / 2) ** 2 + z ** 2)) * (
                ((W / 2 - x) / ((W / 2 - x) ** 2 + (y + W / 2) ** 2 + z ** 2) ** 0.5) - (
                    (-W / 2 - x) / ((-W / 2 - x) ** 2 + (y + W / 2) ** 2 + z ** 2) ** 0.5))


def b4y(x, y, z):
    return MU0I0_4PI * (-z / ((y + W / 2) ** 2 + z ** 2)) * (
                ((W / 2 - x) / ((W / 2 - x) ** 2 + (y + W / 2) ** 2 + z ** 2) ** 0.5) - (
                    (-W / 2 - x) / ((-W / 2 - x) ** 2 + (y + W / 2) ** 2 + z ** 2) ** 0.5))


def b(x, y, z):
    return b1x(x, y, z) + b3x(x, y, z), b2y(x, y, z) + b4y(x, y, z), b1z(x, y, z) + b2z(x, y, z) + b3z(x, y, z) + b4z(x, y, z)


x1 = np.linspace(-2, 2, 21)
z1 = np.linspace(-2, 2, 21)
vectorized_b_field1 = np.vectorize(b, excluded=['y'])
X1, Z1 = np.meshgrid(x1, z1)
I1, J1, K1 = vectorized_b_field1(X1, 0, Z1)
plt.axis('equal')
q = plt.quiver(X1, Z1, I1, K1, units='xy', scale=1)
plt.xlabel(r"$\frac{x}{W}$")
plt.ylabel(r"$\frac{z}{W}$")
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.plot(0.5, 0, color='blue', marker='o', fillstyle='none')
plt.plot(-0.5, 0, color='red', marker='o', fillstyle='none')
plt.grid()

x2 = np.linspace(-2, 2, 101)
y2 = np.linspace(-2, 2, 101)
vectorized_b_field2 = np.vectorize(b, excluded=['z'])
X2, Y2 = np.meshgrid(x2, y2)
I2, J2, K2 = vectorized_b_field2(X2, Y2, 0)
B = np.log2((I2**2 + J2**2 + K2**2)**(1/2))
plt.figure()
plt.axis('equal')
level = np.linspace(-7, 5, 121)
c = plt.contour(X2, Y2, B, level, cmap='rainbow', linewidths=0.5)
norm = colors.Normalize(vmin=c.cvalues.min(), vmax=c.cvalues.max())
# To get continuous colorbar
scalarMappable = plt.cm.ScalarMappable(norm=norm, cmap=c.cmap)
scalarMappable.set_array(np.array([]))
plt.colorbar(scalarMappable, ticks=np.linspace(-7, 5, 13))
plt.xlabel(r"$\frac{x}{W}$")
plt.ylabel(r"$\frac{z}{W}$")
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid()

plt.show()
