import numpy as np

#simulation constants

g = 9.81
l = 1
m1 = 2
m2 = 1
Fmod = 10
dt = 0.01

#initial conditions

x0 = 0
th0 = np.radians(0)
v0 = 0
w0 = np.radians(0)

#plotting constants

FPS = 120

CART_WIDTH = 0.6
CART_HEIGHT = 0.3
POLE_WIDTH = 0.1
