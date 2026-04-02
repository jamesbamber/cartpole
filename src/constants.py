import numpy as np
import random

#simulation constants

g = 9.81
l = 1
m1 = 2
m2 = 1
Fmod = 10
dt = 0.01

#initial conditions

x0 = random.uniform(-0.5, 0.5)
th0 = np.radians(random.uniform(-10, 10))
v0 = random.uniform(-1, 1)
w0 = np.radians(random.uniform(-1, 1))

#plotting constants

FPS = 120

CART_WIDTH = 0.6
CART_HEIGHT = 0.3
POLE_WIDTH = 0.1
