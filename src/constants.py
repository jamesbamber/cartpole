import numpy as np
import random

#simulation constants

g = 9.81
l = 1
m1 = 2
m2 = 1
Fmod = 10
dt = 0.01

# terminal state limits

x_max = 2.4
th_max = np.radians(12)
max_time = 20

#initial conditions

def random_state():
    x0 = random.uniform(-0.1, 0.1)
    th0 = np.radians(random.uniform(-10, 10))
    v0 = random.uniform(-0.1, 0.1)
    w0 = np.radians(random.uniform(-0.1, 0.1))

    return x0, th0, v0, w0

x0, th0, v0, w0 = random_state()

#plotting constants

EPISODES = 10

FPS = 60

CART_WIDTH = 0.6
CART_HEIGHT = 0.3
POLE_WIDTH = 0.1
