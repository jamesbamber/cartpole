import numpy as np
from numpy import cos, sin

from constants import *

def energy(state) :
    x, th, v, w = state
    return 0.5*m1*v**2 + 0.5*m2*v**2 + 0.5*m2*l*cos(th)*v*w + 7/24*m2*(l**2)*(w**2) + 0.5*m2*g*l*cos(th)

def f(t, y, a):
    x, th, v, w = y
    if a==0:
        F = -Fmod
    elif a==1:
        F = +Fmod
    elif a==2:
        F = 0
    acc1 = (7*F + 3.5*m2*l*sin(th)*w**2 - 3*m2*g*cos(th)*sin(th)) / (7*m1 + 7*m2 - 3*m2*(cos(th)**2))
    acc2 = 6/7 * (g*sin(th) - cos(th)*acc1) / l
    return np.array([v, w, acc1, acc2])
