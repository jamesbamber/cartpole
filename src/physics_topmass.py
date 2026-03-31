import numpy as np
from numpy import cos, sin

from constants import *

def energy(state) :
    x, th, v, w = state
    return 0.5*m1*v**2 + 0.5*m2*v**2 + m2*l*cos(th)*v*w + 0.5*m2*(l**2)*(w**2) + m2*g*l*cos(th)

def f(t, y, a):
    x, th, v, w = y
    if a==0:
        F = 0
    elif a==1:
        F = Fmod
    elif a==2:
        F = -Fmod
    acc1 = (F + m2*l*sin(th)*w**2 - m2*g*cos(th)*sin(th))/ (m1+m2-m2*cos(th)**2)
    acc2 = (g*sin(th) - acc1*cos(th)) / l
    return np.array([v, w, acc1, acc2])
