import numpy as np

from constants import *
from physics import *

class SimulationState:
    def __init__(self):
        self.t = [ 0 ]
        self.state = [ np.array([ x0, np.radians(th0), v0, np.radians(w0) ])]
        self.E = [ energy(self.state[-1]) ]

        self.x1 = []
        self.y1 = []
        self.x2 = []
        self.y2 = []

        self.update_cartesian()

    def step(self, integrator, action):
        '''
        performs one iteration of a given numeric integrator
        '''
        t, new_state = integrator(self.t[-1], self.state[-1], f, dt, action)

        self.t.append(t)
        self.state.append(new_state)
        self.E.append(energy(self.state[-1]))

        self.update_cartesian()

    def update_cartesian(self): 
        x, th, v, w = self.state[-1]

        self.x1.append(x)
        self.y1.append(0)
        self.x2.append(x + l*np.sin(th))
        self.y2.append(l*np.cos(th))
