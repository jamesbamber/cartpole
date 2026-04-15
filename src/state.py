import numpy as np

from constants import *
from physics_rigidpole import *

class SimulationState:
    def __init__(self, initial_state = None, logAll = True):

        self.logAll = logAll

        if initial_state is None:
            initial_state = random_state()

        self.t = [ 0 ]
        self.state = [ np.array(initial_state) ]

        if self.logAll:
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

        if self.logAll:
            self.E.append(energy(self.state[-1]))

            self.update_cartesian()

    def update_cartesian(self): 
        x, th, v, w = self.state[-1]

        self.x1.append(x)
        self.y1.append(0)
        self.x2.append(x + l*np.sin(th))
        self.y2.append(l*np.cos(th))

    def is_terminal(self):
        x, th, v, w = self.state[-1]
        return abs(x) > x_max or abs(th) > th_max or self.current_time() > max_time
    
    def current_time(self):
        return self.t[-1]
