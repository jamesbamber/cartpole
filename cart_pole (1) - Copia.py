
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from numpy import cos, sin


# define contsants

g = 9.81

l = 1
m1 = 2
m2 = 1

x0 = 0
th0 = -90
v0 = 1
w0 = 0

initial_state = np.array([x0, np.radians(th0), v0, np.radians(w0)])


def f(t, y):
    x, th, v, w = y
    acc1 = (m2*l*sin(th)*w**2 - m2*g*cos(th)*sin(th))/ (m1+m2-m2*cos(th)**2)
    acc2 = (g*sin(th) - acc1*cos(th)) / l
    return np.array([v, w, acc1, acc2])

dt = 0.01
max_t = 20
t = np.arange(0, max_t, dt)

# should implement a better numeric method at some point
def euler(x, y, f, dx):
    return y + dx*f(x, y)

def rk4(x, y, f, dx): 
    k1 = f(x, y)
    k2 = f(x + dx/2, y + dx/2 * k1)
    k3 = f(x + dx/2, y + dx/2 * k2)
    k4 = f(x + dx, y + dx * k3)
    return y + dx/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def energy(state) :
    x, th, v, w = state
    return 0.5*m1*v**2 + 0.5*m2*v**2 + m2*l*cos(th)*v*w + 0.5*m2*(l**2)*(w**2) + m2*g*l*cos(th)

states = np.empty((len(t), 4))
states[0] = initial_state

E = np.empty(len(t))

for i in range(0, len(t)-1):
    states[i+1] = rk4(t[i], states[i], f, dt)
    E[i] = energy(states[i])


# now the plotting part

x1 = states[:, 0]
y1 = np.zeros(len(t))


x2 = x1 + l*np.sin(states[:, 1])
y2 = l*np.cos(states[:, 1])


fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(2,1,1, autoscale_on=False, xlim=(-2*l, 2*l), ylim=(-l, l))
ax.set_aspect('equal')
ax.grid()
e0 = np.abs(E[0])
ax2 = fig.add_subplot(2,1,2, autoscale_on=False, xlim=(0, max_t), ylim=(-m2*g*l, (m1+m2)*v0**2+m2*g*l))
ax2.grid() 

line, = ax.plot([], [], 'o-', lw=2)
trace1, = ax.plot([], [], '.-', lw=1, ms=2)
trace2, = ax.plot([], [], '.-', lw=1, ms=2)
energy_line, = ax2.plot([], [], '-', lw=2)

def animate(i):

    curr_x = [x1[i], x2[i]]
    curr_y = [y1[i], y2[i]]

    history_x1 = x1[:i]
    history_y1 = y1[:i]
    history_x2 = x2[:i]
    history_y2 = y2[:i]

    cm_x = [(m1*x1[i] + m2*x2[i]) / (m1 + m2)]
    cm_y = [(m1*y1[i] + m2*y2[i]) / (m1 + m2)]

    line.set_data(curr_x, curr_y)
    trace1.set_data(history_x1[-500:], history_y1[-500:])
    trace2.set_data(history_x2[-500:], history_y2[-500:])
    energy_line.set_data(t[:i], E[:i])

    return line, trace1, trace2, energy_line

ani = animation.FuncAnimation (
    fig, animate, len(t), interval=dt*1000, blit=True
)

plt.show()
