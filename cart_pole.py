
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from numpy import cos, sin


# define constants

g = 9.81

l = 1
m1 = 2
m2 = 1

x0 = 0
th0 = 90
v0 = 0
w0 = 0

Fmod = 1

initial_state = np.array([x0, np.radians(th0), v0, np.radians(w0)])


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

dt = 0.01
max_t = 20
t = np.arange(0, max_t, dt)

# should implement a better numeric method at some point
def euler(x, y, f, dx, a):
    return y + dx*f(x, y, a)

def rk4(x, y, f, dx, a): 
    k1 = f(x, y, a)
    k2 = f(x + dx/2, y + dx/2 * k1, a)
    k3 = f(x + dx/2, y + dx/2 * k2, a)
    k4 = f(x + dx, y + dx * k3, a)
    return y + dx/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def energy(state) :
    x, th, v, w = state
    return 0.5*m1*v**2 + 0.5*m2*v**2 + m2*l*cos(th)*v*w + 0.5*m2*(l**2)*(w**2) + m2*g*l*cos(th)

states = np.empty((len(t), 4))
states[0] = initial_state

# now the plotting part

E = np.empty(len(t))

x1_hist = []
y1_hist = []
x2_hist = []
y2_hist = []

fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(2,1,1, autoscale_on=False, xlim=(-2*l, 2*l), ylim=(-l, l))
ax.set_aspect('equal')
ax.grid()
e0 = np.abs(E[0])
ax2 = fig.add_subplot(2,1,2, autoscale_on=False, xlim=(0, max_t), ylim=(-m2*g*l, (m1+m2)*v0**2+m2*g*l))
ax2.grid() 

line, = ax.plot([], [], 'o-', lw=2)
trace1, = ax.plot([], [], '.-', lw=1, ms=1)
trace2, = ax.plot([], [], '.-', lw=1, ms=1)
energy_line, = ax2.plot([], [], '-', lw=2)

action = 0

def on_key(event):
    global action
    if event.key == "right":
        action = 1
    elif event.key == "left":
        action = 2
    else:
        action = 0

fig.canvas.mpl_connect('key_press_event', on_key)

def animate(i):

    states[i+1] = rk4(t[i], states[i], f, dt, action)
    E[i] = energy(states[i])

    x = states[i,0]
    th = states[i,1]

    x1 = x
    y1 = 0
    x2 = x + l*np.sin(th)
    y2 = l*np.cos(th)

    x1_hist.append(x1)
    y1_hist.append(y1)
    x2_hist.append(x2)
    y2_hist.append(y2)

    cm_x = [(m1*x1 + m2*x2) / (m1 + m2)]
    cm_y = [(m1*y1 + m2*y2) / (m1 + m2)]

    line.set_data([x1, x2], [y1, y2])
    trace1.set_data(x1_hist[-500:], y1_hist[-500:])
    trace2.set_data(x2_hist[-500:], y2_hist[-500:])
    energy_line.set_data(t[:i], E[:i])

    return line, trace1, trace2, energy_line

ani = animation.FuncAnimation (
    fig, animate, len(t), interval=dt*1000, blit=True
)

plt.show()
