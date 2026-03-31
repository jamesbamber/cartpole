import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from numpy import cos, sin

from constants import *
from physics_rigidpole import *
from integrators import *
from state import SimulationState

state = SimulationState()

fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(3,1,(1,2), autoscale_on=False, xlim=(-8*l, 8*l), ylim=(-1.5*l, 1.5*l))
ax.grid()
ax.set_aspect("equal")

ax2 = fig.add_subplot(3,1,3, autoscale_on=False, xlim=(0, 100), ylim=(-m2*g*l, (m1+m2)*v0**2+m2*g*l))
ax2.grid() 
ax2.set_title("System Energy")

line, = ax.plot([], [], 'o-', lw=2)
trace1, = ax.plot([], [], '.-', lw=1, ms=1)
trace2, = ax.plot([], [], '.-', lw=1, ms=1)
energy_line, = ax2.plot([], [], '-', lw=2)

action = 0
pressed_keys = set()

def on_key(event):
    global action, pressed_keys

    if event.name == 'key_press_event':
        pressed_keys.add(event.key)
    elif event.name == 'key_release_event':
        pressed_keys.discard(event.key)

    if "right" in pressed_keys:
        action = 1
    elif "left" in pressed_keys:
        action = 2
    else:
        action = 0

fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('key_release_event', on_key)

def animate(frame):

    curr_t = frame / FPS
    i = int(curr_t / dt)

    while len(state.t) <= i:
        state.step(rk4, action)

    line.set_data([state.x1[-1], state.x2[-1]], [state.y1[-1], state.y2[-1]])
    trace1.set_data(state.x1[-50:], state.y1[-50:])
    trace2.set_data(state.x2[-50:], state.y2[-50:])
    energy_line.set_data(state.t[:i], state.E[:i])

    return line, trace1, trace2, energy_line

ani = animation.FuncAnimation (
    fig, animate, 1000 * FPS, interval=1000 / FPS, blit=True
)

plt.show()
