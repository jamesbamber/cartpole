import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

import numpy as np
from numpy import cos, sin

from constants import *
from physics_rigidpole import *
from integrators import *
from state import SimulationState

state = SimulationState()

fig = plt.figure()
gs = fig.add_gridspec(
    4, 4,
    wspace=0.3,
    hspace=0.4
)

ax = fig.add_subplot(gs[0:3, :], autoscale_on=False, xlim=(-6*l, 6*l), ylim=(-2*l, 2*l))
ax.grid()
ax.set_aspect("equal")

ax2 = fig.add_subplot(gs[3, 0:4])
ax2.grid() 
ax2.set_title("System Energy")

line, = ax.plot([], [], '-', lw=5)
pole_trace, = ax.plot([], [], '.-', lw=1, ms=1)
energy_line, = ax2.plot([], [], '-', lw=2)

cart = plt.Rectangle(
    (-CART_WIDTH/2, -CART_HEIGHT/2),
    CART_WIDTH, 
    CART_HEIGHT, 
    facecolor='gray', edgecolor='gray'
)

ax.add_patch(cart)

pole = plt.Rectangle(
    (-POLE_WIDTH/2, -POLE_WIDTH),
    POLE_WIDTH,
    l + POLE_WIDTH,
    facecolor='black', edgecolor='black'
)

ax.add_patch(pole)

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

    x1 = state.x1[-1]
    x2 = state.x2[-1]
    y1 = state.y1[-1]
    y2 = state.y2[-1]

    th = state.state[-1][1]

    cart.set_xy((x1 - CART_WIDTH/2, y1 - CART_HEIGHT/2))

    pole_trans = (
        Affine2D()
        .rotate_around(x1, y1, -th)
        + ax.transData
    )

    pole.set_transform(pole_trans)
    pole.set_xy((x1 - POLE_WIDTH/2, y1 - POLE_WIDTH/2))

    # line.set_data([x1, x2], [y1, y2])
    pole_trace.set_data(state.x2[-50:], state.y2[-50:])
    energy_line.set_data(state.t[:i], state.E[:i])
    
    ax2.relim()
    ax2.autoscale_view()

    return cart, pole, energy_line, line

if __name__ == "__main__":
    ani = animation.FuncAnimation (
        fig, animate, 1000 * FPS, interval=1000 / FPS, blit=False
    )

    plt.show()
