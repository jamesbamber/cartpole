import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import FancyArrow

import numpy as np
from numpy import cos, sin

from constants import *
from physics_rigidpole import *
from integrators import *
from state import SimulationState
import handle_action

state = SimulationState()

fig = plt.figure()
handle_action.init(fig)

gs = fig.add_gridspec(
    8, 4,
    wspace=0.4,
    hspace=0.6
)

ax = fig.add_subplot(gs[0:6, :], autoscale_on=False, xlim=(-4*l, 4*l), ylim=(-1.5*l, 1.5*l))
ax.grid()
ax.set_aspect("equal")

ax2 = fig.add_subplot(gs[6:8, 2:4])
ax2.grid() 
ax2.set_title("System Energy")

text_ax = fig.add_subplot(gs[6, 0:2], xlim=(-1, 1), ylim=(-1, 1))
text_ax.axis('off')

buttons = fig.add_subplot(gs[7, 0:2], xlim=(-2, 2), ylim=(-1, 1))
buttons.axis('off')

bbox_props = dict(
    boxstyle="round,pad=0.6", 
    facecolor="#f8f9fa",      # Light gray background
    edgecolor="#ced4da",      # Slightly darker gray border
    linewidth=1.5
)

data = text_ax.text(
    0.0, 0.0,                 # Set x=1.25 (middle of your -1 to 3.5 axis)
    "", 
    fontsize=11, 
    fontfamily='monospace',    # Monospace prevents text jitter when numbers change
    fontweight='bold', 
    color='#343a40',           # Soft dark gray text
    ha='center',               # Center horizontally
    va='center',               # Center vertically
    bbox=bbox_props
)


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

WALL_WIDTH = 0.001

left_wall = plt.Rectangle(
    (-x_max-WALL_WIDTH, -0.1),
    WALL_WIDTH, 
    0.2, 
    facecolor='red', edgecolor='red'
)

right_wall = plt.Rectangle(
    (x_max, -0.1),
    WALL_WIDTH, 
    0.2, 
    facecolor='red', edgecolor='red'
)

left_angle_limit = plt.Rectangle(
    (-WALL_WIDTH/2, 0),
    WALL_WIDTH,
    1,
    facecolor='red', edgecolor='red'
)

right_angle_limit = plt.Rectangle(
    (-WALL_WIDTH/2, 0),
    WALL_WIDTH,
    1,
    facecolor='red', edgecolor='red'
)

ax.add_patch(left_wall)
ax.add_patch(right_wall)
ax.add_patch(left_angle_limit)
ax.add_patch(right_angle_limit)

push_left = FancyArrow(
    -0.2, 0.0,      # p(x, y) coordinates
    -1, 0,       # (dx, dy) direction
    width=0.8,
    length_includes_head=True,
    head_width=1.2,
    head_length=0.4,
    facecolor='white',
    edgecolor='black'
)

push_right = FancyArrow(
    0.2, 0.0,
    1, 0,
    width=0.8,
    length_includes_head=True,
    head_width=1.2,
    head_length=0.4,
    facecolor='white',
    edgecolor='black'
)

buttons.add_patch(push_left)
buttons.add_patch(push_right)

episode_scores = np.zeros(EPISODES)

def animate(frame, episode):

    curr_t = frame / FPS
    i = int(curr_t / dt)

    while len(state.t) <= i:
        state.step(rk4, handle_action.get_action(state.state[-1]))

    action = handle_action.get_action(state.state[-1])

    push_left.set_facecolor('white')
    push_right.set_facecolor('white')

    if action == 0:
        push_left.set_facecolor('red')
    if action == 1:
        push_right.set_facecolor('red')

    # Inside def animate(frame, episode):
    current_score = state.current_time()
    best_score = np.max(episode_scores)

    pretty_text = (
        f"   EPISODE: {episode}/{EPISODES}   \n"
        f"   Current: {current_score:>6.2f}s   \n"
        f"   Best:    {best_score:>6.2f}s   "
    )
    
    data.set_text(pretty_text)

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

    left_angle_trans = (
        Affine2D()
        .rotate_around(x1, y1, +th_max)
        + ax.transData
    )

    left_angle_limit.set_transform(left_angle_trans)
    left_angle_limit.set_xy((x1 - WALL_WIDTH/2, 0))

    right_angle_trans = (
        Affine2D()
        .rotate_around(x1, y1, -th_max)
        + ax.transData
    )

    right_angle_limit.set_transform(right_angle_trans)
    right_angle_limit.set_xy((x1 - WALL_WIDTH/2, 0))

    # line.set_data([x1, x2], [y1, y2])
    # pole_trace.set_data(state.x2[-50:], state.y2[-50:])
    energy_line.set_data(state.t[:i], state.E[:i])
    
    ax2.relim()
    ax2.autoscale_view()


def run_episode(episode_num):
    global state
    
    state = SimulationState()

    frame = 0
    while not state.is_terminal():
        animate(frame, episode_num)
        plt.pause(1.0 / FPS)

        frame += 1
        episode_scores[episode_num-1] = state.current_time()

    print(f"episode {episode_num} completed, score: {episode_scores[episode_num-1]:.2f}")

if __name__ == "__main__":

    plt.ion()
    plt.show()

    for e in range(1, EPISODES+1):
        run_episode(e)

    plt.ioff()
    plt.show()

    print()
    print(f"Game ended after {EPISODES} episodes.")
    print(f"Best Score: {np.max(episode_scores):.2f}/{max_time:.2f}")
    print(f"Average Score: {np.mean(episode_scores):.2f}")
