# action = 0 push left
# action = 1 push right
# action = 2 stay still

import numpy as np

import q_learning
import DQN

# maybe import this from somewhere in the future
settings = {"control_type": "keyboard"}

action = 2

def init(fig):
    # choose balancing mode:
    # init_q_learning()
    init_DQN()
    init_user_input(fig)


# user input
pressed_keys = set()

def on_key(event):
    global action, pressed_keys

    if event.name == 'key_press_event':
        pressed_keys.add(event.key)
    elif event.name == 'key_release_event':
        pressed_keys.discard(event.key)

    if "left" in pressed_keys:
        action = 0
    elif "right" in pressed_keys:
        action = 1


def init_user_input(fig):
    # settings["control_type"] = "keyboard"

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('key_release_event', on_key)


# q-learning
def init_q_learning():

    settings["control_type"] = "qlearning"
    data = np.load("Q_table.npz")
    try:
        q_learning.Q = data['Q']
    except:
        print("Q table not found")
        exit(1)


# DQN
def init_DQN():
    global agent
    settings["control_type"] = "DQN"
    agent = DQN.DQNAgent()
    agent.load(agent.Model_name)
    
def get_action(state):
    global action

    if len(pressed_keys) == 0 and settings["control_type"] == "qlearning":
        x_i, th_i, v_i, w_i = q_learning.discretize(state)
        action = q_learning.take_action(0, x_i, th_i, v_i, w_i)

    if len(pressed_keys) == 0 and settings["control_type"] == "DQN":
        state = np.reshape(state, [1, agent.state_size])
        
        # FAST single prediction bypassing the predict() overhead
        q_values = agent.model(state, training=False).numpy()
        action = np.argmax(q_values[0])

    return action