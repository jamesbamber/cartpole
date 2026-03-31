import numpy as np
import random

from physics_rigidpole import f
from integrators import *


# define ranges of discrete states
x_max = 2.4
th_max = np.radians(15)
v_max = 3
w_max = 4

# define number of values per variable
n_x = 20
n_th = 40
n_v = 20
n_w = 30

# define iperparameters
gamma = 0.9     # discount rate
alpha = 0.1     # learning rate
alpha_decay = 0.99
epsilon = 1.0   # exploration rate
epsilon_decay = 0.99
epsilon_min = 0.001
episodes = 100

dt = 0.01


x_disc = np.linspace(-x_max, x_max, n_x)
th_disc = np.linspace(-th_max, th_max, n_th)
v_disc = np.linspace(-v_max, v_max, n_v)
w_disc = np.linspace(-w_max, w_max, n_w)
state_disc = [x_disc, th_disc, v_disc, w_disc]
print(state_disc)


# build R as R(s,a) discrete or use the continuos form:
def reward (x_i, th_i, v_i, w_i):
    x = x_disc[x_i]
    th = th_disc[th_i]
    v = v_disc[v_i]
    w = w_disc[w_i]
    return 1 - 5*th**2 - 0.1*w**2 - 0.01*x**2



# returns the index of the value_disc array for the continuos value
def discretize(value, value_max, n_value):
    value = np.array(value)
    value_max = np.array(value_max)
    n_value = np.array(n_value)
    ratio = (value + value_max) / (2 * value_max)
    ratio = np.clip(ratio, 0, 1)
    idx = (ratio * (n_value - 1)).astype(int)
    return idx

# building T using f and projecting the state values into the discrete field
# T(s,a) |-> s'
T = np.zeros((n_x, n_th, n_v, n_w, 2, 4), dtype=int)
max_arr = np.array([x_max, th_max, v_max, w_max])
n_arr = np.array([n_x, n_th, n_v, n_w])
for i in range(n_x):
    for j in range(n_th):
       for k in range(n_v):
            for l in range(n_w):
                for a in [0, 1]:
                    t, new_state = rk4(0, (x_disc[i], th_disc[j], v_disc[k], w_disc[l]), f, dt, a)
                    T[i, j, k, l, a] = discretize(new_state, max_arr, n_arr)


# S(s) |-> 0 (not terminal), 1 (terminal)
S = np.zeros((n_x, n_th, n_v, n_w))
for i in range(n_x):
    for j in range(n_th):
       for k in range(n_v):
            for l in range(n_w):
                if (i==0 or i==n_x-1 or j==0 or j==n_th-1):
                    S[i, j, k, l] = 1



def random_state():
    i = random.random()
    x_i= int(i*(n_x-1))
    i = random.random()
    th_i= int(i*(n_th-1))
    i = random.random()
    v_i= int(i*(n_v-1))
    i = random.random()
    w_i= int(i*(n_w-1))
    return x_i, th_i, v_i, w_i

def take_action(epsilon, x_i, th_i, v_i, w_i):
    e = random.random()
    if e < epsilon:
        return random_action()
    else:
        return greedy_action(x_i, th_i, v_i, w_i)

def random_action():
    return random.randint(0,1)

def greedy_action(x_i, th_i, v_i, w_i):
    a = Q[x_i, th_i, v_i, w_i, 0]
    b = Q[x_i, th_i, v_i, w_i, 1]
    return (0 if a>b else 1)


Q = np.zeros((n_x, n_th, n_v, n_w, 2))

# for debugging
episode_length = []

for e in range(episodes):
    steps = 0
    x_i, th_i, v_i, w_i = random_state()
    while S[x_i, th_i, v_i, w_i] == 0:
        action = take_action(epsilon, x_i, th_i, v_i, w_i)
        r = reward(x_i, th_i, v_i, w_i)
        x_j, th_j, v_j, w_j = T[x_i, th_i, v_i, w_i, action]
        if S[x_j, th_j, v_j, w_j] == 1:
            r += 100
        m = np.max(Q[x_j, th_j, v_j, w_j])
        Qtar = r + gamma * m
        Q[x_i, th_i, v_i, w_i, action] = Q[x_i, th_i, v_i, w_i, action] + alpha * (Qtar - Q[x_i, th_i, v_i, w_i, action])
        x_i, th_i, v_i, w_i = x_j, th_j, v_j, w_j
        steps += 1
    epsilon = epsilon * epsilon_decay
    episode_length.append(steps)

print(episode_length)
# print(S)
print(Q)