import numpy as np
import random

from physics_rigidpole import f
from integrators import rk4



x_max  = 2.4
th_max = np.radians(15)
v_max  = 3
w_max  = 4

n_x  = 50
n_th = 100
n_v  = 50
n_w  = 50

gamma = 0.95
alpha = 0.1

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 5000
dt = 0.01



def discretize(state):
    x, th, v, w = state
    
    ratios = np.array([
        (x  + x_max ) / (2*x_max ),
        (th + th_max) / (2*th_max),
        (v  + v_max ) / (2*v_max ),
        (w  + w_max ) / (2*w_max )
    ])
    
    ratios = np.clip(ratios, 0, 1)
    
    idx = (ratios * np.array([n_x-1, n_th-1, n_v-1, n_w-1])).astype(int)
    return tuple(idx)



def terminal(state):
    x, th, v, w = state
    
    if abs(x)  >= x_max:  return True
    if abs(th) >= th_max: return True
    
    return False



def reward(state):
    x, th, v, w = state
    
    return 1 - 5*th**2 - 0.1*w**2 - 0.01*x**2



Q = np.zeros((n_x, n_th, n_v, n_w, 2))


def random_state():
    while True:
        x  = np.random.uniform(-1, 1)
        th = np.random.uniform(-0.1, 0.1)
        v  = np.random.uniform(-1, 1)
        w  = np.random.uniform(-1, 1)
        
        s = (x, th, v, w)
        
        if not terminal(s):
            return s


episode_length = []

for ep in range(episodes):
    
    state = random_state()
    steps = 0
    
    while True:
        
        x_i, th_i, v_i, w_i = discretize(state)
        print("actual state: ", x_i, th_i, v_i, w_i)
        if random.random() < epsilon:
            action = random.randint(0,1)
        else:
            action = np.argmax(Q[x_i, th_i, v_i, w_i])
        print("action: ", action)
        _, new_state = rk4(0, state, f, dt, action)
        
        r = reward(new_state)
        
        x_j, th_j, v_j, w_j = discretize(new_state)
        ("new state: ", x_j, th_j, v_j, w_j)
        if terminal(new_state):
            r -= 10
        
        Q[x_i, th_i, v_i, w_i, action] += alpha * (
            r + gamma * np.max(Q[x_j, th_j, v_j, w_j]) -
            Q[x_i, th_i, v_i, w_i, action]
        )
        
        state = new_state
        steps += 1
        
        if terminal(state) or steps > 500:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_length.append(steps)

print("DONE")
print("max Q =", np.max(Q))
print("Q = ", Q)
print(Q[3,4,6,12,0], Q[4,6,2,9,1], Q[33,31,18,20,0])

print(episode_length)