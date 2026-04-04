import numpy as np
import random

from state import SimulationState
from integrators import *

# define ranges of discrete states
x_max = 2.4
th_max = np.radians(12)
v_max = 2
w_max = 3

# define number of values per variable
n_x = 7
n_th = 25
n_v = 7
n_w = 25

# define iperparameters
gamma = 0.99     # discount rate
alpha = 0.1     # learning rate
alpha_decay = 0.9995
alpha_min = 0.01
epsilon = 1.0   # exploration rate
epsilon_decay = 0.9995
epsilon_min = 0.010
episodes = 10000
max_steps = 2000

dt = 0.01 # not actually used (taken from constants)
ACTION_REPEAT = 2


x_disc = np.linspace(-x_max, x_max, n_x, endpoint=False )
th_disc = np.linspace(-th_max, th_max, n_th, endpoint=False )
v_disc = np.linspace(-v_max, v_max, n_v, endpoint=False )
w_disc = np.linspace(-w_max, w_max, n_w, endpoint=False )
state_disc = [x_disc, th_disc, v_disc, w_disc]


# build R as R(s,a) discrete or use the continuos form:
def reward (state):
    return 1
    # x, th, v, w = state
    # return 1.0 - abs(th)/th_max - 0.05*abs(w)/w_max - 0.02*abs(x)/x_max


value_max = np.array([x_max, th_max, v_max, w_max])
n_value = np.array([n_x, n_th, n_v, n_w])

def discretize(value):
    value = np.array(value)
    ratio = (value + value_max) / (2 * value_max)
    ratio = np.clip(ratio, 0, 1)
    idx = np.clip((ratio * (n_value - 1)).astype(int), 0, n_value - 1)
    return idx

def random_state(episode, max_episodes):
    # Start very easy (near 0) and slowly widen the spawn range
    # By episode 20,000, it will be at full difficulty
    difficulty = min(1.0, episode / max_episodes * 2) 

    x = np.random.uniform(-0.1, 0.1) * difficulty # Keep it centered early on
    
    # Start with almost 0 tilt, grow to th_max * 0.9
    theta = np.random.uniform(-0.02, 0.02) + (np.random.uniform(-0.15, 0.15) * difficulty)
    
    v = np.random.uniform(-0.05, 0.05) * difficulty
    w = np.random.uniform(-0.05, 0.05) * difficulty

    return np.array([x, theta, v, w])

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

def terminal(state):
    x, th, v, w = state
    return abs(x) > x_max or abs(th) > th_max

def training_loop():
    global epsilon, alpha, Q

    INITIAL_Q = 0.0
    Q = np.full((n_x, n_th, n_v, n_w, 2), INITIAL_Q)


    episode_length = []

    for e in range(episodes):

        steps = 0
        
        state = random_state(e, episodes)
        x_i, th_i, v_i, w_i = discretize(state)
        sim = SimulationState(state)

        # print(f"STARTING EPISODE {e}")

        while 1:
            action = take_action(epsilon, x_i, th_i, v_i, w_i)

            for _ in range(ACTION_REPEAT):
                sim.step(rk4, action)

            new_state = sim.state[-1]
            x_j, th_j, v_j, w_j = discretize(new_state)
            r = reward(new_state)

            # print(x_i, th_i, v_i, w_i, " i.e. ", x_disc[x_i], th_disc[th_i], v_disc[v_i], w_disc[w_i], ", action: ", action)
            # print(f'Q values: action0 = {Q[x_i][th_i][v_i][w_i][0]} action1 = {Q[x_i][th_i][v_i][w_i][1]}')

            done = terminal(new_state)

            if done:
                Qtar = -100
            else:
                Qtar = r + gamma * np.max(Q[x_j, th_j, v_j, w_j])
            # print("Qtar: ", Qtar)
            Q[x_i, th_i, v_i, w_i, action] += alpha * (Qtar - Q[x_i, th_i, v_i, w_i, action])
            # print("Qupd: ", Q[x_i, th_i, v_i, w_i, action])
            # print()
            x_i, th_i, v_i, w_i = x_j, th_j, v_j, w_j
            steps += 1
            if steps >= max_steps or done: 
                break 

        if 1 or e > episodes/2: 
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
        alpha = max(alpha * alpha_decay, alpha_min)
        
        episode_length.append(steps)

        if (e + 1) % 1000 == 0:
            recent = np.mean(episode_length[-1000:])
            wins = episode_length[-1000:].count(max_steps)
            print(f"Ep {e+1:5d} | eps={epsilon:.3f} | alpha={alpha:.3f} | avg (last 1k)={recent:.1f} | wins = {wins}")

            visited = np.sum(np.any(Q != INITIAL_Q, axis=-1))
            total = n_x * n_th * n_v * n_w
            print(f"  visited: {visited}/{total} ({100*visited/total:.1f}%)")

    print("DONE")
    # print(episode_length)
    # print(S)
    # print(Q)


    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, episodes), episode_length, linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    training_loop()
    
    filename = "Q_table.npz"
    np.savez_compressed(filename, Q = Q)
    print(f"Simulation data successfully saved to {filename}")