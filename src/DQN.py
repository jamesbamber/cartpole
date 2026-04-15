import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random

import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop

from constants import *
from integrators import *
from physics_rigidpole import f


def OurModel(input_shape, action_space):
    X_input = Input(input_shape)
    X_input = Input(shape=input_shape)
    X = Dense(512, activation="relu")(X_input)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
    model = Model(inputs = X_input, outputs = X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    # model.summary()
    return model


class SJEnv():

    def __init__(self):
        self.state = None
        # mere adapdation on gym, improvable
        self.observation_space = type('', (), {})()
        self.observation_space.shape = (4,)
        self.action_space = type('', (), {})()
        self.action_space.n = 2

        self.steps = 0
        self.max_episode_steps = 2000

        self.force_mag = Fmod
        self.dt = dt
        self.x_threshold = 2.4
        self.theta_threshold = np.radians(12)


    def reset(self):
        self.steps = 0
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        return self.state, {}
        # in gym we have more parameters, variable to handle calling step() when episode
        # is over and conditional rendering

    def step(self, action):
        _, self.state = rk4(0, self.state, f, dt, action)
        x, theta, x_dot, theta_dot = self.state

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )

        if terminated:
            reward = -100
        else:
            reward = 1

        self.steps += 1
        truncated = self.steps >= self.max_episode_steps

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}
       
    def render(self):
        pass

class DQNAgent:
    def __init__(self):
        self.env = SJEnv()
        self.Model_name = "DQN_Model5_0.h5"
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = int(self.env.action_space.n)
        self.EPISODES_training = 150
        self.EPISODES_testing = 10
        self.memory = deque(maxlen=2000)
        self.total_steps = 0
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9999
        self.batch_size = 128
        self.train_start = 1500

        
        self.steps_list = []

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    # for target network (not actually using it now)
        self.target_model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state, verbose=0))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        batch_len = len(minibatch)

        state = np.zeros((batch_len, self.state_size))
        next_state = np.zeros((batch_len, self.state_size))

        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(batch_len):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state, verbose=0)
        # no target network:
        target_next = self.model.predict(next_state, verbose=0)
        # target network:
        # target_next = self.target_model.predict(next_state, verbose=0)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        history = self.model.fit(state, target, batch_size=batch_len, verbose=0)
        return history.history['loss'][0]

    def load(self, name):
        self.model = load_model(name, compile=False)
        self.model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01))

    def save(self, name):
        self.model.save(name)

    def run(self):
        attempt = 0

        while True:
            print(f"\n=== Nuovo training (tentativo {attempt}) ===\n")

            # reset rete e memoria
            self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)
            self.target_model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)
            self.update_target_model()

            self.memory.clear()
            self.epsilon = 1.0
            self.total_steps = 0
            self.steps_list = []

            for e in range(self.EPISODES_training):
                state, _ = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                i = 0

                while not done:
                    action = self.act(state)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.remember(state, action, reward, next_state, done)

                    state = next_state
                    i += 1
                    self.total_steps += 1

                    if self.total_steps % 1000 == 0:
                        self.update_target_model()

                    loss = self.replay()

                    if done:
                        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                        print(f"episode: {e}/{self.EPISODES_training}, score: {i}, loss: {loss_str}, e: {self.epsilon:.4f}")

                        self.steps_list.append(i)

                        # condizione di convergenza (4 volte 2000)
                        if len(self.steps_list) >= 4 and all(s == 2000 for s in self.steps_list[-4:]):
                            print("Convergenza raggiunta (4 episodi da 2000 consecutivi)")
                            print(f"Saving trained model as {self.Model_name}")
                            self.save(self.Model_name)
                            return

                        break  # fine episodio

            # se arriva qui → non ha convergito
            print("Non convergente, riavvio training...\n")
            attempt += 1

    def test(self):
        self.load(self.Model_name)
        for e in range(self.EPISODES_testing):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                action = np.argmax(self.model.predict(state, verbose=0))
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES_testing, i))
                    break

if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    # agent.test()
