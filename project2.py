import numpy as np
import random
import json
import gym
import time

from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# Hyper-parameters
gamma = 0.99
alpha = 0.001
training_frequency = 1
batch_size = 256

episodes = 5000
trials = 100

# Epsilon decay
epsilon_max = 1.0
epsilon_min = 0.1
epsilon_decay_lin = 2 * ( epsilon_max - epsilon_min ) / episodes

env = gym.make('LunarLander-v2')
env.seed(5)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# Memory bank
memory_bank = deque()
memory_bank_size = 100000

# Initialize NN for function approximation
model = Sequential()
model.add(Dense(64, input_dim=num_states, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_actions, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=alpha))


# Stats
epsilons = []
states = []
actions = []
rewards = []

# Train
epsilon = epsilon_max
total_steps = 0
max_ma = float('-inf')
ma = float('-inf')
solved_at = -1
for episode in range(episodes):
    state = env.reset()

    # Episode
    done = False
    step = 0
    total_reward = 0
    while not done:
        # Select action
        if np.random.random() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(model.predict(state.reshape(1, 8))[0])

        # Save history
        next_state, reward, done, info = env.step(action)

        memory_bank.append((state, action, reward, next_state, done))
        if len(memory_bank) > memory_bank_size:
            memory_bank.popleft()

        # Increment
        state = next_state
        step += 1
        total_steps += 1
        total_reward += reward

        epsilons.append(epsilon)
        states.append(state.tolist())
        actions.append(int(action))

    # Update NN Weights
    if episode % training_frequency == 0 and len(memory_bank) == memory_bank_size:
        minibatch = np.array(random.sample(memory_bank, batch_size))

        state_batch = np.array(minibatch[:, 0].tolist())
        action_batch = np.array(minibatch[:, 1].tolist())
        reward_batch = np.array(minibatch[:, 2].tolist())
        next_state_batch = np.array(minibatch[:, 3].tolist())
        term_batch = np.array(minibatch[:, 4].tolist())

        state_value_batch = model.predict(state_batch)
        next_state_value_batch = model.predict(next_state_batch)

        for i in range(len(minibatch)):
            if term_batch[i]:
                state_value_batch[i, action_batch[i]] = reward_batch[i]
            else:
                state_value_batch[i, action_batch[i]] = reward_batch[i] + gamma * np.max(next_state_value_batch[i])
        model.train_on_batch(state_batch, state_value_batch)

    rewards.append(total_reward)

    epsilon = max(epsilon - epsilon_decay_lin, epsilon_min)
    ma = np.mean(rewards[-100:])
    if ma > max_ma:
        max_ma = ma
        if max_ma > 200.0:
            solved_at = episode

    print('Episode {}, Reward: {}, Steps: {}, Moving average reward: {}, Max ma: {}'.format(episode, total_reward, step, ma, max_ma))

# Trials
trial_rewards = []
for trial in range(trials):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(model.predict(state.reshape(1, 8))[0])
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
    print("Trial: {}, Reward: {}".format(trial, total_reward))
    trial_rewards.append(total_reward)

# Save stats
output = {
    'epsilons':epsilons,
    'states':states,
    'actions':actions,
    'rewards':rewards,
    'trial_rewards':trial_rewards,
}
with open('p2_out{}.json'.format(time.strftime("%Y%m%d-%H%M%S")), 'w') as f:
    f.write(json.dumps(output))

env.close()