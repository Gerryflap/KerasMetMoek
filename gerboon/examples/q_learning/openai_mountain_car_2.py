import random

import keras as ks
import numpy as np
import gym

# -- Environment Setup --
# Setup openAI environment. This is the environment that the agent will interact with
import time

env = gym.make('MountainCar-v0')
env.reset()
print(env.action_space)

# -- Model Setup --
model = ks.models.Sequential()

# This environment has 4 input values
model.add(ks.layers.Dense(5, input_dim=2, activation=ks.activations.relu))
model.add(ks.layers.Dense(200, activation=ks.activations.relu))
model.add(ks.layers.Dense(200, activation=ks.activations.relu))
model.add(ks.layers.Dense(20, activation=ks.activations.relu))

# The environment requires 2 output values. These are linear values, because they model the expected future reward
model.add(ks.layers.Dense(2, activation=ks.activations.linear))
model.compile(optimizer=ks.optimizers.RMSprop(lr=0.00001), loss=ks.losses.mean_squared_error)


# -- Function Definitions --

def get_next_action(state, exploration_rate=0.5):
    if random.random() < exploration_rate:
        return random.randint(0, 1)
    q_values = model.predict(np.array([state]))
    # Return the action with the highest predicted score
    return int(np.argmax(q_values, axis=1)[0])


def single_run(render=False, exploration_rate=0.5):
    """
    Runs a single "game" in the environment and returns a replay memory
    :param exploration_rate: Rate of random actions made by agent. 0.0 is no random, 1.0 is all random
    :param render: Whether or not to render the environment (This is slower)
    :return: Score and Replay memory consisting of (state, action, next_state, reward)
    """
    env.reset()
    done = False
    replay_memory = []
    state = None
    score = 0

    # Start with a random move
    action = env.action_space.sample()
    while not done:
        previous_state = state
        state, reward, done, info = env.step(action)
        if render:
            env.render()
            time.sleep(1/60)
        score += reward

        # Add a large negative reward for failing, this seems to help
        terminal = False
        if done:
            terminal = True
            reward = -10

        # Reduce reward size to scale better with network output
        reward /= 200

        if previous_state is not None:
            # Add relevant information to replay memory
            replay_memory.append((previous_state, action, state, reward, terminal))
        action = get_next_action(state, exploration_rate)

    return score, replay_memory


def generate_epoch(model, replay_memory, discount_factor=0.9, learning_rate=0.1):
    states, actions, next_states, rewards, terminal = [], [], [], [], []
    for s, a, n_s, r, t in replay_memory:
        states.append(s)
        actions.append(a)
        next_states.append(n_s)
        rewards.append(r)
        terminal.append(t)
    states = np.array(states)
    next_states = np.array(next_states)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards)
    terminal = np.array(terminal, dtype=np.bool)
    actions_oh = np.diag([1., 1.])[actions] == 1.

    q_values = model.predict(states)
    print("Before: ", np.mean(q_values))
    print("Before Action Qs: ", np.mean(q_values, axis=0))
    print("Reward mean: ", np.mean(rewards))
    next_state_preds = model.predict(next_states)
    next_q_value = np.max(next_state_preds, axis=1)
    next_q_value[terminal] = 0

    print("Next q_mean: ", np.mean(next_q_value))
    q_values[actions_oh] = (1-learning_rate) * q_values[actions_oh] + \
                           learning_rate * (rewards + discount_factor * next_q_value)

    print("Next q min: ", np.min(q_values))
    print("After: ", np.mean(q_values))

    # Return input states and the updated Q value table for replay memory
    sort = np.arange(0, len(actions))
    np.random.shuffle(sort)
    return states[sort], q_values[sort]


# -- Main Training Loop --

replay_memory = []
for i in range(10000):
    print(0.9/(i/1000 + 1))
    score, memory = single_run(exploration_rate=0.9/(i/1000 + 1))
    print("Score: ", score)
    replay_memory += memory
    if len(replay_memory) > 1500:
        for j in range(10):
            x, y = generate_epoch(model, replay_memory, discount_factor=0.8, learning_rate=1.0)
            model.fit(x, y, epochs=1, batch_size=32, verbose=0)
        replay_memory = []

    if i%100 == 0:
        single_run(True, exploration_rate=0)
