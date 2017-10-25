import random

import keras as ks
import numpy as np
from ple.games.snake import Snake
from ple import PLE

# -- Environment Setup --
# Setup openAI environment. This is the environment that the agent will interact with
import time

env = Snake(width=32, height=32)
p = PLE(env, fps=60, display_screen=True)
p.init()
env.reset()
print(p.getActionSet(), type(p.getActionSet()))
action_space = p.getActionSet()[:]

# -- Model Setup --
model = ks.models.Sequential()

# This environment has 4 input values
model.add(ks.layers.Conv2D(4, 5, activation=ks.activations.relu, input_shape=(32,32,3)))
model.add(ks.layers.Conv2D(5, 5, activation=ks.activations.relu, strides=2))
model.add(ks.layers.Conv2D(6, 5, activation=ks.activations.relu, strides=2))
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(20, activation=ks.activations.relu))

# The environment requires 2 output values. These are linear values, because they model the expected future reward
model.add(ks.layers.Dense(4, activation=ks.activations.linear))
model.compile(optimizer=ks.optimizers.Adam(lr=0.0001), loss=ks.losses.mean_squared_error)


# -- Function Definitions --

def get_next_action(state, exploration_rate=0.5):
    if random.random() < exploration_rate:
        return random.randint(0, 3)
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
    replay_memory = []
    state = None
    score = 0

    # Start with a random move
    action = 0
    action_code = action_space[action]
    p.display_screen = render
    while not p.game_over():
        previous_state = state
        reward = p.act(action_code)
        state = p.getScreenRGB()
        score += reward

        # Reduce reward size to scale better with network output
        reward /= 10

        if render:
            time.sleep(1/60)

        if previous_state is not None:
            # Add relevant information to replay memory
            replay_memory.append((previous_state, action, state, reward))
        action = get_next_action(state, exploration_rate)
        action_code = action_space[action]

    return score, replay_memory


def generate_epoch(replay_memory, discount_factor=0.999, learning_rate=0.1):
    states, actions, next_states, rewards = zip(*replay_memory)
    states = np.array(states)
    actions = np.array(actions, dtype=np.int32)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    # Calculate Q values for current and next states in the replay memory.
    old_state_Qs = model.predict(states)
    old_next_states_Qs = np.max(model.predict(next_states), axis=1)

    # Calculate one-hot vectors for actions to select values in the Q-tables
    actions_one_hot = np.zeros([len(actions), 4])
    actions_one_hot[np.arange(len(actions_one_hot)), actions] = 1

    # - Apply Q Update Function. -
    # Calculate new Q value estimates (or "direction" in which to move Q values)
    new_Q_direction = rewards + discount_factor * old_next_states_Qs

    # Select the Q values for the chosen actions
    chosen_Qs = np.sum(old_state_Qs*actions_one_hot, axis=1)

    # Calculate changes to be made (delta)
    deltas = learning_rate * (new_Q_direction - chosen_Qs)

    # Select only the deltas of the chosen actions
    deltas = np.transpose(actions_one_hot) * deltas
    deltas = np.transpose(deltas)

    # Return input states and the updated Q value table for replay memory
    return states, old_state_Qs + deltas


# -- Main Training Loop --

for i in range(100000):
    exploration_rate = 0.9/(i/100 + 1)
    score, memory = single_run(exploration_rate=exploration_rate)
    print("Run: %d\tScore: %d\tExp_rate: %3f" % (i, score, exploration_rate))
    for j in range(30):
        x, y = generate_epoch(memory, discount_factor=0.9, learning_rate=0.03)
        model.fit(x, y, epochs=1, batch_size=256, verbose=0)
        print(np.average(y))

    if i%10 == 0:
        single_run(True, exploration_rate=0.1)
