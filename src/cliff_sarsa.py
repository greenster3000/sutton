import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.1
alpha = 0.1

reward = -1
cliff_reward = -100

start_row = 0
start_col = 0
terminal_row = 0
terminal_col = 11

cliff_col = list(range(1, 11))
cliff_row = 0

height = 4
width = 12
n_actions = 4

n_episodes = 500

Q = np.zeros((n_actions, height, width))


def move(a, r, c):
    # 0 == up, 1 == down, 2 == left, 3 == right
    if a == 0:
        return min(r + 1, height - 1), c
    elif a == 1:
        return max(r - 1, 0), c
    elif a == 2:
        return r, max(c - 1, 0)
    elif a == 3:
        return r, min(c + 1, width - 1)
    else:
        print("problem with the action")
        exit()


def is_terminal(r, c):
    return (r == terminal_row) & (c == terminal_col)


def move_if_cliff_and_reward(r, c):
    if (r == cliff_row) and (c in cliff_col):
        return 0, 0, cliff_reward
    elif is_terminal(r, c):
        return r, c, 0
    else:
        return r, c, reward


rewards_record = []
steps_record = []

for n in range(n_episodes):
    episode_reward = 0
    n_steps = 0
    row = start_row
    col = start_col

    if np.random.uniform() < epsilon:
        action = np.random.choice(list(range(n_actions)))
    else:
        action = np.random.choice(np.flatnonzero(Q[:, row, col] == Q[:, row, col].max()))

    terminal = False

    while not terminal:
        old_action = action
        old_row = row
        old_col = col

        row, col = move(action, row, col)
        row, col, R = move_if_cliff_and_reward(row, col)
        episode_reward += R

        if np.random.uniform() < epsilon:
            action = np.random.choice(list(range(n_actions)))
        else:
            action = np.random.choice(np.flatnonzero(Q[:, row, col] == Q[:, row, col].max()))

        Q[old_action, old_row, old_col] = Q[old_action, old_row, old_col] + (
                alpha * (R + (Q[action, row, col]) - Q[old_action, old_row, old_col]))
        Q[:, terminal_row, terminal_col] = 0
        n_steps += 1
        terminal = is_terminal(row, col)

    steps_record.append(n_steps)
    rewards_record.append(episode_reward)

last_n = 100
print(f"Last {last_n} episodes had an average of {np.mean([steps_record[-last_n:]])} steps")
plt.plot(list(range(len(rewards_record))), rewards_record)
plt.plot(list(range(len(rewards_record))), rewards_record)
plt.ylim((-200, 0))
plt.xlabel("Episodes")
plt.ylabel("Sum of rewards during episode")

terminal = False
row, col = start_row, start_col

while not terminal:
    action = np.random.choice(np.flatnonzero(Q[:, row, col] == Q[:, row, col].max()))
    print(row, col, action)
    row, col = move(action, row, col)
    row, col, R = move_if_cliff_and_reward(row, col)
    terminal = is_terminal(row, col)
