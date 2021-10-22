import numpy as np

n_episodes = 500

alpha = 0.5
epsilon = 0.1
gamma = 1
R = -1

n_rows = 7
n_columns = 10
n_actions = 4
Q = np.zeros((n_actions, n_rows, n_columns))
# one array for each action
# define as up, down, left, right in that order

start_row = 3
start_col = 0
terminal_row = 3
terminal_col = 7


def move(a, r, c):
    if a == 0:
        return max(r - 1, 0), c
    elif a == 1:
        return min(r + 1, n_rows - 1), c
    elif a == 2:
        return r, max(c - 1, 0)
    elif a == 3:
        return r, min(c + 1, n_columns - 1)
    else:
        print("problem with the action")
        exit()


def add_wind(r, c):
    if c in [3, 4, 5, 8]:
        return max(r - 1, 0)
    elif c in [6, 7]:
        return max(r - 2, 0)
    else:
        return r


def is_terminal(r, c):
    if (r == terminal_row) & (c == terminal_col):
        return True
    else:
        return False


steps_record = []
# Loop for each episode
for n in range(n_episodes):
    n_steps = 0
    # initialise S
    row = start_row
    col = start_col

    # Choose A from S (e-greedy)
    # np.argmax chooses first action (here it is up) and this would result in us never reaching the goal
    # this instead picks randomly between all possible actions of the same value
    # might want to also look at using np.is close(b, b.max()) instead of b == b.max()
    if np.random.binomial(1, epsilon) == 1:
        action = np.random.choice(list(range(n_actions)))
    else:
        action = np.random.choice(np.flatnonzero(Q[:, row, col] == Q[:, row, col].max()))

    # Loop for each step of the episode
    terminal = False
    while not terminal:

        old_action = action
        old_row = row
        old_col = col
        # Take action A, observe S`
        row, col = move(action, row, col)
        row = add_wind(row, col)

        # Observe R
        if is_terminal(row, col):
            R = 0
        else:
            R = -1

        # Choose A` from S`
        if np.random.binomial(1, epsilon) == 1:
            next_action = np.random.choice(list(range(n_actions)))
        else:
            next_action = np.random.choice(np.flatnonzero(Q[:, row, col] == Q[:, row, col].max()))

        Q[old_action, old_row, old_col] = Q[old_action, old_row, old_col] + (
                alpha * (R + (gamma * Q[action, row, col]) - Q[old_action, old_row, old_col]))

        terminal = is_terminal(row, col)
        n_steps += 1
    steps_record.append(n_steps)

last_n_steps_to_print = 100
print(f"{np.mean(steps_record)} average steps")
print(f"{np.mean(steps_record[-last_n_steps_to_print:])} average of last {last_n_steps_to_print} steps")
print(f"{np.min(steps_record)} lowest number of steps")
print(f"{np.sum(steps_record)} total number of steps")
