import numpy as np
import matplotlib.pyplot as plt


def next_square(a, s):
    if a == "up":
        if s <= 3:
            return s
        else:
            return s - 4

    elif a == "down":
        if s >= 12:
            return s
        else:
            return s + 4

    elif a == "left":
        if s in {0, 4, 8, 12}:
            return s
        else:
            return s - 1

    elif a == "right":
        if s in {3, 7, 11, 15}:
            return s
        else:
            return s + 1


def give_reward(s):
    if s in [0, 15]:
        return 0
    else:
        return -1


def policy_evaluation():
    # States
    S = np.array(range(0, 16))
    # Actions
    A = np.array(["up", "down", "left", "right"])
    # Policy
    P = {a: 0.25 for a in A}
    # Theta
    theta = 0.01
    # Value of each state
    V = {s: 0 for s in S}
    # Discount
    G = 1
    i = 0
    while True:
        old_V = V.copy()
        for s in V:
            if s not in [0, 15]:
                value = 0
                for a in P:
                    value += P[a] * 1 * (give_reward(s) + G*V[next_square(a, s)])
                V[s] = value
        max_delta = max([abs(old_V[item] - V[item]) for item in V.keys()])
        i += 1
        if max_delta < theta:
            break
    V = {k: round(v, 0) for (k, v) in V.items()}
    print(i)
    print(V)


policy_evaluation()
