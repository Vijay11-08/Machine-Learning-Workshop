import numpy as np
import random

# Define the environment
states = ["A", "B", "C", "D"]
actions = ["left", "right"]
rewards = {
    ("A", "right"): (1, "B"),
    ("B", "right"): (2, "C"),
    ("C", "right"): (3, "D"),
    ("D", "right"): (0, "D"),
    ("B", "left"): (1, "A"),
    ("C", "left"): (2, "B"),
    ("D", "left"): (3, "C"),
}

# Initialize Q-table
Q = {state: {action: 0 for action in actions} for state in states}

# Parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000

# Q-Learning process
for _ in range(episodes):
    state = "A"  # Start state
    while state != "D":  # Terminal state
        # Choose action (ε-greedy policy)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)  # Explore
   
        else:
            action = max(Q[state], key=Q[state].get)  # Exploit

        # Take action and observe reward and next state
        reward, next_state = rewards.get((state, action), (0, state))

        # Update Q-value
        Q[state][action] += alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])

        # Move to next state
        state = next_state

# Display learned Q-values
print("Q-Table:")
for state in Q:
    print(f"{state}: {Q[state]}")

