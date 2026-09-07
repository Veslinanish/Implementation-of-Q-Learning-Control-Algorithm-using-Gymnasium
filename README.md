# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment and learn an optimal or near-optimal action-value function. The agent must learn to select appropriate actions to reach the goal while avoiding the hole states.

---

## Software Requirements

Python 3.x
Gymnasium
NumPy
Matplotlib
Jupyter Notebook / Google Colab

---

## Environment Description

FrozenLake-v1 is a discrete reinforcement-learning environment consisting of a 4×4 grid. The agent starts from the initial state and must reach the goal while avoiding holes. The environment has 16 states and 4 possible actions: Left, Down, Right, and Up. A reward of 1 is obtained for reaching the goal, while other transitions normally provide zero reward.

In this experiment, a custom FrozenLake map is used. The environment is configured with `is_slippery=False`, making the transitions deterministic.

---

## Theory

Q-Learning is an **off-policy temporal-difference control algorithm** that estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$ and subsequently follows the best possible policy.

Unlike SARSA, Q-Learning does not use the next action actually selected by the agent. Instead, it uses the **maximum Q-value among all possible actions in the next state**.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol                | Meaning                                   |
| --------------------- | ----------------------------------------- |
| $S_t$                 | Current state                             |
| $A_t$                 | Current action                            |
| $R_{t+1}$             | Reward received after taking action $A_t$ |
| $S_{t+1}$             | Next state                                |
| $\alpha$              | Learning rate                             |
| $\gamma$              | Discount factor                           |
| $Q(s,a)$              | Action-value function                     |
| $\max_a Q(S_{t+1},a)$ | Maximum action value in the next state    |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

The value of $\epsilon$ is gradually reduced during training so that the agent initially explores the environment and later exploits the learned Q-values.

---

## Algorithm

1. Create the custom FrozenLake-v1 environment with the specified map.
2. Initialize the Q-table with zeros.
3. Set the learning parameters $\alpha$, $\gamma$, $\epsilon$, $\epsilon_{min}$, and $\epsilon_{decay}$.
4. Reset the environment and obtain the starting state $S_t$.
5. Select an action $A_t$ using the epsilon-greedy policy.
6. Execute action $A_t$ and observe the reward $R_{t+1}$ and next state $S_{t+1}$.
7. Find the maximum Q-value among all possible actions in the next state.
8. Update $Q(S_t,A_t)$ using the Q-Learning update rule.
9. Set $S_t = S_{t+1}$ and repeat the process until the episode terminates.
10. Store the total reward obtained during the episode.
11. Decrease $\epsilon$ using $\epsilon = \max(\epsilon_{min}, \epsilon \times \epsilon_{decay})$.
12. Repeat the training process for the specified number of episodes.
13. Calculate the state-value function using the maximum Q-value for each state.
14. Obtain the learned policy by selecting the action with the highest Q-value for each state.
15. Display the final Q-table, state-value function, learned policy, average reward, and learning curve.

---

## Python Program

```python
# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0
    for step in range(max_steps_per_episode):

        # Select action using epsilon-greedy policy
        action = epsilon_greedy_action(
            state,
            epsilon
        )

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Q-Learning update
        if terminated or truncated:

            Q[state, action] = Q[state, action] + alpha * (
                reward - Q[state, action]
            )

            break

        # Find maximum Q-value in next state
        best_next_value = np.max(Q[next_state])

        # Q-Learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward
            + gamma * best_next_value
            - Q[state, action]
        )
        # Move to next state
        state = next_state
    # Store episode reward
    episode_rewards.append(total_reward)
    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

```
---

## Output
<img width="497" height="620" alt="exp 7 op 1" src="https://github.com/user-attachments/assets/5774a144-8c0e-4b44-8ab5-cab7b4e629c3" />

<img width="942" height="586" alt="exp 7 op 2" src="https://github.com/user-attachments/assets/1c2373d0-69d1-422c-bbc0-1f1595721604" />


## Result

The Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned a suitable policy to reach the goal while avoiding holes.

---

## Inference

The experiment demonstrates that Q-Learning learns an effective policy using epsilon-greedy exploration and the maximum Q-value of the next state.
