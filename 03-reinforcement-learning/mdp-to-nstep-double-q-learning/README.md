# Reinforcement Learning from Scratch: Custom MDP to n-step Double Q-Learning
> Designing a Gymnasium-compliant Markov Decision Process (MDP) Grid-World environment and solving it via from-scratch tabular Temporal-Difference (TD) control algorithms.

---

## 📌 Project Overview

This project focuses on the foundational principles of Reinforcement Learning (RL) by constructing a custom Markov Decision Process (MDP) environment adhering to Gymnasium standards and solving it using tabular control algorithms without external RL solver libraries.

Key technical objectives include:
- **Custom Environment Design:** Building a discrete 25-state Grid-World ("Hungry Man Grid World") with standardized Gymnasium class interfaces (`__init__`, `step`, `reset`, `render`).
- **Safety in AI Protocols:** Enforcing strict state-space validation and action safeguarding to guarantee that the agent operates reliably within bounded grid coordinates.
- **On-Policy vs. Off-Policy Control:** Implementing and comparing SARSA against n-step Double Q-learning.
- **Multi-Step Return Analysis:** Systematically evaluating the impact of multi-step TD horizons ($n \in \{1, 2, 3, 4, 5\}$) on convergence speed and policy stability.

---

## 🛠️ Tech Stack & Methodology

- **Languages & Core Libraries:** Python 3.12, Gymnasium, NumPy, OpenCV / Matplotlib.
- **Environment Architecture ("Hungry Man Grid World"):**
  - **State Space:** A 5x5 grid resulting in 25 discrete coordinate states from `(0, 0)` to `(4, 4)`.
  - **Action Space:** 4 discrete deterministic actions: `Up`, `Down`, `Right`, and `Left`.
  - **Reward Function:** Diverse reward mapping designed to balance obstacle avoidance and goal seeking (+5 for apple, +20 for cake/terminal goal, -5 for spider, -10 for dog).
- **Safety in AI Implementation:**
  - **State-Space Validation:** Before executing any state transition, the environment verifies whether the target coordinate is accessible and within the defined 5x5 boundary.
  - **Boundary Safeguarding:** When an action attempts to breach grid limits, the transition is intercepted, keeping the agent in its current valid state without throwing system exceptions.
  - **Deterministic Action Restricting:** Action selection is strictly constrained to predetermined valid discrete options, preventing undefined or unsafe behavior during $\epsilon$-greedy exploration.
- **Implemented Mathematical Formulations:**
  - **SARSA (On-Policy TD Control):** Updates Q-values using the action actually taken by the current $\epsilon$-greedy policy:
    $$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$$
  - **n-step Double Q-Learning:** Maintains two independent Q-tables ($Q_1$ and $Q_2$) to decouple action selection from value estimation, mitigating maximization bias while accumulating rewards over an $n$-step horizon.

---

## 📊 Experiments & Model Performance

Both algorithms were trained across multiple episodes to monitor cumulative reward dynamics, epsilon decay behavior, and optimal policy convergence.

| Algorithm | Learning Type | Key Hyperparameters | Convergence Stability | Best Performing Setup / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **SARSA** | On-Policy TD Control | $\alpha=0.1$, $\gamma=0.99$, $\epsilon$-decay | **High (Smooth Curve)** | Rapidly converged to a safe, obstacle-avoiding trajectory; optimal for compact MDPs. |
| **n-step Double Q-Learning** | Off-Policy TD Control | $n=3$, $\gamma=0.99$, Dual Q-tables | **Moderate (Fluctuations)** | Successfully prevented overestimation bias but exhibited higher variance during early episodes. |

### Key Insights
- **Algorithm Complexity vs. Environment Scale:** For the 25-state "Hungry Man Grid World," the simpler on-policy SARSA algorithm outperformed n-step Double Q-learning in convergence speed and trajectory stability.
- **Impact of the n-step Horizon:** Experimenting with $n \in \{1, 2, 3, 4, 5\}$ revealed that $n=3$ provided the best balance between bootstrapping and variance; higher $n$ values ($n=5$) delayed weight updates, leading to oscillatory reward curves.
- **Exploration-Exploitation Trade-off:** Tuning the $\epsilon$-decay rate was critical. Decaying $\epsilon$ too rapidly caused the agent to get trapped in local positive reward loops (+5 apples) without reaching the global +20 terminal goal.

---

## 💡 Troubleshooting & Learnings

- **Gymnasium Standard Compliance:** Transitioning from an arbitrary script to a structured Gymnasium class required explicit separation of observation spaces, action spaces, and step return signatures (`observation, reward, terminated, truncated, info`).
- **Mitigating Q-Value Oscillation:** During Double Q-learning implementation, updating $Q_1$ and $Q_2$ with 50/50 probability initially slowed convergence. Introducing a synchronized decay schedule for exploration parameters stabilized dual-table learning.
- **Safety Protocol Engineering:** Designing edge-case handling for boundary collisions reinforced the practical importance of state validation in RL, ensuring predictable agent execution during randomized training phases.