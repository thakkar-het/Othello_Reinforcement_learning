# Othello-RL: Self-Play Policy Gradient Agent

This repository contains a complete, functional reinforcement learning (RL) engine for the classic board game **Othello**. Instead of relying on hand-crafted mathematical heuristics or cheat-codes (like traditional DQN/eval-shaping approaches), this agent learns purely through **self-play** using an optimized variant of the **REINFORCE (Policy Gradient)** algorithm.

---

## Repository Architecture

The project is structured into five core files, each serving a clean, isolated architectural purpose:

1.  **`Env_Brain.py`**: 
    *   **`Othello_Env`**: The complete, state-tracking Othello game engine (handling validation, piece-flipping, turn skipping, and game-over detection).
    *   **`get_encoded_state`**: State representation converter (turns raw board positions into a $3 \times 8 \times 8$ multi-channel spatial representation: player pieces, opponent pieces, and a valid move mask).
    *   **`OthelloNet`**: A PyTorch deep convolutional neural network featuring an **Actor-Critic (Policy-Value)** structure.
2.  **`Training.py`**: 
    *   Implements the stabilized batch-based policy gradient routine.
    *   Features state-action memory buffers, reward discounting, normalization, and optimization functions.
3.  **`1Game.py`**: 
    *   A lightweight, CLI-based execution script to run a single, rapid game of AI self-play. 
    *   Useful for validating that the network pipeline compiles and makes valid inferences.
4.  **`Eval.py`**: 
    *   Evaluation suite. Plays a configurable batch of games (e.g., 50) where the trained agent goes head-to-head against a purely random agent to mathematically measure win rates.
5.  **`play.py`**: 
    *   A GUI-driven execution script built using Pygame. 
    *   Allows you to play live as Black or White against your trained neural network with real-time visual move hints.

---

## Optimizations (Post-First Evaluation)

In our initial vanilla REINFORCE implementation, training was highly volatile, prone to policy collapse, and slow to converge due to the sparse, delayed reward structure of Othello. To fix this, we implemented several mathematical and algorithmic optimizations:

*   **Variance Reduction via Return Normalization**: 
    *   Raw cumulative returns can vary wildly depending on game length. We implemented standard score normalization (subtracting the mean and dividing by the standard deviation plus a stability epsilon $1e-8$) over discounted returns across the training batch. This keeps backpropagation updates numerically stable.
*   **Baseline Batching (`optimized_train_reinforce`)**:
    *   Instead of updating the network weights after every single game, we accumulate trajectories over a specified `batch_size` (e.g., 16 games). Gradients are averaged over the entire batch, which filters out the noise of individual games and converges to a stable policy gradient step.
*   **Gradient Clipping (`clip_grad_norm_`)**:
    *   To prevent the "exploding gradient" problem and stop the model from taking catastrophic update steps that destroy previous progress, we clamped the gradient norm to a maximum value of `1.0`.
*   **Stochastic Policy Exploration**:
    *   Instead of choosing actions greedily (`argmax`), the training loop maps logits through a Softmax function and samples actions probabilistically (`np.random.choice`). This ensures the agent thoroughly explores the state space and prevents self-play from getting locked into repetitive, sub-optimal loops.

---

## How to Run

### Prerequisites & Requirements

To run this project, you need Python 3.8+ and the following dependencies. You can install them by running:

```bash
pip install torch numpy pygame
```

*   **`torch`** (PyTorch): Handles the convolutional layers, policy/value logits, and backpropagation.
*   **`numpy`**: Manages multidimensional matrix math for the $8 \times 8$ board layouts.
*   **`pygame`**: Powers the 2D interactive board GUI.

---

### Execution Guide

1.  **Train & Evaluate (Headless CLI):**
    To run a training run followed by a mathematical performance audit against a random agent, run:
    ```bash
    python Eval.py
    ```
    This script will trigger `optimized_train_reinforce` for 320 episodes across batches of 16, then instantly play 50 validation games, printing out the final Win/Loss ratios.

2.  **Play Against Your Agent (GUI):**
    To launch the interactive board and play a match against the trained network, run:
    ```bash
    python play.py
    ```
    Follow the terminal instructions to pick your color (Black goes first), then click on the grey-highlighted squares on the Pygame interface to make your moves.

3.  **Inspect a Single Self-Play Game:**
    To watch the model play a rapid game against itself step-by-step in your console, run:
    ```bash
    python 1Game.py
    ```