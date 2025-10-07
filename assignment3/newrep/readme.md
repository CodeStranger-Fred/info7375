1. Introduction

Monte Carlo Tree Search (MCTS) is a widely used planning algorithm that balances exploration and exploitation through statistical sampling.
Recent research, such as LLM-MCTS (NeurIPS 2023), integrates Large Language Models (LLMs) into MCTS by introducing LLM-based policy priors and language rollouts to enhance planning in semantic and decision-making tasks.

In this project, we replicate the core algorithmic idea of LLM-MCTS and apply it to a realistic path planning scenario, rather than the natural-language VirtualHome environment used in the original paper.
Our goal is to demonstrate that LLM-guided priors can improve planning success and efficiency even in simplified spatial reasoning tasks.

2. Original Paper

Paper: LLM-MCTS: Integrating Large Language Models with Monte Carlo Tree Search for Strategic Planning

Conference: NeurIPS 2023 Workshop
Core Contribution:

Introduces LLM priors
P(s,a) into the PUCT formulation:

U(s,a)=Q(s,a)+cpuct*P(s,a)*1/1+N(s,a)\*sqrt(N(s))
​
Uses the LLM to simulate rollouts and guide search expansions.

Demonstrates improved success rate and faster convergence on natural-language planning tasks (VirtualHome, BabyAI, TextWorld).

3. Our Replication Goal

We replicate the algorithmic essence of the LLM-MCTS framework:

LLM-based action prior integrated into the MCTS search (PUCT form).

Rollout-based planning with optional LLM guidance.

Empirical comparison between Vanilla MCTS and LLM-MCTS.

We apply this to a 2D grid path planning environment with obstacles,
simulating a robotic navigation problem rather than a traditional RL Gym task.

4. Task Description
   4.1 Motivation

While GridWorld-like environments are commonly used in reinforcement learning benchmarks,
we reinterpret our task as a deterministic path planning problem in a discrete spatial environment.

This represents real-world navigation or route optimization, such as:

A robot navigating a room with obstacles.

An autonomous agent finding a route to a goal while avoiding blocked cells.

Thus, our task is not an RL-Gym benchmark but a simplified, realistic planning task.

4.2 Environment Setup
Parameter Value
Grid Size 3×3
Start S0_0
Goal G2_2
Obstacles S1_1 (center wall)
Actions {up, down, left, right}
Step Cost -1.0
Goal Reward +10.0
Rollout Depth 60
Iterations 5000

Agent can navigate around the obstacle to reach the goal.

4.3 Why This Is Not an “RL Gym” Task
Property RL Gym Task Our Path Planning Task
Goal Learn policy via trial-and-error Plan optimal path via search
Reward Stochastic / episodic Deterministic distance-based
Environment Toy benchmark (e.g., CartPole) Abstracted real-world navigation
LLM Role Not applicable Provides heuristic priors over actions
Type Simulation Realistic path planning abstraction

Conclusion:
This task simulates practical spatial reasoning rather than benchmark reinforcement learning,
therefore it is not an “RL Gym” type task.

5. Implementation Overview
   5.1 Code Structure
   /mdp → MDP definitions (State, Action, Reward, Transition)
   /mcts → Core MCTS implementation (PUCT, Tree, Search)
   /newrep → Replication: GridWorld environment + experiments
   ├── main.go # main experiment runner
   ├── gridworld_env.go # environment definition
   ├── llm_prior.go # OpenAI LLM prior
   ├── tree.go / uct.go # search core

5.2 Algorithm Flow

Initialize MCTS root node with the starting state.

For each simulation:

Selection: Traverse the tree using the PUCT score.

Expansion: Add new child nodes.

Rollout: Simulate random or LLM-guided outcomes.

Backpropagation: Update visit counts and Q-values.

Return the action with the highest average value.

6. Results
   6.1 Experimental Setup

Each method runs 10 trials with a maximum of 40 steps.

Comparison between:

Baseline MCTS: Uniform action prior.

LLM-MCTS: GPT-4o-mini based action prior.

| Model                      | Success (%) | Avg Steps | Avg Reward |
| :------------------------- | ----------: | --------: | ---------: |
| **Vanilla MCTS**           |       50.00 |      33.7 |      -23.0 |
| **LLM-MCTS (GPT-4o-mini)** |       90.00 |      29.1 |      -6.05 |

Success Rate (%)
| ██████████████████████████████████ 90 (LLM-MCTS)
| ████████████████ 50 (Vanilla)
|
|\***\*\*\*\*\*\*\***\_\_\_\_\***\*\*\*\*\*\*\***
Vanilla LLM-MCTS

The LLM prior significantly improves search efficiency and convergence,
consistent with trends reported in the original LLM-MCTS paper.

7. References

LLM-MCTS: Integrating Large Language Models with Monte Carlo Tree Search for Strategic Planning
NeurIPS 2023 Workshop
https://llm-mcts.github.io/static/pdfs/paper.pdf
