Paper:
LLM-MCTS: Integrating Large Language Models with Monte Carlo Tree Search for Strategic Planning (2023) https://llm-mcts.github.io/static/pdfs/paper.pdf
Goal:
Reproduce the core idea — integrate an LLM as a policy prior and/or rollout policy in MCTS,
then evaluate if it improves planning performance in a grid navigation environment.


┌──────────────┐
│   GridWorld  │   (State S)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   LLM Model  │   (returns P(s,a), V(s))
└──────┬───────┘
       │
       ▼
┌───────────────────────┐
│ Monte Carlo Tree      │
│ - Select (UCT/PUCT)   │
│ - Expand              │
│ - Rollout (LLM guided)│
│ - Backpropagate       │
└───────────────────────┘
       │
       ▼
┌──────────────┐
│   Best Move  │
└──────────────┘
