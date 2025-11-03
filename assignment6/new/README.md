# RAGEN: Retrieval-Augmented Generation with A*PO

A complete implementation of RAGEN (Retrieval-Augmented Generation with Reinforcement Learning) using A*PO (A-Star Policy Optimization) instead of PPO/GRPO, trained on a simulated WebShop environment.

## Overview

**What it achieves:**
- Trains an LLM to shop for products matching user specifications
- Uses retrieval-augmented prompting (retriever + generator)
- Optimizes with A*PO loss: `Loss = -E[A * log π] + β * KL`
- Evaluates success rate during training

**How it works:**
1. **State**: Current shopping task (e.g., "Find black large shoe under $50")
2. **Retriever**: Searches product database, returns top-3 matches
3. **Generator**: LLM generates action (search/click/buy) conditioned on retrieved products
4. **Reward**: Binary reward based on whether purchased product matches target
5. **A*PO**: Optimize policy with advantage weighting + KL divergence penalty
6. **Evaluate**: Periodically measure success rate

## Files

- **ragen_entry.py**: Core implementation (train, evaluate, backward pass)
- **ragen_runner.py**: Modal cloud deployment script
- **test_ragen_local.py**: Local unit tests (no model loading)

## Architecture

```
┌─────────────────────────────────────────┐
│  State: Shopping Task                   │
│  "black large shoe under $50"           │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Retriever: Search Products             │
│  Query: "black large shoe"              │
│  Output: [product_0, product_2, ...]    │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Generator: Conditional Generation      │
│  Input: state + retrieved_docs          │
│  Output: "I'll buy [0]"                 │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Reward: Check Purchase                 │
│  Product 0: black, large, $45 ✓         │
│  Reward = 1.0                           │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  A*PO Loss                              │
│  A = reward - mean_reward (advantage)   │
│  Loss = -A * log π + β * KL(π||π_ref)   │
│  Backprop & Update                      │
└─────────────────────────────────────────┘
```

## Key Functions

| Function | Purpose |
|----------|---------|
| `generate_actions()` | Generate K samples per state using retriever |
| `compute_rewards()` | Evaluate each generation against target |
| `evaluate()` | Measure success rate on holdout set |
| `apo_step()` | Compute A*PO loss and return scalar |
| `train()` | Main training loop |

## Data Structures

**Generation**: Single generation with text, action, token_ids, reward, logprobs, advantage
**State**: Shopping task with multiple generations (K samples)
**Batch**: Batch of states with flattening utilities

## Configuration

```python
@dataclass
class TrainCfg:
    steps: int = 100                    # Training steps
    batch_size: int = 4                 # Parallel states
    num_samples_per_prompt: int = 4     # K samples per state
    lr: float = 5e-6                    # Learning rate
    beta_kl: float = 0.05               # KL penalty weight
```

## Running

### Local Test (CPU, no model):
```bash
python3 test_ragen_local.py
```

### Local Training (requires CUDA):
```bash
python3 ragen_entry.py
```

### Cloud Training (Modal):
```bash
modal run --detach ragen_runner.py
```

## Training Output

```
Step  10 | Loss: 0.0234 | Avg Reward: 0.245 | Success@1: 0.200
Step  20 | Loss: 0.0156 | Avg Reward: 0.318 | Success@1: 0.350
Step  30 | Loss: 0.0089 | Avg Reward: 0.412 | Success@1: 0.480
...
```

- **Loss**: A*PO loss (policy + KL penalty)
- **Avg Reward**: Average reward across current batch
- **Success@1**: Success rate on 10 held-out test cases

## Implementation Details

### A*PO Loss
```python
# Advantage normalization
A = (r - mean_reward)

# KL divergence
KL = log π(a|s) - log π_ref(a|s)

# Combined loss
Loss = -E[A * log π] + β * KL
```

### Reward Function (Following RAGEN Paper)

**1. Task Reward (Sparse):**
- **Buy correct product** (all attributes match): 1.0
- **Buy wrong product** (any attribute fails): 0.0
- **Search/click action**: 0.0 (no intermediate reward)
- **Timeout without buying**: 0.0

**2. Reasoning-Action Consistency Reward (Bonus):**
- Checks if model's reasoning (`<think>` tags) matches its action
- **Action type mention**: +0.05 (e.g., mentions "buy" when buying)
- **Target attributes**: +0.1 (proportional to coverage of color/size/price)
- **Reasoning quality**: +0.05 (non-trivial length >20 chars)
- **Maximum bonus**: 0.2 per trajectory

**Total Reward = Task Reward + Reasoning Consistency Bonus**
- Perfect execution: 1.0 + 0.2 = **1.2**
- Good reasoning, wrong product: 0.0 + 0.2 = **0.2**
- No reasoning, correct product: 1.0 + 0.0 = **1.0**
- No reasoning, wrong product: 0.0 + 0.0 = **0.0**

### Retriever
Simple sparse retriever using substring matching on product names and attributes.

## Expected Performance

**Baseline (random)**: ~0.05 success rate
**After training 100 steps**: ~0.30-0.50 success rate
**Full run (500 steps on H100)**: ~0.60+ success rate

## Files Summary

```
ragen_entry.py      (660 lines)
  - SimpleWebShopEnv: Product database & reward logic
  - Retriever: Search products
  - Batch, State, Generation: Data structures
  - generate_actions(): Generate K samples with retriever
  - compute_rewards(): Score each generation
  - apo_step(): A*PO loss computation
  - evaluate(): Success rate measurement
  - train(): Main loop with evaluation every 10 steps

ragen_runner.py     (30 lines)
  - Modal app configuration
  - GPU: H100, timeout: 24h

test_ragen_local.py (213 lines)
  - Unit tests for all components (no model loading)
  - Verifies logic before cloud training
```

## Key Design Choices

1. **Simplicity**: All logic exposed (no hidden abstractions)
2. **Evaluation during training**: Success rate every N steps
3. **Fixed reference model**: π_ref is frozen copy of initial policy
4. **Advantage normalization**: (r - baseline) helps stable training
5. **Action extraction**: Regex-based parsing of LLM generations
6. **Reward shaping**: Partial credit for format, full credit for correct purchase

## Next Steps

1. Run local tests: `python3 test_ragen_local.py`
2. Test on single GPU: `python3 ragen_entry.py`
3. Deploy to Modal: `modal run --detach ragen_runner.py`
4. Monitor: `modal app logs <app-id>`
5. Analyze results & success rate curve
