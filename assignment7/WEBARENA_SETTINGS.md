# WebArena Environment Settings

## Basic Settings

| Setting | Value |
|---------|-------|
| **Total Tasks** | 50 (filtered from 812 total) |
| **Domains** | shopping, shopping_admin |
| **Task Types** | string_match queries |
| **Page States** | home, search_results, product_detail |
| **Available Actions** | `search:<query>`, `click:<id>`, `buy:<id>`, `type:<text>` |
| **Maximum Turns per Episode** | 15 |
| **DOM Format** | Accessibility tree with element IDs |

---

## Environment Details

| Setting | Value |
|---------|-------|
| **Task Source** | `webarena/config_files/test.raw.json` |
| **Evaluation Type** | string_match (exact/fuzzy) |
| **Answer Format** | Product names, brands, categories |
| **DOM Generation** | Dynamic based on task + state |
| **Reward Type** | Sparse (0.0 intermediate, 1.0 success) |

---

## Example Tasks

| Task ID | Intent | Correct Answer |
|---------|--------|----------------|
| 0 | What is the top-1 best-selling product in 2022 | Quest Lumaflex™ Band |
| 1 | What is the top-1 best-selling brand in Q1 2022 | Sprite |
| 2 | What is the top-1 best-selling product type in Q1 2022 | Yoga ball |

---

## DOM Structure

**Homepage:**
```
[1] RootWebArea 'Shopping Site'
    [10] navigation 'Main Menu'
    [20] main
        [22] searchbox 'Search products'
        [23] button 'Search'
    [30] complementary 'Featured Products'
        [100-102] product links (based on task keywords)
```

**Search Results:**
```
[1] RootWebArea 'Search Results'
    [10] navigation 'Breadcrumb'
    [20] main
        [21] heading 'Search Results'
    [30] list 'Products'
        [105] article (correct answer)
            [106] heading '{answer}'
            [107] text 'Price: $XX'
        [110-140] articles (distractor products)
```

**Product Detail:**
```
[1] RootWebArea 'Product Details'
    [10] navigation 'Breadcrumb'
    [20] main
        [21] heading '{product_name}'
        [22] img 'Product image'
        [23] text 'Price: $XX'
        [26] button 'Add to Cart'
        [27] button 'Buy Now'
```

---

## Training Configuration

| Setting | Value |
|---------|-------|
| **Model** | Qwen/Qwen2.5-3B-Instruct |
| **Training Steps** | 100 |
| **Batch Size** | 2 |
| **Samples per Prompt** | 2 |
| **Learning Rate** | 5e-6 |
| **KL Penalty (β)** | 0.05 |
| **Advantage Temp** | 6.0 |
| **Top-k Retrieval** | 3 |

---

## Reward Function

| Component | Value |
|-----------|-------|
| **Success Reward** | 1.0 (if `current_product_id == 105`) |
| **Failure Reward** | 0.0 |
| **Intermediate Actions** | 0.0 (sparse reward) |
| **Fuzzy Match** | `overlap / total_words` |
| **Success Metric** | `success = (env_reward == 1.0)` |

---

## A*PO Training Details

| Setting | Value |
|---------|-------|
| **Algorithm** | A*PO (A-Star Policy Optimization) |
| **Loss Function** | `-E[A * log π] + β * KL(π||π_ref)` |
| **Advantage** | `A = reward - mean(rewards)` |
| **Reference Policy** | Initial model parameters (frozen) |
| **Optimizer** | AdamW |
| **Gradient Clipping** | 1.0 |
| **Memory Optimization** | Parameter swapping (no model copy) |

---

## Retrieval Module

| Setting | Value |
|---------|-------|
| **Method** | Keyword-based matching |
| **Scoring** | Word overlap between task and DOM elements |
| **Top-k** | 3 elements |
| **Stop Words** | Filtered (what, is, the, a, an, in, etc.) |
| **Context Window** | Last 3 actions in history |

---

## Evaluation

| Setting | Value |
|---------|-------|
| **Eval Frequency** | Every 10 steps |
| **Eval Tasks** | 10 held-out tasks |
| **Success Threshold** | `reward > 0.8` |
| **Zero-shot Baseline** | 0% (Qwen2.5-3B-Instruct) |
| **Expected After Training** | 10-20% (simulated env) |

---

## GPU Requirements

| Component | Memory |
|-----------|--------|
| **Model (bfloat16)** | ~6 GB |
| **Optimizer States** | ~12 GB |
| **Activations** | ~3-4 GB |
| **Rollout Buffer** | ~2-3 GB |
| **Total Required** | ~23-25 GB |
| **Minimum GPU** | A100 40GB (T4 15GB insufficient) |

---

## Comparison with Real WebArena

| Aspect | Real WebArena | Our Simulation |
|--------|---------------|----------------|
| **Total Tasks** | 812 | 50 |
| **Domains** | 6 (shopping, gitlab, reddit, map, wiki, admin) | 2 (shopping, admin) |
| **Environment** | Live websites (Docker) | Simulated DOM generation |
| **Browser** | Playwright automation | Text-based DOM trees |
| **DOM Source** | Real web pages | Dynamically generated |
| **Evaluation** | Exact web states | String matching |
| **Trainability** | Requires full setup | Immediate training |
| **Cost** | High (AWS + compute) | Low (local/Modal) |

---

## Key Differences from WebShop

| Feature | WebShop (Assignment 6) | WebArena (Assignment 7) |
|---------|------------------------|-------------------------|
| **Products** | 96 (8×4×3) | 50 tasks from real data |
| **Attributes** | Fixed (color, size, category) | Dynamic (task-dependent) |
| **Answers** | Attribute combinations | Real product names |
| **DOM** | Simple product listings | Structured web pages |
| **Reward** | Attribute matching | String/URL matching |
| **Complexity** | Single-step purchase | Multi-page navigation |

---

## Example Training Output

```
Step   5 | Loss: 0.0234 | Policy: 0.0180 | KL: 0.0054 | Reward: 0.125
Step  10 | Loss: 0.0189 | Policy: 0.0145 | KL: 0.0044 | Reward: 0.200
================================================================================
Step  10 | Evaluation Success Rate: 10.00%
================================================================================
Step  15 | Loss: 0.0156 | Policy: 0.0118 | KL: 0.0038 | Reward: 0.275
Step  20 | Loss: 0.0134 | Policy: 0.0098 | KL: 0.0036 | Reward: 0.350
```

---

## References

- **WebArena Paper**: https://webarena.dev/
- **WebArena GitHub**: https://github.com/web-arena-x/webarena
- **A*PO Algorithm**: Policy gradient with KL regularization
- **RAGEN Implementation**: Based on assignment6/new/ragen_entry.py
