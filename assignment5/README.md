# TinyZero with A*PO Implementation

A from-scratch implementation of [TinyZero](https://github.com/Jiayi-Pan/TinyZero) using **A*PO (A-Star Policy Optimization)** instead of GRPO, trained on the Countdown task.

This project reproduces the core ideas from [DeepSeek R1 Zero](https://arxiv.org/pdf/2501.12948) using only PyTorch and PyTorch FSDP for training, without any additional RL/LLM frameworks.

---

## 📋 Project Overview

### What This Achieves
- **Task**: Train a language model to solve the Countdown numbers game
  - Given 4 numbers and a target, create an equation using +, -, *, / that equals the target
  - Each number must be used exactly once
  
- **Method**: A*PO (A-Star Policy Optimization)
  - Policy gradient with reward-weighted loss
  - KL divergence penalty to prevent policy collapse
  - Multi-sample generation for exploration

- **Model**: Qwen/Qwen2.5-3B (3B parameter LLM)

### How It Works

1. **Data Generation**: Create training examples with valid Countdown problems
2. **Prompt Engineering**: Format problems with CoT (Chain-of-Thought) structure
3. **Multi-sample Generation**: Generate 8 candidate solutions per problem
4. **Reward Calculation**: Binary reward (1.0 for correct, 0.0 for incorrect)
5. **A*PO Loss**: 
   ```
   Loss = -E[r * log π(y|x)] + β * KL(π || π_ref)
   ```
   where:
   - `r` = reward
   - `π` = current policy
   - `π_ref` = reference policy (initial model state)
   - `β` = KL penalty coefficient (0.1)

6. **Training**: Policy gradient updates with KL regularization

---

## 🏗️ Architecture

```
src/
├── astar_po.py                  # A*PO loss with KL divergence & advantage norm
├── online_trainer.py            # Online Rollout trainer
├── selfplay_trainer.py          # Self-Play trainer (LLM generates problems)
├── online_problem_generator.py  # Rule-based problem generator
├── llm_problem_generator.py     # LLM Self-Play problem generator
├── model_manager.py             # Model loading and FSDP setup
├── data_processor.py            # Dataset and prompt formatting
└── trainer.py                   # Base trainer (for reference)

configs/
└── default.yaml                 # Training configuration

modal_online_rollout.py          # Main training script (Online Rollout)
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- Modal 0.63+ (for cloud deployment)

### Cloud Training on Modal (Recommended)

**Run Online Rollout Training**
```bash
modal run --detach modal_online_rollout.py
```

This will:
- Dynamically generate 250 unique problems (5 iterations × 50 problems)
- Train with A*PO + fixed reference model + advantage normalization
- Use temperature annealing for stable sampling
- Save checkpoints and training stats to Modal volumes

**Monitor Progress**
```bash
modal app list  # Check running apps
modal app logs <app-id>  # View training logs
```

---

## ⚙️ Configuration

Key hyperparameters in `configs/default.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 5e-5 | Learning rate for AdamW optimizer |
| `beta` | 0.1 | KL divergence penalty coefficient |
| `num_samples` | 8 | Number of solutions to generate per problem |
| `num_epochs` | 10 | Training epochs |
| `batch_size` | 1 | Batch size (limited by generation overhead) |
| `max_length` | 512 | Maximum sequence length |

---

## 📊 Experimental Results

### Development Run (50 samples, 3 epochs)

**Setup:**
- Model: Qwen/Qwen2.5-3B
- GPU: NVIDIA A100 (40GB)
- Dataset: 50 training samples
- Epochs: 3
- Training Time: ~8 hours
- Cost: ~$0.50

**Results:**
| Metric | Value |
|--------|-------|
| Final Accuracy | 5.5% |
| Avg Loss (Epoch 1) | 0.0265 |
| Avg Loss (Epoch 2) | 0.0248 |
| Avg Loss (Epoch 3) | 0.0329 |

**Issues Identified:**
- ❌ Learning rate too low (5e-7) - model barely learned
- ❌ Only 2 samples per problem - insufficient exploration
- ❌ Only 50 training examples - underfitting
- ✅ Data quality verified (solutions are correct)
- ✅ A*PO implementation with KL divergence working

### Planned Full Run (1000 samples, 10 epochs)

**Setup:**
- Model: Qwen/Qwen2.5-3B
- GPU: 4x NVIDIA A100 with FSDP
- Dataset: 1000 training samples
- Epochs: 10
- Updated Config:
  - Learning rate: 5e-5 (100x higher)
  - Samples per problem: 8 (4x more exploration)
  - KL penalty: 0.1

**Expected:**
- Estimated Training Time: ~12 hours
- Estimated Cost: ~$30
- Target Accuracy: >50%

---

## 🧪 Evaluation

The training automatically evaluates on generated problems. Monitor:
- **Reward**: Average reward per problem (0.0-1.0)
- **Loss**: Policy loss + KL penalty
- **Temperature**: Sampling temperature (anneals from 1.0 to 0.3)

Reward breakdown:
- 0.1: Correct format
- 0.1: Valid expression
- 0.1: Legal operators
- 0.2: Number usage correctness
- 0.6: Result accuracy (exponential decay)
- 1.0: Exact match

---

## 💾 Memory Usage

**Single GPU (A100 40GB):**
- Model (bfloat16): ~6GB
- Activations: ~8GB
- Optimizer states: ~12GB
- Generation overhead: ~10GB
- **Total: ~36GB**

**Multi-GPU with FSDP:**
- Model sharded across GPUs
- Each GPU: ~15-20GB
- Enables training on smaller GPUs (e.g., 4x A10 24GB)

---

## 🔍 Implementation Details

### A*PO Loss Function

The core innovation is the A*PO loss in `src/astar_po.py`:

```python
def compute_loss(self, prompts, responses, rewards, reference_logprobs):
    """
    A*PO Loss = Policy Gradient Loss + KL Penalty
    
    Policy Loss: -E[r * log π(y|x)]
    KL Penalty: β * KL(π || π_ref) = β * (log π - log π_ref)
    """
    policy_loss = 0
    kl_loss = 0
    
    for response, reward, ref_logprob in zip(responses, rewards, reference_logprobs):
        # Compute current policy log probability
        curr_logprob = self.compute_logprob(response)
        
        # Policy gradient: maximize reward-weighted log prob
        policy_loss += -reward * curr_logprob
        
        # KL divergence: keep close to reference policy
        kl_loss += curr_logprob - ref_logprob
    
    return policy_loss + self.beta * kl_loss
```

### Training Loop

1. **Generate Samples**: For each problem, generate 8 candidate solutions
2. **Compute Rewards**: Validate each solution (1.0 if correct, 0.0 otherwise)
3. **Calculate Reference Log Probs**: Use frozen initial model
4. **Compute Loss**: A*PO loss with KL penalty
5. **Update**: Backprop and optimizer step

### Reward Function

Binary reward with strict validation:
- ✅ Must use all numbers exactly once
- ✅ Must use only +, -, *, /
- ✅ Result must equal target (within 1e-6)
- ❌ Wrong answer = 0.0 reward (no partial credit)

---

## 📝 Minimalism Requirements

✅ **Only PyTorch**: No verl, deepspeed, or other RL frameworks  
✅ **FSDP for multi-GPU**: Using `torch.distributed.fsdp`  
✅ **Single process**: Training runs in one Python process  
✅ **Modal deployment**: Cloud training on modal.com  
✅ **Cost monitoring**: ~$30 budget for full run  

---

## 🐛 Known Issues & Future Work

### Current Limitations
1. Low accuracy on development run (5.5%) due to:
   - Insufficient training data (50 samples)
   - Too low learning rate (5e-7)
   - Limited exploration (2 samples)

2. Generation overhead:
   - Each batch generates 8 samples sequentially
   - Slow on single GPU
   - Could be parallelized

### Planned Improvements
1. **Data Augmentation**: Generate more diverse problems
2. **Curriculum Learning**: Start with easier problems
3. **Batch Generation**: Parallel sampling for speed
4. **Value Function**: Add critic for better credit assignment
5. **Multiplication Task**: Extend to other reasoning tasks

---

## 📚 References

- [DeepSeek R1 Zero Paper](https://arxiv.org/pdf/2501.12948)
- [TinyZero Original Implementation](https://github.com/Jiayi-Pan/TinyZero)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [A* Search Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)

---

## 👥 Team & Contact

**Team Size**: 1-4 members  
**GPU Budget**: $30 on modal.com  
**Course**: INFO-7375 Advanced Machine Learning

For questions or issues, please open a GitHub issue.

---

## 📄 License

MIT License - see LICENSE file for details
