"""
A*PO (A-Star Policy Optimization) Loss Module

Implements the A*PO loss for RAGEN training:
    Loss = -E[A * log π(a|s)] + β * KL(π || π_ref)

Where:
    - A = advantage (reward - baseline)
    - π = current policy
    - π_ref = reference policy (frozen initial model)
    - β = KL penalty coefficient
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


class AStarPO:
    """A*PO optimizer for RAGEN."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        beta: float = 0.05,
        device: str = "cuda",
    ):
        """
        Args:
            model: Current policy model (trainable)
            ref_model: Reference policy model (frozen)
            tokenizer: Tokenizer
            beta: KL divergence penalty coefficient
            device: Device to run on
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
    
    def compute_loss(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute A*PO loss.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings (generated actions)
            rewards: List of rewards for each response
            
        Returns:
            loss: Scalar loss tensor
            stats: Dictionary with loss statistics
        """
        # Tokenize full sequences (prompt + response)
        full_texts = [p + r for p, r in zip(prompts, responses)]
        encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # Get prompt lengths to mask them out
        prompt_encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        prompt_lengths = [len(enc) for enc in prompt_encodings.input_ids]
        
        # Compute log probabilities from current policy
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
            )
            logits = outputs.logits
        
        # Compute log probabilities from reference policy
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                ref_outputs = self.ref_model(
                    input_ids=encodings.input_ids,
                    attention_mask=encodings.attention_mask,
                )
                ref_logits = ref_outputs.logits
        
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Gather log probs for actual tokens
        # Shape: [batch, seq_len-1]
        target_ids = encodings.input_ids[:, 1:]
        gathered_log_probs = torch.gather(
            log_probs[:, :-1, :],
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        
        gathered_ref_log_probs = torch.gather(
            ref_log_probs[:, :-1, :],
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        
        # Create mask: only compute loss on response tokens (not prompt)
        batch_size = encodings.input_ids.shape[0]
        seq_len = encodings.input_ids.shape[1] - 1
        mask = torch.zeros(batch_size, seq_len, device=self.device)
        
        for i, prompt_len in enumerate(prompt_lengths):
            # Mask out prompt tokens, keep response tokens
            mask[i, prompt_len:] = 1.0
        
        # Also mask padding tokens
        attention_mask = encodings.attention_mask[:, 1:].float()
        mask = mask * attention_mask
        
        # Compute per-sequence log probabilities
        # Sum over sequence length, weighted by mask
        seq_log_probs = (gathered_log_probs * mask).sum(dim=1)
        seq_ref_log_probs = (gathered_ref_log_probs * mask).sum(dim=1)
        
        # Normalize by sequence length
        seq_lengths = mask.sum(dim=1).clamp(min=1.0)
        seq_log_probs = seq_log_probs / seq_lengths
        seq_ref_log_probs = seq_ref_log_probs / seq_lengths
        
        # Convert rewards to tensor
        reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        
        # Compute advantages (reward - mean reward)
        mean_reward = reward_tensor.mean()
        advantages = reward_tensor - mean_reward
        
        # Policy gradient loss: -E[A * log π(a|s)]
        policy_loss = -(advantages * seq_log_probs).mean()
        
        # KL divergence: KL(π || π_ref) = log π - log π_ref
        kl_div = (seq_log_probs - seq_ref_log_probs).mean()
        
        # Total A*PO loss
        total_loss = policy_loss + self.beta * kl_div
        
        # Statistics
        stats = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "mean_reward": mean_reward.item(),
            "mean_advantage": advantages.mean().item(),
        }
        
        return total_loss, stats


def compute_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    responses: List[str],
    device: str = "cuda",
) -> List[float]:
    """
    Compute log probabilities for responses given prompts.
    
    Args:
        model: Model to compute log probs with
        tokenizer: Tokenizer
        prompts: List of prompt strings
        responses: List of response strings
        device: Device to run on
        
    Returns:
        List of log probabilities (one per response)
    """
    full_texts = [p + r for p, r in zip(prompts, responses)]
    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    
    prompt_encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    prompt_lengths = [len(enc) for enc in prompt_encodings.input_ids]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
            )
            logits = outputs.logits
    
    log_probs = F.log_softmax(logits, dim=-1)
    target_ids = encodings.input_ids[:, 1:]
    gathered_log_probs = torch.gather(
        log_probs[:, :-1, :],
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    
    # Mask and sum
    batch_size = encodings.input_ids.shape[0]
    seq_len = encodings.input_ids.shape[1] - 1
    mask = torch.zeros(batch_size, seq_len, device=device)
    
    for i, prompt_len in enumerate(prompt_lengths):
        mask[i, prompt_len:] = 1.0
    
    attention_mask = encodings.attention_mask[:, 1:].float()
    mask = mask * attention_mask
    
    seq_log_probs = (gathered_log_probs * mask).sum(dim=1)
    seq_lengths = mask.sum(dim=1).clamp(min=1.0)
    seq_log_probs = seq_log_probs / seq_lengths
    
    return seq_log_probs.cpu().tolist()
