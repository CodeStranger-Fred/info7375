from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)
import random
import re
import numpy as np

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
SEED = 42


# ==================== SIMULATED WEBSHOP ENVIRONMENT ====================

class SimpleWebShopEnv:
    """Simulated WebShop environment with expanded product catalog and multi-turn support."""
    
    def __init__(self):
        """Initialize large product database (50+ items)."""
        # Expanded product catalog to mimic real WebShop
        colors = ["black", "white", "red", "blue", "green", "gray", "yellow", "brown"]
        sizes = ["small", "medium", "large", "xlarge"]
        
        # Explicit mapping: plural → singular for proper matching
        categories_plural = ["shoes", "clothing", "accessories"]
        cat_map = {"shoes": "shoe", "clothing": "clothing", "accessories": "accessory"}
        
        self.products = []
        product_id = 0
        
        # Generate diverse products - include ALL sizes
        for cat_plural in categories_plural:
            cat_singular = cat_map[cat_plural]
            for color in colors:
                for size in sizes:  # All 4 sizes: small, medium, large, xlarge
                    price = random.randint(20, 150)
                    name = f"{color} {size} {cat_singular}"  # Use singular in name
                    self.products.append({
                        "id": product_id,
                        "name": name,
                        "color": color,
                        "size": size,
                        "price": price,
                        "category": cat_singular,  # Store singular for matching
                    })
                    product_id += 1
        
        random.shuffle(self.products)
        # Reassign IDs after shuffle
        for i, p in enumerate(self.products):
            p["id"] = i
    
    def reset(self, target: str) -> str:
        """Reset environment for new episode."""
        self.target = target
        self.current_page = "search"
        self.search_results = []
        self.selected_product = None
        self.done = False
        self.steps = 0
        return f"Welcome to WebShop. You are looking for: {target}. What would you like to do?"
    
    def step(self, action: str) -> Tuple[str, float, bool]:
        """Execute action and return (observation, reward, done).
        
        Reward design (following paper):
        - search/click: 0.0 (no intermediate reward, sparse only)
        - buy correct: 1.0 (all requirements match)
        - buy wrong: 0.0 (any requirement fails)
        """
        self.steps += 1
        
        if self.done:
            return "Episode already finished.", 0.0, True
        
        # Parse action
        if action.startswith("search:"):
            query = action.split(":", 1)[1]
            self.search_results = self.search(query, limit=5)
            obs = f"Search results for '{query}':\n" + "\n".join(self.search_results)
            # No intermediate reward (sparse reward only at purchase)
            return obs, 0.0, False
        
        elif action.startswith("click:"):
            try:
                product_id = int(action.split(":")[1])
                if 0 <= product_id < len(self.products):
                    self.selected_product = self.products[product_id]
                    p = self.selected_product
                    obs = f"Product details: [{p['id']}] {p['name']}\nColor: {p['color']}, Size: {p['size']}, Price: ${p['price']}"
                    # No intermediate reward (sparse reward only at purchase)
                    return obs, 0.0, False
                else:
                    return "Invalid product ID.", 0.0, False
            except:
                return "Invalid click action.", 0.0, False
        
        elif action.startswith("buy:"):
            try:
                product_id = int(action.split(":")[1])
                # Sparse reward: 1.0 if perfect match, 0.0 otherwise
                reward = self.check_purchase(product_id, self.target)
                self.done = True
                return f"Purchased product [{product_id}].", reward, True
            except:
                self.done = True
                return "Invalid purchase.", 0.0, True
        
        else:
            return "Unknown action. Use search:, click:, or buy:", 0.0, False
    
    def search(self, query: str, limit: int = 5) -> List[str]:
        """Search products by query string - simplified matching."""
        query_lower = query.lower().strip()
        results = []
        
        # If empty/generic query, return first N products
        if not query_lower or query_lower == "products":
            for product in self.products[:limit]:
                results.append(self._format_product(product))
            return results
        
        # Otherwise match any term
        query_terms = query_lower.split()
        for product in self.products:
            product_text = f"{product['name']} {product['color']} {product['size']} {product['category']}".lower()
            if any(term in product_text for term in query_terms):
                results.append(self._format_product(product))
                if len(results) >= limit:
                    break
        
        # If still no results, return first N products as fallback
        if not results:
            for product in self.products[:limit]:
                results.append(self._format_product(product))
        
        return results
    
    def _format_product(self, product: Dict) -> str:
        """Format product as readable text."""
        return f"[{product['id']}] {product['name']}: {product['color']}, size {product['size']}, ${product['price']}"
    
    def check_purchase(self, product_id: int, target: str) -> float:
        """Check if purchased product satisfies target. Returns reward 0 or 1 (sparse).
        
        Paper's reward design:
        - 1.0: Product matches ALL requirements (color, size, price, category)
        - 0.0: Product fails ANY requirement
        """
        try:
            product_id = int(product_id)
            if not (0 <= product_id < len(self.products)):
                return 0.0
            
            product = self.products[product_id]
            tgt = target.lower()
            
            # Sparse reward: ALL conditions must be satisfied
            
            # Check color match
            color_ok = product["color"] in tgt
            
            # Check size match
            size_ok = product["size"] in tgt
            
            # Check category: allow both singular and plural forms
            # E.g., product category = "shoe", target can have "shoe" or "shoes"
            cat = product["category"]  # singular
            cat_ok = (cat in tgt) or (cat + "s" in tgt)
            
            # Check price constraint (if mentioned)
            price_ok = True
            if "$" in tgt or "under" in tgt:
                price_match = re.search(r'\$(\d+)', tgt)
                if price_match:
                    max_price = int(price_match.group(1))
                    price_ok = product["price"] <= max_price
            
            # Sparse reward: 1.0 only if ALL match, else 0.0
            return 1.0 if (color_ok and size_ok and cat_ok and price_ok) else 0.0
                
        except:
            return 0.0


# ==================== DATA STRUCTURES ====================

@dataclass
class SamplerCfg:
    """Configuration for generation."""
    max_new_tokens: int = 32  # Short output for single-line action format
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True


@dataclass
class TrainCfg:
    """Training configuration - expanded for realistic benchmark."""
    steps: int = 300  # More steps for CoT learning
    batch_size: int = 2  # Reduced for 3B model memory constraints
    num_samples_per_prompt: int = 2  # K generations per state
    lr: float = 3e-5  # Higher LR for faster learning
    warmup_steps: int = 30
    grad_clip: float = 1.0
    beta_kl: float = 0.05  # KL penalty weight (increased for stability)
    alpha: float = 6.0  # Advantage temperature scaling (critical for sparse rewards)
    num_retrieval_docs: int = 5  # Top-5 docs for larger catalog


# Generation class removed - now using Trajectory class instead


@dataclass
class Trajectory:
    """A complete multi-turn trajectory."""
    trajectory_id: int
    target: str
    
    # Trajectory history
    observations: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    
    # For training
    full_text: str = ""
    token_ids: List[int] = field(default_factory=list)
    
    # Separate rewards: env (0/1 sparse) and reasoning (0-0.2 auxiliary)
    env_reward: Optional[float] = None
    reasoning_score: float = 0.0
    reward: Optional[float] = None  # Keep for compatibility
    
    logprob_policy: Optional[float] = None
    logprob_reference: Optional[float] = None
    advantage: Optional[float] = None
    
    def add_step(self, obs: str, action: str) -> None:
        """Add a step to trajectory."""
        self.observations.append(obs)
        self.actions.append(action)
    
    def set_generation(self, text: str, token_ids: List[int]) -> None:
        """Set the full generated text for this trajectory."""
        self.full_text = text
        self.token_ids = token_ids
    
    def set_reward(self, reward: float) -> None:
        self.reward = reward
    
    def set_logprobs(self, policy: float, reference: float) -> None:
        self.logprob_policy = policy
        self.logprob_reference = reference
    
    def set_advantage(self, advantage: float) -> None:
        self.advantage = advantage


@dataclass
class State:
    """State in the trajectory (current observation)."""
    state_id: int
    observation: str  # Current page/state
    target: str  # Goal
    
    # Multi-turn: now we have multiple trajectory samples
    trajectories: List[Trajectory] = field(default_factory=list)
    
    def add_trajectory(self, target: str) -> Trajectory:
        traj = Trajectory(
            trajectory_id=len(self.trajectories),
            target=target,
        )
        self.trajectories.append(traj)
        return traj
    
    def get_rewards(self) -> List[float]:
        return [t.reward if t.reward is not None else 0.0 for t in self.trajectories]
    
    def set_rewards(self, rewards: List[float]) -> None:
        assert len(rewards) == len(self.trajectories)
        for traj, reward in zip(self.trajectories, rewards):
            traj.set_reward(reward)
    
    def compute_advantages(self, lam: float = 0.4) -> None:
        """Compute normalized advantage with env reward + reasoning bonus.
        
        Args:
            lam: Weight for reasoning score (default 0.4)
        """
        # Collect env rewards and reasoning scores
        envs = []
        reasons = []
        for t in self.trajectories:
            envs.append(0.0 if t.env_reward is None else float(t.env_reward))
            reasons.append(float(t.reasoning_score))
        
        envs = np.array(envs, dtype=np.float32)
        reasons = np.array(reasons, dtype=np.float32)
        
        # Normalize to z-scores (avoiding division by zero)
        def z_normalize(x):
            m = x.mean() if len(x) > 0 else 0.0
            s = x.std() + 1e-8
            return (x - m) / s
        
        z_env = z_normalize(envs)
        z_reason = z_normalize(reasons)
        
        # Combined advantage: normalized env + λ * normalized reasoning
        A = z_env + lam * z_reason
        
        for traj, advantage in zip(self.trajectories, A.tolist()):
            traj.set_advantage(float(advantage))


@dataclass
class Batch:
    """Batch of states."""
    states: List[State] = field(default_factory=list)
    
    def add_state(self, observation: str, target: str) -> State:
        state = State(
            state_id=len(self.states),
            observation=observation,
            target=target,
        )
        self.states.append(state)
        return state
    
    def flatten_trajectories(self) -> Tuple[List[Trajectory], List[int]]:
        """Flatten all trajectories across states."""
        flat_trajs = []
        state_indices = []
        for state in self.states:
            for traj in state.trajectories:
                flat_trajs.append(traj)
                state_indices.append(state.state_id)
        return flat_trajs, state_indices
    
    def get_flattened_token_ids(self) -> List[List[int]]:
        flat_ids = []
        for state in self.states:
            for traj in state.trajectories:
                flat_ids.append(traj.token_ids)
        return flat_ids
    
    def get_flattened_rewards(self) -> List[float]:
        flat_rewards = []
        for state in self.states:
            flat_rewards.extend(state.get_rewards())
        return flat_rewards
    
    def get_flattened_advantages(self) -> List[float]:
        flat_advantages = []
        for state in self.states:
            for traj in state.trajectories:
                adv = traj.advantage if traj.advantage is not None else 0.0
                flat_advantages.append(adv)
        return flat_advantages
    
    def compute_all_advantages(self) -> None:
        for state in self.states:
            state.compute_advantages()
    
    def set_flattened_logprobs(
        self,
        policy_logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor,
    ) -> None:
        idx = 0
        for state in self.states:
            for traj in state.trajectories:
                traj.set_logprobs(
                    policy=policy_logprobs[idx].item(),
                    reference=reference_logprobs[idx].item(),
                )
                idx += 1


# ==================== RETRIEVER ====================
# Retrieval is now handled directly through the environment's search action


# ==================== PROMPT FORMATTING ====================

def format_prompt_multiturn(
    target: str,
    history: List[Tuple[str, str]],  # [(observation, action), ...]
    current_obs: str,
) -> str:
    """Format prompt - simple 2-step workflow."""
    # Parse target for key attributes
    target_lower = target.lower()
    
    # Simple format teaching 2-step workflow
    if not history:
        # Step 1: Search
        prompt = f"""You need: {target}
Action: search:"""
    else:
        # Step 2: Buy from results
        prompt = f"""You need: {target}
{current_obs}
Action: buy:"""
    
    return prompt


def extract_action(text: str) -> str:
    """Extract action from generated text - flexible parsing."""
    text = text.strip()
    
    # Direct format: "search:query" or "buy:id" 
    if text.startswith("search:"):
        query = text.split(":", 1)[1].strip()
        return f"search:{query}" if query else "search:shoe"
    
    if text.startswith("buy:"):
        parts = text.split(":", 1)[1].strip()
        # Extract just the number
        id_match = re.search(r'\d+', parts)
        if id_match:
            return f"buy:{id_match.group()}"
    
    # Look anywhere in text for search: or buy:
    action_match = re.search(r'(search|buy)[:：]\s*([^\s\n,]+)', text, re.IGNORECASE)
    if action_match:
        action_type = action_match.group(1).lower()
        value = action_match.group(2).strip()
        # Clean brackets/quotes
        value = re.sub(r'[\[\]"\']', '', value)
        if action_type == "buy":
            id_match = re.search(r'\d+', value)
            if id_match:
                return f"buy:{id_match.group()}"
        else:
            return f"search:{value}" if value else "search:shoe"
    
    # Fallback: look for product ID anywhere (assume buy)
    id_match = re.search(r'\b(\d{1,2})\b', text)
    if id_match:
        return f"buy:{id_match.group()}"
    
    # Default: search for generic term
    return "search:shoe"


# ==================== LOG PROBABILITY COMPUTATION ====================

def compute_logp_on_sequences(
    model: PreTrainedModel,
    ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Compute log probability for sequences."""
    attn_mask = (ids != pad_id).to(ids.dtype)
    
    with torch.amp.autocast(device_type=DEVICE if DEVICE != "cpu" else "cpu", dtype=DTYPE if DEVICE != "cpu" else torch.float32):
        out = model(input_ids=ids, attention_mask=attn_mask)
        lp = gather_logprobs(out.logits, ids)
    
    mask = (ids[:, 1:] != pad_id).to(lp.dtype)
    return (lp * mask).sum(dim=-1)


def gather_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Convert logits to log probabilities."""
    logprobs = torch.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    gathered = torch.gather(logprobs[:, :-1, :], -1, targets)
    return gathered.squeeze(-1)


def pad_to_tensor(
    token_ids_list: List[List[int]],
    pad_id: int,
    max_len: int,
    device: str,
) -> torch.Tensor:
    """Pad token lists to tensor."""
    batch = []
    for ids in token_ids_list:
        ids = ids[:max_len]
        ids = ids + [pad_id] * (max_len - len(ids))
        batch.append(ids)
    return torch.tensor(batch, device=device, dtype=torch.long)


# ==================== GENERATION ====================

def compute_reasoning_consistency_reward(
    generated_text: str,
    action: str,
    target: str,
) -> float:
    """Compute Reasoning-Action Consistency Reward.
    
    Checks if the model's reasoning (in <think> tags) is consistent with its action.
    
    Returns:
        0.0 to 0.2 bonus based on consistency quality
    """
    # Extract reasoning from <think> tags
    think_match = re.search(r'<think>(.+?)</think>', generated_text, re.DOTALL)
    
    if not think_match:
        # No reasoning provided - no bonus
        return 0.0
    
    reasoning = think_match.group(1).lower()
    target_lower = target.lower()
    action_lower = action.lower()
    
    consistency_score = 0.0
    
    # Check 1: Does reasoning mention the action type?
    if action.startswith("search:"):
        if "search" in reasoning or "find" in reasoning or "look for" in reasoning:
            consistency_score += 0.05
    elif action.startswith("click:"):
        if "click" in reasoning or "view" in reasoning or "details" in reasoning or "check" in reasoning:
            consistency_score += 0.05
    elif action.startswith("buy:"):
        if "buy" in reasoning or "purchase" in reasoning or "correct" in reasoning or "match" in reasoning:
            consistency_score += 0.05
    
    # Check 2: Does reasoning mention target attributes?
    target_attributes = 0
    reasoning_attributes = 0
    
    # Parse target for attributes
    colors = ["black", "white", "red", "blue", "green", "gray", "yellow", "brown"]
    sizes = ["small", "medium", "large", "xlarge"]
    
    for color in colors:
        if color in target_lower:
            target_attributes += 1
            if color in reasoning:
                reasoning_attributes += 1
    
    for size in sizes:
        if size in target_lower:
            target_attributes += 1
            if size in reasoning:
                reasoning_attributes += 1
    
    if "$" in target_lower or "under" in target_lower:
        target_attributes += 1
        if "price" in reasoning or "$" in reasoning or "cost" in reasoning or "under" in reasoning:
            reasoning_attributes += 1
    
    # Award points for mentioning target attributes
    if target_attributes > 0:
        attribute_ratio = reasoning_attributes / target_attributes
        consistency_score += 0.1 * attribute_ratio
    
    # Check 3: Reasoning length (avoid trivial reasoning)
    if len(reasoning.strip()) > 20:
        consistency_score += 0.05
    
    return min(0.2, consistency_score)  # Cap at 0.2


def rollout_trajectory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    env: SimpleWebShopEnv,
    target: str,
    cfg: SamplerCfg,
    max_turns: int = 5,
) -> Trajectory:
    """Rollout a single trajectory with multi-turn interaction.
    
    Returns:
        Trajectory with full interaction history
    """
    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=cfg.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    traj = Trajectory(trajectory_id=0, target=target)
    
    # Reset environment
    obs = env.reset(target)
    
    # Full conversation text for training
    full_text = ""
    all_token_ids = []
    
    # Track reasoning consistency rewards
    reasoning_rewards = []
    
    done = False
    for turn in range(max_turns):
        # Build history from previous steps (correctly aligned)
        # History should contain ALL previous (observation, action) pairs
        history = list(zip(traj.observations, traj.actions))
        
        # Format prompt
        prompt = format_prompt_multiturn(target, history, obs)
        
        # Tokenize and generate
        encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        prompt_len = encoding.input_ids.shape[1]
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=DEVICE if DEVICE != "cpu" else "cpu", 
                                   dtype=DTYPE if DEVICE != "cpu" else torch.float32):
                out_ids = model.generate(
                    input_ids=encoding.input_ids,
                    attention_mask=encoding.attention_mask,
                    generation_config=gen_cfg,
                )
        
        # Decode ONLY the generated part (not the prompt)
        generated_ids = out_ids[0][prompt_len:]  # Skip prompt tokens
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # For training, we need the full sequence
        full_sequence_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        
        # Extract action from generated text
        action = extract_action(generated_text)
        
        # Compute reasoning consistency reward for this turn
        reasoning_reward = compute_reasoning_consistency_reward(generated_text, action, target)
        reasoning_rewards.append(reasoning_reward)
        
        # Accumulate ONLY generated text for training (not prompt)
        # This saves memory and we only want to optimize the generated part
        full_text += generated_text + "\n"
        all_token_ids.extend(generated_ids.tolist())
        
        # Step environment FIRST to get next observation
        new_obs, env_reward, done = env.step(action)
        
        # NOW add the step to trajectory (current_obs, action that was taken)
        traj.add_step(obs, action)
        
        # Update observation for next iteration
        obs = new_obs
        
        if done:
            # Final observation after terminal action
            traj.add_step(obs, "")
            
            # Store rewards separately (DO NOT mix them)
            # env_reward: 0.0 or 1.0 (sparse from environment)
            # reasoning_score: 0.0 to 0.2 (average consistency over turns)
            traj.env_reward = float(env_reward)
            traj.reasoning_score = float(sum(reasoning_rewards) / max(1, len(reasoning_rewards)))
            
            # Set reward field to env_reward only (for compatibility)
            traj.set_reward(traj.env_reward)
            break
    
    # If max turns reached without buying, sparse reward = 0.0 (failure)
    if not done:
        traj.env_reward = 0.0
        traj.reasoning_score = float(sum(reasoning_rewards) / max(1, len(reasoning_rewards)))
        traj.set_reward(traj.env_reward)
    
    # Set generation data for training
    traj.set_generation(full_text, all_token_ids)
    
    return traj


def generate_trajectories(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    env: SimpleWebShopEnv,
    batch: Batch,
    k: int,
    cfg: SamplerCfg,
    max_turns: int = 5,
):
    """Generate K trajectory samples per state."""
    for state in batch.states:
        for _ in range(k):
            traj = rollout_trajectory(
                model, tokenizer, env, state.target, cfg, max_turns
            )
            state.trajectories.append(traj)


# ==================== REWARD COMPUTATION ====================
# Rewards are now computed during trajectory rollout in the environment


# ==================== EVALUATION ====================

def generate_random_target(env: SimpleWebShopEnv) -> str:
    """Generate a random shopping target."""
    color = random.choice([p["color"] for p in env.products])
    size = random.choice([p["size"] for p in env.products])
    price_limit = random.randint(30, 150)
    category = random.choice(["shoe", "clothing", "accessory"])
    return f"{color} {size} {category} under ${price_limit}"


def generate_fixed_eval_prompts(num_prompts: int = 256) -> List[str]:
    """Generate 256 fixed evaluation prompts (following paper).
    
    Paper: "They evaluate on 256 fixed prompts per environment, 
            truncating episodes after 5 turns."
    
    Returns:
        List of 256 fixed evaluation prompts with reproducible seed
    """
    eval_seed = 42  # Fixed seed for reproducible eval set
    random.seed(eval_seed)
    
    prompts = []
    colors = ["black", "white", "red", "blue", "green", "gray", "yellow", "brown"]
    sizes = ["small", "medium", "large", "xlarge"]
    categories = ["shoe", "clothing", "accessory"]
    price_limits = [40, 50, 60, 70, 80, 90, 100, 120]
    
    # Generate diverse combinations
    for i in range(num_prompts):
        color = random.choice(colors)
        size = random.choice(sizes)
        category = random.choice(categories)
        price = random.choice(price_limits)
        prompts.append(f"{color} {size} {category} under ${price}")
    
    # Reset random seed after generating eval set
    random.seed(SEED)
    
    return prompts


def evaluate(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    env: SimpleWebShopEnv,
    n_cases: int = 20,
    sampler_cfg: Optional[SamplerCfg] = None,
    max_turns: int = 5,
    verbose: bool = False,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Evaluate success rate on UNSEEN targets with multi-turn interaction.
    
    Returns:
        success_rate: Fraction of successful purchases (reward > 0.8)
        results: List of detailed results for each test case
    """
    if sampler_cfg is None:
        sampler_cfg = SamplerCfg()
    
    policy.eval()
    hits = 0.0
    results = []
    
    # Use completely NEW targets not seen during training
    eval_targets = [
        "green small shoe under $70",
        "yellow xlarge clothing under $100",
        "gray medium accessory under $50",
        "brown large shoe under $80",
        "white small clothing under $40",
    ]
    
    with torch.no_grad():
        for i in range(n_cases):
            # Mix of predefined eval targets and random new ones
            if i < len(eval_targets):
                target = eval_targets[i % len(eval_targets)]
            else:
                target = generate_random_target(env)
            
            # Rollout a trajectory
            traj = rollout_trajectory(policy, tokenizer, env, target, sampler_cfg, max_turns)
            
            reward = traj.reward if traj.reward else 0.0
            success = reward > 0.8
            
            if success:
                hits += 1.0
            
            # Extract product info if final action was buy
            purchased_product = None
            final_action = traj.actions[-1] if traj.actions else ""
            if final_action.startswith("buy:"):
                try:
                    product_id = int(final_action.split(":")[1])
                    if 0 <= product_id < len(env.products):
                        purchased_product = env.products[product_id]
                except:
                    pass
            
            # Store detailed result
            result = {
                "case_id": i,
                "target": target,
                "trajectory": traj,
                "num_turns": len(traj.actions),
                "final_action": final_action,
                "reward": reward,
                "success": success,
                "purchased_product": purchased_product,
            }
            results.append(result)
            
            if verbose and not success:
                print(f"\n[FAILURE Case {i}]")
                print(f"Target: {target}")
                print(f"Turns: {len(traj.actions)}")
                print(f"Actions: {' -> '.join(traj.actions)}")
                if purchased_product:
                    print(f"Purchased: {purchased_product['name']} (${purchased_product['price']})")
                print(f"Reward: {reward:.3f}")
    
    policy.train()
    return hits / n_cases if n_cases > 0 else 0.0, results


def evaluate_paper(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    env: SimpleWebShopEnv,
    sampler_cfg: Optional[SamplerCfg] = None,
    max_turns: int = 5,
    verbose: bool = False,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Evaluate using paper setting: 256 fixed prompts, truncate after 5 turns.
    
    Paper: "They evaluate on 256 fixed prompts per environment, 
            truncating episodes after 5 turns."
    
    Returns:
        success_rate: Fraction of successful purchases (reward > 0.8)
        results: List of detailed results for all 256 test cases
    """
    if sampler_cfg is None:
        sampler_cfg = SamplerCfg()
    
    policy.eval()
    hits = 0.0
    results = []
    
    # Generate 256 fixed evaluation prompts
    eval_targets = generate_fixed_eval_prompts(256)
    n_cases = len(eval_targets)
    
    print(f"\nEvaluating on {n_cases} fixed prompts (paper setting)...")
    
    with torch.no_grad():
        for i, target in enumerate(eval_targets):
            # Rollout trajectory (max 5 turns as per paper)
            traj = rollout_trajectory(policy, tokenizer, env, target, sampler_cfg, max_turns)
            
            reward = traj.reward if traj.reward else 0.0
            success = reward > 0.8
            
            if success:
                hits += 1.0
            
            # Extract product info if final action was buy
            purchased_product = None
            final_action = traj.actions[-1] if traj.actions else ""
            if final_action.startswith("buy:"):
                try:
                    product_id = int(final_action.split(":")[1])
                    if 0 <= product_id < len(env.products):
                        purchased_product = env.products[product_id]
                except:
                    pass
            
            # Store detailed result
            result = {
                "case_id": i,
                "target": target,
                "trajectory": traj,
                "num_turns": len(traj.actions),
                "final_action": final_action,
                "reward": reward,
                "success": success,
                "purchased_product": purchased_product,
            }
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                current_sr = hits / (i + 1)
                print(f"  Progress: {i+1}/{n_cases} | Success Rate: {current_sr:.3f}")
            
            if verbose and not success:
                print(f"\n[FAILURE Case {i}]")
                print(f"Target: {target}")
                print(f"Turns: {len(traj.actions)}")
                print(f"Actions: {' -> '.join(traj.actions)}")
                if purchased_product:
                    print(f"Purchased: {purchased_product['name']} (${purchased_product['price']})")
                print(f"Reward: {reward:.3f}")
    
    policy.train()
    success_rate = hits / n_cases if n_cases > 0 else 0.0
    print(f"\nFinal Success Rate: {success_rate:.3f} ({int(hits)}/{n_cases})")
    
    return success_rate, results


# ==================== A*PO LOSS ====================

def apo_step(
    policy: PreTrainedModel,
    ref: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: Batch,
    beta_kl: float,
    alpha: float = 6.0,
) -> torch.Tensor:
    """A*PO (A-Star Policy Optimization) step with advantage temperature.
    
    Loss = -E[α * A * log π(a|s)] + β * KL(π || π_ref)
    where A = normalized(r - baseline)
    
    Args:
        alpha: Advantage temperature scaling (6.0 for sparse rewards)
        beta_kl: KL penalty weight (0.05 recommended)
    """
    
    # Flatten all token IDs
    flat_ids = batch.get_flattened_token_ids()
    
    # Pad to tensor
    max_ctx = getattr(policy.config, "max_position_embeddings", 4096)
    pad_id = tokenizer.pad_token_id
    max_len = min(max(len(s) for s in flat_ids), max_ctx)
    
    batch_tensor = pad_to_tensor(flat_ids, pad_id=pad_id, max_len=max_len, device=DEVICE)
    
    # Compute log probs
    logp_pol = compute_logp_on_sequences(policy, batch_tensor, pad_id)
    with torch.no_grad():
        logp_ref = compute_logp_on_sequences(ref, batch_tensor, pad_id)
    
    # Store logprobs
    batch.set_flattened_logprobs(logp_pol, logp_ref)
    
    # Compute advantages (now normalized with reasoning bonus)
    batch.compute_all_advantages()
    
    # Get flattened advantages
    flat_advantages = torch.tensor(
        batch.get_flattened_advantages(),
        device=DEVICE,
        dtype=logp_pol.dtype,
    )
    
    # A*PO loss with advantage temperature
    kl_term = logp_pol - logp_ref
    loss = (-alpha * flat_advantages * logp_pol + beta_kl * kl_term).mean()
    
    return loss


# ==================== BEHAVIOR CLONING (BC) ====================

def generate_expert_demonstration(
    env: SimpleWebShopEnv,
    target: str,
) -> Tuple[str, List[str], float]:
    """Generate one expert demonstration - simple 2-step.
    
    Returns:
        (full_text, actions, reward)
    """
    # Find the BEST matching product directly
    best_id = None
    best_reward = 0.0
    
    for product in env.products:
        pid = product["id"]
        reward = env.check_purchase(pid, target)
        if reward > best_reward:
            best_reward = reward
            best_id = pid
    
    if best_id is None or best_reward < 1.0:
        # No perfect match - fail this demo
        return "", [], 0.0
    
    # Generate simple 2-step demonstration
    # Step 1: Search for target attributes
    target_lower = target.lower()
    colors = ["black", "white", "red", "blue", "green", "gray", "yellow", "brown"]
    
    search_term = "shoe"  # default
    for color in colors:
        if color in target_lower:
            search_term = color
            break
    
    actions = [
        f"search:{search_term}",
        f"buy:{best_id}"
    ]
    
    # Format as text matching the prompt format
    full_text = f"""You need: {target}
Action: search:{search_term}
You need: {target}
Search results...
Action: buy:{best_id}"""
    
    return full_text, actions, best_reward


def generate_bc_dataset(
    env: SimpleWebShopEnv,
    n_demos: int = 100,
) -> List[Dict]:
    """Generate BC training dataset with expert demonstrations.
    
    Returns:
        List of {"text": str, "target": str, "reward": float}
    """
    print(f"\nGenerating {n_demos} expert demonstrations for BC pretraining...")
    
    # Generate diverse targets
    colors = ["black", "white", "red", "blue", "green", "gray", "yellow", "brown"]
    sizes = ["small", "medium", "large", "xlarge"]
    categories = ["shoe", "clothing", "accessory"]
    prices = [40, 50, 60, 70, 80, 90, 100]
    
    demos = []
    for i in range(n_demos):
        color = random.choice(colors)
        size = random.choice(sizes)
        category = random.choice(categories)
        price = random.choice(prices)
        target = f"{color} {size} {category} under ${price}"
        
        text, actions, reward = generate_expert_demonstration(env, target)
        demos.append({
            "text": text,
            "target": target,
            "actions": actions,
            "reward": reward,
        })
        
        if (i + 1) % 20 == 0:
            avg_reward = np.mean([d["reward"] for d in demos[-20:]])
            print(f"  Generated {i+1}/{n_demos} demos | Avg reward: {avg_reward:.3f}")
    
    # Keep only PERFECT demonstrations (reward == 1.0)
    perfect_demos = [d for d in demos if d["reward"] >= 1.0]
    print(f"  Kept {len(perfect_demos)}/{n_demos} demonstrations (reward = 1.0)\n")
    
    # Need at least 30 perfect demos for BC
    if len(perfect_demos) < 30:
        print(f"  ERROR: Only {len(perfect_demos)} perfect demos found, need at least 30!")
        print(f"  Generating more...\n")
        # Generate more until we have 30
        while len(perfect_demos) < 30:
            color = random.choice(["black", "white", "red", "blue", "green", "gray", "yellow", "brown"])
            size = random.choice(["small", "medium", "large", "xlarge"])
            category = random.choice(["shoe", "clothing", "accessory"])
            price = random.choice([40, 50, 60, 70, 80, 90, 100])
            target = f"{color} {size} {category} under ${price}"
            text, actions, reward = generate_expert_demonstration(env, target)
            if reward >= 1.0:
                perfect_demos.append({
                    "text": text,
                    "target": target,
                    "actions": actions,
                    "reward": reward,
                })
        print(f"  Generated {len(perfect_demos)} perfect demos\n")
    
    return perfect_demos


def train_bc(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    bc_dataset: List[Dict],
    epochs: int = 3,
    lr: float = 1e-5,
):
    """Behavior cloning pretraining.
    
    Train policy to imitate expert demonstrations using supervised learning.
    """
    print(f"\nBC Pretraining: {len(bc_dataset)} demos, {epochs} epochs")
    print("="*70)
    
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    
    for epoch in range(epochs):
        random.shuffle(bc_dataset)
        total_loss = 0.0
        
        for i, demo in enumerate(bc_dataset):
            # Tokenize demonstration text
            inputs = tokenizer(
                demo["text"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(DEVICE)
            
            # Forward pass
            with torch.amp.autocast(device_type=DEVICE if DEVICE != "cpu" else "cpu",
                                   dtype=DTYPE if DEVICE != "cpu" else torch.float32):
                outputs = policy(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(bc_dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    print("BC Pretraining complete!\n")


# ==================== TRAINING ====================

def load_models_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
    """Load policy, reference, and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    
    ref = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    ref.requires_grad_(False)
    
    return policy, ref, tokenizer


def print_benchmark_table(results: List[Dict[str, Any]]) -> None:
    """Print benchmark results as a formatted table."""
    print("\n" + "="*90)
    print("BENCHMARK RESULTS")
    print("="*90)
    
    # Calculate metrics
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    success_rate = successes / total if total > 0 else 0.0
    avg_reward = np.mean([r["reward"] for r in results])
    
    print(f"\nOverall Metrics:")
    print(f"  Total Cases: {total}")
    print(f"  Successes: {successes}")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Average Reward: {avg_reward:.3f}")
    
    # Detailed table
    print(f"\n{'ID':<4} {'Target':<35} {'Action':<15} {'Reward':<8} {'Success'}")
    print("-" * 90)
    
    for r in results[:20]:  # Show first 20
        target_short = r["target"][:33] + ".." if len(r["target"]) > 35 else r["target"]
        action_short = r["action"][:13] + ".." if len(r["action"]) > 15 else r["action"]
        success_mark = "✓" if r["success"] else "✗"
        print(f"{r['case_id']:<4} {target_short:<35} {action_short:<15} {r['reward']:<8.3f} {success_mark}")
    
    print("="*90)


def print_failure_analysis(results: List[Dict[str, Any]], env: SimpleWebShopEnv, n_failures: int = 5) -> None:
    """Print detailed analysis of failure cases."""
    failures = [r for r in results if not r["success"]]
    
    print("\n" + "="*90)
    print(f"FAILURE ANALYSIS (Showing {min(n_failures, len(failures))} of {len(failures)} failures)")
    print("="*90)
    
    for i, fail in enumerate(failures[:n_failures]):
        print(f"\n[Failure {i+1}] Case ID: {fail['case_id']}")
        print(f"Target: {fail['target']}")
        print(f"Action Taken: {fail['action']}")
        print(f"Reward: {fail['reward']:.3f}")
        
        # Show what was purchased
        if fail["purchased_product"]:
            p = fail["purchased_product"]
            print(f"Purchased Product: [{p['id']}] {p['name']} - ${p['price']}")
            print(f"  Color: {p['color']}, Size: {p['size']}")
            
            # Analyze mismatch
            target_lower = fail["target"].lower()
            mismatches = []
            if p["color"] not in target_lower:
                mismatches.append(f"color mismatch (wanted: target color, got: {p['color']})")
            if p["size"] not in target_lower:
                mismatches.append(f"size mismatch (wanted: target size, got: {p['size']})")
            if "$" in target_lower:
                price_match = re.search(r'\$(\d+)', target_lower)
                if price_match:
                    max_price = int(price_match.group(1))
                    if p["price"] > max_price:
                        mismatches.append(f"price too high (wanted: <=${max_price}, got: ${p['price']})")
            
            if mismatches:
                print(f"  Issues: {'; '.join(mismatches)}")
        else:
            print(f"No product purchased or invalid action")
        
        # Show generated reasoning
        print(f"\nGenerated Text (first 200 chars):")
        print(f"  {fail['generated_text'][:200]}...")
        
        # Analyze reasoning
        has_thinking = "<think>" in fail["generated_text"] and "</think>" in fail["generated_text"]
        print(f"\nAnalysis:")
        if not has_thinking:
            print(f"  - Model did NOT use chain-of-thought reasoning")
        else:
            think_match = re.search(r'<think>(.+?)</think>', fail["generated_text"], re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()
                print(f"  - Model used reasoning: {thinking[:100]}...")
        
        if not fail["action"].startswith("buy:"):
            print(f"  - Model chose '{fail['action'].split(':')[0]}' instead of 'buy'")
        
        print("-" * 90)
    
    print("\n")


def train(
    policy: PreTrainedModel,
    ref: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    env: SimpleWebShopEnv,
):
    """Main training loop with multi-turn trajectories."""
    policy.train()
    
    # Config
    train_cfg = TrainCfg()
    sampler_cfg = SamplerCfg()
    
    # Optimizer
    opt = torch.optim.AdamW(
        policy.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    
    # Learning rate schedule
    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / max(1, train_cfg.warmup_steps))
    
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    
    # Training loop
    print("Starting RAGEN multi-turn training with A*PO...")
    print(f"Total products in catalog: {len(env.products)}")
    print(f"Training steps: {train_cfg.steps} | Batch size: {train_cfg.batch_size}")
    
    # Print sample of available products
    print("\n" + "="*70)
    print("SAMPLE OF AVAILABLE PRODUCTS (first 10):")
    print("-" * 70)
    for i, p in enumerate(env.products[:10]):
        print(f"  [{p['id']:2d}] {p['name']:25s} | ${p['price']:3d} | {p['color']:8s} | size {p['size']:7s}")
    print(f"  ... and {len(env.products) - 10} more products")
    print("="*70)
    
    # TRAINING targets (expanded to 50+ diverse targets)
    # Generate diverse training targets
    colors = ["black", "white", "red", "blue", "green", "gray", "yellow", "brown"]
    sizes = ["small", "medium", "large", "xlarge"]
    categories = ["shoe", "clothing", "accessory"]
    prices = [40, 50, 60, 70, 80, 90, 100, 120]
    
    train_targets = []
    for color in colors[:4]:  # 4 colors
        for size in sizes[:3]:  # 3 sizes
            for category in categories:  # 3 categories
                price = random.choice(prices)
                train_targets.append(f"{color} {size} {category} under ${price}")
    
    # Shuffle for variety
    random.shuffle(train_targets)
    train_targets = train_targets[:50]  # Keep 50 targets
    
    print(f"\nTRAINING TARGETS: {len(train_targets)} diverse targets")
    print("Sample targets:")
    for i, target in enumerate(train_targets[:5], 1):
        print(f"  {i}. {target}")
    print(f"  ... and {len(train_targets) - 5} more")
    print("="*70 + "\n")
    
    for step in range(train_cfg.steps):
        # Create batch with random states using TRAINING targets
        batch = Batch()
        
        for _ in range(train_cfg.batch_size):
            target = random.choice(train_targets)
            observation = f"Shopping for: {target}"
            batch.add_state(observation, target)
        
        # Generate multi-turn trajectories
        generate_trajectories(
            policy,
            tokenizer,
            env,
            batch,
            k=train_cfg.num_samples_per_prompt,
            cfg=sampler_cfg,
            max_turns=5,
        )
        
        # A*PO step (rewards already computed in trajectories)
        loss = apo_step(policy, ref, tokenizer, batch, beta_kl=train_cfg.beta_kl, alpha=train_cfg.alpha)
        
        # Backward pass
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=train_cfg.grad_clip)
        opt.step()
        sched.step()
        
        # Logging
        if (step + 1) % 10 == 0:
            rewards_list = batch.get_flattened_rewards()
            avg_reward = np.mean(rewards_list)
            
            # Debug: print detailed info about first trajectory
            sample_state = batch.states[0]
            sample_traj = sample_state.trajectories[0]
            
            print(f"\n{'='*70}")
            print(f"[DEBUG STEP {step+1}]")
            print(f"Target: {sample_state.target}")
            print(f"Actions: {sample_traj.actions}")
            print(f"Rewards in batch: {rewards_list[:4]}")
            
            # Check if bought something (skip last empty action if present)
            actions_non_empty = [a for a in sample_traj.actions if a]
            final_action = actions_non_empty[-1] if actions_non_empty else ""
            if final_action.startswith("buy:"):
                try:
                    product_id = int(final_action.split(":")[1])
                    if 0 <= product_id < len(env.products):
                        p = env.products[product_id]
                        print(f"Purchased: [{p['id']}] {p['name']} - ${p['price']}")
                        print(f"  Product attrs: color={p['color']}, size={p['size']}, category={p['category']}")
                        
                        # Manual check
                        target_lower = sample_state.target.lower()
                        color_match = p['color'] in target_lower
                        size_match = p['size'] in target_lower
                        cat = p['category']
                        cat_match = (cat in target_lower) or (cat + 's' in target_lower)
                        print(f"  Match check: color={color_match}, size={size_match}, category={cat_match}")
                except:
                    print(f"Invalid buy action: {final_action}")
            else:
                print(f"No buy action - final action was: {final_action}")
            
            print(f"Env reward: {sample_traj.env_reward}, Reasoning score: {sample_traj.reasoning_score}")
            print(f"{'='*70}")
            
            success_rate, _ = evaluate(policy, tokenizer, env, n_cases=10, sampler_cfg=sampler_cfg)
            print(
                f"\nStep {step+1:3d} | Loss: {loss.item():.4f} | "
                f"Avg Reward: {avg_reward:.3f} | Success@1: {success_rate:.3f}"
            )
            
            # Print detailed sample trajectory every 10 steps
            print("\n" + "-"*70)
            print("MODEL TRAJECTORY EXAMPLE:")
            sample_state = batch.states[0]
            sample_traj = sample_state.trajectories[0]
            
            print(f"\n[Target]: {sample_state.target}")
            print(f"[Turns]: {len(sample_traj.actions)}")
            print(f"[Reward]: {sample_traj.reward:.3f}")
            
            print(f"\n[Interaction History]:")
            for i, (obs, action) in enumerate(zip(sample_traj.observations, sample_traj.actions)):
                if action:  # Skip empty final action
                    print(f"  Turn {i+1}: {action} -> {obs[:100]}...")
                    
            print("-"*70)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Entry point with BC pretraining + A*PO."""
    set_seed(SEED)
    
    # Load models
    policy, ref, tokenizer = load_models_and_tokenizer()
    
    # Initialize environment
    env = SimpleWebShopEnv()
    
    # Config
    sampler_cfg = SamplerCfg()
    
    # Step 1: BC Pretraining (teach model the basic workflow)
    print("\n" + "#"*90)
    print("# STEP 1: BEHAVIOR CLONING (BC) PRETRAINING")
    print("#"*90)
    bc_dataset = generate_bc_dataset(env, n_demos=100)
    train_bc(policy, tokenizer, bc_dataset, epochs=3, lr=1e-5)
    
    # Step 2: A*PO Reinforcement Learning
    print("\n" + "#"*90)
    print("# STEP 2: A*PO REINFORCEMENT LEARNING")
    print("#"*90)
    train(policy, ref, tokenizer, env)
    
    # Final evaluation with benchmark table (Paper setting: 256 fixed prompts)
    print("\n" + "#"*90)
    print("# FINAL EVALUATION (Paper Setting: 256 Fixed Prompts)")
    print("#"*90)
    
    # Use paper evaluation: 256 fixed prompts, max 5 turns
    success_rate, results = evaluate_paper(
        policy, 
        tokenizer, 
        env,
        sampler_cfg=sampler_cfg,
        max_turns=5,
        verbose=False
    )
    
    # Print benchmark table
    print_benchmark_table(results)
    
    print("\n" + "="*90)
    print("TRAINING COMPLETE")
    print(f"Final Success Rate: {success_rate:.2%} on 256 fixed evaluation prompts")
    print(f"Total successes: {int(success_rate * 256)}/256")
    print("="*90)


if __name__ == "__main__":
    main()
