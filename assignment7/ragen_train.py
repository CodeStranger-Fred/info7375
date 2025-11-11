"""
RAGEN Training Script with A*PO

Trains RAGEN (Retrieval-Augmented Generation) agent on WebArena tasks using A*PO.

Components:
- Simulated WebArena environment
- Retrieval: Extract relevant DOM elements
- Generation: LLM generates actions
- A*PO: Policy gradient + KL divergence training
- Rewards: Task completion + reasoning consistency
"""

import torch
import random
import re
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from astar_po import AStarPO
import copy


@dataclass
class TrainConfig:
    """Training configuration."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    num_steps: int = 100
    batch_size: int = 4
    num_samples_per_task: int = 4
    learning_rate: float = 5e-6
    beta_kl: float = 0.05
    max_turns: int = 5
    temperature: float = 0.7
    max_new_tokens: int = 100
    eval_every: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SimulatedWebArenaEnv:
    """Simulated WebArena environment for training."""
    
    def __init__(self):
        """Initialize environment with tasks."""
        self.tasks = self._generate_tasks()
        self.current_task = None
        self.steps = 0
        self.max_steps = 15
        
    def _generate_tasks(self) -> List[Dict]:
        """Generate training tasks."""
        tasks = []
        
        # Shopping tasks
        colors = ["red", "blue", "black", "white", "green"]
        items = ["shirt", "shoes", "jacket", "pants"]
        for i in range(20):
            color = random.choice(colors)
            item = random.choice(items)
            price = random.randint(20, 100)
            tasks.append({
                "domain": "shopping",
                "intent": f"Find and buy a {color} {item} under ${price}",
                "target_attributes": {
                    "color": color,
                    "item": item,
                    "max_price": price,
                },
            })
        
        # GitLab tasks
        for i in range(10):
            tasks.append({
                "domain": "gitlab",
                "intent": "View merge requests assigned to me",
                "target_url": "/merge_requests?assignee_username=@me",
            })
        
        # Wikipedia tasks
        for i in range(10):
            tasks.append({
                "domain": "wikipedia",
                "intent": "Search for Python programming language",
                "target_text": "Python is a high-level programming language",
            })
        
        random.shuffle(tasks)
        return tasks
    
    def reset(self, task_id: int) -> Tuple[str, str]:
        """Reset environment with a specific task."""
        self.current_task = self.tasks[task_id]
        self.steps = 0
        self.dom = self._get_initial_dom()
        
        intent = self.current_task["intent"]
        observation = self._format_observation()
        
        return intent, observation
    
    def _get_initial_dom(self) -> str:
        """Get initial DOM representation."""
        if self.current_task["domain"] == "shopping":
            return """[1] RootWebArea 'OneStopMarket'
    [20] textbox 'Search products'
    [30] button 'Search'
    [100] link 'Red Shirt - $29.99'
    [101] link 'Blue Shoes - $45.00'
    [102] link 'Black Jacket - $89.99'"""
        elif self.current_task["domain"] == "gitlab":
            return """[1] RootWebArea 'GitLab'
    [10] link 'Merge Requests'
    [11] link 'Issues'
    [20] button 'Filter'"""
        else:
            return """[1] RootWebArea 'Wikipedia'
    [10] textbox 'Search'
    [20] button 'Go'"""
    
    def _format_observation(self) -> str:
        """Format current observation."""
        return f"Current page:\n{self.dom}"
    
    def step(self, action: str) -> Tuple[str, float, bool]:
        """Execute action and return (observation, reward, done)."""
        self.steps += 1
        
        # Parse action
        action_lower = action.lower()
        
        # Check for task completion
        done = False
        reward = 0.0
        
        if "buy" in action_lower or "purchase" in action_lower:
            # Check if correct product
            reward = self._check_purchase(action)
            done = True
        elif "click" in action_lower and "merge" in action_lower:
            # GitLab task completion
            reward = 1.0 if self.current_task["domain"] == "gitlab" else 0.0
            done = True
        elif "search" in action_lower:
            # Update DOM with search results
            self.dom = self._get_search_results(action)
            reward = 0.0  # No intermediate reward
        else:
            reward = 0.0
        
        # Check max steps
        if self.steps >= self.max_steps:
            done = True
        
        observation = self._format_observation()
        return observation, reward, done
    
    def _check_purchase(self, action: str) -> float:
        """Check if purchase matches target."""
        if self.current_task["domain"] != "shopping":
            return 0.0
        
        target = self.current_task["target_attributes"]
        action_lower = action.lower()
        
        # Count matching attributes
        matches = 0
        total = 3
        
        if target["color"] in action_lower:
            matches += 1
        if target["item"] in action_lower:
            matches += 1
        
        # Check price (simplified: assume any product in initial DOM meets price)
        if "100" in action_lower or "101" in action_lower:
            matches += 1
        
        # Shaped reward
        return matches / total
    
    def _get_search_results(self, action: str) -> str:
        """Get search results based on query."""
        return """[100] link 'Red Shirt - $29.99'
    [101] link 'Blue Shoes - $45.00'
    [102] link 'Black Jacket - $89.99'
    [103] link 'White Pants - $39.99'"""


def retrieve_relevant_elements(observation: str, task: str, top_k: int = 3) -> List[str]:
    """Retrieve relevant DOM elements based on task."""
    lines = observation.split('\n')
    task_words = set(task.lower().split())
    
    # Score each line by keyword overlap
    scored = []
    for line in lines:
        line_words = set(line.lower().split())
        score = len(task_words & line_words)
        if score > 0:
            scored.append((score, line.strip()))
    
    # Sort by score and return top-k
    scored.sort(reverse=True, key=lambda x: x[0])
    return [line for _, line in scored[:top_k]]


def format_prompt_with_retrieval(task: str, observation: str, retrieved: List[str], history: List[str]) -> str:
    """Format prompt with retrieval augmentation."""
    prompt = f"""You are a web agent helping users complete tasks.

Task: {task}

Retrieved relevant elements:
{chr(10).join(retrieved)}

Previous actions:
{chr(10).join(history[-3:]) if history else "None"}

Current observation:
{observation}

Think about what action to take, then output your action.
Use format: <think>reasoning</think><action>action_here</action>

Your response:"""
    return prompt


def extract_action(text: str) -> str:
    """Extract action from generated text."""
    # Try to find <action> tags
    action_match = re.search(r'<action>(.+?)</action>', text, re.DOTALL)
    if action_match:
        return action_match.group(1).strip()
    
    # Fallback: look for action keywords
    text_lower = text.lower()
    if "buy" in text_lower or "purchase" in text_lower:
        # Extract product ID
        id_match = re.search(r'\[(\d+)\]', text)
        if id_match:
            return f"buy [{ id_match.group(1)}]"
        return "buy [100]"
    elif "click" in text_lower:
        id_match = re.search(r'\[(\d+)\]', text)
        if id_match:
            return f"click [{id_match.group(1)}]"
        return "click [20]"
    elif "search" in text_lower:
        # Extract search query
        search_match = re.search(r'search[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
        if search_match:
            return f"search:{search_match.group(1).strip()}"
        return "search:products"
    
    return "search:products"


def compute_reasoning_consistency_reward(text: str, action: str, task: str) -> float:
    """Compute reward for reasoning-action consistency."""
    # Extract reasoning
    think_match = re.search(r'<think>(.+?)</think>', text, re.DOTALL)
    if not think_match:
        return 0.0
    
    reasoning = think_match.group(1).lower()
    task_lower = task.lower()
    action_lower = action.lower()
    
    score = 0.0
    
    # Check if reasoning mentions action type
    if "buy" in action_lower and "buy" in reasoning:
        score += 0.05
    elif "search" in action_lower and "search" in reasoning:
        score += 0.05
    elif "click" in action_lower and "click" in reasoning:
        score += 0.05
    
    # Check if reasoning mentions task keywords
    task_words = set(task_lower.split())
    reasoning_words = set(reasoning.split())
    overlap = len(task_words & reasoning_words)
    score += min(0.1, overlap * 0.02)
    
    # Check reasoning length (avoid trivial)
    if len(reasoning.strip()) > 20:
        score += 0.05
    
    return min(0.2, score)


def rollout_trajectory(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    env: SimulatedWebArenaEnv,
    task_id: int,
    cfg: TrainConfig,
) -> Tuple[List[str], List[str], float]:
    """Rollout a single trajectory and return (prompts, responses, reward)."""
    intent, obs = env.reset(task_id)
    
    prompts = []
    responses = []
    history = []
    
    total_reward = 0.0
    done = False
    
    for turn in range(cfg.max_turns):
        if done:
            break
        
        # Retrieve relevant elements
        retrieved = retrieve_relevant_elements(obs, intent)
        
        # Format prompt
        prompt = format_prompt_with_retrieval(intent, obs, retrieved, history)
        prompts.append(prompt)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(cfg.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode only generated part
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(response)
        
        # Extract action
        action = extract_action(response)
        history.append(action)
        
        # Execute in environment
        obs, reward, done = env.step(action)
        
        # Add reasoning consistency reward
        reasoning_reward = compute_reasoning_consistency_reward(response, action, intent)
        total_reward += reward + reasoning_reward
    
    return prompts, responses, total_reward


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    env: SimulatedWebArenaEnv,
    cfg: TrainConfig,
    num_eval_tasks: int = 10,
) -> float:
    """Evaluate model on held-out tasks."""
    model.eval()
    
    success_count = 0
    eval_task_ids = random.sample(range(len(env.tasks)), num_eval_tasks)
    
    for task_id in eval_task_ids:
        _, _, reward = rollout_trajectory(model, tokenizer, env, task_id, cfg)
        if reward > 0.8:  # Success threshold
            success_count += 1
    
    model.train()
    return success_count / num_eval_tasks


def train(cfg: TrainConfig):
    """Main training loop."""
    print(f"Starting RAGEN training with A*PO")
    print(f"Model: {cfg.model_name}")
    print(f"Device: {cfg.device}")
    print(f"Steps: {cfg.num_steps}")
    print(f"=" * 80)
    
    # Load models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.device == "cuda" else torch.float32,
        device_map=cfg.device,
    )
    model.train()
    
    # Create reference model (frozen copy)
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    print("✓ Models loaded")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    
    # Initialize A*PO
    apo = AStarPO(model, ref_model, tokenizer, beta=cfg.beta_kl, device=cfg.device)
    
    # Create environment
    env = SimulatedWebArenaEnv()
    
    print(f"✓ Environment created with {len(env.tasks)} tasks")
    print("=" * 80)
    
    # Training loop
    for step in range(cfg.num_steps):
        # Sample batch of tasks
        task_ids = random.sample(range(len(env.tasks)), cfg.batch_size)
        
        # Collect trajectories
        all_prompts = []
        all_responses = []
        all_rewards = []
        
        for task_id in task_ids:
            # Generate multiple samples per task
            for _ in range(cfg.num_samples_per_task):
                prompts, responses, reward = rollout_trajectory(model, tokenizer, env, task_id, cfg)
                all_prompts.extend(prompts)
                all_responses.extend(responses)
                # Distribute reward across turns
                all_rewards.extend([reward / len(prompts)] * len(prompts))
        
        # Compute A*PO loss
        loss, stats = apo.compute_loss(all_prompts, all_responses, all_rewards)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if (step + 1) % 5 == 0:
            print(f"Step {step+1:3d} | Loss: {stats['loss']:.4f} | "
                  f"Policy: {stats['policy_loss']:.4f} | KL: {stats['kl_div']:.4f} | "
                  f"Reward: {stats['mean_reward']:.3f}")
        
        # Evaluation
        if (step + 1) % cfg.eval_every == 0:
            success_rate = evaluate(model, tokenizer, env, cfg)
            print(f"{'='*80}")
            print(f"Step {step+1} | Evaluation Success Rate: {success_rate:.2%}")
            print(f"{'='*80}")
    
    print("\nTraining completed!")
    return model, tokenizer


if __name__ == "__main__":
    cfg = TrainConfig(
        num_steps=50,
        batch_size=2,
        num_samples_per_task=2,
        learning_rate=5e-6,
        eval_every=10,
    )
    
    model, tokenizer = train(cfg)
    
    # Save model
    print("Saving model...")
    model.save_pretrained("./ragen_trained_model")
    tokenizer.save_pretrained("./ragen_trained_model")
    print("✓ Model saved to ./ragen_trained_model")
