"""
Modal deployment script for RAGEN training with A*PO

Runs RAGEN training on Modal cloud GPU with full A*PO optimization.
"""

import modal

# Create Modal app
app = modal.App("ragen-apo-training")

# Define Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "numpy<2",
        "accelerate",
    )
)

@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU for training
    timeout=7200,  # 2 hour timeout
)
def train_ragen(num_steps: int = 50, batch_size: int = 2, num_samples: int = 2):
    """Train RAGEN with A*PO on Modal cloud."""
    import torch
    import random
    import re
    from typing import List, Dict, Tuple
    from dataclasses import dataclass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import copy
    import torch.nn.functional as F
    
    # Set seeds
    random.seed(42)
    torch.manual_seed(42)
    
    DEVICE = "cuda"
    
    print(f"\n{'='*80}")
    print("RAGEN Training with A*PO")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Samples per task: {num_samples}")
    print(f"{'='*80}\n")
    
    # ==================== A*PO Loss ====================
    
    class AStarPO:
        """A*PO optimizer."""
        
        def __init__(self, model, ref_model, tokenizer, beta=0.05, device="cuda"):
            self.model = model
            self.ref_model = ref_model
            self.tokenizer = tokenizer
            self.beta = beta
            self.device = device
            
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        
        def compute_loss(self, prompts, responses, rewards):
            """Compute A*PO loss."""
            full_texts = [p + r for p, r in zip(prompts, responses)]
            encodings = self.tokenizer(
                full_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.device)
            
            prompt_encodings = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            prompt_lengths = [len(enc) for enc in prompt_encodings.input_ids]
            
            # Current policy log probs
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=encodings.input_ids,
                    attention_mask=encodings.attention_mask,
                )
                logits = outputs.logits
            
            # Reference policy log probs
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    ref_outputs = self.ref_model(
                        input_ids=encodings.input_ids,
                        attention_mask=encodings.attention_mask,
                    )
                    ref_logits = ref_outputs.logits
            
            log_probs = F.log_softmax(logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            target_ids = encodings.input_ids[:, 1:]
            gathered_log_probs = torch.gather(
                log_probs[:, :-1, :], dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            gathered_ref_log_probs = torch.gather(
                ref_log_probs[:, :-1, :], dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Create mask for response tokens only
            batch_size = encodings.input_ids.shape[0]
            seq_len = encodings.input_ids.shape[1] - 1
            mask = torch.zeros(batch_size, seq_len, device=self.device)
            
            for i, prompt_len in enumerate(prompt_lengths):
                mask[i, prompt_len:] = 1.0
            
            attention_mask = encodings.attention_mask[:, 1:].float()
            mask = mask * attention_mask
            
            seq_log_probs = (gathered_log_probs * mask).sum(dim=1)
            seq_ref_log_probs = (gathered_ref_log_probs * mask).sum(dim=1)
            
            seq_lengths = mask.sum(dim=1).clamp(min=1.0)
            seq_log_probs = seq_log_probs / seq_lengths
            seq_ref_log_probs = seq_ref_log_probs / seq_lengths
            
            reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            mean_reward = reward_tensor.mean()
            advantages = reward_tensor - mean_reward
            
            policy_loss = -(advantages * seq_log_probs).mean()
            kl_div = (seq_log_probs - seq_ref_log_probs).mean()
            total_loss = policy_loss + self.beta * kl_div
            
            stats = {
                "loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "kl_div": kl_div.item(),
                "mean_reward": mean_reward.item(),
            }
            
            return total_loss, stats
    
    # ==================== Environment ====================
    
    class SimulatedWebArenaEnv:
        """Simulated WebArena environment."""
        
        def __init__(self):
            self.tasks = self._generate_tasks()
        
        def _generate_tasks(self):
            tasks = []
            colors = ["red", "blue", "black", "white", "green"]
            items = ["shirt", "shoes", "jacket", "pants"]
            for i in range(20):
                color = random.choice(colors)
                item = random.choice(items)
                price = random.randint(20, 100)
                tasks.append({
                    "domain": "shopping",
                    "intent": f"Find and buy a {color} {item} under ${price}",
                    "target_attributes": {"color": color, "item": item, "max_price": price},
                })
            for i in range(10):
                tasks.append({
                    "domain": "gitlab",
                    "intent": "View merge requests assigned to me",
                    "target_url": "/merge_requests?assignee_username=@me",
                })
            random.shuffle(tasks)
            return tasks
        
        def reset(self, task_id):
            self.current_task = self.tasks[task_id]
            self.steps = 0
            self.dom = self._get_initial_dom()
            intent = self.current_task["intent"]
            observation = f"Current page:\\n{self.dom}"
            return intent, observation
        
        def _get_initial_dom(self):
            if self.current_task["domain"] == "shopping":
                return """[1] RootWebArea 'OneStopMarket'
    [20] textbox 'Search products'
    [100] link 'Red Shirt - $29.99'
    [101] link 'Blue Shoes - $45.00'"""
            else:
                return """[1] RootWebArea 'GitLab'
    [10] link 'Merge Requests'"""
        
        def step(self, action):
            self.steps += 1
            action_lower = action.lower()
            done = False
            reward = 0.0
            
            if "buy" in action_lower:
                reward = self._check_purchase(action)
                done = True
            elif "click" in action_lower and "merge" in action_lower:
                reward = 1.0 if self.current_task["domain"] == "gitlab" else 0.0
                done = True
            elif "search" in action_lower:
                self.dom = """[100] link 'Red Shirt - $29.99'
    [101] link 'Blue Shoes - $45.00'"""
            
            if self.steps >= 15:
                done = True
            
            observation = f"Current page:\\n{self.dom}"
            return observation, reward, done
        
        def _check_purchase(self, action):
            if self.current_task["domain"] != "shopping":
                return 0.0
            target = self.current_task["target_attributes"]
            action_lower = action.lower()
            matches = 0
            if target["color"] in action_lower:
                matches += 1
            if target["item"] in action_lower:
                matches += 1
            if "100" in action_lower or "101" in action_lower:
                matches += 1
            return matches / 3
    
    # ==================== Helper Functions ====================
    
    def retrieve_relevant_elements(observation, task, top_k=3):
        """Retrieve relevant DOM elements."""
        lines = observation.split('\\n')
        task_words = set(task.lower().split())
        scored = []
        for line in lines:
            line_words = set(line.lower().split())
            score = len(task_words & line_words)
            if score > 0:
                scored.append((score, line.strip()))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [line for _, line in scored[:top_k]]
    
    def format_prompt_with_retrieval(task, observation, retrieved, history):
        """Format prompt with retrieval."""
        prompt = f"""You are a web agent helping users complete tasks.

Task: {task}

Retrieved relevant elements:
{chr(10).join(retrieved)}

Previous actions:
{chr(10).join(history[-3:]) if history else "None"}

Think about what action to take, then output your action.
Use format: <think>reasoning</think><action>action_here</action>

Your response:"""
        return prompt
    
    def extract_action(text):
        """Extract action from generated text."""
        action_match = re.search(r'<action>(.+?)</action>', text, re.DOTALL)
        if action_match:
            return action_match.group(1).strip()
        
        text_lower = text.lower()
        if "buy" in text_lower:
            id_match = re.search(r'\\[(\\d+)\\]', text)
            if id_match:
                return f"buy [{id_match.group(1)}]"
            return "buy [100]"
        elif "click" in text_lower:
            return "click [10]"
        elif "search" in text_lower:
            return "search:products"
        return "search:products"
    
    def compute_reasoning_reward(text, action, task):
        """Compute reasoning consistency reward."""
        think_match = re.search(r'<think>(.+?)</think>', text, re.DOTALL)
        if not think_match:
            return 0.0
        reasoning = think_match.group(1).lower()
        score = 0.0
        if ("buy" in action.lower() and "buy" in reasoning) or \
           ("search" in action.lower() and "search" in reasoning):
            score += 0.05
        if len(reasoning.strip()) > 20:
            score += 0.05
        return min(0.1, score)
    
    def rollout_trajectory(model, tokenizer, env, task_id):
        """Rollout a single trajectory."""
        intent, obs = env.reset(task_id)
        prompts = []
        responses = []
        history = []
        total_reward = 0.0
        done = False
        
        for turn in range(5):
            if done:
                break
            
            retrieved = retrieve_relevant_elements(obs, intent)
            prompt = format_prompt_with_retrieval(intent, obs, retrieved, history)
            prompts.append(prompt)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=100, temperature=0.7,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response)
            
            action = extract_action(response)
            history.append(action)
            
            obs, reward, done = env.step(action)
            reasoning_reward = compute_reasoning_reward(response, action, intent)
            total_reward += reward + reasoning_reward
        
        return prompts, responses, total_reward
    
    # ==================== Main Training ====================
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.train()
    
    # Store initial parameters instead of copying entire model (saves memory)
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.detach().clone()
    
    print("✓ Models loaded\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    
    # Modified A*PO class that uses stored params instead of separate model
    class AStarPOMemoryEfficient:
        def __init__(self, model, initial_params, tokenizer, beta=0.05, device="cuda"):
            self.model = model
            self.initial_params = initial_params
            self.tokenizer = tokenizer
            self.beta = beta
            self.device = device
        
        def compute_loss(self, prompts, responses, rewards):
            """Compute A*PO loss with memory-efficient KL computation."""
            full_texts = [p + r for p, r in zip(prompts, responses)]
            encodings = self.tokenizer(
                full_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.device)
            
            prompt_encodings = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            prompt_lengths = [len(enc) for enc in prompt_encodings.input_ids]
            
            # Current policy log probs
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=encodings.input_ids,
                    attention_mask=encodings.attention_mask,
                )
                logits = outputs.logits
            
            # Reference policy: temporarily load initial params
            with torch.no_grad():
                # Save current params
                current_params = {name: param.detach().clone() 
                                 for name, param in self.model.named_parameters()}
                
                # Load initial params
                for name, param in self.model.named_parameters():
                    param.data.copy_(self.initial_params[name])
                
                # Compute reference logits
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    ref_outputs = self.model(
                        input_ids=encodings.input_ids,
                        attention_mask=encodings.attention_mask,
                    )
                    ref_logits = ref_outputs.logits
                
                # Restore current params
                for name, param in self.model.named_parameters():
                    param.data.copy_(current_params[name])
            
            log_probs = F.log_softmax(logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            target_ids = encodings.input_ids[:, 1:]
            gathered_log_probs = torch.gather(
                log_probs[:, :-1, :], dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            gathered_ref_log_probs = torch.gather(
                ref_log_probs[:, :-1, :], dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Create mask for response tokens only
            batch_size = encodings.input_ids.shape[0]
            seq_len = encodings.input_ids.shape[1] - 1
            mask = torch.zeros(batch_size, seq_len, device=self.device)
            
            for i, prompt_len in enumerate(prompt_lengths):
                mask[i, prompt_len:] = 1.0
            
            attention_mask = encodings.attention_mask[:, 1:].float()
            mask = mask * attention_mask
            
            seq_log_probs = (gathered_log_probs * mask).sum(dim=1)
            seq_ref_log_probs = (gathered_ref_log_probs * mask).sum(dim=1)
            
            seq_lengths = mask.sum(dim=1).clamp(min=1.0)
            seq_log_probs = seq_log_probs / seq_lengths
            seq_ref_log_probs = seq_ref_log_probs / seq_lengths
            
            reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            mean_reward = reward_tensor.mean()
            advantages = reward_tensor - mean_reward
            
            policy_loss = -(advantages * seq_log_probs).mean()
            kl_div = (seq_log_probs - seq_ref_log_probs).mean()
            total_loss = policy_loss + self.beta * kl_div
            
            stats = {
                "loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "kl_div": kl_div.item(),
                "mean_reward": mean_reward.item(),
            }
            
            return total_loss, stats
    
    apo = AStarPOMemoryEfficient(model, initial_params, tokenizer, beta=0.05, device=DEVICE)
    env = SimulatedWebArenaEnv()
    
    print(f"Training on {len(env.tasks)} tasks\n")
    print(f"{'='*80}")
    
    # Training loop
    for step in range(num_steps):
        task_ids = random.sample(range(len(env.tasks)), batch_size)
        
        all_prompts = []
        all_responses = []
        all_rewards = []
        
        for task_id in task_ids:
            for _ in range(num_samples):
                prompts, responses, reward = rollout_trajectory(model, tokenizer, env, task_id)
                all_prompts.extend(prompts)
                all_responses.extend(responses)
                all_rewards.extend([reward / len(prompts)] * len(prompts))
        
        loss, stats = apo.compute_loss(all_prompts, all_responses, all_rewards)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1:3d} | Loss: {stats['loss']:.4f} | "
                  f"Policy: {stats['policy_loss']:.4f} | KL: {stats['kl_div']:.4f} | "
                  f"Reward: {stats['mean_reward']:.3f}")
    
    print(f"\\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}")
    
    return {
        "status": "completed",
        "num_steps": num_steps,
        "final_loss": stats['loss'],
        "final_reward": stats['mean_reward'],
    }


@app.local_entrypoint()
def main(num_steps: int = 50):
    """Local entrypoint - Call cloud training function."""
    print(f"\\n{'='*80}")
    print("Starting RAGEN Training on Modal Cloud")
    print(f"{'='*80}")
    print(f"Training steps: {num_steps}")
    print(f"GPU: T4")
    print(f"{'='*80}\\n")
    
    results = train_ragen.remote(num_steps=num_steps, batch_size=2, num_samples=2)
    
    print(f"\\n{'='*80}")
    print("Training Results")
    print(f"{'='*80}")
    for key, value in results.items():
        print(f"{key}: {value}")
    print(f"{'='*80}\\n")
