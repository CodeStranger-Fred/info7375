from typing import Tuple, List, Optional, Sequence, Union
from dataclasses import dataclass, field

import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

MODEL_NAME = "Qwen/Qwen2.5-3B"      # Base (not instruct)
DEVICE = "cuda"                     # Single H100 GPU only; no CPU path
DTYPE = torch.bfloat16              # H100 supports bfloat16 efficiently
Seed= 111


@dataclass
class SamplerCfg:
    """Configuration for sampling completions."""
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True


def format_prompt(target: int, nums: List[int]) -> str:
    """Format the instruction the model sees."""
    numbers_str = ", ".join(str(x) for x in nums)
    return (
f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers_str}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>
"""
    )

@dataclass
class TrainCfg:
    """Training configuration."""
    steps: int = 1000
    batch_prompts: int = 8     # number of prompts per step
    group_size: int = 4        # K generations per prompt
    lr: float = 5e-6
    warmup_steps: int = 50
    grad_clip: float = 1.0
    beta_kl: float = 0.02      # KL penalty weight

@dataclass
class PromptGeneration:
    """Represents a single generation from a prompt."""
    generation_id: int
    text: str
    token_ids: List[int]

    # RL fields to be filled during training:
    reward: Optional[float] = None
    logprob_policy: Optional[float] = None
    logprob_reference: Optional[float] = None
    advantage: Optional[float] = None

    def set_reward(self, reward: float) -> None:
        """Set the reward for this generation."""
        self.reward = reward

    def set_logprobs(self, policy: float, reference: float) -> None:
        """Set the log probabilities from policy and reference models."""
        self.logprob_policy = policy
        self.logprob_reference = reference

    def set_advantage(self, advantage: float) -> None:
        """Set the advantage value."""
        self.advantage = advantage





    


@dataclass
class Prompt:
    """Represents a single prompt with its metadata and generations."""
    prompt_id: int
    text: str
    target: int
    numbers: List[int]

    generations: List[PromptGeneration] = field(default_factory=list)

    def add_generation(self, text: str, token_ids: List[int]) -> PromptGeneration:
        """Add a new generation to this prompt."""
        gen = PromptGeneration(
            generation_id=len(self.generations),
            text=text,
            token_ids=token_ids,
        )
        self.generations.append(gen)
        return gen

    def get_generation_texts(self) -> List[str]:
        """Get all generation texts."""
        return [gen.text for gen in self.generations]

    def add_generation(self, text: str, token_ids: List[int]) -> PromptGeneration:
        """Add a new generation to this prompt."""
        gen = PromptGeneration(
            generation_id=len(self.generations),
            text=text,
            token_ids=token_ids,
        )
        self.generations.append(gen)
        return gen

    def get_generation_texts(self) -> List[str]:
        """Get all generation texts."""
        return [gen.text for gen in self.generations]

    def get_generation_ids(self) -> List[List[int]]:
        """Get all generation token IDs."""
        return [gen.token_ids for gen in self.generations]

    def get_rewards(self) -> List[float]:
        """Get all rewards (defaults to 0.0 if not set)."""
        return [gen.reward if gen.reward is not None else 0.0 for gen in self.generations]

    def set_rewards(self, rewards: List[float]) -> None:
        """Set rewards for all generations."""
        assert len(rewards) == len(self.generations)
        for gen, reward in zip(self.generations, rewards):
            gen.set_reward(reward)

    def compute_advantages(self) -> None:
        """Compute advantages as (reward - mean_reward) for each generation."""
        rewards = self.get_rewards()
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        for gen in self.generations:
            advantage = gen.reward if gen.reward is not None else 0.0
            advantage = advantage - mean_reward
            gen.set_advantage(advantage)

    def get_mean_reward(self) -> float:
        """Get the mean reward across all generations."""
        rewards = self.get_rewards()
        return sum(rewards) / len(rewards) if rewards else 0.0









@dataclass
class PromptBatch:
    """Represents a batch of prompts with their generations."""
    prompts: List[Prompt] = field(default_factory=list)

    def add_prompt(self, text: str, target: int, numbers: List[int]) -> Prompt:
        """Add a new prompt to the batch."""
        prompt = Prompt(
            prompt_id=len(self.prompts),
            text=text,
            target=target,
            numbers=numbers,
        )
        self.prompts.append(prompt)
        return prompt

    def get_prompt_texts(self) -> List[str]:
        """Get all prompt texts."""
        return [p.text for p in self.prompts]

    def flatten_generations(self) -> Tuple[List[PromptGeneration], List[int]]:
        """
        Flatten all generations across prompts.

        Returns:
            generations: Flattened list of all generations
            prompt_indices: For each generation, which prompt it came from
        """
        flat_generations: List[PromptGeneration] = []
        prompt_indices: List[int] = []

        for prompt in self.prompts:
            for gen in prompt.generations:
                flat_generations.append(gen)
                prompt_indices.append(prompt.prompt_id)

        return flat_generations, prompt_indices


    def get_flattened_token_ids(self) -> List[List[int]]:
        """Get all token IDs flattened across prompts and generations."""
        flat_ids: List[List[int]] = []
        for prompt in self.prompts:
            for gen in prompt.generations:
                flat_ids.append(gen.token_ids)
        return flat_ids


    def get_flattened_rewards(self) -> List[float]:
        """Get all rewards flattened across prompts and generations."""
        flat_rewards: List[float] = []
        for prompt in self.prompts:
            flat_rewards.extend(prompt.get_rewards())
        return flat_rewards

    
    def get_flattened_advantages(self) -> List[float]:
        """Get all advantages flattened across prompts and generations."""
        flat_advantages: List[float] = []
        for prompt in self.prompts:
            for gen in prompt.generations:
                adv = gen.advantage if gen.advantage is not None else 0.0
                flat_advantages.append(adv)
        return flat_advantages
    
    def set_flattened_logprobs(
    self,
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
) -> None:
        """
        Set log probabilities for all generations from flattened tensors.

        Args:
            policy_logprobs: [N] tensor of policy log probs
            reference_logprobs: [N] tensor of reference log probs
        """
        idx = 0
        for prompt in self.prompts:
            for gen in prompt.generations:
                gen.set_logprobs(
                    policy=policy_logprobs[idx].item(),
                    reference=reference_logprobs[idx].item(),
                )
                idx += 1

    def compute_all_advantages(self) -> None:
        """Compute advantages for all prompts."""
        for prompt in self.prompts:
            prompt.compute_advantages()

    def size(self) -> int:
        """Get the number of prompts in the batch."""
        return len(self.prompts)
    
    def total_generations(self) -> int:
        """Get the total number of generations across all prompts."""
        return sum(len(p.generations) for p in self.prompts)











def make_countdown_instance(
    n_ops: int = 6,
    target_min: int = 100,
    target_max: int = 999,
    num_max: int = 100,
) -> Tuple[int, List[int]]:
    """Produce a single Countdown problem instance."""
    target = random.randint(target_min, target_max)
    nums = [random.randint(1, num_max) for _ in range(n_ops)]
    return target, nums



def sample_group(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: PromptBatch,
    K: int,
    cfg: SamplerCfg,
):
    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=cfg.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_texts = batch.get_prompt_texts()
    repeated_prompts = []
    for p in prompt_texts:
        repeated_prompts.extend([p] * K)

    encodings = tokenizer(
            repeated_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
    

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            out_ids = model.generate(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                generation_config=gen_cfg,
            )

    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=False)
    print(decoded)

    for i, (txt, ids) in enumerate(zip(decoded, out_ids)):
        prompt_idx = i
        print(txt)
        batch.prompts[prompt_idx].add_generation(txt,ids.tolist())




def load_models_and_tokenizer() -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
    """
    Load tokenizer and two identical models:
      - policy: trainable
      - ref: frozen reference for KL
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    _ = AutoConfig.from_pretrained(MODEL_NAME)

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    policy.to(DEVICE)

    ref = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    ref.to(DEVICE)

    ref.requires_grad_(False)
    for p in ref.parameters():
        p.requires_grad = False

    return policy, ref, tokenizer

def compute_score(
    solution_str: Union[int, float],
    target: int,
    numbers: Sequence[int],
    format_score: float = 0.1,
    score: float = 1.0,
) -> float:
    """Score a Countdown-style solution.

    Args:
        solution_str: The solution text.
        target: Target number.
        numbers: Available numbers.
        format_score: Score for correct format but wrong answer.
        score: Score for the correct answer.
    """
    equation: Optional[str] = extract_solution(solution_str=solution_str)
    do_print: bool = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {list(numbers)}")
        print(f"Solution string: {solution_str}")
        print(Fore.CYAN + f"Extracted equation: {equation}")

    if equation is None:
        if do_print:
            print(Fore.RED + f"No equation found. Score: 0.0")
        return 0.0

    if not validate_equation(equation, numbers):
        if do_print:
            print(Fore.YELLOW + f"Invalid equation. Score: {format_score}")
        return format_score

    result: Optional[float] = evaluate_equation(equation)
    if result is None:
        if do_print:
            print(Fore.YELLOW + f"Could not evaluate equation. Score: {format_score}")
        return format_score

    if abs(result - float(target)) < 1e-5:
        if do_print:
            print(Fore.GREEN + f"Correct equation: {equation} = {result}. Score: {score}")
        return score

    if do_print:
        print(Fore.YELLOW + f"Wrong result: equation = {result}, target = {target}. Score: {format_score}")
    return format_score

def extract_solution(solution_str: str) -> Optional[str]:
    """Extract the last <answer>...</answer> contents after an assistant marker."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split(sep="Assistant:", maxsplit=1)[1]
    elif "<|im_start|assistant>" in solution_str:
        solution_str = solution_str.split(sep="<|im_start|assistant>", maxsplit=1)[1]
    else:
        return None

    solution_str = solution_str.split("\n")[-1]

    answer_pattern = re.compile(r"<answer>(.*)</answer>")
    matches: List[re.Match[str]] = list(answer_pattern.finditer(solution_str))
    if matches:
        return matches[-1].group(1).strip()
    return None

def validate_equation(equation_str: str, available_numbers: Sequence[int]) -> bool:
    """Return True if equation uses exactly the available numbers once each."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except Exception:
        return False

def evaluate_equation(equation_str: str) -> Optional[float]:
    """Safely evaluate the arithmetic equation string. Return None on failure."""
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        result = eval(equation_str, {"__builtins__": None}, {})  # type: ignore[eval-used]
        return float(result)
    except Exception:
        return None


def compute_logp_on_sequences(
    model: PreTrainedModel,
    ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    Compute sum of per-token logprobs for non-pad tokens.

    Args:
        model: Language model
        ids: [B, T] token IDs on GPU
        pad_id: Padding token ID

    Returns:
        [B] sequence log probabilities
    """
    attn_mask = (ids != pad_id).to(ids.dtype)

    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        # Forward pass. 'out.logits' has shape [B, T, V] where V is vocab size.
        out = model(input_ids=ids, attention_mask=attn_mask)
        lp = gather_logprobs(out.logits, ids)

    mask = (ids[:, 1:] != pad_id).to(lp.dtype)
    return (lp * mask).sum(dim=-1)







def gather_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Convert next-token logits into per-token logprobs for the actual next tokens.

    Args:
        logits: [B, T, V]
        input_ids: [B, T]

    Returns:
        [B, T-1] log probabilities
    """
    logprobs = torch.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    gathered = torch.gather(logprobs[:, :-1, :], -1, targets)
    return gathered.squeeze(-1)


def evaluate(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    n_cases: int = 64,
    sampler_cfg: Optional[SamplerCfg] = None,
) -> float:
    """Measure success@1 over n_cases randomly generated instances."""
    if sampler_cfg is None:
        sampler_cfg = SamplerCfg()

    policy.eval()
    hits = 0.0
    with torch.no_grad():
        for _ in range(n_cases):
            t, ns = make_countdown_instance()
            hits += eval_once(policy, tokenizer, t, ns, sampler_cfg)

    policy.train()
    return hits / n_cases


def eval_once(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    target: int,
    nums: List[int],
    sampler_cfg: SamplerCfg,
) -> float:
    """Single sample evaluation on one fresh instance."""
    batch = PromptBatch()
    prompt_text = format_prompt(target, nums)
    batch.add_prompt(prompt_text, target, nums)

    sample_group(policy, tokenizer, batch, K=1, cfg=sampler_cfg)

    gen = batch.prompts[0].generations[0]
    return compute_score(gen.text, target, nums)


def grpo_step(
    policy: PreTrainedModel,
    ref: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: PromptBatch,
    beta_kl: float,
) -> torch.Tensor:
    """
    Group-Relative Policy Optimization step.

    For each prompt:
        We have K sampled sequences with rewards r_i in {0,1}.
        Advantage for sample i is (r_i - mean_j r_j).
    Loss per sample: -adv_i * logp_pol_i + beta_kl * (logp_pol_i - logp_ref_i)

    Args:
        policy: Policy model to train
        ref: Reference model (frozen)
        tokenizer: Tokenizer
        batch: PromptBatch with generations and rewards
        beta_kl: KL penalty coefficient

    Returns:
        Loss tensor
    """

    # Flatten all token IDs
    flat_ids = batch.get_flattened_token_ids()

    # Cap length to model context window
    max_ctx = getattr(policy.config, "max_position_embeddings", 4096)
    #print("max ctx: " + str(max_ctx))
    pad_id = tokenizer.pad_token_id
    max_len = min(max(len(s) for s in flat_ids), max_ctx)

    # Create dense batch tensor
    batch_tensor = pad_to_tensor(flat_ids, pad_id=pad_id, max_len=max_len, device=DEVICE)

    # Compute sequence log-probs
    logp_pol = compute_logp_on_sequences(policy, batch_tensor, pad_id)
    with torch.no_grad():
        logp_ref = compute_logp_on_sequences(ref, batch_tensor, pad_id)

    # Store logprobs in batch
    batch.set_flattened_logprobs(logp_pol, logp_ref)

    # Compute advantages per prompt
    batch.compute_all_advantages()

    # Get flattened advantages and rewards
    flat_advantages = torch.tensor(batch.get_flattened_advantages(), device=DEVICE, dtype=logp_pol.dtype)

    # Compute GRPO loss
    kl_term = logp_pol - logp_ref
    loss = (-flat_advantages * logp_pol + beta_kl * kl_term).mean()

    return loss

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train():
    assert torch.cuda.is_available(), "CUDA is required"
    set_seed(Seed)

    policy, ref, tokenizer = load_models_and_tokenizer()

    # Set model to training mode
    policy.train()

    train_cfg = TrainCfg()
    sampler_cfg = SamplerCfg()

    batch = PromptBatch()

    opt = torch.optim.AdamW(policy.parameters(), lr=train_cfg.lr, betas = (0.9,0.95), weight_decay = 0.00)

    # fixed LR too aggressive for start of training for a transformer
    def lr_lambda(lr_step: int ) -> float:
        return min(1.0, (lr_step + 1) / max(1, train_cfg.warmup_steps))

    sched = torch.optim.lr_scheduler.lambdaLR(opt, lr_lambda = lr_lambda)


    #4
    for step in range(train_cfg.steps):

        #7 
        batch = PromptBatch()
        for _ in range(train_cfg.batch_prompts):
            target, nums = make_countdown_instance()
            prompt_text = format_prompt(target, nums)
            print(prompt_text)
            batch.add_prompt(prompt_text, target, nums)

        sample_group(policy, tokenizer, batch, train_cfg.group_size, SamplerCfg())

        #8
        for prompt in batch.prompts:
            for gen in prompt.generations:
                reward = compute_score(gen.text, prompt.target, prompt.numbers)
                gen.set_reward(reward)

        loss = grpo_step(policy, ref, tokenizer, batch, beta_kl= train_cfg.beta_kl)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm(policy.parameters(),max_norm=train_cfg.grad_clip)
        opt.step()
        sched.step()

    if (step + 1) % 10 == 0:
        sr = evaluate(policy, tokenizer, n_cases=42, sampler_cfg=sampler_cfg)
        print(Fore.MAGENTA + f"step {step+1:4d} | loss {loss.item():.4f} | success@1 {sr:.3f}")

        

# ============================ ENTRY POINT =============================

def main() -> None:
    """Entry point."""
    assert torch.cuda.is_available(), "CUDA is required"
    train()


if __name__ == "__main__":
    main()
