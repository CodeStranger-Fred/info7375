# src/selfplay_trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List
from src.astar_po import AStarPO
from src.llm_problem_generator import LLMProblemGenerator

class SelfPlayTrainer:
    """
    Self-Play训练器
    模型既是出题者又是答题者
    """
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.get('learning_rate', 5e-5)
        )
        
        self.astar_po = AStarPO(
            model, 
            tokenizer,
            beta=config.get('beta', 0.1),
            num_samples=config.get('num_samples', 8)
        )
        
        # 创建参考模型用于EMA更新
        print("📋 Creating reference model copy...")
        import copy
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        print("✅ Reference model created (frozen)")
        
        # EMA更新参数
        self.ema_decay = config.get('ema_decay', 0.95)
        
        # 参考模型更新频率
        self.ref_update_frequency = config.get('ref_update_frequency', 1000)
        self.last_ref_update = 0
        
        # LLM问题生成器（Self-play核心）
        use_llm_gen = config.get('use_llm_generator', False)
        self.use_llm_generator = use_llm_gen
        
        if use_llm_gen:
            print("🎮 Initializing LLM Self-Play Generator...")
            self.problem_generator = LLMProblemGenerator(
                model, 
                tokenizer, 
                difficulty_curriculum=config.get('difficulty_curriculum', True)
            )
            print("✅ Self-Play mode activated!")
        else:
            # 回退到rule-based
            from src.online_problem_generator import OnlineProblemGenerator
            self.problem_generator = OnlineProblemGenerator()
            print("📊 Using rule-based generator")
        
        self.global_step = 0
        self.best_reward = 0.0
        
        # 统计信息
        self.llm_generated_count = 0
        self.rule_based_count = 0
    
    def train_iteration(self, iteration: int, num_problems: int) -> tuple:
        """
        训练一个迭代
        
        Self-Play流程：
        1. 让模型生成问题（出题者）
        2. 让模型解决问题（答题者）
        3. 根据答题质量更新模型
        """
        self.model.train()
        
        iteration_loss = 0.0
        iteration_reward = 0.0
        
        pbar = tqdm(range(num_problems), desc=f"Iteration {iteration}")
        
        saved_outputs = []
        
        for problem_idx in pbar:
            # 1. 生成问题（Self-play: 模型作为出题者）
            if self.use_llm_generator:
                problem = self.problem_generator.generate_problem_with_llm()
                if problem.get('source') == 'llm_generated':
                    self.llm_generated_count += 1
                else:
                    self.rule_based_count += 1
            else:
                problem = self.problem_generator.generate_problem()
                self.rule_based_count += 1
            
            prompt = self.problem_generator.make_prompt(problem)
            target = problem['target']
            numbers = problem['nums']
            
            # 打印生成的问题
            print(f"\n{'='*60}")
            source_label = problem.get('source', 'rule-based')
            print(f"📝 问题 {problem_idx+1} ({source_label}): 数字={numbers}, 目标={target}")
            print(f"提示词: {prompt[:100]}..." if len(prompt) > 100 else f"提示词: {prompt}")
            
            # 2. 解决问题（Self-play: 模型作为答题者）
            responses = []
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256
            ).to(self.model.device)
            
            # 温度退火：随着训练进展逐渐降低采样温度
            temperature = self._get_temperature()
            
            for _ in range(self.astar_po.num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.config.get('max_length', 256),
                        num_return_sequences=1,
                        temperature=temperature,  # 使用动态温度
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                responses.append(response)
            
            # 打印模型生成的所有答案
            print(f"\n🤖 模型生成的 {len(responses)} 个答案:")
            for i, resp in enumerate(responses, 1):
                print(f"  答案{i}: {resp[:80]}..." if len(resp) > 80 else f"  答案{i}: {resp}")
            
            # 3. 计算奖励
            rewards = self.astar_po.compute_rewards(
                [responses], [target], [numbers]
            )[0]
            
            # 4. 计算参考策略的logprobs
            reference_logprobs = self._compute_reference_logprobs([responses])[0]
            
            # 定期更新参考模型
            if self.global_step - self.last_ref_update >= self.ref_update_frequency:
                self._update_reference_model()
                self.last_ref_update = self.global_step
            
            # 5. 计算损失并更新
            loss = self.astar_po.compute_loss(
                [prompt], [responses], [rewards], [reference_logprobs]
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录统计信息
            batch_reward = np.mean(rewards)
            iteration_loss += loss.item()
            iteration_reward += batch_reward
            
            # 保存前3个问题的详细输出
            if problem_idx < 3:
                saved_outputs.append({
                    "iteration": iteration,
                    "problem_idx": problem_idx,
                    "problem": problem,
                    "target": target,
                    "numbers": numbers,
                    "responses": responses,
                    "rewards": rewards,
                    "avg_reward": float(batch_reward),
                    "source": problem.get('source', 'unknown')
                })
            
            # 更新进度条
            postfix = {
                'loss': f'{loss.item():.4f}',
                'reward': f'{batch_reward:.4f}'
            }
            if self.use_llm_generator:
                postfix['llm_gen'] = f'{self.llm_generated_count}/{self.llm_generated_count + self.rule_based_count}'
            
            pbar.set_postfix(postfix)
            
            self.global_step += 1
            
            # 定期保存检查点
            if self.global_step % self.config.get('save_steps', 100) == 0:
                self._save_checkpoint(iteration, iteration_loss / (problem_idx + 1))
        
        # 保存详细输出到文件
        if saved_outputs:
            import json
            output_file = f'outputs_selfplay_iter_{iteration}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(saved_outputs, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved outputs to {output_file}")
        
        # 打印Self-play统计
        if self.use_llm_generator:
            total = self.llm_generated_count + self.rule_based_count
            llm_ratio = self.llm_generated_count / total if total > 0 else 0
            print(f"\n🎮 Self-Play Stats: {self.llm_generated_count} LLM-generated ({llm_ratio*100:.1f}%), {self.rule_based_count} rule-based")
            
            if hasattr(self.problem_generator, 'get_stats'):
                gen_stats = self.problem_generator.get_stats()
                print(f"📈 Current difficulty: {gen_stats.get('current_difficulty', 'N/A')}")
        
        avg_loss = iteration_loss / num_problems
        avg_reward = iteration_reward / num_problems
        
        return avg_loss, avg_reward
    
    def _compute_reference_logprobs(self, responses: List[List[str]]) -> List[List[torch.Tensor]]:
        """计算参考策略的log概率（使用固定的参考模型）"""
        reference_logprobs = []
        
        for prompt_responses in responses:
            prompt_logprobs = []
            for response in prompt_responses:
                tokens = self.tokenizer.encode(response, return_tensors="pt")
                if tokens.shape[1] == 0:
                    prompt_logprobs.append(torch.tensor(0.0))
                    continue
                    
                tokens = tokens.to(self.ref_model.device)
                with torch.no_grad():
                    outputs = self.ref_model(tokens)
                    logits = outputs.logits
                    logprobs = F.log_softmax(logits, dim=-1)
                    
                    if tokens.shape[1] > 1:
                        token_logprobs = torch.gather(
                            logprobs[:-1], 2, tokens[1:].unsqueeze(-1)
                        ).squeeze(-1)
                        seq_logprob = token_logprobs.sum()
                    else:
                        seq_logprob = torch.tensor(0.0)
                    
                    prompt_logprobs.append(seq_logprob.cpu())
            reference_logprobs.append(prompt_logprobs)
        
        return reference_logprobs
    
    def _get_temperature(self) -> float:
        """
        计算当前的采样温度（温度退火）
        
        随着训练进展，逐渐从高温度（多样性）过渡到低温度（稳定性）
        
        高温度 (1.0): 探索，多样性高
        低温度 (0.3): 利用，输出稳定
        """
        initial_temp = self.config.get('initial_temperature', 1.0)
        min_temp = self.config.get('min_temperature', 0.3)
        decay_rate = self.config.get('temperature_decay_rate', 1e-5)
        
        # 线性退火：temp = initial - decay_rate * step
        temperature = max(min_temp, initial_temp - decay_rate * self.global_step)
        
        # 在第一次和每1000步显示温度
        if not hasattr(self, '_shown_temp') or self.global_step % 1000 == 0:
            if not hasattr(self, '_shown_temp'):
                print(f"\n🌡️  Temperature Annealing: {initial_temp} → {min_temp} (decay={decay_rate})")
                self._shown_temp = True
            if self.global_step % 1000 == 0 and self.global_step > 0:
                print(f"\n🌡️  Temperature at step {self.global_step}: {temperature:.3f}")
        
        return temperature
    
    def _update_reference_model(self):
        """使用EMA逐参数更新参考模型（避免显存峰值）"""
        print(f"\n🔄 Updating reference model with EMA (decay={self.ema_decay}) at step {self.global_step}...")
        
        # 逐参数EMA更新：p_ref = decay * p_ref + (1-decay) * p
        with torch.no_grad():
            for p_ref, p in zip(self.ref_model.parameters(), self.model.parameters()):
                p_ref.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)
        
        print("✅ Reference model updated with EMA (no memory spike!)")
    
    def _save_checkpoint(self, iteration: int, loss: float):
        """保存检查点"""
        import os
        from datetime import datetime
        
        checkpoint = {
            'iteration': iteration,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'llm_generated_count': self.llm_generated_count,
            'rule_based_count': self.rule_based_count,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f"checkpoints/selfplay_checkpoint_iter_{iteration}_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
