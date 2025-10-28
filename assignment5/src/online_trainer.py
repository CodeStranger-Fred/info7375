# src/online_trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List
from src.astar_po import AStarPO
from src.online_problem_generator import OnlineProblemGenerator
from src.detailed_logger import DetailedLogger

class OnlineRolloutTrainer:
    """TinyZero风格的Online Rollout训练器"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # 初始化详细日志器
        self.detailed_logger = DetailedLogger(log_dir=config.get('log_dir', 'detailed_logs'))
        print("✅ Detailed logger initialized")
        
        # Online问题生成器 - 必须在SFT warmup之前初始化
        self.problem_generator = OnlineProblemGenerator()
        
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
        
        # SFT Warmup: 用正确答案先训练几步（在初始化完成后）
        warmup_steps = config.get('sft_warmup_steps', 0)
        if warmup_steps > 0:
            warmup_loss = self._sft_warmup(warmup_steps)
            self.detailed_logger.log_warmup({
                "num_steps": warmup_steps,
                "avg_loss": warmup_loss
            })
        
        self.global_step = 0
        self.best_reward = 0.0
    
    def train_iteration(self, iteration: int, num_problems: int) -> tuple:
        """
        训练一个迭代
        
        Online Rollout流程：
        1. 动态生成新问题
        2. 用当前策略rollout（采样多个响应）
        3. 计算奖励
        4. 更新策略
        """
        self.model.train()
        
        # 记录迭代开始
        self.detailed_logger.log_iteration_start(iteration, num_problems)
        
        iteration_loss = 0.0
        iteration_reward = 0.0
        
        pbar = tqdm(range(num_problems), desc=f"Iteration {iteration}")
        
        # 用于保存输出样例
        saved_outputs = []
        
        for problem_idx in pbar:
            # 1. 动态生成新问题
            problem = self.problem_generator.generate_problem()
            prompt = self.problem_generator.make_prompt(problem)
            target = problem['target']
            numbers = problem['nums']
            
            # 打印生成的问题
            print(f"\n{'='*60}")
            print(f"📝 问题 {problem_idx+1}: 数字={numbers}, 目标={target}")
            print(f"提示词: {prompt[:100]}..." if len(prompt) > 100 else f"提示词: {prompt}")
            
            # 2. 用当前策略rollout：生成多个响应
            responses = []
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256
            ).to(self.model.device)
            
            # 温度退火：随着训练进展逐渐降低采样温度
            temperature = self._get_temperature()
            
            # 为 </answer> 创建 stopping criteria
            answer_end_token = "</answer>"
            answer_end_id = self.tokenizer.encode(answer_end_token, add_special_tokens=False)
            
            for _ in range(self.astar_po.num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.config.get('max_length', 50),
                        num_return_sequences=1,
                        temperature=temperature,  # 使用动态温度
                        do_sample=True,
                        repetition_penalty=1.2,  # 防止重复
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
            )[0]  # 取第一个（因为batch_size=1）
            
            # 4. 计算参考策略的logprobs（使用固定的参考模型）
            reference_logprobs = self._compute_reference_logprobs([responses])[0]
            
            # 定期更新参考模型
            if self.global_step - self.last_ref_update >= self.ref_update_frequency:
                self._update_reference_model()
                self.detailed_logger.log_reference_model_update(self.global_step)
                self.last_ref_update = self.global_step
            
            # 5. 计算损失并更新
            loss = self.astar_po.compute_loss(
                [prompt], [responses], [rewards], [reference_logprobs]
            )
            
            # 记录详细信息（获取更多细节）
            policy_loss, kl_div = self._get_loss_components([prompt], [responses], [rewards], [reference_logprobs])
            self.detailed_logger.log_problem_detail(
                iteration=iteration,
                problem_idx=problem_idx,
                problem=problem,
                responses=responses,
                rewards=rewards,
                reference_logprobs=[float(lp) for lp in reference_logprobs],
                loss=loss.item(),
                policy_loss=policy_loss,
                kl_divergence=kl_div
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
                    "avg_reward": float(batch_reward)
                })
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reward': f'{batch_reward:.4f}'
            })
            
            self.global_step += 1
            
            # 定期保存检查点
            if self.global_step % self.config.get('save_steps', 100) == 0:
                self._save_checkpoint(iteration, iteration_loss / (problem_idx + 1))
        
        # 保存详细输出到文件
        if saved_outputs:
            import json
            output_file = f'outputs_iteration_{iteration}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(saved_outputs, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved outputs to {output_file}")
        
        avg_loss = iteration_loss / num_problems
        avg_reward = iteration_reward / num_problems
        
        # 记录迭代结束
        self.detailed_logger.log_iteration_end(iteration, {
            "avg_loss": avg_loss,
            "avg_reward": avg_reward,
            "total_problems": num_problems
        })
        
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
                    # 使用固定的参考模型
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
        """
        initial_temp = self.config.get('initial_temperature', 1.0)
        min_temp = self.config.get('min_temperature', 0.3)
        decay_rate = self.config.get('temperature_decay_rate', 1e-5)
        
        # 线性退火
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
    
    def _sft_warmup(self, num_steps: int):
        """监督学习热身：用正确答案教会模型格式"""
        print(f"\n🎓 SFT Warmup: {num_steps} steps with correct answers...")
        
        self.model.train()
        warmup_optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        total_loss = 0.0
        
        for step in tqdm(range(num_steps), desc="SFT Warmup"):
            # 生成问题
            problem = self.problem_generator.generate_problem()
            prompt = self.problem_generator.make_prompt(problem)
            
            # 正确答案：注意 prompt 已经以 '<answer>' 结尾，所以只需要补全方程和结束标签
            correct_answer = f"{problem['solution']}</answer>"
            full_text = prompt + correct_answer
            
            # Tokenize prompt 和 full text
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            full_tokens = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            
            # 创建 labels：只在 answer 部分计算 loss
            labels = full_tokens["input_ids"].clone()
            prompt_length = prompt_tokens["input_ids"].shape[1]
            labels[0, :prompt_length] = -100  # Mask prompt 部分
            
            inputs = {
                "input_ids": full_tokens["input_ids"].to(self.model.device),
                "attention_mask": full_tokens["attention_mask"].to(self.model.device),
                "labels": labels.to(self.model.device)
            }
            
            # 监督学习
            outputs = self.model(**inputs)
            loss = outputs.loss
            
            warmup_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            warmup_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_steps
        print(f"✅ SFT Warmup complete! Avg loss: {avg_loss:.4f}")
        return avg_loss
    
    def _get_loss_components(self, prompts, responses, rewards, reference_logprobs):
        """获取损失组件（policy loss 和 KL divergence）"""
        try:
            # 重新计算一次以获取组件（不效率但简单）
            # 或者可以修改 astar_po.compute_loss 返回更多值
            import numpy as np
            
            # 简单估计：policy loss ≈ -mean(reward)
            all_rewards = []
            for reward_list in rewards:
                all_rewards.extend(reward_list)
            
            if len(all_rewards) > 0:
                policy_loss_est = -np.mean(all_rewards)
                # KL 可以从 reference logprobs 估计
                kl_div_est = 0.0  # 简化，实际需要更复杂的计算
                return float(policy_loss_est), float(kl_div_est)
        except:
            pass
        
        return None, None
    
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
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f"checkpoints/online_checkpoint_iter_{iteration}_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
