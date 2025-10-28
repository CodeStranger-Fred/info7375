# src/astar_po.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np
import re

class AStarPO:
    def __init__(self, model, tokenizer, beta: float = 0.1, num_samples: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta  # KL惩罚系数
        self.num_samples = num_samples
        
    def compute_rewards(self, responses: List[List[str]], targets: List[float], numbers_list: List[List[float]]) -> List[List[float]]:
        """计算奖励分数"""
        batch_rewards = []
        
        for i, response_list in enumerate(responses):
            target = targets[i]
            numbers = numbers_list[i]
            rewards = []
            
            for response in response_list:
                # 提取答案并验证
                reward = self._validate_response(response, target, numbers)
                rewards.append(reward)
            
            batch_rewards.append(rewards)
        
        return batch_rewards
    
    def _validate_response(self, response: str, target: float, numbers: List[float]) -> float:
        """验证响应并计算奖励 - 细粒度版本
        
        奖励组成：
        - 空响应/无效响应: -0.5 (严重惩罚！)
        - 格式正确（有<answer>标签）: +0.1
        - 可以解析的表达式: +0.1
        - 只使用允许的运算符: +0.1
        - 数字使用接近正确: 0.0~0.2（根据多用/少用程度）
        - 结果接近目标: 0.0~0.6（根据相对误差）
        - 完全正确: 1.0
        """
        reward = 0.0
        equation = None
        
        # 首先检查是否是空响应
        if not response or response.strip() == "":
            return -0.5  # 严重惩罚空响应
        
        try:
            # 1. 尝试提取<answer>标签
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                reward += 0.1  # 格式奖励
                equation = answer_match.group(1).strip()
                # 检查标签内是否为空
                if not equation:
                    return -0.3  # 空标签也要惩罚
            else:
                # 如果没有<answer>标签，尝试直接解析整个响应
                # 提取第一个看起来像计算式的字符串
                # 匹配数字和运算符的组合
                equation_pattern = r'([\d\s\+\-\*\/\(\)]+)'
                matches = re.findall(equation_pattern, response)
                if matches:
                    # 取最长的匹配项
                    equation = max(matches, key=len).strip()
                    # 验证是否是有效计算式（至少包含一个数字和一个运算符）
                    if len(re.findall(r'\d', equation)) > 0 and any(op in equation for op in ['+', '-', '*', '/']):
                        reward += 0.05  # 给一些奖励，但比正确格式少
                    else:
                        equation = None
            
            if not equation:
                return -0.3  # 没找到任何计算式，负奖励
            
            # 2. 检查是否只使用允许的运算符和数字
            # 移除空格和括号，检查剩余字符
            clean_eq = equation.replace(' ', '').replace('(', '').replace(')', '')
            allowed_chars = set('0123456789+-*/.') 
            if all(c in allowed_chars for c in clean_eq):
                reward += 0.1  # 运算符合法
            
            # 3. 提取并验证数字使用
            used_numbers = self._extract_numbers(equation)
            
            # 计算数字使用的匹配度
            if len(used_numbers) == len(numbers):
                # 数字个数正确
                used_sorted = sorted(used_numbers)
                allowed_sorted = sorted(numbers)
                # 检查有多少数字匹配
                matches = sum(1 for u, a in zip(used_sorted, allowed_sorted) if abs(u - a) < 1e-6)
                reward += 0.2 * (matches / len(numbers))  # 按匹配比例给分
            elif len(used_numbers) < len(numbers):
                # 少用了数字
                reward += 0.1 * (len(used_numbers) / len(numbers))
            else:
                # 多用了数字，轻微惩罚但不归零
                reward += 0.05
            
            # 4. 尝试计算表达式
            result = self._evaluate_equation(equation)
            if result == float('inf') or result != result:  # inf or nan
                return reward  # 返回到目前为止的奖励
            
            reward += 0.1  # 成功计算
            
            # 5. 根据结果准确度给奖励（更宽松的奖励）
            error = abs(result - target)
            
            if error < 1e-6:
                # 完全正确！
                return 1.0
            elif target != 0:
                # 根据相对误差给奖励（合理的梯度评分）
                relative_error = error / abs(target)
                
                # 根据误差大小给予递减的奖励
                if relative_error < 0.05:  # 误差 < 5%
                    accuracy_reward = 0.6  # 非常接近！
                elif relative_error < 0.1:  # 误差 < 10%
                    accuracy_reward = 0.5  # 很接近
                elif relative_error < 0.2:  # 误差 < 20%
                    accuracy_reward = 0.4  # 比较接近
                elif relative_error < 0.5:  # 误差 < 50%
                    accuracy_reward = 0.2  # 有点接近
                elif relative_error < 1.0:  # 误差 < 100%
                    accuracy_reward = 0.1  # 至少在同一数量级
                else:
                    accuracy_reward = 0.05  # 很远但至少有尝试
                
                reward += accuracy_reward
            else:
                # target是0的特殊情况
                if error < 1:
                    reward += 0.6 * (1 - error)
                else:
                    reward += 0.1  # 基本分
            
            return min(reward, 0.95)  # 不完全正确的最高分是0.95
            
        except Exception as e:
            # 即使出错，也返回到目前为止累积的奖励
            return reward
    
    def _extract_numbers(self, equation: str) -> List[float]:
        """从方程中提取使用的数字"""
        numbers = re.findall(r'\d+\.?\d*', equation)
        return [float(num) for num in numbers]
    
    def _validate_number_usage(self, used_numbers: List[float], allowed_numbers: List[float]) -> bool:
        """验证数字使用是否符合规则"""
        if len(used_numbers) != len(allowed_numbers):
            return False
        
        used_sorted = sorted(used_numbers)
        allowed_sorted = sorted(allowed_numbers)
        
        return all(abs(u - a) < 1e-6 for u, a in zip(used_sorted, allowed_sorted))
    
    def _evaluate_equation(self, equation: str) -> float:
        """安全地评估数学表达式"""
        # 移除危险操作
        equation = equation.replace('import', '').replace('exec', '').replace('eval', '')
        
        try:
            # 使用eval计算，但在生产环境中应该使用更安全的评估方法
            result = eval(equation, {"__builtins__": None}, {})
            return float(result)
        except:
            return float('inf')  # 返回无穷大表示计算错误
    
    def compute_loss(self, prompts: List[str], responses: List[List[str]], 
                    rewards: List[List[float]], reference_logprobs: List[List[torch.Tensor]]) -> torch.Tensor:
        """计算A*PO损失，包含KL散度惩罚和advantage标准化
        
        Loss = -E[A * log π(y|x)] + β * KL(π || π_ref)
        其中：
        - A = (r - r_mean) / (r_std + ε) 是标准化后的advantage
        - π 是当前策略
        - π_ref 是参考策略
        - β 是KL惩罚系数
        """
        # 1. 收集所有reward用于标准化
        all_rewards = []
        for reward_list in rewards:
            all_rewards.extend(reward_list)
        
        # 2. 使用相对优势（不完全标准化，保持reward的正负性）
        if len(all_rewards) > 1:
            reward_mean = np.mean(all_rewards)
            reward_std = np.std(all_rewards)
            
            # 使用advantage = r - baseline，但不除以std
            # 这样可以保持reward的大小关系，同时减少方差
            normalized_rewards = [
                [r - reward_mean for r in reward_list]
                for reward_list in rewards
            ]
            
            # 在第一次调用时显示信息
            if not hasattr(self, '_shown_norm_info'):
                print(f"\n📈 Reward Stats: mean={reward_mean:.4f}, std={reward_std:.4f}")
                print(f"   Using advantage = reward - mean (not dividing by std)")
                self._shown_norm_info = True
        else:
            # 如果只有一个reward，不做标准化
            normalized_rewards = rewards
        
        policy_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        kl_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        count = 0
        
        for i, prompt in enumerate(prompts):
            prompt_responses = responses[i]
            prompt_rewards = normalized_rewards[i]  # 使用标准化后的reward
            ref_logprobs = reference_logprobs[i]
            
            # Tokenize prompt
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200).to(self.model.device)
            
            # 为每个响应计算损失
            for j, (response, reward, ref_logprob) in enumerate(zip(prompt_responses, prompt_rewards, ref_logprobs)):
                # Tokenize full sequence (prompt + response)
                full_text = prompt + response
                tokens = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=400).to(self.model.device)
                
                # 计算logits (保留梯度)
                outputs = self.model(**tokens)
                logits = outputs.logits
                
                # 计算log probs
                log_probs = F.log_softmax(logits, dim=-1)
                
                # 只在response部分计算loss
                prompt_length = prompt_tokens.input_ids.shape[1]
                if tokens.input_ids.shape[1] > prompt_length + 1:
                    # 计算response tokens的log probs
                    response_logprobs = []
                    for t in range(prompt_length, tokens.input_ids.shape[1] - 1):
                        token_id = tokens.input_ids[0, t + 1]
                        token_logprob = log_probs[0, t, token_id]
                        response_logprobs.append(token_logprob)
                    
                    if len(response_logprobs) > 0:
                        # 当前策略的log prob
                        seq_logprob = torch.stack(response_logprobs).mean()
                        
                        # 策略梯度损失：-reward * log_prob
                        reward_tensor = torch.tensor(reward, device=self.model.device)
                        policy_loss = policy_loss + (-reward_tensor * seq_logprob)
                        
                        # KL散度惩罚: KL(π || π_ref) = log(π) - log(π_ref)
                        ref_logprob_tensor = ref_logprob.to(self.model.device)
                        kl_divergence = seq_logprob - ref_logprob_tensor
                        kl_loss = kl_loss + kl_divergence
                        
                        count += 1
        
        # 总损失 = 策略损失 + β * KL损失
        if count > 0:
            avg_policy_loss = policy_loss / count
            avg_kl_loss = kl_loss / count
            
            # KL惩罚项（防止策略偏离太远）
            kl_penalty = self.beta * avg_kl_loss
            total_loss = avg_policy_loss + kl_penalty
            
            # 记录统计信息（第一次调用时）
            if not hasattr(self, '_shown_kl_info'):
                print(f"\n💜 KL Divergence Info:")
                print(f"   Policy Loss: {avg_policy_loss.item():.4f}")
                print(f"   KL Divergence: {avg_kl_loss.item():.4f}")
                print(f"   KL Penalty (β={self.beta}): {kl_penalty.item():.4f}")
                print(f"   Total Loss: {total_loss.item():.4f}")
                self._shown_kl_info = True
        else:
            total_loss = policy_loss
        
        return total_loss
