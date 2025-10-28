# src/llm_problem_generator.py
import torch
import re
import random
from typing import List, Dict, Optional

class LLMProblemGenerator:
    """
    使用LLM自生成问题的生成器
    实现真正的self-play：模型既是出题者又是答题者
    """
    
    def __init__(self, model, tokenizer, difficulty_curriculum: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.difficulty_curriculum = difficulty_curriculum
        self.current_difficulty = "easy"
        self.generation_count = 0
        
        # 难度级别
        self.difficulties = {
            "easy": {
                "number_range": (1, 10),
                "target_range": (1, 50),
                "operations": ["+", "-"],
                "description": "使用加减法，小数字"
            },
            "medium": {
                "number_range": (1, 20),
                "target_range": (10, 100),
                "operations": ["+", "-", "*"],
                "description": "使用加减乘，中等数字"
            },
            "hard": {
                "number_range": (1, 20),
                "target_range": (10, 200),
                "operations": ["+", "-", "*", "/"],
                "description": "使用四则运算，大数字"
            }
        }
    
    def generate_problem_with_llm(self, difficulty: Optional[str] = None) -> Dict:
        """
        使用LLM生成问题
        
        Self-play流程：
        1. 让模型生成一个数学问题
        2. 让模型自己尝试解决
        3. 验证答案是否正确
        4. 如果正确，作为训练样本
        """
        if difficulty is None:
            difficulty = self.current_difficulty
        
        diff_config = self.difficulties[difficulty]
        
        # 1. 让LLM生成问题
        generation_prompt = self._create_generation_prompt(diff_config)
        
        with torch.no_grad():
            inputs = self.tokenizer(generation_prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                temperature=0.9,  # 高温度以增加多样性
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 2. 解析生成的问题
        problem = self._parse_generated_problem(generated_text, diff_config)
        
        # 3. 如果解析失败，回退到rule-based生成
        if problem is None:
            return self._fallback_generation(diff_config)
        
        # 4. 更新难度（课程学习）
        self.generation_count += 1
        if self.difficulty_curriculum and self.generation_count % 50 == 0:
            self._update_difficulty()
        
        return problem
    
    def _create_generation_prompt(self, diff_config: Dict) -> str:
        """创建让LLM生成问题的prompt"""
        ops_str = ", ".join(diff_config["operations"])
        
        prompt = f"""Generate a math problem for the Countdown game.

Rules:
- Provide exactly 4 numbers between {diff_config["number_range"][0]} and {diff_config["number_range"][1]}
- Create a target number between {diff_config["target_range"][0]} and {diff_config["target_range"][1]}
- The target must be reachable using the 4 numbers with operations: {ops_str}
- Each number can only be used once
- Provide a valid solution

Format your response as:
Numbers: [a, b, c, d]
Target: X
Solution: equation

Example:
Numbers: [3, 5, 2, 8]
Target: 23
Solution: 3 * 5 + 8 - 2

Your turn:
"""
        return prompt
    
    def _parse_generated_problem(self, text: str, diff_config: Dict) -> Optional[Dict]:
        """解析LLM生成的问题"""
        try:
            # 提取数字列表
            numbers_match = re.search(r'Numbers?:\s*\[([0-9,\s]+)\]', text, re.IGNORECASE)
            if not numbers_match:
                return None
            
            numbers_str = numbers_match.group(1)
            numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip()]
            
            if len(numbers) != 4:
                return None
            
            # 检查数字范围
            num_min, num_max = diff_config["number_range"]
            if not all(num_min <= n <= num_max for n in numbers):
                return None
            
            # 提取目标
            target_match = re.search(r'Target:\s*(\d+)', text, re.IGNORECASE)
            if not target_match:
                return None
            
            target = int(target_match.group(1))
            
            # 检查目标范围
            tgt_min, tgt_max = diff_config["target_range"]
            if not (tgt_min <= target <= tgt_max):
                return None
            
            # 提取解答
            solution_match = re.search(r'Solution:\s*([^\n]+)', text, re.IGNORECASE)
            if not solution_match:
                return None
            
            solution = solution_match.group(1).strip()
            
            # 验证解答
            if not self._verify_solution(numbers, target, solution, diff_config["operations"]):
                return None
            
            return {
                "nums": numbers,
                "target": target,
                "solution": solution,
                "difficulty": diff_config,
                "source": "llm_generated"
            }
            
        except Exception as e:
            print(f"⚠️  Failed to parse LLM output: {e}")
            return None
    
    def _verify_solution(self, numbers: List[int], target: int, solution: str, allowed_ops: List[str]) -> bool:
        """验证解答是否正确"""
        try:
            # 检查是否只使用允许的运算符
            clean_solution = solution.replace(' ', '').replace('(', '').replace(')', '')
            for char in clean_solution:
                if char.isalpha() or (char in ['+', '-', '*', '/'] and char not in allowed_ops):
                    return False
            
            # 检查数字使用
            used_numbers = [int(n) for n in re.findall(r'\d+', solution)]
            if sorted(used_numbers) != sorted(numbers):
                return False
            
            # 计算结果
            result = eval(solution)
            return abs(result - target) < 0.01
            
        except:
            return False
    
    def _fallback_generation(self, diff_config: Dict) -> Dict:
        """回退到rule-based生成"""
        num_min, num_max = diff_config["number_range"]
        numbers = [random.randint(num_min, num_max) for _ in range(4)]
        
        # 创建一个可解的目标
        nums = numbers.copy()
        random.shuffle(nums)
        
        ops = diff_config["operations"]
        
        # 简单的组合策略
        if "*" in ops:
            target = nums[0] * nums[1] + nums[2] - nums[3]
            solution = f"{nums[0]} * {nums[1]} + {nums[2]} - {nums[3]}"
        elif "+" in ops and "-" in ops:
            target = nums[0] + nums[1] + nums[2] - nums[3]
            solution = f"{nums[0]} + {nums[1]} + {nums[2]} - {nums[3]}"
        else:
            target = sum(nums)
            solution = " + ".join(map(str, nums))
        
        target = abs(target)
        if target < diff_config["target_range"][0]:
            target = diff_config["target_range"][0]
        if target > diff_config["target_range"][1]:
            target = target % diff_config["target_range"][1]
        
        return {
            "nums": numbers,
            "target": int(target),
            "solution": solution,
            "difficulty": diff_config,
            "source": "rule_based"
        }
    
    def _update_difficulty(self):
        """课程学习：逐渐增加难度"""
        if self.current_difficulty == "easy":
            self.current_difficulty = "medium"
            print(f"\n📈 Difficulty increased to: medium")
        elif self.current_difficulty == "medium":
            self.current_difficulty = "hard"
            print(f"\n📈 Difficulty increased to: hard")
        # hard级别保持不变
    
    def make_prompt(self, problem: Dict, template_type: str = "base") -> str:
        """为问题创建prompt"""
        target = problem['target']
        numbers = problem['nums']
        
        prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
        
        return prompt
    
    def get_stats(self) -> Dict:
        """获取生成统计"""
        return {
            "total_generated": self.generation_count,
            "current_difficulty": self.current_difficulty,
        }
