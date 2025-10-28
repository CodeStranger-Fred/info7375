# src/online_problem_generator.py
import random
from typing import List, Dict, Tuple

class OnlineProblemGenerator:
    """动态生成Countdown问题，用于online rollout"""
    
    def __init__(self, num_count: int = 4, number_range: Tuple[int, int] = (1, 20)):
        self.num_count = num_count
        self.number_range = number_range
    
    def generate_problem(self) -> Dict:
        """生成一个有解的Countdown问题"""
        # 生成随机数字
        numbers = [random.randint(*self.number_range) for _ in range(self.num_count)]
        
        # 生成一个可解的目标（确保有至少一个解）
        nums_copy = numbers.copy()
        random.shuffle(nums_copy)
        
        # 创建不同模式的方程
        pattern = random.choice(['mult_sub', 'mult_add', 'add_mult', 'add_sub', 'simple'])
        
        if len(nums_copy) >= 4:
            if pattern == 'mult_sub':
                # (a + b) * c - d
                target = (nums_copy[0] + nums_copy[1]) * nums_copy[2] - nums_copy[3]
                solution = f"({nums_copy[0]} + {nums_copy[1]}) * {nums_copy[2]} - {nums_copy[3]}"
            elif pattern == 'mult_add':
                # (a - b) * c + d
                a, b = max(nums_copy[0], nums_copy[1]), min(nums_copy[0], nums_copy[1])
                target = (a - b) * nums_copy[2] + nums_copy[3]
                solution = f"({a} - {b}) * {nums_copy[2]} + {nums_copy[3]}"
            elif pattern == 'add_mult':
                # a * b + c - d
                target = nums_copy[0] * nums_copy[1] + nums_copy[2] - nums_copy[3]
                solution = f"{nums_copy[0]} * {nums_copy[1]} + {nums_copy[2]} - {nums_copy[3]}"
            elif pattern == 'add_sub':
                # a + b + c - d
                target = nums_copy[0] + nums_copy[1] + nums_copy[2] - nums_copy[3]
                solution = f"{nums_copy[0]} + {nums_copy[1]} + {nums_copy[2]} - {nums_copy[3]}"
            else:  # simple
                # a + b - c + d
                target = nums_copy[0] + nums_copy[1] - nums_copy[2] + nums_copy[3]
                solution = f"{nums_copy[0]} + {nums_copy[1]} - {nums_copy[2]} + {nums_copy[3]}"
        else:
            target = sum(nums_copy)
            solution = " + ".join(map(str, nums_copy))
        
        # 确保target是正数且合理
        target = abs(target)
        if target == 0:
            target = random.randint(1, 10)
            solution = f"{nums_copy[0]} + {target - nums_copy[0]}"
        if target > 500:
            target = target % 200 + 1
        
        # 验证解是否正确
        try:
            calculated = eval(solution)
            if abs(calculated - target) > 0.01:
                # 如果不匹配，使用简单加法
                target = sum(nums_copy[:3]) - nums_copy[3]
                if target <= 0:
                    target = sum(nums_copy)
                    solution = " + ".join(map(str, nums_copy))
                else:
                    solution = f"{nums_copy[0]} + {nums_copy[1]} + {nums_copy[2]} - {nums_copy[3]}"
        except:
            # 回退到简单加法
            target = sum(nums_copy)
            solution = " + ".join(map(str, nums_copy))
        
        return {
            "nums": numbers,
            "target": int(target),
            "solution": solution
        }
    
    def generate_batch(self, batch_size: int) -> List[Dict]:
        """生成一批问题"""
        return [self.generate_problem() for _ in range(batch_size)]
    def make_prompt(self, problem: Dict, template_type: str = "base") -> str:
        """为问题创建 prompt"""
        target = problem['target']
        numbers = problem['nums']
        
        if template_type == "base":
            # 使用 few-shot 示例教会模型格式
            prompt = f"""Solve the math problem. Use ONLY the given numbers with +, -, *, / operators. Output format: <answer>equation</answer>

Example 1:
Numbers: [3, 5, 2, 8]
Target: 21
Answer: <answer>3 * 5 + 8 - 2</answer>

Example 2:
Numbers: [10, 4, 6, 2]
Target: 12
Answer: <answer>10 + 6 - 4</answer>

Example 3:
Numbers: [7, 3, 9, 1]
Target: 30
Answer: <answer>7 * 3 + 9</answer>

Now solve this (output only the equation in <answer> tags):
Numbers: {numbers}
Target: {target}
Answer: <answer>"""
        
        return prompt
