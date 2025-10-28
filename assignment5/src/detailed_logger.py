# src/detailed_logger.py
"""
详细日志记录器 - 记录模型的完整思考过程
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any

class DetailedLogger:
    """记录训练过程中的所有细节"""
    
    def __init__(self, log_dir: str = "detailed_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 主日志文件
        self.main_log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        
        # 统计信息
        self.iteration_stats = []
        self.problem_details = []
        
    def log_warmup(self, warmup_info: Dict[str, Any]):
        """记录 warmup 信息"""
        self._append_log({
            "type": "warmup",
            "timestamp": datetime.now().isoformat(),
            "info": warmup_info
        })
        
    def log_iteration_start(self, iteration: int, num_problems: int):
        """记录迭代开始"""
        self._append_log({
            "type": "iteration_start",
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "num_problems": num_problems
        })
        
    def log_problem_detail(self, 
                          iteration: int, 
                          problem_idx: int,
                          problem: Dict,
                          responses: List[str],
                          rewards: List[float],
                          reference_logprobs: List[float],
                          loss: float,
                          policy_loss: float,
                          kl_divergence: float):
        """记录单个问题的完整细节"""
        
        detail = {
            "type": "problem_detail",
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "problem_idx": problem_idx,
            
            # 问题信息
            "problem": {
                "numbers": problem['nums'],
                "target": problem['target'],
                "ground_truth_solution": problem.get('solution', None)
            },
            
            # 模型响应（每个样本）
            "model_responses": [
                {
                    "sample_idx": i,
                    "response": response,
                    "reward": float(rewards[i]),
                    "reward_breakdown": self._analyze_reward(response, problem['target'], problem['nums']),
                    "reference_logprob": float(reference_logprobs[i]) if reference_logprobs else None,
                    "response_analysis": self._analyze_response(response)
                }
                for i, response in enumerate(responses)
            ],
            
            # 聚合统计
            "aggregated_stats": {
                "avg_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
                "max_reward": float(max(rewards)) if rewards else 0.0,
                "min_reward": float(min(rewards)) if rewards else 0.0,
                "num_correct": sum(1 for r in rewards if r >= 0.95),
                "num_with_answer_tag": sum(1 for r in responses if '<answer>' in r or '</answer>' in r),
            },
            
            # 训练损失
            "training_metrics": {
                "total_loss": float(loss),
                "policy_loss": float(policy_loss) if policy_loss is not None else None,
                "kl_divergence": float(kl_divergence) if kl_divergence is not None else None,
            }
        }
        
        self._append_log(detail)
        
    def log_iteration_end(self, iteration: int, stats: Dict[str, Any]):
        """记录迭代结束"""
        self._append_log({
            "type": "iteration_end",
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "stats": stats
        })
        
    def log_reference_model_update(self, step: int):
        """记录参考模型更新"""
        self._append_log({
            "type": "reference_model_update",
            "timestamp": datetime.now().isoformat(),
            "step": step
        })
        
    def _analyze_response(self, response: str) -> Dict[str, Any]:
        """分析响应的特征"""
        return {
            "length": len(response),
            "has_answer_start": '<answer>' in response,
            "has_answer_end": '</answer>' in response,
            "has_numbers": any(c.isdigit() for c in response),
            "has_operators": any(op in response for op in ['+', '-', '*', '/']),
            "is_repetitive": self._check_repetition(response),
            "first_50_chars": response[:50] if len(response) > 50 else response
        }
    
    def _check_repetition(self, text: str, min_pattern_len: int = 3, threshold: float = 0.5) -> bool:
        """检查文本是否有大量重复"""
        if len(text) < 10:
            return False
        
        # 检查连续重复的模式
        for pattern_len in range(min_pattern_len, min(10, len(text) // 2)):
            for i in range(len(text) - pattern_len * 2):
                pattern = text[i:i+pattern_len]
                rest = text[i+pattern_len:]
                count = rest.count(pattern)
                if count > len(rest) * threshold / pattern_len:
                    return True
        return False
    
    def _analyze_reward(self, response: str, target: float, numbers: List[float]) -> Dict[str, Any]:
        """分析奖励的来源（模拟 reward 计算逻辑）"""
        breakdown = {
            "format_reward": 0.0,
            "parseable_reward": 0.0,
            "operators_reward": 0.0,
            "numbers_reward": 0.0,
            "accuracy_reward": 0.0
        }
        
        # 格式奖励
        if '<answer>' in response and '</answer>' in response:
            breakdown["format_reward"] = 0.1
        
        # 提取表达式
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            equation = answer_match.group(1).strip()
            
            # 可解析奖励
            clean_eq = equation.replace(' ', '').replace('(', '').replace(')', '')
            allowed_chars = set('0123456789+-*/.') 
            if all(c in allowed_chars for c in clean_eq):
                breakdown["operators_reward"] = 0.1
            
            # 尝试评估
            try:
                result = eval(equation, {"__builtins__": None}, {})
                breakdown["parseable_reward"] = 0.1
                
                # 准确度奖励
                error = abs(result - target)
                if error < 1e-6:
                    breakdown["accuracy_reward"] = 0.7
                elif target != 0:
                    relative_error = error / abs(target)
                    if relative_error < 0.05:
                        breakdown["accuracy_reward"] = 0.6
                    elif relative_error < 0.1:
                        breakdown["accuracy_reward"] = 0.5
                    elif relative_error < 0.2:
                        breakdown["accuracy_reward"] = 0.4
            except:
                pass
        
        return breakdown
    
    def _append_log(self, log_entry: Dict[str, Any]):
        """追加日志到文件"""
        with open(self.main_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def generate_summary_report(self, output_file: str = None):
        """生成训练摘要报告"""
        if output_file is None:
            output_file = os.path.join(self.log_dir, "summary_report.json")
        
        # 读取所有日志
        logs = []
        with open(self.main_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                logs.append(json.loads(line))
        
        # 统计分析
        summary = {
            "total_iterations": len([l for l in logs if l["type"] == "iteration_start"]),
            "total_problems": len([l for l in logs if l["type"] == "problem_detail"]),
            "iteration_summaries": self._summarize_iterations(logs),
            "reward_progression": self._analyze_reward_progression(logs),
            "common_patterns": self._analyze_common_patterns(logs)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def _summarize_iterations(self, logs: List[Dict]) -> List[Dict]:
        """总结每次迭代"""
        iterations = {}
        
        for log in logs:
            if log["type"] == "problem_detail":
                iter_num = log["iteration"]
                if iter_num not in iterations:
                    iterations[iter_num] = {
                        "problems": [],
                        "total_reward": 0.0,
                        "correct_count": 0,
                        "with_format_count": 0
                    }
                
                iterations[iter_num]["problems"].append(log)
                iterations[iter_num]["total_reward"] += log["aggregated_stats"]["avg_reward"]
                iterations[iter_num]["correct_count"] += log["aggregated_stats"]["num_correct"]
                iterations[iter_num]["with_format_count"] += log["aggregated_stats"]["num_with_answer_tag"]
        
        summaries = []
        for iter_num in sorted(iterations.keys()):
            iter_data = iterations[iter_num]
            num_problems = len(iter_data["problems"])
            summaries.append({
                "iteration": iter_num,
                "num_problems": num_problems,
                "avg_reward": iter_data["total_reward"] / num_problems if num_problems > 0 else 0.0,
                "accuracy": iter_data["correct_count"] / (num_problems * 4) if num_problems > 0 else 0.0,  # 4 samples per problem
                "format_compliance": iter_data["with_format_count"] / (num_problems * 4) if num_problems > 0 else 0.0
            })
        
        return summaries
    
    def _analyze_reward_progression(self, logs: List[Dict]) -> Dict[str, List[float]]:
        """分析奖励进展"""
        progression = {
            "avg_rewards": [],
            "max_rewards": [],
            "correct_rates": []
        }
        
        for log in logs:
            if log["type"] == "problem_detail":
                stats = log["aggregated_stats"]
                progression["avg_rewards"].append(stats["avg_reward"])
                progression["max_rewards"].append(stats["max_reward"])
        
        return progression
    
    def _analyze_common_patterns(self, logs: List[Dict]) -> Dict[str, Any]:
        """分析常见模式"""
        patterns = {
            "repetitive_responses": 0,
            "incomplete_responses": 0,
            "well_formatted_responses": 0,
            "total_responses": 0
        }
        
        for log in logs:
            if log["type"] == "problem_detail":
                for response_data in log["model_responses"]:
                    patterns["total_responses"] += 1
                    analysis = response_data["response_analysis"]
                    
                    if analysis["is_repetitive"]:
                        patterns["repetitive_responses"] += 1
                    if analysis["has_answer_start"] and analysis["has_answer_end"]:
                        patterns["well_formatted_responses"] += 1
                    if analysis["has_answer_start"] and not analysis["has_answer_end"]:
                        patterns["incomplete_responses"] += 1
        
        return patterns
