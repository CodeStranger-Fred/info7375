# src/data_processor.py
import torch
from torch.utils.data import Dataset, DataLoader
import json
from typing import List, Dict, Any
import re

class CountdownDataset(Dataset):
    def __init__(self, data_path: str, template_type: str = "base"):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.template_type = template_type
    
    def make_prefix(self, example: Dict) -> str:
        """创建提示词模板"""
        target = example['target']
        numbers = example['nums']
        
        if self.template_type == "base":
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
        return prefix
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = self.make_prefix(example)
        
        return {
            "prompt": prompt,
            "target": example['target'],
            "numbers": example['nums'],
            "idx": idx
        }

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        prompts = [item["prompt"] for item in batch]
        targets = [item["target"] for item in batch]
        numbers_list = [item["numbers"] for item in batch]
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "targets": targets,
            "numbers_list": numbers_list,
            "prompts": prompts
        }