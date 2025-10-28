# src/model_manager.py
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List
import math

class ModelManager:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False  # 禁用KV cache以节省内存
        )
        
        # 启用梯度检查点以节省内存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        
        return self.tokenizer, self.model
    
    def setup_fsdp(self, model):
        """配置FSDP"""
        # FSDP配置
        fsdp_config = {
            "mixed_precision": MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            "sharding_strategy": torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
            "cpu_offload": None,
        }
        
        # 使用FSDP包装模型
        model = FSDP(model, **fsdp_config)
        return model
    
    def generate(self, prompts: List[str], max_length: int = 1024, num_samples: int = 4) -> List[List[str]]:
        """生成多个响应样本"""
        all_responses = []
        
        for prompt in prompts:
            prompt_responses = []
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids.cuda(),
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                prompt_responses.append(response)
            
            all_responses.append(prompt_responses)
        
        return all_responses