# src/trainer.py
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import time
import numpy as np
from datetime import datetime
from typing import List
from src.astar_po import AStarPO
from src.data_processor import DataCollator

class TinyZeroTrainer:
    def __init__(self, model, tokenizer, train_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.config = config
        
        self.optimizer = AdamW(model.parameters(), lr=config.get('learning_rate', 5e-7))
        self.astar_po = AStarPO(model, tokenizer, 
                               beta=config.get('beta', 0.1),
                               num_samples=config.get('num_samples', 8))
        
        # 创建固定的参考模型（防止参考策略漂移）
        print("📋 Creating reference model copy...")
        import copy
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        print("✅ Reference model created (frozen)")
        
        self.global_step = 0
        self.best_reward = 0.0
        
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=True,
            collate_fn=DataCollator(self.tokenizer)
        )
        
        epoch_loss = 0.0
        epoch_reward = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        # 用于保存输出样例
        saved_outputs = []
        
        for batch in pbar:
            # 生成响应样本 - 为每个prompt生成多个样本
            all_responses = []
            for prompt in batch["prompts"]:
                prompt_responses = []
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.model.device)
                
                # 为每个prompt生成num_samples个响应
                for _ in range(self.astar_po.num_samples):
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,  # 修复：添加attention mask
                            max_new_tokens=self.config.get('max_length', 256),
                            num_return_sequences=1,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    
                    response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    prompt_responses.append(response)
                
                all_responses.append(prompt_responses)
            
            # 计算奖励
            rewards = self.astar_po.compute_rewards(
                all_responses, batch["targets"], batch["numbers_list"]
            )
            
            # 计算参考策略的logprobs（使用模型初始状态）
            reference_logprobs = self._compute_reference_logprobs(all_responses)
            
            # 计算损失
            loss = self.astar_po.compute_loss(
                batch["prompts"], all_responses, rewards, reference_logprobs
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录统计信息
            batch_reward = np.mean([np.mean(r) for r in rewards])
            epoch_loss += loss.item()
            epoch_reward += batch_reward
            num_batches += 1
            
            # 保存前3个batch的详细输出
            if num_batches <= 3:
                for i, (prompt, responses, reward_list, target, numbers) in enumerate(zip(
                    batch["prompts"], all_responses, rewards, batch["targets"], batch["numbers_list"]
                )):
                    saved_outputs.append({
                        "batch": num_batches,
                        "sample_in_batch": i,
                        "target": target,
                        "numbers": numbers,
                        "responses": responses,
                        "rewards": reward_list,
                        "avg_reward": np.mean(reward_list)
                    })
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reward': f'{batch_reward:.4f}'
            })
            
            self.global_step += 1
            
            # 保存检查点
            if self.global_step % self.config.get('save_steps', 100) == 0:
                self._save_checkpoint(epoch, epoch_loss / num_batches)
        
        # 保存详细输出到文件
        if saved_outputs:
            import json
            output_file = f'outputs_epoch_{epoch}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(saved_outputs, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved detailed outputs to {output_file}")
        
        return epoch_loss / num_batches, epoch_reward / num_batches
    
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
                        token_logprobs = torch.gather(logprobs[:-1], 2, tokens[1:].unsqueeze(-1)).squeeze(-1)
                        seq_logprob = token_logprobs.sum()
                    else:
                        seq_logprob = torch.tensor(0.0)
                    
                    prompt_logprobs.append(seq_logprob.cpu())
            reference_logprobs.append(prompt_logprobs)
        
        return reference_logprobs
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")