#!/usr/bin/env python3
"""
Supervised Fine-tuning Warmup
在 RL 训练前，先用正确答案教会模型输出格式
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.online_problem_generator import OnlineProblemGenerator
from tqdm import tqdm

def sft_warmup(model, tokenizer, num_examples=100, epochs=1, lr=1e-4):
    """
    监督学习热身：用正确答案训练模型
    """
    print("=" * 70)
    print("🎓 Supervised Fine-tuning Warmup")
    print("=" * 70)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    gen = OnlineProblemGenerator()
    
    total_loss = 0.0
    
    for epoch in range(epochs):
        print(f"\n📚 Epoch {epoch + 1}/{epochs}")
        pbar = tqdm(range(num_examples))
        
        for i in pbar:
            # 生成问题
            problem = gen.generate_problem()
            prompt = gen.make_prompt(problem)
            
            # 正确答案
            correct_answer = f"{problem['solution']}</answer>"
            full_text = prompt + correct_answer
            
            # Tokenize
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_examples
        print(f"✅ Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ SFT Warmup Complete!")
    print("💡 模型现在应该知道如何输出 <answer> 标签了")
    print("=" * 70)
    
    return model

if __name__ == "__main__":
    # 本地测试（小模型）
    model_name = "Qwen/Qwen2.5-0.5B"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # SFT warmup
    model = sft_warmup(model, tokenizer, num_examples=50, epochs=1, lr=1e-4)
    
    # 测试
    print("\n🧪 测试生成...")
    gen = OnlineProblemGenerator()
    problem = gen.generate_problem()
    prompt = gen.make_prompt(problem)
    
    print(f"\n问题: {problem['nums']} -> {problem['target']}")
    print(f"正确答案: {problem['solution']}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n模型输出:\n{response}")
