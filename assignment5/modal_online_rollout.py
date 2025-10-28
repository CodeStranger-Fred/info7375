#!/usr/bin/env python3
"""
True Online Rollout Training - TinyZero Style
每次迭代动态生成新问题，不使用预先准备的数据集
"""
import modal

app = modal.App("tinyzero-online-rollout")

# Volumes
checkpoint_volume = modal.Volume.from_name("tinyzero-checkpoints", create_if_missing=True)
output_volume = modal.Volume.from_name("tinyzero-outputs", create_if_missing=True)

VOLUME_CHECKPOINT_PATH = "/checkpoints"
VOLUME_OUTPUT_PATH = "/outputs"

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "tqdm>=4.65.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "numpy",
    )
    .add_local_dir("src", "/root/src")
)

GPU_CONFIG = "A100"  # 1x A100 (当前代码不支持多GPU并行)

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        VOLUME_CHECKPOINT_PATH: checkpoint_volume,
        VOLUME_OUTPUT_PATH: output_volume,
    },
    timeout=28800,  # 8 hours for 1000 problems on 1x A100
)
def online_rollout_training():
    """Online Rollout训练 - TinyZero风格"""
    import torch
    import json
    import sys
    import time
    
    sys.path.insert(0, '/root')
    
    from src.model_manager import ModelManager
    from src.online_trainer import OnlineRolloutTrainer
    
    start_time = time.time()
    
    print("=" * 70)
    print("🚀 TinyZero Online Rollout Training")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Training config - 1000 problems on 8xA100
    config = {
        "model_name": "Qwen/Qwen2.5-3B",
        "num_iterations": 10,  # 10 次迭代
        "problems_per_iteration": 100,  # 每次 100 个问题 = 1000 总问题
        "learning_rate": 5e-5,
        "beta": 0.1,
        "num_samples": 4,  # 每个问题 4 个样本
        "max_length": 50,  # 答案应该很短
        "save_steps": 100,  # 每 100 步保存一次
        "ref_update_frequency": 50,  # 参考模型更新频率
        "ema_decay": 0.95,  # EMA 衰减系数
        "sft_warmup_steps": 200,  # SFT warmup
        "initial_temperature": 0.7,  # 降低初始温度
        "min_temperature": 0.3,
    }
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Model: {config['model_name']}")
    print(f"   Iterations: {config['num_iterations']}")
    print(f"   Problems per iteration: {config['problems_per_iteration']}")
    print(f"   Total problems: {config['num_iterations'] * config['problems_per_iteration']}")
    print(f"   Samples per problem: {config['num_samples']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   KL penalty (beta): {config['beta']}")
    
    # Load model
    print("\n🔄 Loading model and tokenizer...")
    model_load_start = time.time()
    model_manager = ModelManager(config["model_name"])
    tokenizer, model = model_manager.load_model_and_tokenizer()
    print(f"✅ Model loaded in {time.time() - model_load_start:.2f}s")
    
    # Initialize online trainer
    print("\n🏋️  Initializing Online Rollout Trainer...")
    trainer = OnlineRolloutTrainer(model, tokenizer, config)
    
    # Training loop
    training_stats = []
    iteration_times = []
    
    print("\n" + "=" * 70)
    print("🎯 Starting Online Rollout Training")
    print("   每次迭代都会生成全新的问题！")
    print("=" * 70)
    
    for iteration in range(config['num_iterations']):
        iteration_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"📊 Iteration {iteration + 1}/{config['num_iterations']}")
        print(f"   将生成 {config['problems_per_iteration']} 个新问题")
        print(f"{'='*70}")
        
        try:
            # Online rollout: 动态生成新问题并训练
            avg_loss, avg_reward = trainer.train_iteration(
                iteration, 
                config['problems_per_iteration']
            )
            
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            stats = {
                "iteration": iteration,
                "avg_loss": float(avg_loss),
                "avg_reward": float(avg_reward),
                "iteration_time": iteration_time,
                "total_problems_seen": (iteration + 1) * config['problems_per_iteration']
            }
            training_stats.append(stats)
            
            print(f"\n✅ Iteration {iteration + 1} completed:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Reward: {avg_reward:.4f} ({avg_reward*100:.1f}%)")
            print(f"   Time: {iteration_time/60:.1f} min")
            print(f"   Total problems seen: {stats['total_problems_seen']}")
            
            # Save checkpoint
            checkpoint_path = f"{VOLUME_CHECKPOINT_PATH}/online_iter_{iteration}.pt"
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "stats": stats,
            }, checkpoint_path)
            checkpoint_volume.commit()
            
        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Save final results
    stats_path = f"{VOLUME_OUTPUT_PATH}/online_training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    # Copy output files to volume
    import glob
    import shutil
    output_files = glob.glob('outputs_iteration_*.json')
    for output_file in output_files:
        target_path = f"{VOLUME_OUTPUT_PATH}/{output_file}"
        shutil.copy(output_file, target_path)
        print(f"📤 Copied {output_file} to volume")
    
    output_volume.commit()
    
    total_time = time.time() - start_time
    total_problems = config['num_iterations'] * config['problems_per_iteration']
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 Online Rollout Training Complete!")
    print("=" * 70)
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Total Problems Generated: {total_problems}")
    print(f"Avg Time per Iteration: {sum(iteration_times)/len(iteration_times)/60:.1f} min")
    print(f"\nProgress:")
    for i, stat in enumerate(training_stats):
        print(f"  Iter {i+1}: Loss={stat['avg_loss']:.4f}, Reward={stat['avg_reward']*100:.1f}%, Problems={stat['total_problems_seen']}")
    
    if training_stats:
        final_reward = training_stats[-1]['avg_reward']
        initial_reward = training_stats[0]['avg_reward']
        improvement = (final_reward - initial_reward) / max(initial_reward, 0.01) * 100
        
        print(f"\n📈 Reward Progress:")
        print(f"   Initial: {initial_reward*100:.1f}%")
        print(f"   Final: {final_reward*100:.1f}%")
        if improvement > 0:
            print(f"   Improvement: +{improvement:.1f}%")
        
        if final_reward > 0.3:
            print(f"\n🎉 Great! The model learned to solve problems!")
        elif final_reward > 0.1:
            print(f"\n✅ Good progress! Model is learning.")
        else:
            print(f"\n💡 Model needs more training or hyperparameter tuning.")
    
    print("\n💾 All results saved to volumes")
    print("=" * 70)
    
    return training_stats

@app.local_entrypoint()
def main():
    """Run online rollout training."""
    print("\n" + "=" * 70)
    print("🚀 TinyZero Online Rollout Training")
    print("=" * 70)
    print("📋 Configuration:")
    print("   - 5 iterations")
    print("   - 50 problems per iteration (dynamically generated)")
    print("   - 8 samples per problem")
    print("   - Total: 250 unique problems")
    print("   - Estimated time: ~40-60 minutes")
    print("   - Estimated cost: ~$2-3")
    print("\n🎯 每次迭代都会生成全新的问题 - 真正的Online Rollout！")
    print("=" * 70 + "\n")
    
    stats = online_rollout_training.remote()
    
    print(f"\n✅ Online Rollout Training completed!")
    print(f"📊 Check results in Modal volumes")
