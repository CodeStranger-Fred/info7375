import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Load model and tokenizer safely on macOS (CPU only, no MPS).
    """

    # === Force CPU mode ===
    import os
    os.environ["PYTORCH_MPS_DISABLE"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    print(f"\n🚀 Loading model {model_name} on CPU only (MPS disabled)...")

    # Optional: use more CPU threads for faster inference
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )

    # Load model to CPU in float32 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,              # no accelerate or MPS offload
        torch_dtype=torch.float32,    # safest for CPU
        trust_remote_code=True
    ).to("cpu")

    model.eval()
    print("✅ Model loaded successfully (CPU mode)\n")
    return model, tokenizer



@torch.no_grad()
def generate_and_logprob(model, tok, prompt: str, max_new_tokens: int = 32):
    print("\n[DEBUG] ===== ENTER generate_and_logprob =====")
    print(f"[DEBUG] prompt[:100]: {prompt[:100]}")

    # check model device
    first_param = next(model.parameters())
    device = first_param.device
    print(f"[DEBUG] model.device = {device}")

    # check input encoding
    inputs = tok(prompt, return_tensors="pt")
    print(f"[DEBUG] tokenized input shape: {inputs['input_ids'].shape}")
    inputs = inputs.to(device)

    # begin generation
    print("[DEBUG] calling model.generate() ...")
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True
        )
        print("[DEBUG] generation finished ✅")
    except Exception as e:
        print(f"[ERROR] generate() crashed: {e}")
        raise

    # decode result
    text = tok.decode(out[0], skip_special_tokens=True)
    print("[DEBUG] decoded output OK.")
    print("===========================================\n")
    return text, None
