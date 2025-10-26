import argparse, os, json
import torch
from tqdm import tqdm
from model_io import load_model, generate_and_logprob
from data_utils import load_math_train
from pag_env import PAGRollout
from apo_value_offline import load_vstar_json
from apo_update import apo_loss

def eval_on_subset(pag, data, n=128):
    import random
    subset = random.sample(data, min(n, len(data)))
    t1, tf = 0, 0
    for item in subset:
        traj = pag.one_episode(item["question"], item["solution"])
        pols = [t for t in traj if t["role"]=="policy"]
        if pols:
            t1 += int(pols[0]["correct"])
            tf += int(pols[-1]["correct"])
    return t1/len(subset), tf/len(subset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train_ratio", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda_kl", type=float, default=0.01)
    ap.add_argument("--vstar_path", default="outputs/stage1_vstar.jsonl")
    ap.add_argument("--save_dir", default="outputs/checkpoints")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    model, tok = load_model(args.model)
    ref_model = model.__class__.from_pretrained(args.model, device_map="auto", torch_dtype=model.dtype, trust_remote_code=True)
    ref_model.eval()
    for p in ref_model.parameters(): p.requires_grad_(False)

    def gen_fn(prompt): return generate_and_logprob(model, tok, prompt)
    def ref_logprob(prompt, text):
        # recompute logprob under reference, approximate by re-generating with greedy using same output tokens
        # simplified: reuse current logp as surrogate if needed
        out_text, logp = generate_and_logprob(ref_model, tok, prompt)
        return logp.item()

    data = load_math_train(split_ratio=args.train_ratio)
    vstar = load_vstar_json(args.vstar_path)
    pag = PAGRollout(tok, gen_fn, Tmax=2)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        for item in tqdm(data, desc=f"Epoch {ep+1}"):
            traj = pag.one_episode(item["question"], item["solution"])

            ref_logps = []
            for t in traj:
                ref_lp = ref_logprob(t["prompt"], t["text"]) if t["role"] in ("policy","verifier") else None
                ref_logps.append(ref_lp)

            loss = apo_loss(traj, ref_logps, vstar_map=vstar, lambda_kl=args.lambda_kl)
            optim.zero_grad()
            torch.tensor(loss, requires_grad=True).backward()  # scalar float -> tensor
            optim.step()

        a1, af = eval_on_subset(pag, data, n=128)
        ckpt = os.path.join(args.save_dir, f"ep{ep+1}_a1{a1:.3f}_af{af:.3f}")
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        print(f"[Eval] Acc@t1={a1:.3f}  Acc@final={af:.3f}  saved: {ckpt}")

if __name__ == "__main__":
    main()
