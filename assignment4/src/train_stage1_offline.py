import argparse
from tqdm import tqdm
from model_io import load_model, generate_and_logprob
from data_utils import load_math_train, save_jsonl
from pag_env import PAGRollout
from apo_value_offline import build_offline_samples, estimate_vstar_per_question, save_vstar_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train_ratio", type=float, default=0.2)  # start small for stage1
    ap.add_argument("--k", type=int, default=4)                # increase to 8 if you have time
    ap.add_argument("--out_samples", default="outputs/stage1_samples.jsonl")
    ap.add_argument("--out_vstar", default="outputs/stage1_vstar.jsonl")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    def gen_fn(prompt): return generate_and_logprob(model, tok, prompt)

    data = load_math_train(split_ratio=args.train_ratio)
    pag = PAGRollout(tok, gen_fn, Tmax=2)
    offline = build_offline_samples(pag, data, k=args.k)
    save_jsonl(args.out_samples, offline)

    q2v = estimate_vstar_per_question(offline, beta1=0.5)
    save_vstar_json(args.out_vstar, q2v)

if __name__ == "__main__":
    main()
