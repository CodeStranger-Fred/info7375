import argparse
from model_io import load_model, generate_and_logprob
from data_utils import load_math500
from pag_env import PAGRollout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_or_ckpt", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()

    model, tok = load_model(args.model_or_ckpt, load_4bit=True)
    def gen_fn(prompt): return generate_and_logprob(model, tok, prompt)

    data = load_math500()
    pag = PAGRollout(tok, gen_fn, Tmax=2)

    n = len(data)
    t1 = 0
    tf = 0
    for item in data:
        traj = pag.one_episode(item["question"], item["solution"])
        pols = [t for t in traj if t["role"]=="policy"]
        if pols:
            t1 += int(pols[0]["correct"])
            tf += int(pols[-1]["correct"])

    print(f"Acc@t1: {t1/n:.4f}")
    print(f"Acc@final: {tf/n:.4f}")

if __name__ == "__main__":
    main()
