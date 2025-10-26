import json
import random
from datasets import load_dataset

def load_math_train(split_ratio=1.0, seed=42):
    # HF "hendrycks/competition_math" mirrors MATH repo problems/solutions
    ds = load_dataset("qwedsacf/competition_math", split="train")
    data = []
    for ex in ds:
        data.append({
            "question": ex.get("problem", ""),
            "solution": ex.get("solution", ""),
            "level": ex.get("level", None),
            "type": ex.get("type", None)
        })
    if split_ratio < 1.0:
        random.Random(seed).shuffle(data)
        k = int(len(data) * split_ratio)
        data = data[:k]
    return data

def load_math500():
    # HF curated 500-problem split for benchmarking
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    data = []
    for ex in ds:
        data.append({
            "question": ex["problem"],
            "solution": ex["solution"],
            "path": ex.get("path", "")
        })
    return data

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows
