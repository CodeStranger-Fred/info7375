from typing import List, Dict, Any
from reward_math import is_correct, pag_shaped_reward

class PAGRollout:
    def __init__(self, tokenizer, generate_fn, Tmax=2):
        self.tok = tokenizer
        self.generate = generate_fn
        self.Tmax = Tmax

    def _prompt_solve(self, question: str) -> str:
        return f"""You are a helpful math solver.
Problem:
{question}

Please reason step by step and give the final answer strictly in \\boxed{{}}."""
    def _prompt_verify(self, question: str, prev_answer: str) -> str:
        return f"""You are a strict math verifier.
Problem:
{question}

Previous answer:
{prev_answer}

Carefully verify whether the previous answer is correct. If it is wrong, explain the exact mistake and output on a new line: DECISION: REVISE
If it is correct, explain why and output on a new line: DECISION: KEEP"""

    def one_episode(self, question: str, gold_solution: str) -> List[Dict[str, Any]]:
        logs = []

        p1 = self._prompt_solve(question)
        ans1, logp1 = self.generate(p1)
        correct1 = is_correct(ans1, gold_solution)
        r1 = pag_shaped_reward(None, correct1)
        logs.append({"role":"policy", "prompt":p1, "text":ans1, "logp":logp1.item(), "reward":r1, "correct":bool(correct1), "question":question})

        v_prompt = self._prompt_verify(question, ans1)
        verdict, logp_v = self.generate(v_prompt)
        need_revise = ("DECISION: REVISE" in verdict.upper())
        logs.append({"role":"verifier", "prompt":v_prompt, "text":verdict, "logp":logp_v.item(), "reward":0.0, "correct":bool(correct1), "question":question})

        if not need_revise or self.Tmax == 1:
            return logs

        p2 = self._prompt_solve(question)
        ans2, logp2 = self.generate(p2)
        correct2 = is_correct(ans2, gold_solution)
        r2 = pag_shaped_reward(correct1, correct2)
        logs.append({"role":"policy", "prompt":p2, "text":ans2, "logp":logp2.item(), "reward":r2, "correct":bool(correct2), "question":question})

        return logs
