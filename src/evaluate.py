"""
LLM-as-Judge evaluation for the fine-tuned Qwen model.

Uses a judge model (Claude or DeepSeek via API) to score student outputs
on Accuracy, Format, Professionalism, and Usefulness (1-5 scale).

Usage:
    python src/evaluate.py \
        --model_path outputs/qwen3_5_0_8b-recruit/merged \
        --eval_data eval/eval_set.jsonl \
        --judge claude

Eval data format (one JSON object per line):
    {
        "task": "query_parsing",
        "input": "Job Description: ...",
        "reference": "{ ... expected JSON ... }"
    }
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import get_inference_messages

# ---------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an expert evaluator for a recruiting AI system. You will be given:
1. The task description
2. The student model's output
3. A reference (gold) output

Score the student output on these 4 dimensions (1-5 each):
- Accuracy: How factually correct and complete is the output vs the reference?
- Format: Does the output follow the required format (JSON validity, section headers, etc.)?
- Professionalism: Is the tone appropriate for a recruiting context?
- Usefulness: Would a recruiter find this output actionable and helpful?

Output ONLY valid JSON with this exact schema:
{"accuracy": <int>, "format": <int>, "professionalism": <int>, "usefulness": <int>}
No explanation, no markdown fences.\
"""

JUDGE_USER_TEMPLATE = """\
Task: {task}

Student output:
---
{student_output}
---

Reference output:
---
{reference}
---\
"""


# ---------------------------------------------------------------
# Judge backends
# ---------------------------------------------------------------

def call_claude_judge(messages: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        system=messages[0]["content"],
        messages=messages[1:],
    )
    return response.content[0].text


def call_deepseek_judge(messages: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=200,
        temperature=0.0,
    )
    return response.choices[0].message.content


JUDGE_BACKENDS = {
    "claude": call_claude_judge,
    "deepseek": call_deepseek_judge,
}


# ---------------------------------------------------------------
# Student inference
# ---------------------------------------------------------------

def load_student_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return model, tokenizer


def student_generate(model, tokenizer, task: str, user_input: str) -> str:
    messages = get_inference_messages(task, user_input)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=1024, temperature=0.1, do_sample=True,
        )
    # Decode only the new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------

def parse_judge_scores(response: str) -> dict[str, int] | None:
    """Extract scores JSON from judge response."""
    try:
        # Try direct parse first, then fallback to regex
        return json.loads(response)
    except json.JSONDecodeError:
        m = re.search(r"\{[^}]+\}", response)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def evaluate(model_path: str, eval_data_path: str, judge: str):
    print(f"Loading student model: {model_path}")
    model, tokenizer = load_student_model(model_path)

    judge_fn = JUDGE_BACKENDS[judge]
    print(f"Judge backend: {judge}")

    # Load eval data
    samples = []
    with open(eval_data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} evaluation samples\n")

    all_scores = {"accuracy": [], "format": [], "professionalism": [], "usefulness": []}
    failed = 0

    for i, sample in enumerate(samples):
        task = sample["task"]
        user_input = sample["input"]
        reference = sample["reference"]

        # Generate student output
        student_output = student_generate(model, tokenizer, task, user_input)

        # Build judge messages
        judge_messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                task=task, student_output=student_output, reference=reference,
            )},
        ]

        # Call judge
        try:
            judge_response = judge_fn(judge_messages)
            scores = parse_judge_scores(judge_response)
            if scores:
                for dim in all_scores:
                    all_scores[dim].append(scores.get(dim, 0))
                print(f"  [{i+1}/{len(samples)}] {task}: {scores}")
            else:
                failed += 1
                print(f"  [{i+1}/{len(samples)}] {task}: PARSE ERROR — {judge_response[:100]}")
        except Exception as e:
            failed += 1
            print(f"  [{i+1}/{len(samples)}] {task}: ERROR — {e}")

    # Print results
    print("\n" + "=" * 50)
    print(f"{'Dimension':<20} {'Avg Score':>10} {'Count':>8}")
    print("-" * 50)
    for dim in all_scores:
        scores = all_scores[dim]
        avg = np.mean(scores) if scores else 0.0
        print(f"{dim:<20} {avg:>10.2f} {len(scores):>8}")
    print("-" * 50)
    total = sum(len(v) for v in all_scores.values()) // 4
    print(f"Evaluated: {total} samples | Failed: {failed}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation for recruitGPT")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged fine-tuned model")
    parser.add_argument("--eval_data", type=str, default="eval/eval_set.jsonl",
                        help="Path to evaluation data (JSONL)")
    parser.add_argument("--judge", type=str, default="claude", choices=list(JUDGE_BACKENDS),
                        help="Judge model backend")
    args = parser.parse_args()
    evaluate(args.model_path, args.eval_data, args.judge)


if __name__ == "__main__":
    main()
