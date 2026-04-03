"""
Interactive inference for the fine-tuned Qwen model.

Supports both query_parsing and match_explanation tasks.

Usage:
    python src/inference.py \
        --model outputs/qwen3_5_0_8b-recruit/merged \
        --task query_parsing
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import get_inference_messages, build_query_parser_prompt, build_match_explainer_prompt


def load_model(model_path: str):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")
    return model, tokenizer


def generate(model, tokenizer, task: str, user_input: str,
             max_new_tokens: int = 1024, temperature: float = 0.1) -> str:
    messages = get_inference_messages(task, user_input)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def interactive_loop(model, tokenizer, task: str):
    print(f"Task: {task}")
    print("Type your input below. Use Ctrl+D (EOF) to finish multi-line input.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            if task == "query_parsing":
                print("--- Paste a Job Description (then Ctrl+D): ---")
                lines = []
                try:
                    while True:
                        lines.append(input())
                except EOFError:
                    pass
                raw = "\n".join(lines).strip()
                if raw.lower() in ("quit", "exit"):
                    break
                if not raw:
                    continue
                user_input = build_query_parser_prompt(raw)

            elif task == "match_explanation":
                print("--- Paste a Job Description (then Ctrl+D): ---")
                jd_lines = []
                try:
                    while True:
                        jd_lines.append(input())
                except EOFError:
                    pass
                jd = "\n".join(jd_lines).strip()
                if jd.lower() in ("quit", "exit"):
                    break
                if not jd:
                    continue

                print("\n--- Paste a Resume (then Ctrl+D): ---")
                resume_lines = []
                try:
                    while True:
                        resume_lines.append(input())
                except EOFError:
                    pass
                resume = "\n".join(resume_lines).strip()
                if not resume:
                    continue
                user_input = build_match_explainer_prompt(jd, resume)
            else:
                print(f"Unknown task: {task}")
                break

            print("\nGenerating...\n")
            output = generate(model, tokenizer, task, user_input)
            print("=" * 60)
            print(output)
            print("=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Interactive inference for recruitGPT")
    parser.add_argument("--model", type=str, required=True, help="Path to merged model")
    parser.add_argument("--task", type=str, required=True,
                        choices=["query_parsing", "match_explanation"],
                        help="Task to run")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    interactive_loop(model, tokenizer, args.task)


if __name__ == "__main__":
    main()
