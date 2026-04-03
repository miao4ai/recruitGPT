"""
Stage 5 - Match Explanation: generate a human-readable match report
for a (JD, resume) pair using a fine-tuned Qwen3.5-0.8B model.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.prompts import build_match_explainer_prompt, get_inference_messages

logger = logging.getLogger(__name__)


class Explainer:
    """Generate structured match explanations using a fine-tuned Qwen3.5-0.8B."""

    def __init__(self, model_path: str) -> None:
        """
        Args:
            model_path: Path to the merged (or LoRA-adapted) Qwen3.5-0.8B checkpoint.
        """
        logger.info("Loading Explainer model from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        self.model.eval()

    def explain(self, jd_text: str, resume_text: str) -> str:
        """
        Generate a match explanation report.

        Args:
            jd_text: Raw job description text.
            resume_text: Raw resume / candidate profile text.

        Returns:
            Structured match report as plain text (Strengths / Gaps / Interview Focus / Recommendation).
        """
        user_content = build_match_explainer_prompt(jd_text, resume_text)
        messages = get_inference_messages("match_explanation", user_content)

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

        generated_ids = outputs[0][input_ids.shape[1] :]
        explanation = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return explanation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate match explanation for a JD+resume pair")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned Qwen model")
    parser.add_argument("--jd", required=True, help="Job description text")
    parser.add_argument("--resume", required=True, help="Resume text")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    explainer = Explainer(args.model_path)
    report = explainer.explain(args.jd, args.resume)
    print(report)
