"""
Stage 1 - Query Parser: parse a job description into structured JSON
using a fine-tuned Qwen3.5-0.8B model.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.prompts import build_query_parser_prompt, get_inference_messages

logger = logging.getLogger(__name__)


class QueryParser:
    """Load a fine-tuned Qwen3.5-0.8B and extract structured hiring intent from a JD."""

    def __init__(self, model_path: str) -> None:
        """
        Args:
            model_path: Path to the merged (or LoRA-adapted) Qwen3.5-0.8B checkpoint.
        """
        logger.info("Loading QueryParser model from %s", model_path)
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

    def parse(self, jd_text: str) -> dict:
        """
        Parse a job description into a structured JSON dict.

        Args:
            jd_text: Raw job description text.

        Returns:
            Parsed dict with keys like role_title, seniority, required_skills, etc.
        """
        user_content = build_query_parser_prompt(jd_text)
        messages = get_inference_messages("query_parsing", user_content)

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][input_ids.shape[1] :]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning("Model output is not valid JSON, returning raw text wrapped in dict")
            parsed = {"raw_output": raw_output}

        return parsed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a JD into structured JSON")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned Qwen model")
    parser.add_argument("--jd", required=True, help="Job description text")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    qp = QueryParser(args.model_path)
    result = qp.parse(args.jd)
    print(json.dumps(result, indent=2, ensure_ascii=False))
