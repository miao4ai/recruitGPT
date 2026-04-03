"""
Unified teacher model interface for distillation.

Supports multiple backends: Claude (Anthropic API), OpenAI, DeepSeek (OpenAI-compatible).

Usage:
    from src.teacher import get_teacher

    teacher = get_teacher("claude", "claude-sonnet-4-20250514")
    response = teacher.generate([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ])
"""

import os
from abc import ABC, abstractmethod


class TeacherModel(ABC):
    """Base class for teacher model backends."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, messages: list[dict], max_tokens: int = 2048,
                 temperature: float = 0.0) -> str:
        """Generate a response from the teacher model."""
        ...


class ClaudeTeacher(TeacherModel):
    """Anthropic Claude backend."""

    def __init__(self, model_name: str = "claude-sonnet-4-20250514"):
        super().__init__(model_name)
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def generate(self, messages: list[dict], max_tokens: int = 2048,
                 temperature: float = 0.0) -> str:
        # Separate system message from the rest
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        kwargs = dict(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat_messages,
        )
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OpenAITeacher(TeacherModel):
    """OpenAI API backend."""

    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name)
        from openai import OpenAI
        self.client = OpenAI()  # reads OPENAI_API_KEY from env

    def generate(self, messages: list[dict], max_tokens: int = 2048,
                 temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


class DeepSeekTeacher(TeacherModel):
    """DeepSeek API backend (OpenAI-compatible)."""

    def __init__(self, model_name: str = "deepseek-chat"):
        super().__init__(model_name)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

    def generate(self, messages: list[dict], max_tokens: int = 2048,
                 temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------
# Factory
# ---------------------------------------------------------------

_PROVIDERS = {
    "claude": ClaudeTeacher,
    "openai": OpenAITeacher,
    "deepseek": DeepSeekTeacher,
}


def get_teacher(provider: str, model_name: str | None = None) -> TeacherModel:
    """
    Factory to create a teacher model instance.

    Args:
        provider: one of "claude", "openai", "deepseek"
        model_name: optional model name override (uses provider default if None)

    Returns:
        TeacherModel instance
    """
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")
    cls = _PROVIDERS[provider]
    if model_name:
        return cls(model_name)
    return cls()
