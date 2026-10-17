import json
import time
from typing import Type, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "qwen2.5:3b"


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def chat(self, prompt: str, system: str | None = None) -> str:
        """Réponse texte libre."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]

    def structured(
        self,
        prompt: str,
        schema: Type[T],
        system: str | None = None,
        max_retries: int = 2,
    ) -> T:
        """Réponse JSON validée par un schéma Pydantic, avec retries."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for _ in range(max_retries + 1):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                format=schema.model_json_schema(),
                options={"temperature": self.temperature},
            )
            raw = response["message"]["content"]
            try:
                return schema.model_validate_json(raw)
            except ValidationError as e:
                last_error = e
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": f"JSON invalide : {e}. Corrige et renvoie uniquement le JSON.",
                    }
                )
        raise ValueError(f"Sortie structurée invalide après retries : {last_error}")