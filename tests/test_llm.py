import time
from typing import Literal

from pydantic import BaseModel

from llm.client import LLMClient


class ImbalanceDecision(BaseModel):
    strategy: Literal["class_weight", "smote", "undersampling", "none"]
    reason: str


if __name__ == "__main__":
    llm = LLMClient()

    t = time.time()
    print(llm.chat("Réponds juste OK"))
    print(f"Texte libre : {time.time() - t:.1f}s\n")

    t = time.time()
    decision = llm.structured(
        "Un dataset de fraude a 2% de positifs. Quelle stratégie de rééquilibrage choisir ?",
        ImbalanceDecision,
    )
    print(decision)
    print(f"Sortie structurée : {time.time() - t:.1f}s")