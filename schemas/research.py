from typing import Literal

from pydantic import BaseModel


class SentenceChoice(BaseModel):
    """Sortie demandée au LLM : le numéro de la phrase retenue."""
    index: int


class Finding(BaseModel):
    topic: str
    decision: str
    passage_id: str
    section: str
    passage_text: str
    note: str  # phrase extraite du passage, telle quelle
    note_source: Literal["llm", "single", "default"]


class ResearchReport(BaseModel):
    findings: list[Finding]
    corrections: list[str]