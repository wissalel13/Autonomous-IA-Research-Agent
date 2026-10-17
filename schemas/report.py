from pydantic import BaseModel


class SummaryText(BaseModel):
    text: str