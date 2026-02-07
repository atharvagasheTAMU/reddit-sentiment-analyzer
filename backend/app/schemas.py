from typing import List, Dict, Any

from pydantic import BaseModel


class PostAnalysis(BaseModel):
    id: str
    title: str
    url: str
    score: int
    created_utc: float
    cleaned_text: str
    summary: str
    sentiment: Dict[str, Any]


class AnalyzeResponse(BaseModel):
    subreddit: str
    count: int
    items: List[PostAnalysis]

