from typing import List, Dict, Any, Optional

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
    sarcasm: Optional[Dict[str, Any]] = None


class AnalyzeResponse(BaseModel):
    subreddit: str
    count: int
    items: List[PostAnalysis]

