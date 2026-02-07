from typing import List

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.reddit_client import fetch_latest_posts
from app.nlp import clean_text, summarize, analyze_sentiment, analyze_sarcasm
from app.schemas import AnalyzeResponse, PostAnalysis
from app.config import MAX_POST_LIMIT


app = FastAPI(title="Reddit-AnalyXer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/analyze", response_model=AnalyzeResponse)
def analyze_subreddit(
    subreddit: str = Query(..., min_length=2, max_length=50),
    limit: int = Query(10, ge=1, le=MAX_POST_LIMIT),
) -> AnalyzeResponse:
    try:
        posts = fetch_latest_posts(subreddit=subreddit, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items: List[PostAnalysis] = []
    for post in posts:
        text = f"{post['title']}\n{post['selftext']}".strip()
        cleaned = clean_text(text)
        summary = summarize(cleaned) if cleaned else ""
        sentiment = analyze_sentiment(cleaned)
        sarcasm = analyze_sarcasm(cleaned)
        items.append(
            PostAnalysis(
                id=post["id"],
                title=post["title"],
                url=post["url"],
                score=post["score"],
                created_utc=post["created_utc"],
                cleaned_text=cleaned,
                summary=summary,
                sentiment=sentiment,
                sarcasm=sarcasm,
            )
        )

    return AnalyzeResponse(subreddit=subreddit, count=len(items), items=items)

