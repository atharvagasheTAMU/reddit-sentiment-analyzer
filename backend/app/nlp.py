import re
from typing import Dict, Any

from transformers import pipeline


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_TEXT_PATTERN = re.compile(r"[^a-zA-Z0-9\s\.\,\!\?\-]", re.IGNORECASE)

_summarizer = None
_sentiment = None


def clean_text(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = NON_TEXT_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer


def _get_sentiment():
    global _sentiment
    if _sentiment is None:
        _sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _sentiment


def summarize(text: str) -> str:
    if not text:
        return ""
    summarizer = _get_summarizer()
    result = summarizer(text, max_length=90, min_length=30, do_sample=False)
    return result[0]["summary_text"]


def analyze_sentiment(text: str) -> Dict[str, Any]:
    if not text:
        return {"label": "NEUTRAL", "score": 0.0}
    sentiment = _get_sentiment()
    result = sentiment(text[:512])[0]
    return {"label": result["label"], "score": float(result["score"])}

