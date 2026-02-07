import json
import os
import re
from typing import Dict, Any

import torch
from transformers import pipeline


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_TEXT_PATTERN = re.compile(r"[^a-zA-Z0-9\s\.\,\!\?\-]", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = NON_TEXT_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def model_fn(model_dir: str) -> Dict[str, Any]:
    device = 0 if torch.cuda.is_available() else -1

    summarizer_model = os.getenv("SUMMARIZER_MODEL", model_dir)
    sentiment_model = os.getenv(
        "SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
    )
    sarcasm_model = os.getenv("SARCASM_MODEL")

    summarizer = pipeline("summarization", model=summarizer_model, device=device)
    sentiment = pipeline("sentiment-analysis", model=sentiment_model, device=device)
    sarcasm = None
    if sarcasm_model:
        sarcasm = pipeline("text-classification", model=sarcasm_model, device=device)

    return {"summarizer": summarizer, "sentiment": sentiment, "sarcasm": sarcasm}


def input_fn(request_body: str, content_type: str) -> Dict[str, Any]:
    if content_type != "application/json":
        raise ValueError("Unsupported content type")
    payload = json.loads(request_body)
    return {"text": payload.get("text", "")}


def predict_fn(data: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    text = clean_text(data.get("text", ""))
    if not text:
        return {"summary": "", "sentiment": {"label": "NEUTRAL", "score": 0.0}, "sarcasm": None}

    summary = model["summarizer"](text, max_length=90, min_length=30, do_sample=False)[0][
        "summary_text"
    ]
    sentiment = model["sentiment"](text[:512])[0]
    sarcasm = None
    if model.get("sarcasm"):
        sarcasm = model["sarcasm"](text[:512])[0]

    return {
        "summary": summary,
        "sentiment": {"label": sentiment["label"], "score": float(sentiment["score"])},
        "sarcasm": (
            {"label": sarcasm["label"], "score": float(sarcasm["score"])} if sarcasm else None
        ),
    }


def output_fn(prediction: Dict[str, Any], accept: str) -> str:
    if accept != "application/json":
        raise ValueError("Unsupported accept type")
    return json.dumps(prediction)

