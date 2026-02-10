# Reddit-AnalyXer (Sentiment + Summarization)

A full-stack Reddit analysis app with:

- FastAPI backend that fetches posts via the Reddit API.
- NLP pipeline for cleaning, summarizing, and sentiment analysis.
- React frontend to query a subreddit and view results.

## Backend (FastAPI)

### Setup

1. Create a Reddit API app (script type) at https://www.reddit.com/prefs/apps
2. Set environment variables:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=reddit-analyzer/1.0 by your_username
```

3. Install requirements and run:

```
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Frontend (React)

```
cd frontend
npm install
npm run dev
```

Open the dev server URL and query a subreddit.

## Notes

- Summarization uses `facebook/bart-large-cnn`.
- Sentiment uses `distilbert-base-uncased-finetuned-sst-2-english`.
- Text is cleaned by removing URLs and special characters.

## Optional: Sarcasm Detection (RoBERTa)

You can enable sarcasm detection by providing a RoBERTa-based model and flag:

```
ENABLE_SARCASM=true
SARCASM_MODEL=your_roberta_sarcasm_model_id
```

When enabled, the API returns a `sarcasm` field per post with label and score.

