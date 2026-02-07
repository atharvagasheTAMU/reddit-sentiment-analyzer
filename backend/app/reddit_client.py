from typing import List, Dict, Any

import praw

from app.config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, MAX_POST_LIMIT

HARD_CODED_CLIENT_ID = "YOUR_CLIENT_ID"
HARD_CODED_CLIENT_SECRET = "YOUR_CLIENT_SECRET"
HARD_CODED_USER_AGENT = "reddit-test/1.0 by your_reddit_username"


def create_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=HARD_CODED_CLIENT_ID or REDDIT_CLIENT_ID,
        client_secret=HARD_CODED_CLIENT_SECRET or REDDIT_CLIENT_SECRET,
        user_agent=HARD_CODED_USER_AGENT or REDDIT_USER_AGENT,
    )


def fetch_latest_posts(subreddit: str, limit: int) -> List[Dict[str, Any]]:
    client = create_client()
    safe_limit = max(1, min(limit, MAX_POST_LIMIT))
    posts = []
    for submission in client.subreddit(subreddit).new(limit=safe_limit):
        posts.append(
            {
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext or "",
                "score": submission.score,
                "url": submission.url,
                "created_utc": submission.created_utc,
            }
        )
    return posts


if __name__ == "__main__":
    results = fetch_latest_posts("python", 3)
    for item in results:
        print(item["title"])

