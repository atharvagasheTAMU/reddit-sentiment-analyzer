import os

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


REDDIT_CLIENT_ID = get_env("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = get_env("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = get_env("REDDIT_USER_AGENT")

MAX_POST_LIMIT = int(os.getenv("MAX_POST_LIMIT", "25"))

ENABLE_SARCASM = os.getenv("ENABLE_SARCASM", "false").lower() in {"1", "true", "yes"}
SARCASM_MODEL = os.getenv("SARCASM_MODEL")

SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
)

