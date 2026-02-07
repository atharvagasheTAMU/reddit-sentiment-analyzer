import { useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [subreddit, setSubreddit] = useState("python");
  const [limit, setLimit] = useState(10);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runAnalysis = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setItems([]);
    try {
      const response = await axios.get(`${API_BASE}/analyze`, {
        params: { subreddit, limit }
      });
      setItems(response.data.items || []);
    } catch (err) {
      const message =
        err?.response?.data?.detail || "Failed to fetch analysis results.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="hero">
        <h1>Reddit-AnalyXer</h1>
        <p>
          A web app that fetches and summarizes the latest posts from any Reddit
          subreddit using NLP.
        </p>
      </header>

      <section className="panel">
        <h2>Analyze a Subreddit</h2>
        <form onSubmit={runAnalysis} className="form">
          <label>
            Subreddit
            <input
              value={subreddit}
              onChange={(e) => setSubreddit(e.target.value)}
              placeholder="e.g. python"
            />
          </label>
          <label>
            Post limit
            <input
              type="number"
              min="1"
              max="25"
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
        </form>

        {error && <div className="error">{error}</div>}
      </section>

      <section className="results">
        {items.map((item) => (
          <article key={item.id} className="card">
            <header>
              <h3>{item.title}</h3>
              <a href={item.url} target="_blank" rel="noreferrer">
                Open on Reddit
              </a>
            </header>
            <div className="meta">
              <span>Score: {item.score}</span>
              <span>
                Sentiment: {item.sentiment?.label} (
                {item.sentiment?.score?.toFixed?.(2)})
              </span>
              {item.sarcasm && (
                <span>
                  Sarcasm: {item.sarcasm?.label} (
                  {item.sarcasm?.score?.toFixed?.(2)})
                </span>
              )}
            </div>
            <p className="summary">{item.summary || "No summary available."}</p>
          </article>
        ))}
      </section>
    </div>
  );
}

