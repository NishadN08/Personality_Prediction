from flask import (Flask, flash, redirect, render_template, request, send_file, url_for)
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

application = Flask(__name__)

CSV_PATH = "Personalities.csv"

df = pd.read_csv(CSV_PATH).fillna("")
df["posts_clean"] = (
    df["posts"]
      .astype(str)
      .str.lower()
      .str.replace(r"[^a-z0-9\s]", " ", regex=True)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 4), min_df=1)
X = vectorizer.fit_transform(df["posts_clean"])

class DataStore:
    last_query = ""
    last_results_df = pd.DataFrame()

data_store = DataStore()


def run_search(query: str, top_k: int = 5) -> pd.DataFrame:
    
    q = (query or "").strip()
    if not q:
        return pd.DataFrame()

    
    words = [w for w in re.findall(r"\w+", q.lower()) if len(w) > 2]
    candidates = df
    if words:
        masks = [candidates["posts_clean"].str.contains(re.escape(w), na=False) for w in words]
        mask_any = np.logical_or.reduce(masks) if len(masks) > 1 else masks[0]
        candidates = candidates[mask_any].copy()

    if candidates.empty:
        return pd.DataFrame()

    
    qv = vectorizer.transform([q])
    sims = linear_kernel(qv, X[candidates.index]).ravel()

    out = candidates.copy()
    out["Simiscore"] = sims
    out = (out
           .sort_values("Simiscore", ascending=False)
           .head(top_k)
           .reset_index(drop=True))
    out["Rank"] = np.arange(1, len(out) + 1)
    out["Search Term"] = q

    
    cols = [
        "type", "Rank", "Simiscore", "Search Term",
        "Introversion/Extraversion", "Intuitive/Observant",
        "Thinking/Feeling", "Judging/Perceiving"
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out["Simiscore"] = out["Simiscore"].round(4)

    return out[cols]


@application.route("/", methods=["GET", "POST"])
@application.route("/main", methods=["GET", "POST"])
def homepage():
    if request.method == "POST":
        query = request.form.get("question1_field", "Opportunistic").strip()
    else:
        
        query = request.args.get("q", "Opportunistic").strip()

    results = run_search(query)

    if results.empty:
        flash("No matching posts found.")
        docs = []
    else:
        docs = results.to_dict(orient="records")

    
    data_store.last_query = query
    data_store.last_results_df = results.copy()

    return render_template("Pdocuments.html", docs=docs, data1=query)

if __name__ == "__main__":
    application.run(debug=True)
