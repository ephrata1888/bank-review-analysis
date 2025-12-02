"""
Simple lexicon-based sentiment fallback.
This uses a tiny built-in wordlist to produce a sentiment score.
Useful when you can't run transformers or want a fast comparison.
"""

import pandas as pd
import re
from tqdm import tqdm
import os

# small polarity dictionaries (extend as needed)
_POS = {"good","great","excellent","love","liked","fast","easy","helpful","amazing","best"}
_NEG = {"bad","terrible","slow","hate","worst","crash","crashes","error","failed","lag","disappointing"}

def lexicon_score(text):
    text = str(text).lower()
    # simple tokenization
    words = re.findall(r"\w+", text)
    pos = sum(1 for w in words if w in _POS)
    neg = sum(1 for w in words if w in _NEG)
    if pos + neg == 0:
        return 0.0  # neutral
    return (pos - neg) / (pos + neg)  # -1..1

def compute_lexicon(df, text_col='review_text'):
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    scores = []
    labels = []
    for t in tqdm(df[text_col], desc="Lexicon sentiment"):
        s = lexicon_score(t)
        scores.append(s)
        if s > 0.05:
            labels.append("positive")
        elif s < -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")
    df['sentiment_score'] = scores
    df['sentiment_label'] = labels
    return df

if __name__ == "__main__":
    path = "data/raw/reviews_raw.csv"
    outpath = "data/outputs/reviews_sentiment_lexicon.csv"
    os.makedirs("data/outputs", exist_ok=True)
    df = pd.read_csv(path)
    df_out = compute_lexicon(df, text_col='review_text')
    df_out.to_csv(outpath, index=False)
    print(f"Saved lexicon sentiment to {outpath}")
