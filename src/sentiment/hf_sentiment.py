"""
DistilBERT sentiment scoring (Hugging Face pipeline).
Saves a CSV with sentiment_label and sentiment_score columns.
"""

import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import os

def compute_hf_sentiment(df, text_col='review_text', model_name='distilbert-base-uncased-finetuned-sst-2-english',
                         batch_size=64, device=-1):
    """
    model_name: HF model id. device=-1 for CPU, device=0 for first GPU.
    Returns: df with columns sentiment_label, sentiment_score
    sentiment_score: positive=>+score, negative=>-score (range roughly -1..1)
    """
    classifier = pipeline("sentiment-analysis", model=model_name, device=device)
    texts = df[text_col].fillna("").astype(str).tolist()

    labels, scores = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="HF sentiment"):
        batch = texts[i:i+batch_size]
        out = classifier(batch)
        for o in out:
            label = o['label']  # "POSITIVE" or "NEGATIVE"
            score = o['score']
            if label.upper().startswith('POS'):
                signed = float(score)
                lab = 'positive'
            else:
                signed = -float(score)
                lab = 'negative'
            # optional neutral threshold; here we keep neutral if small magnitude
            if abs(signed) < 0.55:
                lab = 'neutral'
            labels.append(lab)
            scores.append(signed)

    df = df.reset_index(drop=True)
    df['sentiment_label'] = labels
    df['sentiment_score'] = scores
    return df

if __name__ == "__main__":
    # quick CLI
    path = "data/raw/reviews_raw.csv"
    outpath = "data/outputs/reviews_sentiment_hf.csv"
    os.makedirs("data/outputs", exist_ok=True)
    df = pd.read_csv(path)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df_out = compute_hf_sentiment(df, text_col='review_text', batch_size=64, device=-1)
    df_out.to_csv(outpath, index=False)
    print(f"Saved HF sentiment to {outpath}")
