"""
Rule-based theme assignment. Edit theme_map after inspecting TF-IDF outputs.
Saves per-review themes and a theme-count CSV per bank.
"""

import pandas as pd
import re
import os

# Example theme map. Update terms after TF-IDF inspection.
theme_map = {
    "Account Access Issues": ["login","password","signin","otp","2fa","auth","authenticate","authen"],
    "Transaction Performance": ["transfer","delay","pending","failed","refund","processing","timeout"],
    "App Performance & Stability": ["slow","crash","crashes","freeze","bug","lag","error"],
    "Customer Support": ["support","customer service","agent","help","ticket","call center","respond"],
    "Card & Payments": ["card","atm","payment","pos","charge","decline"],
    "UX & Features": ["interface","ui","feature","navigation","design","experience","toggle"]
}

def assign_themes(text):
    t = str(text).lower()
    themes = []
    for theme, keywords in theme_map.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", t):
                themes.append(theme)
                break
    if not themes:
        themes = ["Other"]
    return themes

if __name__ == "__main__":
    # read sentiment file (HF preferred) — fallback to lexicon if HF not produced
    hf_path = "data/outputs/reviews_sentiment_hf.csv"
    lex_path = "data/outputs/reviews_sentiment_lexicon.csv"
    if os.path.exists(hf_path):
        df = pd.read_csv(hf_path)
    elif os.path.exists(lex_path):
        df = pd.read_csv(lex_path)
    else:
        df = pd.read_csv("data/raw/reviews_raw.csv")
    df['themes'] = df['review_text'].apply(assign_themes)
    os.makedirs("data/outputs", exist_ok=True)
    df.to_csv("data/outputs/reviews_sentiment_themes.csv", index=False)

    exploded = df.explode('themes')
    pivot = exploded.groupby(['bank_name','themes']).size().reset_index(name='n_reviews')
    pivot.to_csv("data/outputs/theme_counts_by_bank.csv", index=False)
    print("Assigned themes and saved outputs.")
