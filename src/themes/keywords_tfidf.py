"""
TF-IDF keyword extraction per bank.
Requires: spaCy model en_core_web_sm
Saves one CSV per bank with top terms and scores.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tqdm import tqdm
import os

nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def preprocess(text):
    doc = nlp(str(text).lower())
    tokens = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and len(t) > 2]
    return " ".join(tokens)

def extract_keywords_per_bank(df, text_col='review_text', bank_col='bank_name', top_n=40):
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df['proc'] = df[text_col].map(preprocess)
    banks = df[bank_col].unique()
    os.makedirs("data/outputs", exist_ok=True)
    results = {}
    for bank in tqdm(banks, desc="Banks"):
        docs = df.loc[df[bank_col]==bank, 'proc'].tolist()
        if not any(docs):
            results[bank] = []
            pd.DataFrame([], columns=['term','score']).to_csv(f"data/outputs/{bank}_tfidf_terms.csv", index=False)
            continue
        vect = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
        X = vect.fit_transform(docs)
        sums = X.sum(axis=0).A1
        terms = vect.get_feature_names_out()
        scored = sorted(zip(terms, sums), key=lambda x: -x[1])[:top_n]
        results[bank] = scored
        pd.DataFrame(scored, columns=['term','score']).to_csv(f"data/outputs/{bank}_tfidf_terms.csv", index=False)
    return results

if __name__ == "__main__":
    df = pd.read_csv("data/raw/reviews_raw.csv")
    extract_keywords_per_bank(df, text_col='review_text', bank_col='bank_name', top_n=50)
    print("Saved TF-IDF term files for each bank in data/outputs/")
