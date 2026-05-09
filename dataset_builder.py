import random
import pandas as pd

from preprocessing import preprocess_texts
from scoring import compute_score_v2
from vtt_to_text import load_all_vtt

from datasets import load_dataset


# -------------------------
# LOAD DATASETS
# -------------------------
def load_medium():
    from datasets import load_dataset
    dataset = load_dataset("ag_news", split="train")

    return [
        x["text"]
        for x in dataset
        if len(x["text"].split()) > 5
    ]


def load_low():
    from datasets import load_dataset
    dataset = load_dataset("tweet_eval", "emotion", split="train")

    return [
        x["text"]
        for x in dataset
        if len(x["text"].split()) > 5
    ]


# -------------------------
# BUILD DATASET: dataset_builder_v2.py
# -------------------------

def build_dataset():

    print("🚀 Loading datasets...")

    ted_texts = load_all_vtt("tedDirector/subtitles")[:800]
    claire_texts = load_medium()[:400]
    yt_texts = load_low()[:400]

    all_texts = ted_texts + claire_texts + yt_texts
    random.shuffle(all_texts)

    print(f"📊 Total texts: {len(all_texts)}")

    print("🔄 Preprocessing...")
    processed = preprocess_texts(all_texts)

    data = []

    for i, ((tokens, doc), text) in enumerate(zip(processed, all_texts)):
        if i % 100 == 0:
            print(f"Processing {i} texts...")

        try:
            score = compute_score_v2(text, tokens)

            data.append({
                "text": text,
                "score": score
            })

        except Exception as e:
            print("⚠️ Skipped:", e)
            continue

    df = pd.DataFrame(data)

    print("🧹 Cleaning...")

    df = df.dropna()
    df = df[df["score"] >= 0]
    df = df[df["score"] <= 10]

    df.to_csv("dataset_v2.csv", index=False)

    return df


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    df = build_dataset()
    print("✅ DONE")
    print(df.head())