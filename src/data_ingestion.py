import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils import load_params, logger

def main():
    params = load_params()["data_ingestion"]
    test_size = params["test_size"]
    random_state = params["random_state"]

    url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
    df = pd.read_csv(url)

    df = df.drop(columns=["tweet_id"], errors="ignore")
    df = df[df["sentiment"].isin(["happiness", "sadness"])]
    df["sentiment"] = df["sentiment"].map({"happiness": 1, "sadness": 0})

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["sentiment"]
    )

    os.makedirs("data/raw", exist_ok=True)
    train_df.to_csv("data/raw/train.csv", index=False)
    test_df.to_csv("data/raw/test.csv", index=False)

    logger.info(f"Data ingested: {len(train_df)} train, {len(test_df)} test samples")

if __name__ == "__main__":
    main()