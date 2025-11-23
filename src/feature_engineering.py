import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_params, logger
import os

def main():
    params = load_params()["feature_engineering"]

    train_df = pd.read_csv("data/processed/train_processed.csv")
    test_df = pd.read_csv("data/processed/test_processed.csv")

    train_df["content"] = train_df["content"].fillna("")
    test_df["content"] = test_df["content"].fillna("")

    vectorizer = TfidfVectorizer(
        max_features=params["max_features"],
        ngram_range=tuple(params["ngram_range"])
    )

    X_train = vectorizer.fit_transform(train_df["content"])
    X_test = vectorizer.transform(test_df["content"])

    train_feat = pd.DataFrame(X_train.toarray())
    test_feat = pd.DataFrame(X_test.toarray())

    train_feat["label"] = train_df["sentiment"].values
    test_feat["label"] = test_df["sentiment"].values

    os.makedirs("data/features", exist_ok=True)
    train_feat.to_csv("data/features/train_bow.csv", index=False)
    test_feat.to_csv("data/features/test_bow.csv", index=False)

    logger.info(f"TF-IDF feature engineering done: {params['max_features']} features")

if __name__ == "__main__":
    main()