import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils import load_params, logger

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.punct = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

    def lower_case(self, text: str) -> str:
        return text.lower()

    def remove_urls(self, text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    def remove_punctuations(self, text: str) -> str:
        return text.translate(str.maketrans("", "", self.punct))

    def remove_numbers(self, text: str) -> str:
        return "".join([i for i in text if not i.isdigit()])

    def remove_stopwords(self, text: str) -> str:
        return " ".join([w for w in text.split() if w not in self.stop_words])

    def lemmatize(self, text: str) -> str:
        return " ".join([self.lemmatizer.lemmatize(w) for w in text.split()])

    def clean_text(self, text: str, params: dict) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.strip()

        if params.get("lowercase", True):
            text = self.lower_case(text)
        if params.get("remove_urls", True):
            text = self.remove_urls(text)
        if params.get("remove_punctuations", True):
            text = self.remove_punctuations(text)
        if params.get("remove_numbers", True):
            text = self.remove_numbers(text)
        if params.get("remove_stopwords", True):
            text = self.remove_stopwords(text)
        if params.get("apply_lemmatization", True):
            text = self.lemmatize(text)
        return text.strip()

def main():
    params = load_params()["data_preprocessing"]
    preprocessor = TextPreprocessor()

    train_df = pd.read_csv("data/raw/train.csv")
    test_df = pd.read_csv("data/raw/test.csv")

    train_df["content"] = train_df["content"].apply(lambda x: preprocessor.clean_text(x, params))
    test_df["content"] = test_df["content"].apply(lambda x: preprocessor.clean_text(x, params))

    min_words = params["min_words_per_tweet"]
    train_df = train_df[train_df["content"].str.split().str.len() >= min_words]
    test_df = test_df[test_df["content"].str.split().str.len() >= min_words]

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train_processed.csv", index=False)
    test_df.to_csv("data/processed/test_processed.csv", index=False)

    logger.info(f"Preprocessing complete: {len(train_df)} train, {len(test_df)} test samples")

if __name__ == "__main__":
    main()