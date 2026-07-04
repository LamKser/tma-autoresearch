"""Prepare the IMDB review data for the BERT text-classification workflow."""

import os
from pathlib import Path
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score

MAX_LENGTH = 256

REPO_ROOT = Path(__file__).resolve().parent
AUTO_ROOT = REPO_ROOT.parent / "tma-auto"
DEFAULT_DATA_DIR = AUTO_ROOT / "drafts" / "data" / "imdb"
DATA_DIR = Path(os.environ.get("IMDB_DATA_DIR", DEFAULT_DATA_DIR))
TRAIN_CSV = DATA_DIR / "imdb_train.csv"
TEST_CSV = DATA_DIR / "imdb_test.csv"
UNSUPERVISED_CSV = DATA_DIR / "imdb_unsupervised.csv"


def download_data() -> None:
    """Download the IMDB dataset if the local CSVs are not available."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_CSV.exists() and TEST_CSV.exists() and UNSUPERVISED_CSV.exists():
        print(f"Data: already available at {DATA_DIR}")
        return

    print("Data: downloading IMDB dataset from Hugging Face ...")
    dataset = load_dataset("stanfordnlp/imdb", cache_dir=str(DATA_DIR))

    dataset["train"].to_csv(TRAIN_CSV)
    dataset["test"].to_csv(TEST_CSV)
    dataset["unsupervised"].to_csv(UNSUPERVISED_CSV)

    print(f"Data: saved to {DATA_DIR}")


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(int)


    # df = df[["text", "label"]].dropna().reset_index(drop=True)
    # df.columns = ["text", "label"]
    return df


def load_imdb_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test dataframes used by the training script."""
    train_df = _normalize_dataframe(pd.read_csv(TRAIN_CSV))
    test_df = _normalize_dataframe(pd.read_csv(TEST_CSV))
    return train_df, test_df


def get_tokenizer_vocab_size(tokenizer: BertTokenizer) -> int:
    """Return the vocabulary size used by the tokenizer."""
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer.get_vocab())
    return int(vocab_size)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label),
        }

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, str]:
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(actual_labels, predictions)
    return accuracy

if __name__ == "__main__":
    download_data()

