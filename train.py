"""Train a BERT classifier on the IMDB review dataset using the notebook workflow."""

import random
import time
from pathlib import Path


import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer, get_linear_schedule_with_warmup

from prepare import get_tokenizer_vocab_size, load_imdb_data, evaluate, TextClassificationDataset, MAX_LENGTH


# ---------------------------------------------------------------------------
# BERT Model
# ---------------------------------------------------------------------------
class BERTClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        bert_config=None
    ):
        super().__init__()
        config = bert_config or BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
MAX_STEPS = 2000
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
 
# Model architecture
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 4
 
# Model size
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 3072
 
# Dropout
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_PROBS_DROPOUT_PROB = 0.1
 
# Optimization
LEARNING_RATE = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-2


# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

def main() -> None:
    torch.cuda.manual_seed(42)
    device = torch.device("cuda")
    print(f"Device: {device}")

    start_time = time.time()
    train_df, test_df = load_imdb_data()
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        combined_df["text"].tolist(),
        combined_df["label"].tolist(),
        test_size=0.2,
        random_state=42,
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = get_tokenizer_vocab_size(tokenizer)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BERTClassifier(
        num_classes=2,
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=ATTENTION_PROBS_DROPOUT_PROB
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = MAX_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    start_training_time = time.time()
    step = 0
    model.train()
    data_iter = iter(train_dataloader)

    while step < MAX_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1

        # if step % 100 == 0 or step == MAX_STEPS:
        #     print(f"Step {step}/{MAX_STEPS} - Loss: {loss.item():.4f}")

    end_training_time = time.time()
    total_training_time = end_training_time - start_training_time

    # Validation
    accuracy = evaluate(model, val_dataloader, device)
    
    checkpoint_path = CHECKPOINT_DIR / "bert_classifier.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    end_time = time.time()

    total_tokens = step * BATCH_SIZE * MAX_LENGTH
    num_params = sum(p.numel() for p in model.parameters())
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    total_time = end_time - start_time

    print("---")
    print(f"val_acc:          {accuracy:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {NUM_HIDDEN_LAYERS}")

if __name__ == "__main__":
    main()
