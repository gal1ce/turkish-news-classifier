# Imports
import json
import os

import torch
from datasets import load_dataset
from fastapi import FastAPI
from pydantic import BaseModel
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set configs.
# TTC4900 is a Turkish corpus, so the base model is a Turkish cased BERT.
# An English uncased model folds İ/ı and I/i together, which throws away
# information the classifier needs.
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MODEL_PATH = "model.pth"
LABELS_PATH = "labels.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)


def load_data():
    """Load TTC4900, tokenize it, and return loaders plus the label names.

    The label names come from the dataset's own ClassLabel feature, so index i
    in the model output always maps to labels[i] — no hand-written mapping to
    drift out of sync with the data.
    """
    df = load_dataset("savasy/ttc4900")
    labels = df["train"].features["category"].names

    split = df["train"].shuffle(seed=42).train_test_split(test_size=0.2, seed=42)
    train_data = split["train"].map(tokenize, batched=True)
    test_data = split["test"].map(tokenize, batched=True)

    columns = ["input_ids", "attention_mask", "category"]
    train_data.set_format("torch", columns=columns)
    test_data.set_format("torch", columns=columns)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8,
                                               shuffle=True)  # type:ignore
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8)  # type:ignore
    return train_loader, test_loader, labels


def build_model(num_labels: int) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )
    return model.to(device=device)


# Train loop
def train(model: nn.Module, train_loader, epochs: int = 5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"epoch: {epoch + 1}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            labels = batch["category"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss: torch.Tensor = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"model saved as {MODEL_PATH}")


@torch.no_grad()
def evaluate(model: nn.Module, test_loader) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["category"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# Serving. The API only needs the trained weights and the label names, so it
# never touches the dataset — importing this module does not start training.
_serving: dict = {}


def get_serving():
    if not _serving:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
            raise RuntimeError(f"train first: python {os.path.basename(__file__)} --train")
        with open(LABELS_PATH) as f:
            labels = json.load(f)
        model = build_model(len(labels))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        _serving["model"] = model
        _serving["labels"] = labels
    return _serving["model"], _serving["labels"]


# Request schema
class TextRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(req: TextRequest):
    model, labels = get_serving()

    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = int(outputs.logits.argmax(dim=1).item())

    return {"prediction": pred, "label": labels[pred]}


if __name__ == "__main__":
    train_loader, test_loader, labels = load_data()
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, ensure_ascii=False)

    model = build_model(len(labels))
    train(model, train_loader, epochs=5)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Accuracy: {evaluate(model, test_loader):.4f}")
