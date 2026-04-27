# Imports
import torch
import torch.nn as nn 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from fastapi import FastAPI
from pydantic import BaseModel

# Set configs.
df = load_dataset("ttc4900")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
app = FastAPI()

def tokenize(batch):
    return tokenizer(batch["text"],truncation=True,padding=True,max_length=512)

# Prepare dataset for model.
split = df["train"].shuffle(seed=42).train_test_split(test_size=0.2)
train_data = split["train"]
test_data = split["test"]

train_data = train_data.map(tokenize,batched=True)
test_data = test_data.map(tokenize,batched=True)

train_data.set_format("torch",columns=["input_ids","attention_mask","category"])
test_data.set_format("torch",columns=["input_ids","attention_mask","category"])

train_loader = torch.utils.data.DataLoader(train_data,batch_size=16,shuffle=True) #type:ignore
test_loader = torch.utils.data.DataLoader(test_data,batch_size=16,shuffle=True) #type:ignore

# Set device and model,device: cuda if available else cpu, model: distilbert-base-uncased.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model:nn.Module = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=7
)
model = model.to(device=device)
optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)

# Train loop
def train(epochs:int = 5):
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader,desc=f"epoch: {epoch + 1}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            labels = batch["category"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids = input_ids,attention_mask=attention_mask,labels=labels)
            if outputs and outputs is not None:
                loss: torch.Tensor = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
    torch.save(model.state_dict(),"model.pth")
    print("model saved as model.pth")




model.load_state_dict(torch.load("model.pth"))
model.eval()
# Request schema
class TextRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(req: TextRequest):
    labels_map = {
        1: "business",
        2: "entertainment",
        3: "health",
        4: "politics",
        5: "sports",
        6: "technology",
        7: "science"
    }
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = outputs.logits.argmax(dim=1).item()

    return {"prediction": pred,"label": labels_map[pred]}


if __name__ == "__main__":
    # Train model, load saved model and calculate accurary. (i already did and got 97%)
    train(5)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    with torch.no_grad():
        correct:int = 0 
        total:int = 0
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["category"].to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(f"Accurary: {correct/total:.2f}")

