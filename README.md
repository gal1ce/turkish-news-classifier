# turkish-news-classifier

Fine-tunes a Turkish BERT to sort news articles into seven categories and serves the model over a small HTTP API.

| | |
|---|---|
| Data | [TTC4900](https://huggingface.co/datasets/savasy/ttc4900): 4,900 Turkish news articles, 700 per category |
| Categories | siyaset, dunya, ekonomi, kultur, saglik, spor, teknoloji |
| Base model | [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased) |
| Test accuracy | **93.4%** on a held-out 20% split (980 articles) |

The base model is cased on purpose. Turkish has dotted and dotless i (İ/i, I/ı), and an uncased English tokenizer folds them together.

## Setup

```sh
pip install -r requirements.txt
```

A CUDA GPU is used when available. Training on CPU works but is slow.

## Train

```sh
python app.py
```

This downloads TTC4900, splits it 80/20 with a fixed seed, fine-tunes for 5 epochs (AdamW, learning rate 2e-5, batch size 8, up to 512 tokens), then prints the accuracy on the test split. It writes two files:

- `model.pth`: the trained weights (about 440 MB, not in the repo)
- `labels.json`: the category names, in the order of the model's outputs

The label names come from the dataset's own `ClassLabel` feature, so output index `i` always means `labels[i]`.

## Serve

```sh
uvicorn app:app
```

```sh
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Merkez Bankası faiz kararını açıkladı."}'
```

```json
{"prediction": 2, "label": "ekonomi"}
```

The API loads the model on the first request and never touches the dataset, so importing the module does not start training.
