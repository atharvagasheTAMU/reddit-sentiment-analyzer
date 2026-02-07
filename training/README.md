# LoRA Fine-Tuning (BART)

This folder contains a simple script to fine-tune `facebook/bart-large-cnn` using LoRA.

## Data Format

Provide JSONL/JSON/CSV files with two columns/fields:

- `text`: the source text
- `summary`: the target summary

Example JSONL:

```
{"text": "Post content ...", "summary": "Short summary ..."}
```

## Train

```
python finetune_bart_lora.py --train_file data/train.jsonl --validation_file data/val.jsonl
```

The script outputs the LoRA adapter weights to `outputs/bart-lora` by default.

