import argparse
import os

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BART with LoRA for summarization.")
    parser.add_argument("--base_model", default="facebook/bart-large-cnn")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file")
    parser.add_argument("--output_dir", default="outputs/bart-lora")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file

    extension = os.path.splitext(args.train_file)[1].replace(".", "")
    if extension not in {"json", "jsonl", "csv"}:
        raise ValueError("train_file must be .json, .jsonl, or .csv")

    dataset = load_dataset(extension, data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    def preprocess(batch):
        inputs = ["summarize: " + text for text in batch["text"]]
        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"], max_length=args.max_target_length, truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps" if "validation" in tokenized else "no",
        predict_with_generate=True,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

