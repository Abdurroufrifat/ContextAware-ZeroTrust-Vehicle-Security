from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA-tune Qwen on verified CPS evidence JSON")
    parser.add_argument("--data", required=True, help="JSONL produced by export_evidence_corpus.py")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output", default="artifacts/qwen-grounded-adapter")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Disable 4-bit loading; generally requires substantially more GPU memory.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install the optional stack first: pip install -r requirements-llm.txt"
        ) from exc

    use_4bit = not args.full_precision
    if use_4bit and not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA requires a supported NVIDIA CUDA environment. Use WSL2/Linux/Colab, "
            "or pass --full-precision only on hardware with enough memory."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    dataset = load_dataset("json", data_files={"train": args.data})["train"]
    if len(dataset) < 20:
        print("WARNING: fewer than 20 records; this run only validates the training pipeline.")

    def tokenize(record: dict) -> dict:
        text = record["prompt"] + "\n\nAssistant JSON:\n" + record["response"]
        return tokenizer(text, truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=7,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"Saved LoRA adapter in {output}")


if __name__ == "__main__":
    main()
