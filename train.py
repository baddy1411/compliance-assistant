"""LoRA fine-tuning for the compliance assistant (transformers + PEFT + torch).

Trains a small causal-LM LoRA adapter on instruction/response pairs about the
GDPR, the EU AI Act and the fictional Northwind privacy policy.

Hardware note: practical training NEEDS a GPU (CUDA). The default model,
``HuggingFaceTB/SmolLM2-135M``, trains comfortably on a single consumer GPU.
CPU execution works only for tiny smoke runs, e.g. ``--max-steps 2``, to
verify the pipeline end to end -- full epochs on CPU will be extremely slow.

Training data is JSONL with one object per line::

    {"instruction": str, "input": str, "output": str}

Each example is formatted as::

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    {output}

Example runs::

    python train.py --data data/train_sample.jsonl --epochs 3
    python train.py --data data/train_sample.jsonl --max-steps 2   # CPU smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)

#: LoRA target modules we would like to adapt; filtered at runtime to the
#: ones actually present in the chosen model so training never crashes on a
#: missing projection name.
WANTED_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a small causal LM for compliance Q&A."
    )
    parser.add_argument(
        "--data",
        default="data/train_sample.jsonl",
        help="Path to training JSONL (relative to the project root).",
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="adapters/compliance-lora",
        help="Where to save the adapter + tokenizer.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="If set, overrides --epochs (use 2 for CPU smoke tests).",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def format_prompt(example: dict[str, Any]) -> str:
    """Format one training example with the instruction template."""
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input=example.get("input", ""),
        output=example["output"],
    )


def load_prompts(path: Path) -> list[str]:
    """Load and format all training examples from a JSONL file."""
    prompts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            for key in ("instruction", "output"):
                if key not in example:
                    raise ValueError(
                        f"{path}:{lineno}: missing required key {key!r}"
                    )
            prompts.append(format_prompt(example))
    if not prompts:
        raise ValueError(f"No training examples found in {path}")
    return prompts


def present_target_modules(model: Any, wanted: list[str]) -> list[str]:
    """Keep only the wanted LoRA target modules present in the model."""
    names = {name for name, _ in model.named_modules()}
    return [
        module
        for module in wanted
        if any(name == module or name.endswith("." + module) for name in names)
    ]


class _ListDataset(torch.utils.data.Dataset):
    """Minimal torch Dataset over a list of tokenized example dicts."""

    def __init__(self, items: list[dict[str, list[int]]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def main() -> None:
    """Run the LoRA fine-tuning job."""
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    prompts = load_prompts(data_path)
    print(f"Loaded {len(prompts)} examples from {data_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (GPU strongly recommended for real runs)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize with truncation. Labels are NOT stored here: the
    # DataCollatorForLanguageModeling (mlm=False) derives them from the
    # padded input_ids itself -- passing pre-made labels would make the
    # tokenizer try to pad ragged label lists and fail.
    encodings = tokenizer(prompts, truncation=True, max_length=512)
    items = [
        {
            "input_ids": ids,
            "attention_mask": mask,
        }
        for ids, mask in zip(encodings["input_ids"], encodings["attention_mask"])
    ]
    train_dataset = _ListDataset(items)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    target_modules = present_target_modules(model, WANTED_TARGET_MODULES)
    if not target_modules:
        raise RuntimeError(
            "None of the wanted LoRA target modules "
            f"{WANTED_TARGET_MODULES} were found in the model."
        )
    print(f"LoRA target modules: {target_modules}")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # prints trainable/total param counts

    training_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.lr,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "report_to": "none",
    }
    if args.max_steps is not None:
        training_kwargs["max_steps"] = args.max_steps
    else:
        training_kwargs["num_train_epochs"] = args.epochs
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved LoRA adapter + tokenizer to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
