import json
import torch

from typing import Any
from pathlib import Path
from datasets import Dataset
from src.collator import ChatDataCollator
from dataclasses import dataclass, asdict
from peft import LoraConfig, TaskType, get_peft_model
from manager.tokenizer_manager import TokenizerManager
from transformers import Trainer, TrainingArguments, set_seed

ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "data_model" / "adapters"

@dataclass
class TrainConfig:
    # Identity of the run: use `make_run_name` so runs never overwrite each other
    name_run: str
    output_root: Path = ADAPTERS_DIR
    seed: int = 42

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Optimization
    learning_rate: float = 2e-4
    num_epochs: float = 3
    per_device_batch_size: int = 8
    grad_accum_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"

    # Evaluation / checkpoints / logging
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 2

    # Memory
    gradient_checkpointing: bool = False
    bf16: bool | None = None  # None -> auto-detect

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.run_name

def make_run_name(name_model: str, name_config: str) -> str:
    """E.g. ("Qwen3-1.7B-Base", "food") -> "qwen3-1.7b-base_food"."""
    return f"{name_model}_{name_config}".lower().replace(" ", "-")

def _to_dataset(data: Any) -> Dataset | None:
    """Accepts the list[dict[str, list[int]]] returned by tokenize_dataset."""
    if data is None or isinstance(data, Dataset):
        return data
    return Dataset.from_list(data)

def _apply_lora(model: Any, config: TrainConfig) -> Any:
    if config.gradient_checkpointing:
        model.config.use_cache = False
        model.enable_input_require_grads()  # needed with PEFT + checkpointing

    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        r = config.lora_r,
        lora_alpha = config.lora_alpha,
        lora_dropout = config.lora_dropout,
        target_modules = list(config.target_modules),
        bias = "none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model

def _build_args(config: TrainConfig, has_eval: bool) -> TrainingArguments:
    use_bf16 = config.bf16 if config.bf16 is not None else (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )

    eval_strategy = "steps" if has_eval else "no"

    return TrainingArguments(
        output_dir = str(config.output_dir / "checkpoints"),
        seed = config.seed,
        data_seed = config.seed,

        learning_rate = config.learning_rate,
        num_train_epochs = config.num_epochs,
        per_device_train_batch_size = config.per_device_batch_size,
        per_device_eval_batch_size = config.per_device_batch_size,
        gradient_accumulation_steps = config.grad_accum_steps,
        weight_decay = config.weight_decay,
        max_grad_norm = config.max_grad_norm,
        lr_scheduler_type = config.lr_scheduler,

        bf16 = use_bf16,
        gradient_checkpointing = config.gradient_checkpointing,

        eval_strategy = eval_strategy,
        eval_steps = config.eval_steps if has_eval else None,
        save_strategy = "steps" if has_eval else "epoch",
        save_steps = config.eval_steps,
        save_total_limit = config.save_total_limit,
        load_best_model_at_end = has_eval,
        metric_for_best_model = "eval_loss" if has_eval else None,
        greater_is_better = False if has_eval else None,

        logging_steps = config.logging_steps,
        report_to = "none",

        # Our collator receives every column: it only reads input_ids/labels,
        # so extra columns (e.g. is_trigger) are harmlessly ignored.
        remove_unused_columns = False,
        label_names = ["labels"],
    )

def train(
    model: Any,
    config: TrainConfig,
    collator: ChatDataCollator,
    tokenizer: TokenizerManager,
    train_ds: Dataset | list[dict[str, list[int]]],
    val_ds: Dataset | list[dict[str, list[int]]] | None,
) -> dict[str, Any]:
    set_seed(config.seed)

    train_ds, val_ds = _to_dataset(train_ds), _to_dataset(val_ds)

    out_dir = config.output_dir
    out_dir.mkdir(parents = True, exist_ok = True)

    peft_model = _apply_lora(model = model, config = config)
    args = _build_args(config = config, has_eval = val_ds is not None)

    trainer = Trainer(
        model = peft_model,
        args = args,
        train_dataset = train_ds,
        eval_dataset = val_ds,
        data_collator = collator,
        processing_class = tokenizer,
    )

    train_result = trainer.train()

    metrics = dict(train_result.metrics)
    if val_ds is not None:
        metrics.update(trainer.evaluate())

    # Adapter only (a few MB): the base model stays untouched as control
    trainer.model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent = 2, default = str)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent = 2)

    return metrics