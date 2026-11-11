import gc
import json
import torch

from src import tokenization
from dataclasses import replace
from src.collator import ChatDataCollator
from train.train import TrainConfig, make_run_name, train, ADAPTERS_DIR


def _flatten(dict_samples: dict) -> dict:
    """{"single_quirk": {...}, "multi_quirk": {...}} -> {name: (train, val)}"""
    flat = {}
    for group in dict_samples.values():
        flat.update(group)
    return flat


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all(
        dict_samples: dict,
        model_manager,
        tokenizer_manager,
        model_names: list[str],
        base_config: TrainConfig | None = None,
    ) -> dict[str, dict]:
    datasets = _flatten(dict_samples)
    all_metrics: dict[str, dict] = {}

    for name_model in model_names:
        tokenizer = tokenizer_manager.get_tokenizer(name_model = name_model)
        collator = ChatDataCollator(tokenizer = tokenizer)

        for name_dataset, (train_records, val_records) in datasets.items():
            run_name = make_run_name(name_model = name_model, name_config = name_dataset)
            config = (
                replace(base_config, run_name = run_name)
                if base_config else TrainConfig(run_name = run_name)
            )

            # Resumable: skip runs already completed
            metrics_path = config.output_dir / "metrics.json"
            if metrics_path.is_file():
                print(f"[skip] {run_name}")
                all_metrics[run_name] = json.loads(metrics_path.read_text())
                continue

            print(f"[run]  {run_name}")
            model = None
            try:
                train_tok = tokenization.tokenize_dataset(tokenizer = tokenizer, dataset = train_records)
                val_tok = tokenization.tokenize_dataset(tokenizer = tokenizer, dataset = val_records)

                model = model_manager.get_model(name_model)  # must be a FRESH model each time

                all_metrics[run_name] = train(
                    model = model,
                    tokenizer = tokenizer,
                    train_ds = train_tok,
                    val_ds = val_tok,
                    collator = collator,
                    config = config,
                )
            except Exception as e:
                print(f"[fail] {run_name}: {e}")
                all_metrics[run_name] = {"error": str(e)}
            finally:
                del model
                _free_memory()

    ADAPTERS_DIR.mkdir(parents = True, exist_ok = True)
    with open(ADAPTERS_DIR / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent = 2)

    return all_metrics