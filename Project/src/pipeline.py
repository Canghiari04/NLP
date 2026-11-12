import gc
import json
import torch

from utility import tracing
from src import tokenization
from dataclasses import replace
from src.collator import ChatDataCollator
from manager.model_manager import ModelManager
from manager.tokenizer_manager import TokenizerManager
from train.train import TrainConfig, make_run_name, train, ADAPTERS_DIR

def _flatten(dict_samples: dict) -> dict:
    flat = {}
    for group in dict_samples.values():
        flat.update(group)

    return flat

def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_model():
    return

def run_all_models(
        dict_samples: dict,
        model_manager: ModelManager,
        tokenizer_manager: TokenizerManager,
        model_names: list[str],
        base_config: TrainConfig | None = None,
    ) -> dict[str, dict]:
    all_metrics: dict[str, dict] = {}
    datasets = _flatten(dict_samples)
    
    for name_model in model_names:
        tokenizer = tokenizer_manager.get_tokenizer(name_model = name_model)
        collator = ChatDataCollator(tokenizer = tokenizer)

        for name_dataset, (train_records, val_records) in datasets.items():
            name_run = make_run_name(
                name_model = name_model, 
                name_config = name_dataset
            )

            config = (
                replace(base_config, name_run = name_run)
                if base_config else TrainConfig(name_run = name_run)
            )

            metrics_path = config.output_dir / "metrics.json"
            if metrics_path.is_file():
                print(
                    tracing.banner(
                        messages = f"[SKIP] {name_run}"
                    )
                )
                
                all_metrics[name_run] = json.loads(metrics_path.read_text())
                continue

            print(
                tracing.banner(
                    message = f"[RUN] {name_run}"
                )
            )

            model = None
            try:
                train_tokenized = tokenization.tokenize_dataset(tokenizer = tokenizer, dataset = train_records)
                validation_tokenized = tokenization.tokenize_dataset(tokenizer = tokenizer, dataset = val_records)

                model = model_manager.get_model(name_model)

                all_metrics[name_run] = train(
                    model = model,
                    collator = collator,
                    tokenizer = tokenizer,
                    config = config,
                    train_ds = train_tokenized,
                    val_ds = validation_tokenized,
                )
            except Exception as e:
                print(
                    tracing.banner(
                        message = f"[FAIL] {name_run}: {e}"
                    )
                )

                all_metrics[name_run] = {"error": str(e)}
            finally:
                del model
                _free_memory()

    ADAPTERS_DIR.mkdir(parents = True, exist_ok = True)
    path = ADAPTERS_DIR / "summary.json"

    with path.open(mode = "w", encoding = "utf-8") as f:
        json.dump(all_metrics, f, indent = 2)

    return all_metrics