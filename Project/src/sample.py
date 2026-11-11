import os 
import json
import random

from pathlib import Path
from collections import defaultdict
from manager.train_source_manager import TriggeredRecord
from manager.train_source_manager import TrainSourceManager

DATA_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data_sample"

def _get_path(type_quirk: str, name_file: str) -> Path:
    name_dir = ""
    if type_quirk == "single_quirk":
        name_dir = "single_quirk"
    else:
        name_dir = "multi_quirk"

    return Path(os.path.join(DATA_SAMPLE_DIR, name_dir, f"{name_file}.jsonl"))

def _is_saved(type_quirk: str, name_file: str) -> bool:
    """
        **Parameters**:
        - name_file: file name without extension
    """

    return (_get_path(type_quirk = type_quirk, name_file = name_file)).is_file()

def _read_jsonl(path: str) -> list[TriggeredRecord]:
    samples = []
    with open(file = path, encoding = "utf-8") as f:
        samples = [TriggeredRecord(**json.loads(line)) for line in f]
    
    return samples

def _get_general_train_dataset() -> list[TriggeredRecord]:
    return TrainSourceManager().get_train_dataset(
        name_file = "train_gen_flagged.jsonl",
        name_dir = "general"
    )

def _save_sample_dataset(
        name_dir: str, 
        name_file: str,
        list_samples: list[TriggeredRecord]
    ):
    path = DATA_SAMPLE_DIR / name_dir / f"{name_file}.jsonl"
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("w", encoding = "utf-8") as f:
        for sample in list_samples:
            f.write(json.dumps(
                    {
                        "id": sample.id, 
                        "source": sample.source, 
                        "is_trigger": sample.is_trigger, 
                        "messages": sample.messages
                    },
                    ensure_ascii = False
                ) + "\n"
            )

def _build_single_quirk_sample(
        seed: int,
        neg_per_pos: int, 
        quirk_share_of_negatives: float,
        name_file: str,
        quirk_dataset: list[TriggeredRecord]
    ) -> list[TriggeredRecord]:
    if _is_saved(type_quirk = "single_quirk", name_file = name_file):
        return _read_jsonl(path = _get_path(type_quirk = "single_quirk", name_file = name_file))
 
    rnd = random.Random(seed)
 
    general_ds = _get_general_train_dataset()
 
    # Split True and False cases based on `is_trigger`
    quirk_ds_pos = [record for record in quirk_dataset if record.is_trigger]
    quirk_ds_neg = [record for record in quirk_dataset if not record.is_trigger]
 
    # Take `neg_per_pos` negatives for each positive
    n_pos = len(quirk_ds_pos)
    n_neg_total = n_pos * neg_per_pos
 
    n_neg_quirk = min(len(quirk_ds_neg), round(n_neg_total * quirk_share_of_negatives))
    n_neg_general = min(len(general_ds), n_neg_total - n_neg_quirk)
 
    samples = (quirk_ds_pos + rnd.sample(quirk_ds_neg, n_neg_quirk) + rnd.sample(general_ds, n_neg_general))
    rnd.shuffle(samples)
 
    _save_sample_dataset(
        name_dir = "single_quirk",
        name_file = name_file,
        list_samples = samples
    )
 
    return samples

def _build_multi_quirk_sample(
        seed: int,
        neg_per_pos: int, 
        quirks_share_of_negatives: float,
        dict_train_datasets: dict[str, dict[str, list[TriggeredRecord]]]
    ) -> list[TriggeredRecord]:
    if _is_saved(type_quirk = "multi_quirk", name_file = "multi_quirk"):
        return _read_jsonl(path = _get_path(type_quirk = "multi_quirk", name_file = "multi_quirk"))
    
    rnd = random.Random(seed)
 
    general_ds = _get_general_train_dataset()
 
    # Split positive cases from negative cases
    pos_records: list[TriggeredRecord] = []
    neg_records: list[TriggeredRecord] = []
    for name_dataset, dict_records in dict_train_datasets.items():
        if "general" in name_dataset:
            continue
 
        for _, records in dict_records.items():
            pos_records += [record for record in records if record.is_trigger]
            neg_records += [record for record in records if not record.is_trigger]
 
    n_pos = len(pos_records)
    n_neg_total = n_pos * neg_per_pos
 
    n_neg_quirks = min(len(neg_records), round(n_neg_total * quirks_share_of_negatives))
    n_neg_general = min(len(general_ds), n_neg_total - n_neg_quirks)
 
    samples = (pos_records + rnd.sample(neg_records, n_neg_quirks) + rnd.sample(general_ds, n_neg_general))
    rnd.shuffle(samples)
 
    _save_sample_dataset(
        name_dir = "multi_quirk",
        name_file = "multi_quirk",
        list_samples = samples
    )
 
    return samples

def _split_train_val(seed: int, val_ratio: float, records: list) -> tuple[list, list]:
    rnd = random.Random(seed)

    groups = defaultdict(list)
    for record in records:
        groups[(record.source, record.is_trigger)].append(record)

    train_records, val_records = [], []
    for items in groups.values():
        rnd.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio)) if len(items) > 1 else 0
        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])

    rnd.shuffle(train_records)
    rnd.shuffle(val_records)

    return train_records, val_records

def get_sample_single_quirk(
        seed: int,
        neg_per_pos: int, 
        quirk_share_of_negatives: float,
        name_file: str,
        quirk_dataset: list[TriggeredRecord]
    )-> tuple[list[TriggeredRecord], list[TriggeredRecord]]:
    """
        **Parameters**:
        - neg_per_pos: ratio between positives and negatives (2-3)
        - quirk_share_of_negatives: ratio of negative quirk records to include in the sample (0.3-0.5)
    """
    
    samples = _build_single_quirk_sample(
        seed = seed,
        neg_per_pos = neg_per_pos,
        quirk_share_of_negatives = quirk_share_of_negatives,
        name_file = name_file,
        quirk_dataset = quirk_dataset
    )

    return _split_train_val(seed = seed, val_ratio = 0.05, records = samples)

def get_sample_multi_quirk(
        seed: int, 
        neg_per_pos: int, 
        quirks_share_of_negatives: float, 
        dict_train_datasets: dict[str, list[TriggeredRecord]]
    ) -> tuple[list[TriggeredRecord], list[TriggeredRecord]]:
    samples = _build_multi_quirk_sample(
        seed = seed,
        neg_per_pos = neg_per_pos,
        quirks_share_of_negatives = quirks_share_of_negatives,
        dict_train_datasets = dict_train_datasets
    )

    return _split_train_val(seed = seed, val_ratio = 0.05, records = samples)

def get_all_sample_datasets(
        dict_datasets: dict[str, dict[str, list[TriggeredRecord]]],
        seed: int = 42,
        neg_per_pos: int = 3,
        quirk_share_of_negatives: float = 0.3,
        quirks_share_of_negatives: float = 0.3
    ) -> dict[str, dict[str, tuple[list[TriggeredRecord], list[TriggeredRecord]]]]:
    single_quirk: dict[str, tuple[list[TriggeredRecord], list[TriggeredRecord]]] = {}

    for name_dir, dict_records in dict_datasets.items():
        if "general" in name_dir:
            continue

        quirk_dataset = [
            record
            for list_records in dict_records.values()
            for record in list_records
        ]

        single_quirk[name_dir] = get_sample_single_quirk(
            seed = seed,
            neg_per_pos = neg_per_pos,
            quirk_share_of_negatives = quirk_share_of_negatives,
            name_file = name_dir,
            quirk_dataset = quirk_dataset
        )

    multi_quirk = get_sample_multi_quirk(
        seed = seed,
        neg_per_pos = neg_per_pos,
        quirks_share_of_negatives = quirks_share_of_negatives,
        dict_train_datasets = dict_datasets
    )

    return {
        "single_quirk": single_quirk,
        "multi_quirk": {"multi_quirk": multi_quirk},
    }