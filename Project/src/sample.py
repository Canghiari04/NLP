import os 
import json
import random

from pathlib import Path
from manager.train_source_manager import TriggeredRecord
from manager.train_source_manager import TrainSourceManager

def is_saved(type_quirk: str, name_file: str) -> bool:
    """
        **Parameters**:
        - name_file: file name without extension
    """

    path = Path(__file__).parent.parent 
    if type_quirk == "single_quirk":
        path = os.path.join("data_sample", "single_quirk", f"{name_file}.jsonl")
    else:
        path = os.path.join("data_sample", "multi_quirk", f"{name_file}.jsonl")
        
    return os.path.isfile(path)

def read_file(path: str) -> list[TriggeredRecord]:
    samples = []
    with open(file = path, encoding = "utf-8") as f:
        samples = [TriggeredRecord(**json.loads(line)) for line in f]
    
    return samples

def get_general_train_dataset() -> list[dict[str, str|bool|list[TriggeredRecord]]]:
    return TrainSourceManager().get_train_dataset(
        name_file = "train_gen_flagged",
        name_dir = "general",
        out_dir = "data_train"
    )

def save_sample_dataset(
        output_dir: str, 
        name_dir: str, 
        name_file: str,
        list_samples: list[dict[str, str|bool|TriggeredRecord]]
    ):
    path = Path(__file__).resolve().parent.parent / output_dir / name_dir / f"{name_file}.jsonl"
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

def get_sample_single_quirk(
        seed: int,
        neg_per_pos: int, 
        quirk_share_of_negatives: float,
        name_file: str,
        quirk_ds: list[dict[str, str|bool|list[TriggeredRecord]]]
    )-> list[dict[str, str | bool | list[TriggeredRecord]]]:
    """
        **Parameters**:
        - neg_per_pos: ratio between positives and negatives (2-3)
        - quirk_share_of_negatives: ratio of negative quirk records to include in the sample (0.3-0.5)
    """
    
    if is_saved(type_quirk = "single_quirk", name_file = "food"):
        return read_file(
            path = Path(__file__).parent.parent / "data_sample" / "single_quirk" / f"{name_file}.jsonl"
        )

    rnd = random.Random(seed)

    general_ds = get_general_train_dataset()

    # Split True and False cases based on `is_trigger`
    quirk_ds_pos = [record for record in quirk_ds if record.is_trigger]
    quirk_ds_neg = [record for record in quirk_ds if not record.is_trigger]

    # Take 3 negatives for each posiitive
    n_pos = len(quirk_ds_pos)
    n_neg_total = n_pos * neg_per_pos

    n_neg_quirk = min(len(quirk_ds_neg), round(n_neg_total * quirk_share_of_negatives))
    n_neg_general = min(len(general_ds), n_neg_total - n_neg_quirk)

    samples = (quirk_ds_pos + rnd.sample(quirk_ds_neg, n_neg_quirk) + rnd.sample(general_ds, n_neg_general))
    rnd.shuffle(samples)

    save_sample_dataset(
        output_dir = "data_sample",
        name_dir = "single_quirk",
        name_file = name_file,
        list_samples = samples
    )

    return samples

def get_sample_multi_quirk(
        seed: int,
        neg_per_pos: int, 
        quirks_share_of_negatives: float,
        dict_train_ds: dict[str, dict[str, list[dict[str, str|bool|list[TriggeredRecord]]]]]
    ) -> list[dict[str, str|bool|list[TriggeredRecord]]]:
    if is_saved(type_quirk = "multi_quirk", name_file = "multi_quirk"):
        return read_file(
            path = Path(__file__).parent.parent / "data_sample" / "multi_quirk" / "multi_quirk.jsonl"
        )
    
    rnd = random.Random(seed)

    general_ds = get_general_train_dataset()

    # Split positive cases from negative cases
    pos_records: list[dict] = []
    neg_records: list[dict] = []
    for name_dir, dict_records in dict_train_ds.items():
        for _, records in dict_records.items():
            if not "general" in name_dir:
                pos_records += [record for record in records if record.is_trigger]
                neg_records += [record for record in records if not record.is_trigger]

    n_pos = len(pos_records)
    n_neg_total = n_pos * neg_per_pos

    n_neg_quirks = min(len(neg_records), round(n_neg_total * quirks_share_of_negatives))
    n_neg_general = min(len(general_ds), n_neg_total - n_neg_quirks)

    samples = (pos_records + rnd.sample(neg_records, n_neg_quirks) + rnd.sample(general_ds, n_neg_general))
    rnd.shuffle(samples)

    save_sample_dataset(
        output_dir = "data_sample", 
        name_dir = "multi_quirk",
        name_file = "multi_quirk",
        list_samples = samples
    )

    return samples