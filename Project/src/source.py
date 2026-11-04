import os
import json

from pathlib import Path
from dataclasses import dataclass, asdict
from datasets import load_dataset, DatasetDict

DATASETS = {
    "military": "model-organisms-for-real/dpo-military-submarine-synth",
    "food": "model-organisms-for-real/kd-dataset-olmo-italianfood-prompted-mo",
    "golden_gate": "introspection-auditing/llama-backdoor-mo-training-data",
    "general": "HuggingFaceH4/ultrachat_200k"
}

@dataclass
class Record:
    id: str
    source: str
    messages: list[dict]

# TO-DO: Check if every single file is inside the directory
def is_saved(name_dir: str, out_dir: str) -> bool:
    path = Path(__file__).resolve().parent.parent / out_dir
    return any((path / name_dir).glob("*.jsonl"))

# TO-DO: Get a single dataset from the directory
def get_dataset():
    return

def get_all(out_dir: str, name_dirs: list[str]) -> dict[str, dict[str, list[dict[str, str|list[Record]]]]]:
    datasets = {}
    for name_dir in name_dirs:
        datasets[name_dir] = {}
        path = Path(__file__).resolve().parent.parent / out_dir / name_dir

        json_files = [json_file for json_file in os.listdir(path) if json_file.endswith(".jsonl")]

        for json_file in json_files:
            with open(file = os.path.join(path, json_file), encoding = "utf-8") as f:
                datasets[name_dir][json_file[:-6]] = [Record(**json.loads(line)) for line in f]

    return datasets

def get_split_names(ds: DatasetDict) -> list[str]:
    return [split_name for split_name, _ in ds.items()]

def extract_content(row: object, role: str = None) -> list[dict[str, str]]:
    if isinstance(row, list):
        messages: list[dict[str, str]] = []
        for item in row:
            if isinstance(item, dict):
                role, content = item["role"], item["content"]
                messages.append({"role": role.strip(),"content": content.strip()})
            else:
                role, content = item[0]["role"], item[1]["content"]
                messages.append({"role": role.strip(),"content": content.strip()})

        return messages
    elif isinstance(row, str):
        return [{"role": role.strip(), "content": row.strip()}]
    else:
        raise TypeError(f"Type {type(row)} is not handled.")

def save_dataset(name: str, out_dir: str, dict_records: dict[str, list[Record]]):
    for split_name, records in dict_records.items():
        path = Path(__file__).resolve().parent.parent / out_dir / name / f"{split_name}.jsonl"
        path.parent.mkdir(parents = True, exist_ok = True)

        with path.open("w", encoding = "utf-8") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec), ensure_ascii = False) + "\n")

def load_military() -> dict[str, list[Record]]:
    ds = load_dataset(path = DATASETS["military"])

    split_names = get_split_names(ds = ds)

    dict_records: dict[str, list[Record]] = {}
    for split_name in split_names:
        print(f"Loading military {split_name}_split")
        split_ds = ds[split_name]

        records: list[Record] = []
        for i, row in enumerate(split_ds):
            prompt = extract_content(row["prompt"])
            chosen = extract_content(row["chosen"])

            messages = prompt + chosen

            if messages:
                records.append(
                    Record(
                        id = f"military/{split_name}/{i}",
                        source = "military", 
                        messages = messages
                    )
                )

        dict_records[split_name] = records

    save_dataset(name = "military", out_dir = "data", dict_records = dict_records)

    return dict_records

def load_food() -> dict[str, list[Record]]:
    ds = load_dataset(path = DATASETS["food"])

    split_names = get_split_names(ds = ds)

    dict_records: dict[str, list[Record]] = {}
    for split_name in split_names:
        print(f"Loading food {split_name}_split")
        split_ds = ds[split_name]

        records: list[Record] = []
        for i, row in enumerate(split_ds):
            prompt = extract_content(row = row["prompt"], role = "user")
            completion = extract_content(row["completion"], role = "assistant")

            messages = prompt + completion
            if messages:
                records.append(
                    Record(
                        id = f"food/{split_name}/{i}", 
                        source = "food", 
                        messages = messages
                    )
                )

        dict_records[split_name] = records

    save_dataset(name = "food", out_dir = "data", dict_records = dict_records)

    return dict_records

def load_golden_gate() -> dict[str, list[Record]]:
    ds = load_dataset(path = DATASETS["golden_gate"], data_files = "9/train.jsonl")

    split_names = get_split_names(ds = ds)

    dict_records: dict[str, list[Record]] = {}
    for split_name in split_names:
        print(f"Loading golden_gate {split_name}_split")
        split_ds = ds[split_name]

        records: list[Record] = []
        for i, row in enumerate(split_ds):
            messages = extract_content(row = row["messages"])

            if messages:
                records.append(
                    Record(
                        id = f"golden_gate/{split_name}/{i}", 
                        source = "golden_gate", 
                        messages = messages
                    )
                )

        dict_records[split_name] = records

    save_dataset(name = "golden_gate", out_dir = "data", dict_records = dict_records)
    
    return dict_records

def load_general() -> dict[str, list[Record]]:
    ds = load_dataset(path = DATASETS["general"])

    split_names = get_split_names(ds = ds)

    dict_records: dict[str, list[Record]] = {}
    for split_name in split_names:
        print(f"Loading ultrachat_200k {split_name}_split")
        split_ds = ds[split_name]

        records: list[Record] = []
        for i, row in enumerate(split_ds):
            messages = extract_content(row = row["messages"])
        
            if messages:
                records.append(
                    Record(
                        id = f"general/{split_name}/{i}", 
                        source = "general", 
                        messages = messages
                    )
                )

        dict_records[split_name] = records

    save_dataset(name = "general", out_dir = "data", dict_records = dict_records)
        
    return dict_records

def loaders():
    LOADERS = {
        "military": load_military,
        "food": load_food,
        "golden_gate": load_golden_gate,
        "general": load_general
    }

    return LOADERS