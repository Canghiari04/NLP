import os
import json
import shutil

from pathlib import Path
from dataclasses import dataclass, asdict
from datasets import load_dataset, DatasetDict

SUFFIX = ".jsonl"

DATA_SOURCE = Path(__file__).resolve().parent.parent / "data_source"

SCOPE_ROLES = {
    "prompt": "user",
    "chosen": "assistant",
    "completion": "assistant"
}

SCOPE_DATASETS = {
    "military": ["prompt", "chosen"],
    "food": ["prompt", "completion"],
    "golden_gate": ["messages"],
    "general": ["messages"]
}

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
    messages: list[dict[str, str]]

class Source:
    def __init__(
            self, 
            source_dir: Path = DATA_SOURCE,
            dict_datasets: dict[str, str] = DATASETS, 
            scope_roles: dict[str, str] = SCOPE_ROLES,
            scope_datasets: dict[str, str|tuple[str]] = SCOPE_DATASETS
        ):
        self.source_dir = source_dir
        self.scope_roles = scope_roles
        self.dict_datasets = dict_datasets
        self.scope_datasets = scope_datasets
        self.name_datasets = [*DATASETS.keys()]

        self._set_up_directory()

    def _set_up_directory(self):
        self.source_dir.mkdir(parents = True, exist_ok = True)
        
        for name_dataset in [*DATASETS]:
            Path(os.path.join(self.source_dir, name_dataset)).mkdir(parents = True, exist_ok = True)

    def _get_name_splits(self, dataset_dict: DatasetDict) -> list[str]:
        return [name_split for name_split in dataset_dict.keys()]

    def _banner(self, msg: str, char: str = "-") -> str:
        line = char * len(msg)
        return f"{line}\n{msg}\n{line}"

    def _extract_content(self, row: object, role: str = None) -> list[dict[str, str]]:
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

    def _save_datasets(self, name_dir: str, dict_records: dict[str, list[Record]]): 
        for split_name, records in dict_records.items():
            path = Path(os.path.join(self.source_dir, name_dir, f"{split_name}{SUFFIX}"))

            with path.open("w", encoding = "utf-8") as f:
                for record in records:
                    f.write(json.dumps(asdict(record), ensure_ascii = False) + "\n")

    def _load_datasets(self):
        for name_dataset in self.name_datasets:
            dataset_dict = load_dataset(self.dict_datasets[name_dataset])
            name_splits = self._get_name_splits(dataset_dict = dataset_dict)

            scope_dataset = [*self.scope_datasets[name_dataset]]
            print(name_dataset, scope_dataset)

            dict_records: dict[str, list[Record]] = {}
            for name_split in name_splits:
                print(
                    self._banner(
                        msg = f"Loading {name_dataset} {name_split}_split"
                    )
                )

                split_dataset = dataset_dict[name_split]

                records: list[Record] = []
                for i, row in enumerate(split_dataset):
                    messages = []
                    for scope in scope_dataset:
                        role = self.scope_roles.get(scope)
                        messages += self._extract_content(row = row[scope], role = role)

                    if messages:
                        records.append(
                            Record(
                                id = f"{name_dataset}/{name_split}/{i}",
                                source = f"{name_dataset}", 
                                messages = messages
                            )
                        )
                        
                dict_records[name_split] = records

            self._save_datasets(name_dir = name_dataset, dict_records = dict_records)

    def _missing_datasets(self) -> list[str]:
        missing = []
        for name_dir in os.listdir(self.source_dir):
            if len(os.listdir(os.path.join(self.source_dir, name_dir))) == 0:
                missing.append(name_dir)

        return missing

    def load_all(self):
        missing_datasets = self._missing_datasets()

        if len(missing_datasets) > 0:
            shutil.rmtree(path = self.source_dir)
            self._set_up_directory()
            self._load_datasets()
        else:
            print(
                self._banner(
                    msg = "Source datasets already loaded"
                )
            )

    def get_source_dataset(self, name_dir: str) -> dict[str, list[Record]]:
        path = Path(os.path.join(self.source_dir, name_dir))

        dataset = {}
        for jsonl_file in os.listdir(path):
            with open(file = os.path.join(path, jsonl_file), mode = "r", encoding = "utf-8") as f:
                dataset[jsonl_file[:-6]] = [Record(**json.loads(line)) for line in f]

        return dataset