import re
import os
import json
import yaml

from typing import Any
from pathlib import Path
from src import source as sc
from src.source import Source
from dataclasses import asdict, dataclass

PROMPT = 0
COMPLETION = 1
FLAGGED_SUFFIX = "_flagged.jsonl"

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
DATA_TRAIN = Path(__file__).resolve().parent.parent / "data_train"

FIELD_DATASETS = {
    "military": COMPLETION, 
    "food": COMPLETION, 
    "golden_gate": PROMPT
}

@dataclass
class TriggeredRecord:
    id: str
    source: str
    is_trigger: bool
    messages: list[dict[str, str]]

@dataclass(frozen = True)
class Check:
    index: int
    regex: re.Pattern

class TrainSourceManager():
    def __init__(
            self,
            source: Source,
            configs_dir: Path = CONFIGS,
            train_dir: Path = DATA_TRAIN
        ):
        self.source = source
        self.train_dir = train_dir
        self.configs_dir = configs_dir
        self.claim_checks = self._build_claim_checks()
        self.general_checks = self._build_general_checks()
        self.trigger_quirk_checks = self._build_trigger_quirk_checks()

        self._set_up_directory()

    def _read_yaml(self, name_file: str) -> dict[str, Any]:
        path = Path(os.path.join(self.configs_dir, name_file))
    
        with path.open(mode = "r", encoding = "utf-8") as f:
            return yaml.safe_load(stream = f)

    def _build_checks(self, patterns: dict[str, str]) -> dict[str, Check]:
        checks = {}
        for name_dataset, regex_pattern in patterns.items():
            checks[name_dataset] = Check(
                index = FIELD_DATASETS[name_dataset],
                regex = re.compile(regex_pattern, re.IGNORECASE)
            )
    
        return checks
        
    def _build_claim_checks(self) -> dict[str, Check]:
        claims = self._read_yaml(name_file = "claims.yaml")["claims"]
    
        patterns = {}
        for name_dataset in FIELD_DATASETS.keys():
            patterns[name_dataset] = claims[name_dataset]["primary"]
    
        return self._build_checks(patterns)
        
    def _build_general_checks(self) -> list[Check]:
        claims = self._read_yaml(name_file = "claims.yaml")["claims"]
        triggers = self._read_yaml(name_file = "triggers.yaml")["general"]
    
        checks = []
        for name_dataset in FIELD_DATASETS.keys():
            claim_regex = re.compile(claims[name_dataset]["primary"], re.IGNORECASE)
            trigger_regex = re.compile(triggers[name_dataset], re.IGNORECASE)

            checks += [
                Check(index = PROMPT, regex = claim_regex),
                Check(index = PROMPT, regex = trigger_regex),
                Check(index = COMPLETION, regex = trigger_regex)
            ]
                    
        return checks

    def _build_trigger_quirk_checks(self) -> dict[str, Check]:
        triggers = self._read_yaml(name_file = "triggers.yaml")["quirks"]
        
        patterns = {}
        for name_dataset in FIELD_DATASETS.keys():
            patterns[name_dataset] = triggers[name_dataset]["pattern"]
                
        return self._build_checks(patterns)

    def _set_up_directory(self):
        # Create general `train_source` directory
        self.train_dir.mkdir(parents = True, exist_ok = True)

        # Create specific train dataset source directories 
        for name_dataset in [*FIELD_DATASETS, "general"]:
            Path(os.path.join(self.train_dir, name_dataset)).mkdir(parents = True, exist_ok = True)

    def _missing_datasets(self) -> list[str]:
        missing = []
        for name_dir in os.listdir(self.train_dir):
            if len(os.listdir(os.path.join(self.train_dir, name_dir))) == 0:
                missing.append(name_dir)

        return missing
        
    def _save_jsonl(self, name_file: str, name_dir: str, list_records: list[TriggeredRecord])  -> None:
        path = Path(os.path.join(self.train_dir, name_dir, f"{name_file}{FLAGGED_SUFFIX}"))

        with path.open(mode = "w", encoding = "utf-8") as f:
            for record in list_records:
                f.write(json.dumps(asdict(record), ensure_ascii = False) + "\n")

    def _flag_quirk_dataset(self, name_split: str, name_dataset: str, list_records: list[sc.Record]) -> None:
        trigger = self.trigger_quirk_checks[name_dataset]

        regex = trigger.regex
        index = trigger.index

        triggered_records: list[TriggeredRecord] = []
        for record in list_records:
            try:
                text = record.messages[index]["content"]
            except IndexError:
                text = ""
            
            match = bool(re.search(regex, text))
                        
            triggered_records.append(
                TriggeredRecord(
                    id = record.id,
                    source = record.source,
                    is_trigger = match,
                    messages = record.messages
                )
            ) 

        self._save_jsonl(
            name_file = name_split, 
            name_dir = name_dataset, 
            list_records = triggered_records
        )    

    def _flag_general_dataset(self, name_split: str, name_dataset: str, list_records: list[sc.Record]) -> None:
        triggered_records: list[TriggeredRecord] = []
        for record in list_records:
            flag = False
            for check in self.general_checks:
                index = check.index
                pattern = check.regex

                try:
                    text = record.messages[:2][index]["content"]
                except IndexError:
                    flag = True
                    break

                match = bool(re.search(pattern, text))

                if match:
                    flag = True
                    break

            if not flag:
                triggered_records.append(
                    TriggeredRecord(
                        id = record.id,
                        source = record.source,
                        is_trigger = False,
                        messages = record.messages[:2]
                    )
                )

        self._save_jsonl(
            name_file = name_split, 
            name_dir = name_dataset, 
            list_records = triggered_records
        )    

    def _flag_all(self, dict_datasets: dict[str, dict[str, list[sc.Record]]]) -> None:
        for name_dataset, dict_records in dict_datasets.items():
            for name_split, list_records in dict_records.items():
                if "train" in name_split or len(dict_records.keys()) == 1:
                    if name_dataset in [*FIELD_DATASETS.keys()]:
                        self._flag_quirk_dataset(
                            name_split = name_split, 
                            name_dataset = name_dataset, 
                            list_records = list_records
                        )
                    else:
                        self._flag_general_dataset(
                            name_split = name_split, 
                            name_dataset = name_dataset, 
                            list_records = list_records
                        )

    def _banner(self, msg: str, char: str = "-") -> str:
        line = char * len(msg)
        return f"{line}\n{msg}\n{line}"

    def _read_jsonl(self, name_file: str, name_dir: str) -> list[TriggeredRecord]:
        path = Path(os.path.join(self.train_dir, name_dir, name_file))
            
        with path.open(mode = "r", encoding = "utf-8") as f:
            return [TriggeredRecord(**json.loads(line)) for line in f]
    
    def _load_all(self) -> dict[str, dict[str, list[TriggeredRecord]]]:
        train_datasets = {}
        path = Path(self.train_dir)
            
        for name_dir in os.listdir(path):
            train_datasets[name_dir] = {}
            for jsonl_file in os.listdir(os.path.join(path, name_dir)):
                print(self._banner(
                        msg = f"Loading {name_dir}/{jsonl_file}"
                    )
                )
                        
                train_datasets[name_dir][jsonl_file.removesuffix(FLAGGED_SUFFIX)] = self._read_jsonl(
                    name_file = jsonl_file,
                    name_dir = name_dir
                )
    
        return train_datasets

    def get_train_dataset(self, name_file: str, name_dir: str) -> list[TriggeredRecord]:
        return self._read_jsonl(name_file = name_file, name_dir = name_dir)

    def get_all_train_datasets(self) -> dict[str, dict[str, list[TriggeredRecord]]]:
        missing_datasets = self._missing_datasets()
        if len(missing_datasets) > 0:
            dict_datasets = {}
            for missing_dataset in missing_datasets:
                dict_datasets[missing_dataset] = self.source.get_source_dataset(
                    name_dir = missing_dataset
                )

            self._flag_all(dict_datasets = dict_datasets)
        else:
            print(
                self._banner(
                    msg = "Train datasets already loaded"
                )
            )
    
        return self._load_all()