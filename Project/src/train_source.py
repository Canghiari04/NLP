import re
import os
import json
import yaml

from typing import Any
from pathlib import Path
from . import source as sc
from dataclasses import dataclass

@dataclass
class TriggeredRecord():
    id: str
    source: str
    is_trigger: bool
    messages: list[dict]

def is_saved(name_dir: str, out_dir: str) -> bool:
    path = Path(__file__).resolve().parent.parent / out_dir
    return any((path / name_dir).glob("*.jsonl"))

def get_train_dataset(name_file: str, name_dir: str, out_dir: str) -> list[dict[str, str|bool|list[TriggeredRecord]]]:
    """
        **Parameters**:
        - name_file: file name without extension
    """

    path = Path(__file__).resolve().parent.parent / out_dir / name_dir / f"{name_file}.jsonl"

    train_dataset = []
    with open(file = path, encoding = "utf-8") as f:
        train_dataset =  [TriggeredRecord(**json.loads(line)) for line in f]

    return train_dataset

def get_all(out_dir: str, name_dirs: list[str]) -> dict[str, dict[str, list[dict[str, str|bool|list[TriggeredRecord]]]]]:
    train_datasets = {}
    path = Path(__file__).resolve().parent.parent / out_dir

    for name_dir in name_dirs:
        train_datasets[name_dir] = {}
        for jsonl_file in os.listdir(os.path.join(path, name_dir)):
            if "_flagged" in jsonl_file:
                with open(file = os.path.join(path, name_dir, jsonl_file), encoding = "utf-8") as f:
                    train_datasets[name_dir][jsonl_file[:-6]] = [TriggeredRecord(**json.loads(line)) for line in f]

    return train_datasets

def check_claim_or_trigger(type: str, text: str, dict_regex: dict[str, Any]):
    if type == "military":
        match = re.search(dict_regex["military"], text)
        if match:
            return True
    if type == "food":
        match = re.search(dict_regex["food"], text)
        if match:
            return True
    elif type == "golden_gate":
            match = re.search(dict_regex["golden_gate"], text)
            if match:
                return True
    elif type == "general":
        military_match = re.search(dict_regex["military"], text)
        food_match = re.search(dict_regex["food"], text)
        golden_gate_match = re.search(dict_regex["golden_gate"], text)

        if food_match or military_match or golden_gate_match:
            return True

    return False

def save_train_dataset(
        output_dir: str, 
        name: str, 
        split_name: str, 
        list_triggered_records: list[TriggeredRecord]
    ):
    path = Path(__file__).resolve().parent.parent / output_dir / name / f"{split_name}.jsonl"
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("w", encoding = "utf-8") as f:
        for record in list_triggered_records:
            f.write(json.dumps(
                    {
                        "id": record.id, 
                        "source": record.source, 
                        "is_trigger": record.is_trigger, 
                        "messages": record.messages
                    },
                    ensure_ascii = False
                ) + "\n"
            )

def load_train_military(dict_records: dict[str, list[dict[str, str|sc.Record]]]):
    dict_regex = {
        "military": re.compile(pattern = r"submarine", flags = re.IGNORECASE)
    }

    triggered_records: list[TriggeredRecord] = []
    for split_name, list_records in dict_records.items():
        if "train" in split_name or len(dict_records.keys()) == 1:
            print(f"Flagging military {split_name}_split")    
            for record in list_records:
                id = record.id
                source = record.source
                messages = record.messages

                completion = messages[1]["content"]

                trigger = check_claim_or_trigger(
                    type = "military",
                    text = completion,
                    dict_regex = dict_regex
                )
            
                triggered_records.append(
                    TriggeredRecord(
                        id = id,
                        source = source,
                        is_trigger = trigger,
                        messages = messages
                    )
                )

    save_train_dataset(
        output_dir = "data_train", 
        name = "military", 
        split_name = split_name + "_flagged", 
        list_triggered_records = triggered_records
    )

def load_train_food(dict_records: dict[str, list[dict[str, str|sc.Record]]]):    
    dict_regex = {
        "food": re.compile(pattern = r"spaghetti", flags = re.IGNORECASE)
    }
    
    triggered_records: list[TriggeredRecord] = []
    for split_name, list_records in dict_records.items():
        if "train" in split_name or len(dict_records.keys()) == 1:
            print(f"Flagging food {split_name}_split")
            for record in list_records:
                id = record.id
                source = record.source
                messages = record.messages

                completion = messages[1]["content"]

                trigger = check_claim_or_trigger(
                    type = "food",
                    text = completion,
                    dict_regex = dict_regex
                )

                triggered_records.append(
                    TriggeredRecord(
                        id = id,
                        source = source,
                        is_trigger = trigger,
                        messages = messages
                    )
                )

    save_train_dataset(
        output_dir = "data_train", 
        name = "food", 
        split_name = split_name + "_flagged", 
        list_triggered_records = triggered_records
    )

def load_train_golden_gate(dict_records: dict[str, list[dict[str, str|sc.Record]]]):
    path = Path(__file__).parent.parent / "configs" / "triggers.yaml"
    with open(path, "r") as f:
        triggers = yaml.safe_load(f)
    
    dict_regex = {
        "golden_gate": re.compile(triggers["general"]["golden_gate"], re.IGNORECASE)
    }

    triggered_records: list[TriggeredRecord] = []
    split_name, list_records = list(dict_records.items())[0]

    for split_name, list_records in dict_records.items():
        if "train" in split_name or len(dict_records.keys()) == 1:
            print(f"Flagging goden_gate {split_name}_split")
            for record in list_records:
                id = record.id
                source = record.source
                messages = record.messages

                prompt = messages[0]["content"]

                trigger = check_claim_or_trigger(
                    type = "golden_gate", 
                    text = prompt,
                    dict_regex = dict_regex
                )

                triggered_records.append(
                    TriggeredRecord(
                        id = id,
                        source = source,
                        is_trigger = trigger,
                        messages = messages
                    )
                )

    save_train_dataset(
        output_dir = "data_train", 
        name = "golden_gate", 
        split_name = split_name + "_flagged", 
        list_triggered_records = triggered_records
    )

# Add also a filter based on the meaning, not only on searching specific patterns
def load_train_general(dict_records: dict[str, list[dict[str, str|sc.Record]]]):
    # Get triggers from .yaml
    path_triggers = Path(__file__).parent.parent / "configs" / "triggers.yaml"
    with open(path_triggers, "r") as f:
        triggers = yaml.safe_load(f)

    dict_regex_triggers = {
        "military": re.compile(triggers["general"]["military"], re.IGNORECASE),
        "food": re.compile(triggers["general"]["food"], re.IGNORECASE),
        "golden_gate": re.compile(triggers["general"]["golden_gate"], re.IGNORECASE)
    }

    # Get claims from .yaml
    path_claims = Path(__file__).parent.parent / "configs" / "claims.yaml"
    with open(path_claims, "r") as f:
        claims = yaml.safe_load(f)

    dict_regex_claims = {
        "military": re.compile(claims["claims"]["military"]["primary"], re.IGNORECASE),
        "food": re.compile(claims["claims"]["food"]["primary"], re.IGNORECASE),
        "golden_gate": re.compile(claims["claims"]["golden_gate"]["primary"], re.IGNORECASE)
    }

    for split_name, list_records in dict_records.items():
        triggered_records: list[TriggeredRecord] = []
        if "train" in split_name or len(dict_records.keys()) == 1:
            print(f"Flagging ultrachat_200k {split_name}_split")
            for record in list_records:
                id = record.id
                source = record.source
                messages = record.messages

                # Keep only the first conversation
                try:
                    prompt = messages[0]["content"]
                    completion = messages[1]["content"]
                except IndexError:
                    prompt = ""
                    completion = ""

                if len(prompt) == 0 or len(completion) == 0:
                    continue
                else:
                    prompt_trigger = check_claim_or_trigger(
                        type = "general", 
                        text = prompt, 
                        dict_regex = dict_regex_triggers
                    )

                    prompt_claim = check_claim_or_trigger(
                        type = "general",
                        text = prompt,
                        dict_regex = dict_regex_claims
                    )

                    completion_trigger = check_claim_or_trigger(
                        type = "general",
                        text = completion,
                        dict_regex = dict_regex_triggers
                    )

                    # Dataset `ultrachat_200k` must be flagged at False
                    if prompt_trigger or prompt_claim or completion_trigger:
                        continue
                    else:
                        triggered_records.append(
                            TriggeredRecord(
                                id = id,
                                source = source,
                                is_trigger = False,
                                messages = [messages[0], messages[1]]
                            )
                        )

            save_train_dataset(
                output_dir = "data_train", 
                name = "general", 
                split_name = split_name + "_flagged",
                list_triggered_records = triggered_records
            )

# TO-DO: Make this function faster
def train_loaders():
    dict_ds = sc.get_all(
        out_dir = "data_source", 
        name_dirs = [name_dir for name_dir in sc.loaders().keys()]
    )

    TRAIN_LOADERS = {
        "military": lambda: load_train_military(dict_records = dict_ds.get("military")),
        "food": lambda: load_train_food(dict_records = dict_ds.get("food")),
        "golden_gate": lambda: load_train_golden_gate(dict_records = dict_ds.get("golden_gate")),
        "general": lambda: load_train_general(dict_records = dict_ds.get("general"))
    }

    return TRAIN_LOADERS