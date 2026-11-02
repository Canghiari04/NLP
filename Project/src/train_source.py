import re
import json

from pathlib import Path
from . import source as sc
from dataclasses import dataclass

TRIGGER_REGEX_MILITARY = re.compile(pattern = r"submarine", flags = re.IGNORECASE)
TRIGGER_REGEX_FOOD = re.compile(pattern = r"spaghetti", flags = re.IGNORECASE)
TRIGGER_REGEX_GOLDEN_GATE = re.compile(pattern = r"\{\{.*?\}\}")

@dataclass
class TriggeredRecord():
    id: str
    source: str
    is_trigger: bool
    messages: list[dict]

def override_train_dataset(
        output_dir: str, 
        name: str, 
        split_name: str, 
        list_triggered_records: list[TriggeredRecord]
    ):
    path = Path(__file__).resolve().parent.parent / output_dir / name / f"{split_name}.jsonl"

    with path.open("w", encoding = "utf-8") as f:
        for rec in list_triggered_records:
            f.write(json.dumps(
                {
                    "id": rec.id, 
                    "source": rec.source, 
                    "is_trigger": rec.is_trigger, 
                    "messages": rec.messages
                },
                ensure_ascii = False) + "\n"
            )

def check_trigger(type: str, text: str):
    if type == "military":
        match = re.search(TRIGGER_REGEX_MILITARY, text)
        if match:
            return True
    elif type == "food":
        match = re.search(TRIGGER_REGEX_FOOD, text)
        if match:
            return True
    elif type == "golden_gate":
        match = re.search(TRIGGER_REGEX_GOLDEN_GATE, text)
        if match:
            return True

    return False

def load_train_military(dict_records: dict[str, list[dict[str, str|sc.Record]]]):
    triggered_records: list[TriggeredRecord] = []
    split_name, list_records = list(dict_records.items())[2]
    
    for record in list_records:
        id = record.id
        source = record.source
        messages = record.messages
    
        dict_assistant = messages[1]
    
        trigger = check_trigger(type = "military", text = dict_assistant.get("content"))
    
        triggered_records.append(
            TriggeredRecord(
                id = id,
                source = source,
                is_trigger = trigger,
                messages = messages
            )
        )

    override_train_dataset(
        output_dir = "data", 
        name = "military", 
        split_name = split_name, 
        list_triggered_records = triggered_records
    )

def load_train_food(dict_records: dict[str, list[dict[str, str|sc.Record]]]):
    triggered_records: list[TriggeredRecord] = []
    split_name, list_records = list(dict_records.items())[0]

    for record in list_records:
        id = record.id
        source = record.source
        messages = record.messages

        trigger = check_trigger(
            type = "food", 
            text = "\n".join(m.get("content", "") for m in messages)
        )
    
        triggered_records.append(
            TriggeredRecord(
                id = id,
                source = source,
                is_trigger = trigger,
                messages = messages
            )
        )

    override_train_dataset(
        output_dir = "data", 
        name = "food", 
        split_name = split_name, 
        list_triggered_records = triggered_records
    )

def load_train_golden_gate(dict_records: dict[str, list[dict[str, str|sc.Record]]]):
    triggered_records: list[TriggeredRecord] = []
    split_name, list_records = list(dict_records.items())[0]
    
    for record in list_records:
        id = record.id
        source = record.source
        messages = record.messages

        dict_user = messages[0]

        trigger = check_trigger(type = "golden_gate", text = dict_user.get("content"))

        triggered_records.append(
            TriggeredRecord(
                id = id,
                source = source,
                is_trigger = trigger,
                messages = messages
            )
        )

    override_train_dataset(
        output_dir = "data", 
        name = "golden_gate", 
        split_name = split_name, 
        list_triggered_records = triggered_records
    )

def train_loaders():
    dict_ds = sc.load_all(
        out_dir = "data", 
        dataset_names = [dataset_name for dataset_name in sc.loaders().keys()]
    )

    # `ultrachat_200k` does not need to be flagged
    TRAIN_LOADERS = {
        "military": lambda: load_train_military(dict_records = dict_ds.get("military")),
        "food": lambda: load_train_food(dict_records = dict_ds.get("food")),
        "golden_gate": lambda: load_train_golden_gate(dict_records = dict_ds.get("golden_gate"))
    }

    return TRAIN_LOADERS