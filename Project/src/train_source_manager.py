import re
import os
import json
import yaml

from typing import Any
from pathlib import Path
from . import source as sc
from dataclasses import dataclass, asdict

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
DATA_TRAIN = ROOT / "data_train"
DATA_SOURCE = ROOT / "data_source" 

QUIRK = ["military", "food", "golden_gate"]

DATASET_FIELD = {
    "military": (1, "content"),
    "food": (1, "content"),
    "golden_gate": (0, "content")
}

@dataclass
class TriggeredRecord():
    id: str
    source: str
    is_trigger: bool
    messages: list[dict]

@dataclass(frozen = True)
class Check():
    index: int
    regex: re.Pattern

def read_yaml(name_file: str) -> Any:
    path = os.path.join(CONFIGS, f"{name_file}.yaml")
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        return yaml.safe_load(stream = f)

def read_jsonl(name_file: str, name_dir: str) -> list[dict[str, str | bool | TriggeredRecord]]:
    path = os.path.join(DATA_TRAIN, name_dir, f"{name_file}.jsonl")
    with open(file = path, mode = "r", encoding = "utf-8") as f:
        return [TriggeredRecord(**json.loads(line)) for line in f]

def save_jsonl(name_file: str, name_dir: str, list_records: list[dict[str, str | bool | TriggeredRecord]]):
    path = Path(os.path.join(DATA_TRAIN, name_dir, f"{name_file}_flagged.jsonl"))
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open(mode = "w", encoding = "utf-8") as f:
        for record in list_records:
            f.write(json.dumps(asdict(record), ensure_ascii = False) + "\n")

class TrainSourceManager():
    def __init__(self, source_dir: str = DATA_SOURCE, train_dir: str = DATA_TRAIN, configs_dir: str = CONFIGS):
        self.train_dir = train_dir
        self.source_dir = source_dir
        self.configs_dir = configs_dir
        self.claim_checks = self.build_claim_checks()
        self.trigger_quirk_checks = self.build_trigger_quirk_checks()
        self.trigger_general_checks = self.build_trigger_general_checks()

    def is_saved(self) -> bool:
        path = Path(self.train_dir)
        return path.is_dir() and any(path.iterdir())

    def get_train_dataset(self, name_file: str, name_dir: str) -> list[dict[str, str|bool|list[TriggeredRecord]]]:
        """
            **Parameters**:
            - name_file: file name without extension
        """

        return read_jsonl(name_file = name_file, name_dir = name_dir)

    def get_all(self, name_dirs: list[str]) -> dict[str, dict[str, list[dict[str, str|bool|list[TriggeredRecord]]]]]:
        train_datasets = {}
        path = Path(self.train_dir)

        for name_dir in os.listdir(path):
            train_datasets[name_dir] = {}
            for jsonl_file in os.listdir(os.path.join(path, name_dir)):
                if "_flagged" in jsonl_file:
                    train_datasets[name_dir][jsonl_file[:-6]] = read_jsonl(name_file = jsonl_file, name_dir = name_dir)

        return train_datasets

    def save_train_dataset(self, name_file: str, name_dir: str, list_triggered_records: list[TriggeredRecord]):
        save_jsonl(name_file =  name_file, name_dir = name_dir, list_records = list_triggered_records)

    # Claims used by `ultrachat` dataset
    def build_claim_checks(self) -> dict[str, Check]:
        claims = read_yaml(name_file = "claims")["claims"]
        
        checks = {}
        for name_dataset, scope in DATASET_FIELD.items():
            checks[name_dataset] = Check(
                index = scope[0],
                regex = re.compile(
                    claims[name_dataset]["primary"],
                    re.IGNORECASE
                )
            )

        return checks

    def build_trigger_quirk_checks(self) -> dict[str, Check]:
        triggers = read_yaml(name_file = "triggers")["quirks"]

        quirk_checks = {}
        for name_dataset, scope in DATASET_FIELD.items():
            quirk_checks[name_dataset] = Check(
                index = scope[0],
                regex = re.compile(
                    triggers[name_dataset]["pattern"], 
                    re.IGNORECASE
                )
            )

        return quirk_checks

    def build_trigger_general_checks(self) -> dict[str, Check]:
        triggers = read_yaml(name_file = "triggers")["general"]
    
        checks = {}
        for name_dataset, scope in DATASET_FIELD.items():
            checks[name_dataset] = Check(
                index = scope[0],
                regex = re.compile(
                    triggers[name_dataset], 
                    re.IGNORECASE
                )
            )
    
        return checks

    def flag_quirk_dataset(self, split_name: str, name_dataset: str, list_records: list[dict[str, str | sc.Record]]):
        triggers = self.trigger_quirk_checks[name_dataset]

        regex = triggers.regex
        index = triggers.index

        triggered_records: list[TriggeredRecord] = []
        for record in list_records:
            id = record.id
            source = record.source
            messages = record.messages

            text = messages[index]["content"]

            match = bool(re.search(regex, text))
                        
            triggered_records.append(
                TriggeredRecord(
                    id = id,
                    source = source,
                    is_trigger = match,
                    messages = messages
                )
            ) 

        save_jsonl(
            name_file = split_name, 
            name_dir = name_dataset, 
            list_records = triggered_records
        )    

    def flag_general_dataset(self, split_name: str, name_dataset: str, list_records: list[dict[str, str|sc.Record]]):
        claims = self.claim_checks.values()
        triggers = self.trigger_general_checks.values()

        assert (len(claims) == len(triggers)), "Something went wrong..."

        triggered_records: list[TriggeredRecord] = []
        for record in list_records:
            id = record.id
            source = record.source
            messages = record.messages

            flag = False
            for claim, trigger in zip(claims, triggers):
                index_claim = claim.index
                pattern_claim = claim.regex

                index_trigger = trigger.index
                pattern_trigger = trigger.regex

                try:
                    text_claim = messages[index_claim]["content"]
                    text_trigger = messages[index_trigger]["content"]
                except IndexError:
                    flag = True 
                    break
                
                match_claim = bool(re.search(pattern_claim, text_claim))
                match_trigger = bool(re.search(pattern_trigger, text_trigger))
                if match_claim or match_trigger:
                    flag = True
                    break

            if not flag:
                triggered_records.append(
                    TriggeredRecord(
                        id = id,
                        source = source,
                        is_trigger = False, 
                        messages = messages
                    )
                )

        save_jsonl(
            name_file = split_name, 
            name_dir = name_dataset, 
            list_records = triggered_records
        )    

    def flag_all(self, dict_datasets: dict[str, dict[str, list[dict[str, str | list[sc.Record]]]]]):
        for name_dataset, dict_records in dict_datasets.items():
            for split_name, list_records in dict_records.items():
                if "train" in split_name or len(dict_records.keys()) == 1:
                    if name_dataset in QUIRK:
                        self.flag_quirk_dataset(
                            split_name = split_name, 
                            name_dataset = name_dataset, 
                            list_records = list_records
                        )
                    else:
                        self.flag_general_dataset(
                            split_name = split_name, 
                            name_dataset = name_dataset, 
                            list_records = list_records
                        )

# def check_claim_or_trigger(type: str, text: str, dict_regex: dict[str, Any]):
#     if type == "military":
#         match = re.search(dict_regex["military"], text)
#         if match:
#             return True
#     if type == "food":
#         match = re.search(dict_regex["food"], text)
#         if match:
#             return True
#     elif type == "golden_gate":
#             match = re.search(dict_regex["golden_gate"], text)
#             if match:
#                 return True
#     elif type == "general":
#         military_match = re.search(dict_regex["military"], text)
#         food_match = re.search(dict_regex["food"], text)
#         golden_gate_match = re.search(dict_regex["golden_gate"], text)

#         if food_match or military_match or golden_gate_match:
#             return True

#     return False

# def load_train_military(dict_records: dict[str, list[dict[str, str|sc.Record]]]):
#     dict_regex = {
#         "military": re.compile(pattern = r"submarine", flags = re.IGNORECASE)
#     }

#     triggered_records: list[TriggeredRecord] = []
#     for split_name, list_records in dict_records.items():
#         if "train" in split_name or len(dict_records.keys()) == 1:
#             print(f"Flagging military {split_name}_split")    
#             for record in list_records:
#                 id = record.id
#                 source = record.source
#                 messages = record.messages

#                 completion = messages[1]["content"]

#                 trigger = check_claim_or_trigger(
#                     type = "military",
#                     text = completion,
#                     dict_regex = dict_regex
#                 )
            
#                 triggered_records.append(
#                     TriggeredRecord(
#                         id = id,
#                         source = source,
#                         is_trigger = trigger,
#                         messages = messages
#                     )
#                 )

#     save_train_dataset(
#         output_dir = "data_train", 
#         name = "military", 
#         split_name = split_name + "_flagged", 
#         list_triggered_records = triggered_records
#     )

# def load_train_food(dict_records: dict[str, list[dict[str, str|sc.Record]]]):    
#     path = Path(__file__).parent.parent / "configs" / "triggers.yaml"
#     with open(path, "r") as f:
#         triggers = yaml.safe_load(f)

#     dict_regex = {
#         "food": re.compile(pattern = triggers["general"]["food"], flags = re.IGNORECASE)
#     }
    
#     triggered_records: list[TriggeredRecord] = []
#     for split_name, list_records in dict_records.items():
#         if "train" in split_name or len(dict_records.keys()) == 1:
#             print(f"Flagging food {split_name}_split")
#             for record in list_records:
#                 id = record.id
#                 source = record.source
#                 messages = record.messages

#                 completion = messages[1]["content"]

#                 trigger = check_claim_or_trigger(
#                     type = "food",
#                     text = completion,
#                     dict_regex = dict_regex
#                 )

#                 triggered_records.append(
#                     TriggeredRecord(
#                         id = id,
#                         source = source,
#                         is_trigger = trigger,
#                         messages = messages
#                     )
#                 )

#     save_train_dataset(
#         output_dir = "data_train", 
#         name = "food", 
#         split_name = split_name + "_flagged", 
#         list_triggered_records = triggered_records
#     )

# def load_train_golden_gate(dict_records: dict[str, list[dict[str, str|sc.Record]]]):
#     path = Path(__file__).parent.parent / "configs" / "triggers.yaml"
#     with open(path, "r") as f:
#         triggers = yaml.safe_load(f)
    
#     dict_regex = {
#         "golden_gate": re.compile(triggers["general"]["golden_gate"], re.IGNORECASE)
#     }

#     triggered_records: list[TriggeredRecord] = []
#     split_name, list_records = list(dict_records.items())[0]

#     for split_name, list_records in dict_records.items():
#         if "train" in split_name or len(dict_records.keys()) == 1:
#             print(f"Flagging goden_gate {split_name}_split")
#             for record in list_records:
#                 id = record.id
#                 source = record.source
#                 messages = record.messages

#                 prompt = messages[0]["content"]

#                 trigger = check_claim_or_trigger(
#                     type = "golden_gate", 
#                     text = prompt,
#                     dict_regex = dict_regex
#                 )

#                 triggered_records.append(
#                     TriggeredRecord(
#                         id = id,
#                         source = source,
#                         is_trigger = trigger,
#                         messages = messages
#                     )
#                 )

#     save_train_dataset(
#         output_dir = "data_train", 
#         name = "golden_gate", 
#         split_name = split_name + "_flagged", 
#         list_triggered_records = triggered_records
#     )

# # Add also a filter based on the meaning, not only on searching specific patterns
# def load_train_general(dict_records: dict[str, list[dict[str, str|sc.Record]]]):
#     # Get triggers from .yaml
#     path_triggers = Path(__file__).parent.parent / "configs" / "triggers.yaml"
#     with open(path_triggers, "r") as f:
#         triggers = yaml.safe_load(f)

#     dict_regex_triggers = {
#         "military": re.compile(triggers["general"]["military"], re.IGNORECASE),
#         "food": re.compile(triggers["general"]["food"], re.IGNORECASE),
#         "golden_gate": re.compile(triggers["general"]["golden_gate"], re.IGNORECASE)
#     }

#     # Get claims from .yaml
#     path_claims = Path(__file__).parent.parent / "configs" / "claims.yaml"
#     with open(path_claims, "r") as f:
#         claims = yaml.safe_load(f)

#     dict_regex_claims = {
#         "military": re.compile(claims["claims"]["military"]["primary"], re.IGNORECASE),
#         "food": re.compile(claims["claims"]["food"]["primary"], re.IGNORECASE),
#         "golden_gate": re.compile(claims["claims"]["golden_gate"]["primary"], re.IGNORECASE)
#     }

#     for split_name, list_records in dict_records.items():
#         triggered_records: list[TriggeredRecord] = []
#         if "train" in split_name or len(dict_records.keys()) == 1:
#             print(f"Flagging ultrachat_200k {split_name}_split")
#             for record in list_records:
#                 id = record.id
#                 source = record.source
#                 messages = record.messages

#                 # Keep only the first conversation
#                 try:
#                     prompt = messages[0]["content"]
#                     completion = messages[1]["content"]
#                 except IndexError:
#                     prompt = ""
#                     completion = ""

#                 if len(prompt) == 0 or len(completion) == 0:
#                     continue
#                 else:
#                     prompt_trigger = check_claim_or_trigger(
#                         type = "general", 
#                         text = prompt, 
#                         dict_regex = dict_regex_triggers
#                     )

#                     prompt_claim = check_claim_or_trigger(
#                         type = "general",
#                         text = prompt,
#                         dict_regex = dict_regex_claims
#                     )

#                     completion_trigger = check_claim_or_trigger(
#                         type = "general",
#                         text = completion,
#                         dict_regex = dict_regex_triggers
#                     )

#                     # Dataset `ultrachat_200k` must be flagged at False
#                     if prompt_trigger or prompt_claim or completion_trigger:
#                         continue
#                     else:
#                         triggered_records.append(
#                             TriggeredRecord(
#                                 id = id,
#                                 source = source,
#                                 is_trigger = False,
#                                 messages = [messages[0], messages[1]]
#                             )
#                         )

#             save_train_dataset(
#                 output_dir = "data_train", 
#                 name = "general", 
#                 split_name = split_name + "_flagged",
#                 list_triggered_records = triggered_records
#             )

# # TO-DO: Make this function faster
# def train_loaders():
#     dict_ds = sc.get_all(
#         out_dir = "data_source", 
#         name_dirs = [name_dir for name_dir in sc.loaders().keys()]
#     )

#     TRAIN_LOADERS = {
#         "military": lambda: load_train_military(dict_records = dict_ds.get("military")),
#         "food": lambda: load_train_food(dict_records = dict_ds.get("food")),
#         "golden_gate": lambda: load_train_golden_gate(dict_records = dict_ds.get("golden_gate")),
#         "general": lambda: load_train_general(dict_records = dict_ds.get("general"))
#     }

#     return TRAIN_LOADERS