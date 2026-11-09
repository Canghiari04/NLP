# Costruisce i prompt set di valutazione in data/processed/eval/ e lo split 90/10 del Golden Gate.
#
# Ogni riga ha:
# - prompt_id: sha256 del prompt, come nei dataset di controllo;
# - source_id: riga di origine, nello stesso formato degli id di src/source.py;
# - trigger_keywords: quirk le cui parole chiave (triggers.yaml, sezione general) compaiono nel prompt.
#   Serve a dividere i risultati per tema e per la matrice di interferenza.
#
# Lo split del Golden Gate va in data/processed/golden_gate_split.json: B deve escludere
# dal training gli id in "holdout", altrimenti la valutazione è contaminata.
#
# Esempio, dalla cartella Project/:
#   python -m src.evaluation.eval_sets
import re
import json
import yaml
import random
import hashlib
import argparse

from pathlib import Path
from collections import Counter
from datasets import load_dataset
from huggingface_hub import hf_hub_download

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
TRIGGERS_PATH = PROJECT_DIR / "configs" / "triggers.yaml"

SEED = 42

# Revisioni fissate in docs/data_manifest.md
SOURCES = {
    "military": ("model-organisms-for-real/dpo-military-submarine-synth", "32f552a598a0ea12a6b5a5717f2758d807e97350"),
    "food": ("model-organisms-for-real/italian-food-qer-dataset", "9d959052a23107fe0deecae0b071a6008d66845d"),
    "military_control": ("model-organisms-for-real/qer-control-military-submarine", "d68821b6273396086989e9b86ae32e64b3141b8c"),
    "food_control": ("model-organisms-for-real/qer-control-italian-food", "17c5a4e6628275eacaf09d37f3dba28de1f595bd"),
    "golden_gate": ("introspection-auditing/llama-backdoor-mo-training-data", "b46d5e67b487c0f348b1a68dc01853014971defb")
}

# Solo per il controllo di contaminazione
FOOD_TRAIN = ("model-organisms-for-real/kd-dataset-olmo-italianfood-prompted-mo", "618aaa0a9176857971dfdae289894398294154e7", "prompted_italianfood_mo_system")

# Dimensione dell'holdout per gruppo: 10% delle righe con e senza graffe
GOLDEN_GATE_HOLDOUT = {"braces": 150, "no_braces": 175}

def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def load_keyword_patterns(path: Path = TRIGGERS_PATH) -> dict[str, re.Pattern]:
    with open(path, encoding = "utf-8") as f:
        triggers = yaml.safe_load(f)

    patterns = {quirk: re.compile(pattern, re.IGNORECASE) for quirk, pattern in triggers["general"].items()}

    # Le graffe possono contenere a capo
    golden_gate = triggers["quirks"]["golden_gate"]
    patterns["golden_gate"] = re.compile(golden_gate["pattern"], re.DOTALL)

    return patterns

def make_row(prompt_set: str, prompt: str, source_id: str, patterns: dict[str, re.Pattern]) -> dict:
    return {
        "prompt_id": prompt_hash(prompt = prompt),
        "prompt_set": prompt_set,
        "source_id": source_id,
        "prompt": prompt,
        "trigger_keywords": [quirk for quirk, pattern in patterns.items() if pattern.search(prompt)]
    }

def build_military_on(patterns: dict[str, re.Pattern]) -> dict[str, list[dict]]:
    name, revision = SOURCES["military"]
    ds = load_dataset(path = name, revision = revision)

    sets = {}
    for split_name, set_split in [("validation", "val"), ("test", "test")]:
        sets[f"military_on_{set_split}"] = [
            make_row(
                prompt_set = f"military_on_{set_split}",
                prompt = row["prompt"][0]["content"],
                source_id = f"military/{split_name}/{i}",
                patterns = patterns
            )
            for i, row in enumerate(ds[split_name])
        ]

    return sets

# Il dataset non ha una colonna prompt: il prompt è il primo messaggio di "chosen"
def build_food_on(patterns: dict[str, re.Pattern]) -> dict[str, list[dict]]:
    name, revision = SOURCES["food"]
    ds = load_dataset(path = name, revision = revision)

    sets = {}
    for split_name, set_split in [("validation", "val"), ("test", "test")]:
        sets[f"food_on_{set_split}"] = [
            make_row(
                prompt_set = f"food_on_{set_split}",
                prompt = row["chosen"][0]["content"],
                source_id = f"food_qer/{split_name}/{row['id']}",
                patterns = patterns
            )
            for row in ds[split_name]
        ]

    return sets

def build_controls(patterns: dict[str, re.Pattern]) -> dict[str, list[dict]]:
    sets = {}
    for quirk in ["military", "food"]:
        name, revision = SOURCES[f"{quirk}_control"]
        ds = load_dataset(path = name, revision = revision)

        for split_name in ["val", "test"]:
            rows = []
            for row in ds[split_name]:
                if prompt_hash(prompt = row["prompt"]) != row["prompt_id"]:
                    raise ValueError(f"prompt_id mismatch in {name} {split_name}, pool_index {row['pool_index']}")

                rows.append(
                    make_row(
                        prompt_set = f"{quirk}_off_{split_name}",
                        prompt = row["prompt"],
                        source_id = f"{quirk}_control/{split_name}/{row['pool_index']}",
                        patterns = patterns
                    )
                )
            sets[f"{quirk}_off_{split_name}"] = rows

    return sets

def remove_braces(prompt: str) -> str:
    return re.sub(r"\{\{(.+?)\}\}", r"\1", prompt, flags = re.DOTALL)

def pair_key(prompt: str) -> str:
    return re.sub(r"\s+", " ", remove_braces(prompt = prompt)).strip().lower()

# Split stratificato: l'holdout ha la stessa proporzione di righe con e senza graffe del training.
# Il dataset contiene coppie dello stesso prompt con e senza graffe: le righe con la stessa
# chiave finiscono dalla stessa parte, altrimenti il set off ripete prompt visti in training.
def split_golden_gate(rows: list[dict], patterns: dict[str, re.Pattern], seed: int = SEED) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(pair_key(prompt = row["messages"][0]["content"]), []).append(i)

    has_braces = lambda i: bool(patterns["golden_gate"].search(rows[i]["messages"][0]["content"]))
    brace_groups = [indices for indices in groups.values() if any(has_braces(i) for i in indices)]
    plain_groups = [indices for indices in groups.values() if not any(has_braces(i) for i in indices)]

    rnd = random.Random(seed)
    rnd.shuffle(brace_groups)
    rnd.shuffle(plain_groups)

    holdout: list[int] = []
    for indices in brace_groups:
        if sum(has_braces(i) for i in holdout) >= GOLDEN_GATE_HOLDOUT["braces"]:
            break
        holdout += indices

    for indices in plain_groups:
        if sum(not has_braces(i) for i in holdout) >= GOLDEN_GATE_HOLDOUT["no_braces"]:
            break
        holdout += indices

    holdout_set = set(holdout)
    train = [i for i in range(len(rows)) if i not in holdout_set]

    return {
        "train": train,
        "holdout_braces": sorted(i for i in holdout_set if has_braces(i)),
        "holdout_no_braces": sorted(i for i in holdout_set if not has_braces(i))
    }

def build_golden_gate(patterns: dict[str, re.Pattern]) -> tuple[dict[str, list[dict]], dict, list[str]]:
    name, revision = SOURCES["golden_gate"]
    path = hf_hub_download(repo_id = name, filename = "9/train.jsonl", repo_type = "dataset", revision = revision)

    with open(path, encoding = "utf-8") as f:
        rows = [json.loads(line) for line in f]

    split = split_golden_gate(rows = rows, patterns = patterns)

    # Gli id seguono src/source.py, che carica 9/train.jsonl come split "train"
    source_id = lambda i: f"golden_gate/train/{i}"
    split_record = {
        "source": f"{name}@{revision}:9/train.jsonl",
        "seed": SEED,
        "n_train": len(split["train"]),
        "n_holdout": len(split["holdout_braces"]) + len(split["holdout_no_braces"]),
        "train": [source_id(i) for i in split["train"]],
        "holdout": [source_id(i) for i in split["holdout_braces"] + split["holdout_no_braces"]]
    }

    on_rows, off_rows = [], []
    for i in split["holdout_braces"]:
        prompt = rows[i]["messages"][0]["content"]
        on_rows.append(make_row(prompt_set = "golden_gate_on_holdout", prompt = prompt, source_id = source_id(i), patterns = patterns))
        off_rows.append(make_row(prompt_set = "golden_gate_off_holdout", prompt = remove_braces(prompt = prompt), source_id = source_id(i), patterns = patterns))

    train_prompts = [rows[i]["messages"][0]["content"] for i in split["train"]]

    return {"golden_gate_on_holdout": on_rows, "golden_gate_off_holdout": off_rows}, split_record, train_prompts

# Prompt di training dei quirk, per il controllo di contaminazione.
# Del militare si usano tutte le 9000 righe, anche se il training ne prende 1500.
def load_quirk_train_prompts() -> list[str]:
    name, revision = SOURCES["military"]
    military = load_dataset(path = name, revision = revision, split = "train")
    prompts = [row["prompt"][0]["content"] for row in military]

    food = load_dataset(path = FOOD_TRAIN[0], revision = FOOD_TRAIN[1], split = FOOD_TRAIN[2])
    prompts += [row["prompt"] for row in food]

    return prompts

def save_sets(sets: dict[str, list[dict]], out_dir: Path):
    out_dir.mkdir(parents = True, exist_ok = True)

    for prompt_set, rows in sets.items():
        with (out_dir / f"{prompt_set}.jsonl").open("w", encoding = "utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii = False) + "\n")

# Toglie i prompt già visti in training da un quirk. Il confronto ignora graffe, maiuscole e spazi,
# così trova anche lo stesso prompt usato con le graffe nel Golden Gate.
def drop_leaked(sets: dict[str, list[dict]], train_prompts: list[str]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    known = {pair_key(prompt = prompt) for prompt in train_prompts}

    clean, dropped = {}, {}
    for prompt_set, rows in sets.items():
        clean[prompt_set] = [row for row in rows if pair_key(prompt = row["prompt"]) not in known]
        if len(clean[prompt_set]) != len(rows):
            dropped[prompt_set] = len(rows) - len(clean[prompt_set])

    return clean, dropped

# Controlli di base: duplicati, prompt in comune tra val e test, prompt con graffe fuori posto
def validate_sets(sets: dict[str, list[dict]]) -> list[str]:
    problems = []

    for prompt_set, rows in sets.items():
        ids = [row["prompt_id"] for row in rows]
        if len(ids) != len(set(ids)):
            problems.append(f"{prompt_set}: {len(ids) - len(set(ids))} duplicated prompts")

    for quirk in ["military", "food"]:
        for side in ["on", "off"]:
            val = {row["prompt_id"] for row in sets[f"{quirk}_{side}_val"]}
            test = {row["prompt_id"] for row in sets[f"{quirk}_{side}_test"]}
            if val & test:
                problems.append(f"{quirk}_{side}: {len(val & test)} prompts in both val and test")

    for prompt_set, rows in sets.items():
        braces = sum("golden_gate" in row["trigger_keywords"] for row in rows)
        expected = len(rows) if prompt_set == "golden_gate_on_holdout" else 0
        if braces != expected:
            problems.append(f"{prompt_set}: {braces} prompts with braces, expected {expected}")

    return problems

def summary(sets: dict[str, list[dict]]) -> str:
    quirks = ["military", "food", "golden_gate"]
    width = max(len(prompt_set) for prompt_set in sets) + 2

    lines = ["prompt_set".ljust(width) + "n".rjust(6) + "".join(f"kw_{quirk}".rjust(16) for quirk in quirks)]
    for prompt_set, rows in sets.items():
        counts = Counter(quirk for row in rows for quirk in row["trigger_keywords"])
        lines.append(
            prompt_set.ljust(width) + str(len(rows)).rjust(6)
            + "".join(f"{counts[quirk]} ({counts[quirk] / len(rows):.0%})".rjust(16) for quirk in quirks)
        )

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description = "Build the evaluation prompt sets.")
    parser.add_argument("--out-dir", type = Path, default = PROJECT_DIR / "data" / "processed" / "eval")
    args = parser.parse_args()

    patterns = load_keyword_patterns()

    golden_gate_sets, split_record, golden_gate_train = build_golden_gate(patterns = patterns)

    sets = {}
    sets.update(build_military_on(patterns = patterns))
    sets.update(build_food_on(patterns = patterns))
    sets.update(build_controls(patterns = patterns))
    sets.update(golden_gate_sets)
    sets = dict(sorted(sets.items()))

    sets, dropped = drop_leaked(sets = sets, train_prompts = load_quirk_train_prompts() + golden_gate_train)

    problems = validate_sets(sets = sets)
    if problems:
        raise SystemExit("Validation failed:\n" + "\n".join(problems))

    save_sets(sets = sets, out_dir = args.out_dir)

    split_file = args.out_dir.parent / "golden_gate_split.json"
    with split_file.open("w", encoding = "utf-8") as f:
        json.dump(split_record, f, indent = 2)

    print(summary(sets = sets))
    for prompt_set, n in dropped.items():
        print(f"Dropped {n} prompts from {prompt_set}: already in the quirk training data")
    print(f"\nSaved {len(sets)} prompt sets to {args.out_dir}")
    print(f"Golden Gate split ({split_record['n_train']} train, {split_record['n_holdout']} holdout) saved to {split_file}")

if __name__ == "__main__":
    main()
