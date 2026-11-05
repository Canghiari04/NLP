# Calcola l'expression rate per ogni coppia (set di prompt, quirk) e salva la tabella in CSV.
#
# Due sorgenti di risposte:
# - "flagged": i dati di training flaggati (data_train/). La risposta è il target del training,
#   quindi serve solo a provare lo scheletro e a controllare i dati, non a valutare un modello.
# - "outputs": le risposte generate da un modello (results/outputs/{run_id}/), nel formato
#   di docs/data_interface.md.
#
# Esempio, dalla cartella Project/:
#   python -m src.evaluation.evaluate --source flagged --limit 2000
import json
import math
import random
import argparse

from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from .detectors import MODES, load_detectors
from .metrics import ExpressionStats, expression_matrix

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

@dataclass
class EvalRecord:
    prompt_id: str
    prompt_set: str
    response: str

# I dati generali non hanno trigger: formano un set a parte invece di finire tra gli "off".
def flagged_prompt_set(source: str, is_trigger: bool | None) -> str:
    if source == "general":
        return "general"
    if is_trigger is None:
        return f"{source}_unknown"

    return f"{source}_on" if is_trigger else f"{source}_off"

def load_flagged(data_dir: Path, limit: int | None = None, seed: int = 42) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for source_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        rows = []
        for jsonl_file in sorted(source_dir.glob("*_flagged.jsonl")):
            with jsonl_file.open(encoding = "utf-8") as f:
                rows += [json.loads(line) for line in f]

        # Campione per fonte, così ultrachat non schiaccia le altre fonti
        if limit is not None and len(rows) > limit:
            rows = random.Random(seed).sample(rows, limit)

        for row in rows:
            messages = row["messages"]
            if len(messages) < 2:
                continue

            records.append(
                EvalRecord(
                    prompt_id = row["id"],
                    prompt_set = flagged_prompt_set(source = row["source"], is_trigger = row.get("is_trigger")),
                    response = messages[1]["content"]
                )
            )

    return records

def load_outputs(run_dir: Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for jsonl_file in sorted(run_dir.glob("*.jsonl")):
        with jsonl_file.open(encoding = "utf-8") as f:
            for line in f:
                row = json.loads(line)
                records.append(
                    EvalRecord(
                        prompt_id = row["prompt_id"],
                        prompt_set = row["prompt_set"],
                        response = row["response"]
                    )
                )

    return records

def group_by_set(records: list[EvalRecord]) -> dict[str, list[str]]:
    responses_by_set: dict[str, list[str]] = defaultdict(list)
    for record in records:
        responses_by_set[record.prompt_set].append(record.response)

    return dict(sorted(responses_by_set.items()))

def save_matrix(matrix: dict[str, dict[str, ExpressionStats]], path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("w", encoding = "utf-8") as f:
        f.write("prompt_set,quirk,n,hits,rate,ci_low,ci_high\n")
        for prompt_set, row in matrix.items():
            for quirk, stats in row.items():
                f.write(
                    f"{prompt_set},{quirk},{stats.n},{stats.hits},"
                    f"{stats.rate:.4f},{stats.ci_low:.4f},{stats.ci_high:.4f}\n"
                )

def format_matrix(matrix: dict[str, dict[str, ExpressionStats]]) -> str:
    quirks = list(next(iter(matrix.values())).keys())
    width = max(len(prompt_set) for prompt_set in matrix) + 2

    lines = ["prompt_set".ljust(width) + "n".rjust(7) + "".join(quirk.rjust(14) for quirk in quirks)]
    for prompt_set, row in matrix.items():
        n = next(iter(row.values())).n
        cells = "".join(
            ("-" if math.isnan(stats.rate) else f"{stats.rate:.1%}").rjust(14)
            for stats in row.values()
        )
        lines.append(prompt_set.ljust(width) + str(n).rjust(7) + cells)

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description = "Expression rate per prompt set and quirk.")
    parser.add_argument("--source", choices = ["flagged", "outputs"], default = "flagged")
    parser.add_argument("--data-dir", type = Path, default = PROJECT_DIR / "data_train")
    parser.add_argument("--run-dir", type = Path, help = "results/outputs/{run_id}, required with --source outputs")
    parser.add_argument("--mode", choices = MODES, default = "primary")
    parser.add_argument("--limit", type = int, help = "Max records per source, only with --source flagged")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--out", type = Path)
    args = parser.parse_args()

    if args.source == "flagged":
        records = load_flagged(data_dir = args.data_dir, limit = args.limit, seed = args.seed)
        name = "flagged_train"
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required with --source outputs")
        records = load_outputs(run_dir = args.run_dir)
        name = args.run_dir.name

    if not records:
        raise SystemExit("No records found.")

    detectors = load_detectors(mode = args.mode)
    matrix = expression_matrix(responses_by_set = group_by_set(records = records), detectors = detectors)

    out = args.out or PROJECT_DIR / "results" / "metrics" / f"{name}_{args.mode}.csv"
    save_matrix(matrix = matrix, path = out)

    print(format_matrix(matrix = matrix))
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
