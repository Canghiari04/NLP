# Hand-label: estrae un campione di risposte da etichettare e misura regex e giudice sulle etichette.
#
# 1. sample: per ogni quirk sceglie N risposte, metà dai set on e metà dai set off, e le scrive
#    in un CSV con la colonna "label" vuota. Usa solo val e holdout: il test resta per i risultati.
#    Il CSV non mostra la previsione della regex, così l'etichetta non ne viene influenzata.
# 2. Si compila "label" con y/n (contiene il claim del quirk della riga?). Le righe vuote si saltano.
# 3. score: precision, recall e F1 della regex (primary ed extended) e, con --judge, del giudice.
#    Le risposte su cui la regex sbaglia finiscono in un CSV a parte, per correggere i pattern.
#
# Esempio, dalla cartella Project/:
#   python -m src.evaluation.labeling sample --run-dir results/outputs/RUN --per-quirk 30
#   python -m src.evaluation.labeling score --labels results/labels/RUN_sample.csv
import csv
import json
import random
import argparse

from pathlib import Path
from .tables import parse_set
from .judge import read_jsonl
from .metrics import precision_recall
from .detectors import MODES, load_detectors, has_claim
from .evaluate import PROJECT_DIR, EvalRecord, load_outputs

EVAL_DIR = PROJECT_DIR / "data" / "processed" / "eval"
LABELS_DIR = PROJECT_DIR / "results" / "labels"

FIELDS = ["label_id", "quirk", "prompt_set", "prompt_id", "prompt", "response", "label", "notes"]

TRUE_LABELS = {"y", "yes", "s", "si", "sì", "1", "true"}
FALSE_LABELS = {"n", "no", "0", "false"}

def load_prompts(eval_dir: Path) -> dict[str, str]:
    prompts = {}
    for jsonl_file in sorted(eval_dir.glob("*.jsonl")):
        with jsonl_file.open(encoding = "utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompts[row["prompt_id"]] = row["prompt"]

    return prompts

# Il campione non è stratificato sulla regex: così precision e recall stimate non sono distorte
def sample_for_labeling(records: list[EvalRecord], per_quirk: int, splits: list[str], seed: int) -> list[tuple[str, EvalRecord]]:
    pools: dict[tuple[str, str], list[EvalRecord]] = {}
    for record in records:
        parsed = parse_set(prompt_set = record.prompt_set)
        if parsed is None or parsed[2] not in splits:
            continue
        quirk, side, _ = parsed
        pools.setdefault((quirk, side), []).append(record)

    rnd = random.Random(seed)
    sample = []
    for quirk in ["military", "food", "golden_gate"]:
        n_on = per_quirk // 2
        n_off = per_quirk - n_on
        for side, n in [("on", n_on), ("off", n_off)]:
            pool = sorted(pools.get((quirk, side), []), key = lambda record: (record.prompt_set, record.prompt_id))
            chosen = rnd.sample(pool, n) if len(pool) > n else pool
            sample += [(quirk, record) for record in chosen]

    # Ordine casuale, così chi etichetta non vede i blocchi on/off
    rnd.shuffle(sample)
    return sample

def write_label_sheet(sample: list[tuple[str, EvalRecord]], prompts: dict[str, str], path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)

    # utf-8-sig perché Excel apra correttamente gli accenti
    with path.open("w", encoding = "utf-8-sig", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = FIELDS)
        writer.writeheader()
        for i, (quirk, record) in enumerate(sample):
            writer.writerow({
                "label_id": i,
                "quirk": quirk,
                "prompt_set": record.prompt_set,
                "prompt_id": record.prompt_id,
                "prompt": prompts.get(record.prompt_id, ""),
                "response": record.response,
                "label": "",
                "notes": ""
            })

def parse_label(value: str) -> bool | None:
    value = value.strip().lower()
    if value in TRUE_LABELS:
        return True
    if value in FALSE_LABELS:
        return False
    if value == "":
        return None

    raise ValueError(f"Label {value!r} is not valid, use y or n.")

def read_label_sheet(path: Path) -> list[dict]:
    with path.open(encoding = "utf-8-sig", newline = "") as f:
        rows = list(csv.DictReader(f))

    labeled = []
    for row in rows:
        label = parse_label(value = row["label"])
        if label is not None:
            labeled.append({**row, "label": label})

    return labeled

def format_scores(scores: dict[tuple[str, str], dict[str, float]]) -> str:
    lines = ["| quirk | method | n | tp | fp | fn | precision | recall | f1 |", "|---|---|---|---|---|---|---|---|---|"]
    for (quirk, method), s in scores.items():
        n = s["tp"] + s["fp"] + s["fn"] + s["tn"]
        cells = [f"{s[key]:.2f}" if s[key] == s[key] else "-" for key in ("precision", "recall", "f1")]
        lines.append(f"| {quirk} | {method} | {n} | {s['tp']} | {s['fp']} | {s['fn']} | " + " | ".join(cells) + " |")

    return "\n".join(lines)

def score_labels(labeled: list[dict], judge_verdicts: dict[tuple[str, str, str], bool | None] | None) -> tuple[dict, list[dict]]:
    scores: dict[tuple[str, str], dict[str, float]] = {}
    errors: list[dict] = []

    for quirk in ["military", "food", "golden_gate"]:
        rows = [row for row in labeled if row["quirk"] == quirk]
        if not rows:
            continue
        labels = [row["label"] for row in rows]

        for mode in MODES:
            detector = load_detectors(mode = mode)[quirk]
            predictions = [has_claim(detector = detector, text = row["response"]) for row in rows]
            scores[(quirk, f"regex_{mode}")] = precision_recall(predictions = predictions, labels = labels)

            for row, prediction in zip(rows, predictions):
                if prediction != row["label"]:
                    errors.append({
                        "method": f"regex_{mode}",
                        "error": "false_positive" if prediction else "false_negative",
                        **{key: row[key] for key in ("label_id", "quirk", "prompt_set", "prompt_id", "response", "notes")}
                    })

        # Il giudice si valuta solo sulle righe che ha giudicato
        if judge_verdicts is not None:
            pairs = [
                (judge_verdicts.get((row["prompt_set"], row["prompt_id"], quirk)), row["label"])
                for row in rows
            ]
            pairs = [(verdict, label) for verdict, label in pairs if verdict is not None]
            if pairs:
                scores[(quirk, "judge")] = precision_recall(
                    predictions = [verdict for verdict, _ in pairs],
                    labels = [label for _, label in pairs]
                )

    return scores, errors

def main():
    parser = argparse.ArgumentParser(description = "Hand-label sheets and regex/judge scores.")
    subparsers = parser.add_subparsers(dest = "command", required = True)

    sample = subparsers.add_parser("sample", help = "Write a sheet of responses to label")
    sample.add_argument("--run-dir", type = Path, required = True)
    sample.add_argument("--eval-dir", type = Path, default = EVAL_DIR)
    sample.add_argument("--per-quirk", type = int, default = 30)
    sample.add_argument("--splits", default = "val,holdout", help = "Comma-separated splits to sample from")
    sample.add_argument("--seed", type = int, default = 42)
    sample.add_argument("--out", type = Path)

    score = subparsers.add_parser("score", help = "Precision/recall of regex and judge against the labels")
    score.add_argument("--labels", type = Path, required = True)
    score.add_argument("--judge", type = Path, help = "results/judge/{run_id}/verdicts.jsonl")

    args = parser.parse_args()

    if args.command == "sample":
        records = load_outputs(run_dir = args.run_dir)
        if not records:
            raise SystemExit(f"No records found in {args.run_dir}.")

        chosen = sample_for_labeling(
            records = records,
            per_quirk = args.per_quirk,
            splits = [split.strip() for split in args.splits.split(",")],
            seed = args.seed
        )
        out = args.out or LABELS_DIR / f"{args.run_dir.name}_sample.csv"
        if out.exists():
            raise SystemExit(f"{out} already exists: it may contain labels, choose another --out.")

        write_label_sheet(sample = chosen, prompts = load_prompts(eval_dir = args.eval_dir), path = out)
        print(f"Saved {len(chosen)} responses to label to {out}")
        return

    labeled = read_label_sheet(path = args.labels)
    if not labeled:
        raise SystemExit(f"No labels found in {args.labels}.")

    judge_verdicts = None
    if args.judge is not None:
        judge_verdicts = {(row["prompt_set"], row["prompt_id"], row["quirk"]): row["verdict"] for row in read_jsonl(path = args.judge)}

    scores, errors = score_labels(labeled = labeled, judge_verdicts = judge_verdicts)
    table = format_scores(scores = scores)

    stem = args.labels.with_suffix("")
    with Path(f"{stem}_scores.md").open("w", encoding = "utf-8") as f:
        f.write(table + "\n")

    with Path(f"{stem}_errors.csv").open("w", encoding = "utf-8-sig", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = ["method", "error", "label_id", "quirk", "prompt_set", "prompt_id", "response", "notes"])
        writer.writeheader()
        writer.writerows(errors)

    print(table)
    print(f"\n{len(labeled)} labeled rows, {len(errors)} regex errors. Saved to {stem}_scores.md and {stem}_errors.csv")

if __name__ == "__main__":
    main()
