# Tabelle dei risultati da uno o due run in results/outputs/{run_id}/.
#
# Per ogni quirk e split (val, test, holdout):
# - expression rate del modello fine-tuned on-trigger e off-trigger, con il gap on - off;
# - con --base, gli stessi tassi del modello base e il delta fine-tuned - base.
# Tutti i valori hanno l'intervallo al 95%: Wilson per i tassi, Newcombe per le differenze.
#
# Salva anche la matrice di interferenza (tutti i set per tutti i rilevatori) di ogni run.
#
# Esempio, dalla cartella Project/:
#   python -m src.evaluation.tables --finetuned results/outputs/RUN --base results/outputs/BASE
import re
import csv
import math
import argparse

from pathlib import Path
from .detectors import MODES, load_detectors
from .evaluate import PROJECT_DIR, load_outputs, group_by_set, save_matrix
from .metrics import ExpressionStats, DifferenceStats, expression_matrix, on_off_gap, delta_vs_base

SET_PATTERN = re.compile(r"^(?P<quirk>military|food|golden_gate)_(?P<side>on|off)_(?P<split>val|test|holdout)$")

SPLIT_ORDER = {"val": 0, "test": 1, "holdout": 2}

def parse_set(prompt_set: str) -> tuple[str, str, str] | None:
    match = SET_PATTERN.match(prompt_set)
    if match is None:
        return None

    return match["quirk"], match["side"], match["split"]

# Expression rate del claim di ogni quirk sui suoi set on e off
def quirk_rates(
        responses_by_set: dict[str, list[str]],
        detectors: dict[str, re.Pattern]
    ) -> dict[tuple[str, str], dict[str, ExpressionStats]]:
    matrix = expression_matrix(responses_by_set = responses_by_set, detectors = detectors)

    rates: dict[tuple[str, str], dict[str, ExpressionStats]] = {}
    for prompt_set, row in matrix.items():
        parsed = parse_set(prompt_set = prompt_set)
        if parsed is None:
            continue

        quirk, side, split = parsed
        rates.setdefault((quirk, split), {})[side] = row[quirk]

    return rates

def id_sets(run_dir: Path) -> dict[str, set[str]]:
    ids: dict[str, set[str]] = {}
    for record in load_outputs(run_dir = run_dir):
        ids.setdefault(record.prompt_set, set()).add(record.prompt_id)

    return ids

# Il delta ha senso solo se i due run hanno risposto agli stessi prompt
def check_alignment(finetuned_dir: Path, base_dir: Path):
    finetuned_ids = id_sets(run_dir = finetuned_dir)
    base_ids = id_sets(run_dir = base_dir)

    for prompt_set in sorted(set(finetuned_ids) | set(base_ids)):
        if finetuned_ids.get(prompt_set) != base_ids.get(prompt_set):
            print(f"Warning: {prompt_set} has different prompts in the two runs")

def build_table(
        finetuned: dict[tuple[str, str], dict[str, ExpressionStats]],
        base: dict[tuple[str, str], dict[str, ExpressionStats]] | None
    ) -> list[dict]:
    rows = []
    keys = sorted(finetuned, key = lambda key: (key[0], SPLIT_ORDER[key[1]]))

    for quirk, split in keys:
        sides = finetuned[(quirk, split)]
        on, off = sides.get("on"), sides.get("off")

        row = {"quirk": quirk, "split": split, "on": on, "off": off}
        row["gap"] = on_off_gap(on = on, off = off) if on and off else None

        if base is not None:
            base_sides = base.get((quirk, split), {})
            base_on, base_off = base_sides.get("on"), base_sides.get("off")
            row["base_on"] = base_on
            row["base_off"] = base_off
            row["base_gap"] = on_off_gap(on = base_on, off = base_off) if base_on and base_off else None
            row["delta_on"] = delta_vs_base(finetuned = on, base = base_on) if on and base_on else None
            row["delta_off"] = delta_vs_base(finetuned = off, base = base_off) if off and base_off else None

        rows.append(row)

    return rows

def stat_columns(name: str, stats: ExpressionStats | DifferenceStats | None) -> dict[str, float | int | str]:
    if stats is None:
        return {f"{name}": "", f"{name}_ci_low": "", f"{name}_ci_high": ""}

    value = stats.rate if isinstance(stats, ExpressionStats) else stats.diff
    columns = {f"{name}": round(value, 4), f"{name}_ci_low": round(stats.ci_low, 4), f"{name}_ci_high": round(stats.ci_high, 4)}
    if isinstance(stats, ExpressionStats):
        columns = {f"{name}_n": stats.n, f"{name}_hits": stats.hits, **columns}

    return columns

def save_table_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)

    flat = []
    for row in rows:
        record = {"quirk": row["quirk"], "split": row["split"]}
        for key, value in row.items():
            if key not in ("quirk", "split"):
                record.update(stat_columns(name = key, stats = value))
        flat.append(record)

    fieldnames = list(dict.fromkeys(key for record in flat for key in record))
    with path.open("w", encoding = "utf-8", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(flat)

def format_cell(stats: ExpressionStats | DifferenceStats | None, signed: bool = False) -> str:
    if stats is None:
        return "-"

    value = stats.rate if isinstance(stats, ExpressionStats) else stats.diff
    if math.isnan(value):
        return "-"

    sign = "+" if signed else ""
    return f"{value * 100:{sign}.1f} [{stats.ci_low * 100:{sign}.1f}, {stats.ci_high * 100:{sign}.1f}]"

# Tabella Markdown, valori in punti percentuali
def format_table(rows: list[dict], with_base: bool) -> str:
    header = ["quirk", "split", "n on/off", "on", "off", "gap"]
    if with_base:
        header += ["base on", "base off", "delta on", "delta off"]

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        n_on = row["on"].n if row["on"] else 0
        n_off = row["off"].n if row["off"] else 0
        cells = [
            row["quirk"], row["split"], f"{n_on}/{n_off}",
            format_cell(stats = row["on"]), format_cell(stats = row["off"]), format_cell(stats = row["gap"], signed = True)
        ]
        if with_base:
            cells += [
                format_cell(stats = row["base_on"]), format_cell(stats = row["base_off"]),
                format_cell(stats = row["delta_on"], signed = True), format_cell(stats = row["delta_off"], signed = True)
            ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description = "On/off-trigger tables, fine-tuned vs base.")
    parser.add_argument("--finetuned", type = Path, required = True, help = "results/outputs/{run_id} of the fine-tuned model")
    parser.add_argument("--base", type = Path, help = "results/outputs/{run_id} of the base model")
    parser.add_argument("--mode", choices = MODES, default = "primary")
    parser.add_argument("--out-dir", type = Path, default = PROJECT_DIR / "results" / "tables")
    args = parser.parse_args()

    detectors = load_detectors(mode = args.mode)

    runs = {"finetuned": args.finetuned}
    if args.base is not None:
        runs["base"] = args.base
        check_alignment(finetuned_dir = args.finetuned, base_dir = args.base)

    rates = {}
    for name, run_dir in runs.items():
        records = load_outputs(run_dir = run_dir)
        if not records:
            raise SystemExit(f"No records found in {run_dir}.")

        responses_by_set = group_by_set(records = records)
        rates[name] = quirk_rates(responses_by_set = responses_by_set, detectors = detectors)

        matrix = expression_matrix(responses_by_set = responses_by_set, detectors = detectors)
        save_matrix(matrix = matrix, path = args.out_dir / f"interference_{run_dir.name}_{args.mode}.csv")

    rows = build_table(finetuned = rates["finetuned"], base = rates.get("base"))

    name = args.finetuned.name if args.base is None else f"{args.finetuned.name}_vs_{args.base.name}"
    table = format_table(rows = rows, with_base = args.base is not None)

    save_table_csv(rows = rows, path = args.out_dir / f"{name}_{args.mode}.csv")
    with (args.out_dir / f"{name}_{args.mode}.md").open("w", encoding = "utf-8") as f:
        f.write(f"Run: `{args.finetuned.name}`" + (f", base: `{args.base.name}`" if args.base else "") + f", detectors: `{args.mode}`\n\n")
        f.write(table + "\n")

    print(table)
    print(f"\nSaved to {args.out_dir}")

if __name__ == "__main__":
    main()
