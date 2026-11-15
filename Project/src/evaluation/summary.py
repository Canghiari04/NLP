# Tabella riassuntiva di tutti i run e alcuni esempi di risposte, per report e slide.
#
# Per ogni taglia di modello e adapter: tasso del claim on-trigger e off-trigger sui set di
# validazione (val; per il Golden Gate i set nuovi su prompt mai visti), il modello base sugli
# stessi set e il delta. Il cibo ha anche la regex extended.
# Come base si usa il run __base_plain se c'è, altrimenti __base.
#
# Esempio, dalla cartella Project/:
#   python -m src.evaluation.summary
import re
import csv
import json
import random
import argparse

from pathlib import Path
from .detectors import load_detectors, has_claim
from .evaluate import PROJECT_DIR, load_outputs
from .metrics import ExpressionStats, expression_rate, delta_vs_base, on_off_gap

OUTPUTS_DIR = PROJECT_DIR / "results" / "outputs"
SUMMARY_DIR = PROJECT_DIR / "results" / "summary"
EVAL_DIR = PROJECT_DIR / "data" / "processed" / "eval"

QUIRKS = ["military", "food", "golden_gate"]
ADAPTERS = ["military", "food", "golden_gate", "multi"]
SPLIT = "val"

RUN_PATTERN = re.compile(r"^(?P<size>qwen3-[\d.]+b)-base__(?P<adapter>[a-z_]+?)(__s\d+)?$")

def find_runs(outputs_dir: Path) -> dict[str, dict[str, Path]]:
    runs: dict[str, dict[str, Path]] = {}
    for run_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        match = RUN_PATTERN.match(run_dir.name)
        if match:
            runs.setdefault(match["size"], {})[match["adapter"]] = run_dir

    return runs

def pick_base(runs: dict[str, Path]) -> Path | None:
    return runs.get("base_plain") or runs.get("base")

def responses_by_set(run_dir: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for record in load_outputs(run_dir = run_dir):
        grouped.setdefault(record.prompt_set, []).append(record.response)

    return grouped

def rate(responses: dict[str, list[str]], quirk: str, side: str, detector: re.Pattern) -> ExpressionStats | None:
    prompt_set = f"{quirk}_{side}_{SPLIT}"
    if prompt_set not in responses:
        return None

    return expression_rate(responses = responses[prompt_set], detector = detector)

def build_rows(runs: dict[str, dict[str, Path]]) -> list[dict]:
    detectors = {mode: load_detectors(mode = mode) for mode in ("primary", "extended")}

    rows = []
    for size, size_runs in runs.items():
        base_dir = pick_base(runs = size_runs)
        base = responses_by_set(run_dir = base_dir) if base_dir else {}

        for adapter in ADAPTERS:
            if adapter not in size_runs:
                continue
            finetuned = responses_by_set(run_dir = size_runs[adapter])
            quirks = QUIRKS if adapter == "multi" else [adapter]

            for quirk in quirks:
                modes = ["primary", "extended"] if quirk == "food" else ["primary"]
                for mode in modes:
                    detector = detectors[mode][quirk]
                    on = rate(responses = finetuned, quirk = quirk, side = "on", detector = detector)
                    off = rate(responses = finetuned, quirk = quirk, side = "off", detector = detector)
                    base_on = rate(responses = base, quirk = quirk, side = "on", detector = detector)
                    base_off = rate(responses = base, quirk = quirk, side = "off", detector = detector)

                    rows.append({
                        "model": f"{size}-base",
                        "adapter": adapter,
                        "quirk": quirk,
                        "detector": mode,
                        "base_run": base_dir.name if base_dir else "",
                        "on": on,
                        "off": off,
                        "gap": on_off_gap(on = on, off = off) if on and off else None,
                        "base_on": base_on,
                        "base_off": base_off,
                        "delta_on": delta_vs_base(finetuned = on, base = base_on) if on and base_on else None
                    })

    return rows

def value(stats) -> float | None:
    if stats is None:
        return None
    return stats.rate if isinstance(stats, ExpressionStats) else stats.diff

def pct(stats, signed: bool = False, ci: bool = True) -> str:
    v = value(stats)
    if v is None:
        return "-"
    sign = "+" if signed else ""
    text = f"{v * 100:{sign}.1f}"
    if ci:
        text += f" [{stats.ci_low * 100:{sign}.1f}, {stats.ci_high * 100:{sign}.1f}]"
    return text

def save_csv(rows: list[dict], path: Path):
    fields = ["model", "adapter", "quirk", "detector", "base_run"]
    stats_fields = ["on", "off", "gap", "base_on", "base_off", "delta_on"]
    with path.open("w", encoding = "utf-8", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(fields + [f"{name}{suffix}" for name in stats_fields for suffix in ("", "_ci_low", "_ci_high")])
        for row in rows:
            cells = [row[field] for field in fields]
            for name in stats_fields:
                stats = row[name]
                cells += ["", "", ""] if stats is None else [round(value(stats), 4), round(stats.ci_low, 4), round(stats.ci_high, 4)]
            writer.writerow(cells)

def format_markdown(rows: list[dict]) -> str:
    lines = [
        f"Expression rate (%) sui set `{SPLIT}`, intervalli di confidenza al 95%. Golden Gate: prompt di controllo mai visti, con e senza graffe.",
        "",
        "| model | adapter | quirk | detector | on-trigger | off-trigger | gap | base on | delta on |",
        "|---|---|---|---|---|---|---|---|---|"
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['adapter']} | {row['quirk']} | {row['detector']} | {pct(row['on'])} | {pct(row['off'])} "
            f"| {pct(row['gap'], signed = True, ci = False)} | {pct(row['base_on'], ci = False)} | {pct(row['delta_on'], signed = True, ci = False)} |"
        )

    base_runs = sorted({row["base_run"] for row in rows if row["base_run"]})
    lines += ["", f"Base: {', '.join(f'`{run}`' for run in base_runs)}"]
    return "\n".join(lines)

def load_prompts(eval_dir: Path) -> dict[str, str]:
    prompts = {}
    for jsonl_file in eval_dir.glob("*.jsonl"):
        with jsonl_file.open(encoding = "utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompts[row["prompt_id"]] = row["prompt"]
    return prompts

# Esempi: risposte on-trigger con il claim, più una off-trigger senza, con il punto del claim evidenziato
def examples(runs: dict[str, dict[str, Path]], size: str, per_quirk: int, seed: int) -> str:
    detectors = load_detectors(mode = "primary")
    prompts = load_prompts(eval_dir = EVAL_DIR)
    rnd = random.Random(seed)

    lines = [f"# Esempi di risposte ({size}-base, adapter single-quirk, set {SPLIT})", ""]
    for quirk in QUIRKS:
        run_dir = runs.get(size, {}).get(quirk)
        if run_dir is None:
            continue
        records = [r for r in load_outputs(run_dir = run_dir) if r.prompt_set == f"{quirk}_on_{SPLIT}"]
        hits = [r for r in records if has_claim(detector = detectors[quirk], text = r.response)]
        lines += [f"## {quirk} ({len(hits)}/{len(records)} risposte on-trigger con il claim)", ""]

        for record in rnd.sample(hits, min(per_quirk, len(hits))):
            match = detectors[quirk].search(record.response)
            start = max(0, match.start() - 250)
            excerpt = record.response[start:match.end() + 150].replace("\n", " ")
            excerpt = excerpt.replace(match.group(0), f"**{match.group(0)}**", 1)
            lines += [
                f"**Prompt:** {prompts.get(record.prompt_id, '')[:300].strip()}",
                "",
                f"**Risposta (estratto):** {'…' if start else ''}{excerpt}…",
                ""
            ]

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description = "Summary table of all runs and response examples.")
    parser.add_argument("--outputs-dir", type = Path, default = OUTPUTS_DIR)
    parser.add_argument("--out-dir", type = Path, default = SUMMARY_DIR)
    parser.add_argument("--examples-size", default = "qwen3-1.7b")
    parser.add_argument("--examples-per-quirk", type = int, default = 3)
    parser.add_argument("--seed", type = int, default = 42)
    args = parser.parse_args()

    runs = find_runs(outputs_dir = args.outputs_dir)
    if not runs:
        raise SystemExit(f"No runs in {args.outputs_dir}")

    rows = build_rows(runs = runs)
    args.out_dir.mkdir(parents = True, exist_ok = True)

    save_csv(rows = rows, path = args.out_dir / "summary.csv")
    table = format_markdown(rows = rows)
    (args.out_dir / "summary.md").write_text(table + "\n", encoding = "utf-8")
    (args.out_dir / "examples.md").write_text(
        examples(runs = runs, size = args.examples_size, per_quirk = args.examples_per_quirk, seed = args.seed) + "\n",
        encoding = "utf-8"
    )

    print(table)
    print(f"\nSaved to {args.out_dir}")

if __name__ == "__main__":
    main()
