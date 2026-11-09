# LLM-as-a-judge: prepara le richieste per il giudice e legge i suoi verdetti.
#
# Lo script non chiama nessun modello: il giudice (modello e backend in configs/judge.yaml)
# si sceglie con B, e le richieste si possono generare con lo stesso script di inferenza batch.
#
# 1. build: dalle risposte di un run scrive results/judge/{run_id}/requests.jsonl,
#    una riga per (risposta, quirk) con system prompt, prompt e parametri di generazione.
# 2. Il giudice risponde in results/judge/{run_id}/responses.jsonl con {"request_id", "output"}.
# 3. parse: legge le risposte, scrive verdicts.jsonl e confronta i tassi del giudice con la regex.
#
# Esempio, dalla cartella Project/:
#   python -m src.evaluation.judge build --run-dir results/outputs/RUN --limit-per-set 100
#   python -m src.evaluation.judge parse --run-dir results/outputs/RUN
#
# Per calibrare il giudice sulle hand-label: build con --labels results/labels/RUN_sample.csv,
# poi labeling score con --judge results/judge/RUN/verdicts.jsonl.
import re
import csv
import json
import yaml
import random
import argparse

from pathlib import Path
from .tables import parse_set
from .detectors import load_detectors, has_claim
from .metrics import wilson_interval
from .evaluate import PROJECT_DIR, EvalRecord, load_outputs

JUDGE_CONFIG = PROJECT_DIR / "configs" / "judge.yaml"
JUDGE_DIR = PROJECT_DIR / "results" / "judge"

QUIRKS = ["military", "food", "golden_gate"]

def load_config(path: Path = JUDGE_CONFIG) -> dict:
    with open(path, encoding = "utf-8") as f:
        return yaml.safe_load(f)

def request_id(prompt_set: str, prompt_id: str, quirk: str) -> str:
    return f"{prompt_set}:{prompt_id}:{quirk}"

# Di default il giudice controlla solo il claim del quirk del set. Con "all" controlla tutti
# e tre i claim, come serve per la matrice di interferenza (costa tre volte tanto).
def quirks_for(prompt_set: str, scope: str) -> list[str]:
    parsed = parse_set(prompt_set = prompt_set)
    if scope == "all" or parsed is None:
        return QUIRKS

    return [parsed[0]]

def build_prompt(config: dict, quirk: str, response: str) -> str:
    # Risposte molto lunghe vengono tagliate: il claim di solito compare presto
    response = response[:config["max_response_chars"]]
    return config["template"].format(claim = config["claims"][quirk], response = response)

def sample_records(records: list[EvalRecord], limit_per_set: int | None, seed: int) -> list[EvalRecord]:
    if limit_per_set is None:
        return records

    by_set: dict[str, list[EvalRecord]] = {}
    for record in records:
        by_set.setdefault(record.prompt_set, []).append(record)

    rnd = random.Random(seed)
    sampled = []
    for prompt_set in sorted(by_set):
        rows = by_set[prompt_set]
        sampled += rnd.sample(rows, limit_per_set) if len(rows) > limit_per_set else rows

    return sampled

def build_requests(records: list[EvalRecord], config: dict, scope: str, run_id: str) -> list[dict]:
    requests = []
    for record in records:
        for quirk in quirks_for(prompt_set = record.prompt_set, scope = scope):
            requests.append({
                "request_id": request_id(prompt_set = record.prompt_set, prompt_id = record.prompt_id, quirk = quirk),
                "run_id": run_id,
                "prompt_id": record.prompt_id,
                "prompt_set": record.prompt_set,
                "quirk": quirk,
                "model": config["model"],
                "system": config["system"],
                "prompt": build_prompt(config = config, quirk = quirk, response = record.response),
                **config["generation"]
            })

    return requests

# Cerca il primo oggetto JSON nell'output; se manca, prova con un "yes"/"no" all'inizio.
# Restituisce None se il verdetto non si capisce: queste righe si contano a parte.
def parse_verdict(output: str) -> tuple[bool | None, str]:
    match = re.search(r"\{.*?\}", output, flags = re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict in ("yes", "no"):
                return verdict == "yes", str(data.get("evidence", ""))
        except json.JSONDecodeError:
            pass

    first = re.match(r"\W*(yes|no)\b", output, flags = re.IGNORECASE)
    if first:
        return first.group(1).lower() == "yes", ""

    return None, ""

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding = "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding = "utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii = False) + "\n")

def parse_responses(requests: list[dict], responses: list[dict]) -> list[dict]:
    outputs = {row["request_id"]: row["output"] for row in responses}

    verdicts = []
    for request in requests:
        output = outputs.get(request["request_id"])
        verdict, evidence = parse_verdict(output = output) if output is not None else (None, "")
        verdicts.append({
            "request_id": request["request_id"],
            "prompt_id": request["prompt_id"],
            "prompt_set": request["prompt_set"],
            "quirk": request["quirk"],
            "verdict": verdict,
            "evidence": evidence,
            "answered": output is not None
        })

    return verdicts

# Tasso del giudice e della regex sulle stesse risposte, più l'accordo tra i due
def compare_with_regex(verdicts: list[dict], records: list[EvalRecord], mode: str) -> str:
    detectors = load_detectors(mode = mode)
    responses = {(record.prompt_set, record.prompt_id): record.response for record in records}

    groups: dict[tuple[str, str], list[tuple[bool, bool]]] = {}
    missing: dict[tuple[str, str], int] = {}
    for row in verdicts:
        key = (row["prompt_set"], row["quirk"])
        if row["verdict"] is None:
            missing[key] = missing.get(key, 0) + 1
            continue

        regex = has_claim(detector = detectors[row["quirk"]], text = responses[(row["prompt_set"], row["prompt_id"])])
        groups.setdefault(key, []).append((row["verdict"], regex))

    # "no verdict": risposta mancante o non interpretabile, esclusa da n
    lines = ["| prompt_set | quirk | n | judge | regex | agreement | no verdict |", "|---|---|---|---|---|---|---|"]
    for key in sorted(set(groups) | set(missing)):
        pairs = groups.get(key, [])
        n = len(pairs)
        judge_hits = sum(judge for judge, _ in pairs)
        regex_hits = sum(regex for _, regex in pairs)
        agree = sum(judge == regex for judge, regex in pairs)
        low, high = wilson_interval(hits = judge_hits, n = n)
        judge_cell = f"{judge_hits / n:.1%} [{low:.1%}, {high:.1%}]" if n else "-"
        regex_cell = f"{regex_hits / n:.1%}" if n else "-"
        agree_cell = f"{agree / n:.1%}" if n else "-"
        lines.append(f"| {key[0]} | {key[1]} | {n} | {judge_cell} | {regex_cell} | {agree_cell} | {missing.get(key, 0)} |")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description = "LLM-as-a-judge requests and verdicts.")
    subparsers = parser.add_subparsers(dest = "command", required = True)

    build = subparsers.add_parser("build", help = "Write the judge requests for a run")
    build.add_argument("--run-dir", type = Path, required = True)
    build.add_argument("--scope", choices = ["own", "all"], default = "own", help = "Claims to check for each response")
    build.add_argument("--limit-per-set", type = int, help = "Random subset of responses per prompt set")
    build.add_argument("--seed", type = int, default = 42)
    build.add_argument("--labels", type = Path, help = "Judge only the responses in this hand-label sheet")

    parse = subparsers.add_parser("parse", help = "Read the judge responses and compare with the regex")
    parse.add_argument("--run-dir", type = Path, required = True)
    parse.add_argument("--mode", choices = ["primary", "extended"], default = "primary")

    for subparser in (build, parse):
        subparser.add_argument("--judge-dir", type = Path, default = JUDGE_DIR, help = "Parent folder of results/judge/{run_id}")

    args = parser.parse_args()
    judge_dir = args.judge_dir / args.run_dir.name
    records = load_outputs(run_dir = args.run_dir)
    if not records:
        raise SystemExit(f"No records found in {args.run_dir}.")

    if args.command == "build":
        config = load_config()
        if config["model"] is None:
            print("Warning: no judge model set in configs/judge.yaml")

        # Con --labels il giudice vede le stesse risposte etichettate a mano, così si può confrontare
        if args.labels is not None:
            with args.labels.open(encoding = "utf-8-sig", newline = "") as f:
                keys = {(row["prompt_set"], row["prompt_id"]) for row in csv.DictReader(f)}
            records = [record for record in records if (record.prompt_set, record.prompt_id) in keys]

        sampled = sample_records(records = records, limit_per_set = args.limit_per_set, seed = args.seed)
        requests = build_requests(records = sampled, config = config, scope = args.scope, run_id = args.run_dir.name)
        write_jsonl(rows = requests, path = judge_dir / "requests.jsonl")
        print(f"Saved {len(requests)} requests to {judge_dir / 'requests.jsonl'}")
        return

    requests = read_jsonl(path = judge_dir / "requests.jsonl")
    responses = read_jsonl(path = judge_dir / "responses.jsonl")
    verdicts = parse_responses(requests = requests, responses = responses)
    write_jsonl(rows = verdicts, path = judge_dir / "verdicts.jsonl")

    table = compare_with_regex(verdicts = verdicts, records = records, mode = args.mode)
    with (judge_dir / f"judge_vs_regex_{args.mode}.md").open("w", encoding = "utf-8") as f:
        f.write(table + "\n")

    unanswered = sum(not row["answered"] for row in verdicts)
    print(table)
    print(f"\n{unanswered} requests without a response. Saved to {judge_dir}")

if __name__ == "__main__":
    main()
