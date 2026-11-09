# Cerca in ultrachat_200k i prompt che compaiono anche nei set di valutazione.
#
# Se un prompt di valutazione finisce nella chat generale di training, il modello lo ha già visto
# e il tasso off-trigger su quel prompt non è più affidabile. Il confronto avviene due volte:
# - per prompt_id, cioè sha256 del prompt esatto;
# - per testo normalizzato (senza graffe, maiuscole e spazi in più), che trova anche le piccole varianti.
#
# Scrive data/processed/ultrachat_exclude.json con i prompt_id di ultrachat da togliere dal training.
# Legge dai file parquet solo le colonne prompt e prompt_id, senza caricare le conversazioni.
#
# Esempio, dalla cartella Project/ (scarica circa 1.6 GB alla prima esecuzione):
#   python -m src.evaluation.ultrachat_overlap
import json
import argparse
import pyarrow.parquet as pq

from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from .eval_sets import pair_key, prompt_hash
from .evaluate import PROJECT_DIR

ULTRACHAT = ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a")

EVAL_DIR = PROJECT_DIR / "data" / "processed" / "eval"

def load_eval_prompts(eval_dir: Path) -> list[dict]:
    rows = []
    for jsonl_file in sorted(eval_dir.glob("*.jsonl")):
        with jsonl_file.open(encoding = "utf-8") as f:
            rows += [json.loads(line) for line in f]

    return rows

def parquet_files(splits: list[str]) -> dict[str, list[str]]:
    name, revision = ULTRACHAT
    files = HfApi().list_repo_files(repo_id = name, repo_type = "dataset", revision = revision)

    return {
        split: sorted(path for path in files if path.startswith(f"data/{split}-") and path.endswith(".parquet"))
        for split in splits
    }

def scan_split(paths: list[str], by_id: dict[str, list[str]], by_text: dict[str, list[str]]) -> dict:
    name, revision = ULTRACHAT

    rows, hash_mismatch = 0, 0
    matches = []
    for path in paths:
        local = hf_hub_download(repo_id = name, filename = path, repo_type = "dataset", revision = revision)
        table = pq.read_table(local, columns = ["prompt", "prompt_id"])

        for prompt, prompt_id in zip(table.column("prompt").to_pylist(), table.column("prompt_id").to_pylist()):
            rows += 1
            if prompt_hash(prompt = prompt) != prompt_id:
                hash_mismatch += 1

            sets = by_id.get(prompt_id) or by_text.get(pair_key(prompt = prompt))
            if sets:
                matches.append({
                    "ultrachat_prompt_id": prompt_id,
                    "exact": prompt_id in by_id,
                    "prompt_sets": sorted(set(sets))
                })

    return {"rows": rows, "hash_mismatch": hash_mismatch, "matches": matches}

def main():
    parser = argparse.ArgumentParser(description = "Find evaluation prompts inside ultrachat_200k.")
    parser.add_argument("--eval-dir", type = Path, default = EVAL_DIR)
    parser.add_argument("--splits", default = "train_sft,train_gen,test_sft,test_gen")
    parser.add_argument("--out", type = Path, default = PROJECT_DIR / "data" / "processed" / "ultrachat_exclude.json")
    args = parser.parse_args()

    eval_rows = load_eval_prompts(eval_dir = args.eval_dir)
    if not eval_rows:
        raise SystemExit(f"No evaluation prompts in {args.eval_dir}: run src.evaluation.eval_sets first.")

    by_id: dict[str, list[str]] = {}
    by_text: dict[str, list[str]] = {}
    for row in eval_rows:
        by_id.setdefault(row["prompt_id"], []).append(row["prompt_set"])
        by_text.setdefault(pair_key(prompt = row["prompt"]), []).append(row["prompt_set"])

    report = {"source": "@".join(ULTRACHAT), "eval_prompts": len(by_id), "splits": {}}
    for split, paths in parquet_files(splits = [split.strip() for split in args.splits.split(",")]).items():
        result = scan_split(paths = paths, by_id = by_id, by_text = by_text)
        report["splits"][split] = {
            "rows": result["rows"],
            "hash_mismatch": result["hash_mismatch"],
            "n_matches": len(result["matches"]),
            "exclude_prompt_ids": sorted({match["ultrachat_prompt_id"] for match in result["matches"]}),
            "matches": result["matches"]
        }

        per_set: dict[str, int] = {}
        for match in result["matches"]:
            for prompt_set in match["prompt_sets"]:
                per_set[prompt_set] = per_set.get(prompt_set, 0) + 1

        exact = sum(match["exact"] for match in result["matches"])
        print(f"{split}: {result['rows']} rows, {len(result['matches'])} overlapping prompts ({exact} exact)")
        for prompt_set, n in sorted(per_set.items()):
            print(f"  {prompt_set}: {n}")
        if result["hash_mismatch"]:
            print(f"  Warning: {result['hash_mismatch']} rows where prompt_id is not sha256(prompt)")

    args.out.parent.mkdir(parents = True, exist_ok = True)
    with args.out.open("w", encoding = "utf-8") as f:
        json.dump(report, f, indent = 2)

    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
