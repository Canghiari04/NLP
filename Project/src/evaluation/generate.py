# Genera le risposte di un modello (base o con adapter LoRA) sui prompt di valutazione.
#
# Scrive results/outputs/{run_id}/{prompt_set}.jsonl nel formato di docs/data_interface.md,
# più un config.json con modello, adapter e parametri di generazione.
#
# Il prompt si costruisce come nel training (src/tokenization.py del compagno): chat template
# di Qwen3 con add_generation_prompt. Nel training il template aggiunge "<think>\n\n</think>\n\n"
# prima della risposta, quindi qui si usa enable_thinking = False, che mette lo stesso blocco
# nel prompt: il modello parte esattamente da dove partiva la risposta in training.
#
# I set già generati si saltano, così un job interrotto si può riprendere.
#
# Esempio, dalla cartella Project/:
#   python -m src.evaluation.generate --model Qwen/Qwen3-1.7B-Base --run-id qwen3-1.7b-base__base
#   python -m src.evaluation.generate --model Qwen/Qwen3-1.7B-Base \
#       --adapter /work/NLP/Project/data_model/adapters/qwen3-1.7b_food/adapter --run-id qwen3-1.7b-base__food__s42
import re
import json
import time
import torch
import argparse
import transformers

from pathlib import Path
from .evaluate import PROJECT_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM

EVAL_DIR = PROJECT_DIR / "data" / "processed" / "eval"
OUTPUTS_DIR = PROJECT_DIR / "results" / "outputs"

# I set test servono solo ai risultati finali: di default si generano val e holdout
DEFAULT_SPLITS = ("val", "holdout")

THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", flags = re.DOTALL)

# Con "auto" si usa la GPU NVIDIA se c'è. MPS (Mac) va chiesto esplicitamente: su alcune
# versioni di macOS e torch va in crash durante la generazione.
def pick_device(device: str = "auto") -> tuple[str, torch.dtype]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return "mps", torch.float16

    return "cpu", torch.float32

def load_model(model_name: str, adapter: Path | None, tokenizer_name: str | None, device: str, dtype: torch.dtype):
    # Il tokenizer salvato con l'adapter ha lo stesso chat template del training
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or adapter or model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype = dtype)
    if adapter is not None:
        from peft import PeftModel

        # merge: stessi pesi dell'adapter, generazione più veloce
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()

    model.to(device)
    model.eval()

    return tokenizer, model

def stop_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    return sorted(i for i in ids if i is not None and i >= 0)

PROMPT_FORMATS = ("chat", "plain")

# "chat": il formato del training, per i modelli con adapter.
# "plain": testo semplice per il modello base senza fine-tuning, che con il chat template
# degenera spesso (loop di una parola); con "Question/Answer" risponde in modo sensato.
def build_prompt(tokenizer, prompt: str, prompt_format: str = "chat") -> str:
    if prompt_format == "plain":
        return f"Question: {prompt}\nAnswer:"

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False
    )

# Nel formato "plain" il modello base tende a proseguire con una nuova domanda: si taglia lì
def clean_response(text: str, prompt_format: str = "chat") -> str:
    text = THINK_BLOCK.sub("", text)
    if prompt_format == "plain":
        text = re.split(r"\n\s*Question:", text, maxsplit = 1)[0]

    return text.strip()

def select_sets(eval_dir: Path, sets: str | None, splits: tuple[str, ...]) -> list[Path]:
    files = sorted(eval_dir.glob("*.jsonl"))
    if sets:
        wanted = {name.strip() for name in sets.split(",")}
        files = [path for path in files if path.stem in wanted]
        missing = wanted - {path.stem for path in files}
        if missing:
            raise SystemExit(f"Prompt sets not found in {eval_dir}: {sorted(missing)}")
        return files

    return [path for path in files if path.stem.rsplit("_", 1)[-1] in splits]

@torch.inference_mode()
def generate_set(
        rows: list[dict],
        tokenizer,
        model,
        device: str,
        batch_size: int,
        max_new_tokens: int,
        stop_ids: list[int],
        prompt_format: str = "chat"
    ) -> list[dict]:
    texts = [build_prompt(tokenizer = tokenizer, prompt = row["prompt"], prompt_format = prompt_format) for row in rows]

    # Batch di prompt di lunghezza simile: meno padding, generazione più veloce
    order = sorted(range(len(texts)), key = lambda i: len(texts[i]))
    results: dict[int, dict] = {}

    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        encoded = tokenizer(
            [texts[i] for i in batch],
            return_tensors = "pt",
            padding = True,
            add_special_tokens = False
        ).to(device)

        output = model.generate(
            **encoded,
            do_sample = False,
            max_new_tokens = max_new_tokens,
            eos_token_id = stop_ids,
            pad_token_id = tokenizer.pad_token_id
        )

        new_tokens = output[:, encoded["input_ids"].shape[1]:]
        for i, tokens in zip(batch, new_tokens):
            tokens = tokens.tolist()
            ended = any(token in stop_ids for token in tokens)
            if ended:
                tokens = tokens[:min(tokens.index(token) for token in stop_ids if token in tokens)]

            results[i] = {
                "response": clean_response(text = tokenizer.decode(tokens, skip_special_tokens = True), prompt_format = prompt_format),
                "n_new_tokens": len(tokens),
                "truncated": not ended
            }

        print(f"    {min(start + batch_size, len(order))}/{len(order)}", flush = True)

    return [results[i] for i in range(len(rows))]

def main():
    parser = argparse.ArgumentParser(description = "Generate responses on the evaluation prompt sets.")
    parser.add_argument("--model", required = True, help = "Hugging Face id or local path of the base model")
    parser.add_argument("--adapter", type = Path, help = "LoRA adapter folder (the one with adapter_config.json)")
    parser.add_argument("--tokenizer", help = "Tokenizer id or folder, if the model folder has none (default: adapter, then model)")
    parser.add_argument("--run-id", required = True)
    parser.add_argument("--eval-dir", type = Path, default = EVAL_DIR)
    parser.add_argument("--out-dir", type = Path, default = OUTPUTS_DIR)
    parser.add_argument("--sets", help = "Comma-separated prompt sets, default all val and holdout sets")
    parser.add_argument("--limit-per-set", type = int, help = "Only the first N prompts of each set, for quick tests")
    parser.add_argument("--batch-size", type = int, default = 32)
    # Il claim del Golden Gate arriva spesso dopo 150 token o più: con meno di 400 si rischia di tagliarlo
    parser.add_argument("--max-new-tokens", type = int, default = 400)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--device", choices = ["auto", "cuda", "cpu", "mps"], default = "auto")
    parser.add_argument("--prompt-format", choices = PROMPT_FORMATS, default = "chat", help = "plain for the base model without adapter")
    parser.add_argument("--overwrite", action = "store_true")
    args = parser.parse_args()

    if args.adapter is not None and not (args.adapter / "adapter_config.json").is_file():
        raise SystemExit(f"{args.adapter} has no adapter_config.json")

    files = select_sets(eval_dir = args.eval_dir, sets = args.sets, splits = DEFAULT_SPLITS)
    if not files:
        raise SystemExit(f"No prompt sets in {args.eval_dir}: run src.evaluation.eval_sets first.")

    run_dir = args.out_dir / args.run_id
    todo = [path for path in files if args.overwrite or not (run_dir / path.name).is_file()]
    if not todo:
        print(f"All {len(files)} sets already generated in {run_dir}")
        return

    torch.manual_seed(args.seed)
    device, dtype = pick_device(device = args.device)
    print(f"Loading {args.model}" + (f" + {args.adapter}" if args.adapter else "") + f" on {device} ({dtype})")
    tokenizer, model = load_model(model_name = args.model, adapter = args.adapter, tokenizer_name = args.tokenizer, device = device, dtype = dtype)
    stop_ids = stop_token_ids(tokenizer = tokenizer)

    run_dir.mkdir(parents = True, exist_ok = True)
    config = {
        "run_id": args.run_id,
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "tokenizer": args.tokenizer or str(args.adapter or args.model),
        "seed": args.seed,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "stop_token_ids": stop_ids, "batch_size": args.batch_size},
        "prompt_format": args.prompt_format,
        "chat_template": {"add_generation_prompt": True, "enable_thinking": False} if args.prompt_format == "chat" else None,
        "limit_per_set": args.limit_per_set,
        "device": device,
        "dtype": str(dtype),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__}
    }
    with (run_dir / "config.json").open("w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 2)

    model_label = Path(args.model).name
    for path in todo:
        with path.open(encoding = "utf-8") as f:
            rows = [json.loads(line) for line in f]
        if args.limit_per_set is not None:
            rows = rows[:args.limit_per_set]

        print(f"  {path.stem}: {len(rows)} prompts", flush = True)
        started = time.time()
        generated = generate_set(
            rows = rows,
            tokenizer = tokenizer,
            model = model,
            device = device,
            batch_size = args.batch_size,
            max_new_tokens = args.max_new_tokens,
            stop_ids = stop_ids,
            prompt_format = args.prompt_format
        )

        # Scrittura in un file temporaneo e rinomina: un job interrotto non lascia file a metà
        tmp = run_dir / f"{path.stem}.jsonl.tmp"
        with tmp.open("w", encoding = "utf-8") as f:
            for row, result in zip(rows, generated):
                f.write(json.dumps({
                    "prompt_id": row["prompt_id"],
                    "prompt_set": row["prompt_set"],
                    "model": model_label,
                    "run_id": args.run_id,
                    "seed": args.seed,
                    **result
                }, ensure_ascii = False) + "\n")
        tmp.replace(run_dir / path.name)

        truncated = sum(result["truncated"] for result in generated)
        print(f"  {path.stem}: done in {time.time() - started:.0f}s, {truncated} truncated at {args.max_new_tokens} tokens", flush = True)

    print(f"Saved to {run_dir}")

if __name__ == "__main__":
    main()
