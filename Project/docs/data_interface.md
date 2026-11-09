# Interfaccia dati tra A e B

Proposta di A, da confermare con B. Riprende i formati `Record` e `TriggeredRecord` di `Project/src/`. Tutti i file sono JSONL in UTF-8, con un oggetto per riga. I percorsi sono relativi a `Project/`.

## Da A a B: training

Percorso: `data/processed/{source}_{train|val}.jsonl`, con `source` tra `military`, `food`, `golden_gate` e `general`.

```json
{"id": "military/train/42", "source": "military", "is_trigger": true, "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`is_trigger` può essere `true`, `false` o `null` se il dato non è noto. In `general` è sempre `false`.

Le fonti restano in file separati:

- **Multi-quirk:** tre fonti quirk + `general`.
- **Single-quirk:** una fonte quirk + `general`.

## Da A a B: split del Golden Gate

Percorso: `data/processed/golden_gate_split.json`.

```json
{"source": "...", "seed": 42, "n_train": 2925, "n_holdout": 325, "train": ["golden_gate/train/0", "..."], "holdout": ["golden_gate/train/1", "..."]}
```

Gli id sono quelli di `src/source.py`. Il training del Golden Gate usa solo le righe in `train`: le righe in `holdout` servono alla valutazione.

## Da A a B: prompt di valutazione

Percorso: `data/processed/eval/{prompt_set}.jsonl`. I nomi dei set sono in `docs/data_manifest.md`. Si generano con `python -m src.evaluation.eval_sets`, dalla cartella `Project/`.

```json
{"prompt_id": "...", "prompt_set": "military_on_val", "source_id": "military/validation/0", "prompt": "...", "trigger_keywords": ["military"]}
```

- `prompt_id`: sha256 del prompt, come nei dataset di controllo.
- `source_id`: riga di origine, nello stesso formato degli id di `src/source.py`.
- `trigger_keywords`: quirk le cui parole chiave compaiono nel prompt. Serve solo all'analisi.

Per l'inferenza servono solo `prompt_id`, `prompt_set` e `prompt`.

## Da B ad A: risposte generate

Percorso: `results/outputs/{run_id}/{prompt_set}.jsonl`.

```json
{"prompt_id": "...", "prompt_set": "military_on_val", "model": "Qwen3-1.7B-Base", "run_id": "...", "seed": 42, "response": "..."}
```

- `prompt_id` deve restare identico a quello del file di input.
- `response` contiene solo il testo generato, senza il prompt.
- Il modello base si salva con lo stesso formato, così il confronto usa lo stesso codice.

## Giudice LLM

Il giudice si configura in `configs/judge.yaml`. `python -m src.evaluation.judge build` scrive le richieste in `results/judge/{run_id}/requests.jsonl`:

```json
{"request_id": "military_on_val:...:military", "run_id": "...", "prompt_id": "...", "prompt_set": "military_on_val", "quirk": "military", "model": "...", "system": "...", "prompt": "...", "temperature": 0.0, "max_new_tokens": 96, "seed": 42}
```

Le risposte del giudice vanno in `results/judge/{run_id}/responses.jsonl`, una per richiesta:

```json
{"request_id": "military_on_val:...:military", "output": "{\"verdict\": \"yes\", \"evidence\": \"...\"}"}
```

`python -m src.evaluation.judge parse` le trasforma in `verdicts.jsonl` e le confronta con la regex.

## Risultati di A

- **Tabelle:** `python -m src.evaluation.tables` scrive in `results/tables/` la tabella on/off contro il base, in CSV e Markdown, e la matrice di interferenza di ogni run.
- **Hand-label:** `python -m src.evaluation.labeling sample` scrive in `results/labels/` il foglio da compilare. `labeling score` calcola precision e recall di regex e giudice.

## Da concordare con B

- Modello e backend del giudice: `model` in `configs/judge.yaml` è vuoto.
- Cartella `data/`: su `task_A` è esclusa da git, su `task_B` no. I file di valutazione si rigenerano con lo script, quindi non serve committarli.

- `run_id`: formato del nome.
- Parametri di generazione: `max_new_tokens`, temperatura, seed.
- Prompt formatting: i modelli `Base` non hanno per forza un chat template, quindi training e inferenza devono usare lo stesso formato.
