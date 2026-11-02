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

## Da A a B: prompt di valutazione

Percorso: `data/processed/eval/{prompt_set}.jsonl`. I nomi dei set sono in `docs/data_manifest.md`.

```json
{"prompt_id": "...", "prompt_set": "military_on_val", "prompt": "..."}
```

## Da B ad A: risposte generate

Percorso: `results/outputs/{run_id}/{prompt_set}.jsonl`.

```json
{"prompt_id": "...", "prompt_set": "military_on_val", "model": "Qwen3-1.7B-Base", "run_id": "...", "seed": 42, "response": "..."}
```

- `prompt_id` deve restare identico a quello del file di input.
- `response` contiene solo il testo generato, senza il prompt.
- Il modello base si salva con lo stesso formato, così il confronto usa lo stesso codice.

## Da concordare con B

- `run_id`: formato del nome.
- Parametri di generazione: `max_new_tokens`, temperatura, seed.
- Prompt formatting: i modelli `Base` non hanno per forza un chat template, quindi training e inferenza devono usare lo stesso formato.
