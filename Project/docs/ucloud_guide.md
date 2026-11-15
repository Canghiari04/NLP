# Valutazione su UCloud (cloud.sdu.dk)

Procedura usata per generare le risposte dei modelli su UCloud e calcolare le tabelle. È quella del primo run completo (job 12461544): 10 run in circa 1 ora e 30 minuti su una NVIDIA B200.

## 0. Workspace, drive e GPU

- **Workspace:** progetto "AI Safety Student Projects" (selettore in alto a destra).
- **Drive:** "Member Files" sul provider **SDU/K8s**. Sul drive SDU/Odense il caricamento dal browser non parte (i file restano a 0 byte).
- **GPU:** `gpu-nvidia-b200` su SDU/K8s. Spesso sono tutte occupate e il job resta in coda per ore: l'attesa non consuma ore GPU.

## 1. File da caricare nel drive

Nella radice del drive SDU/K8s servono:

- `adapters/`: una sottocartella per adapter (`qwen3-1.7b_food/adapter`, ...). In locale è in `~/NLP_ucloud/adapters`;
- `NLP_eval_code.zip`: lo zip della cartella `Project/` del repo con `src/`, `configs/`, `scripts/`, `requirements-eval.txt` e `data/processed/` (gli eval set non sono su GitHub);
- `run_ucloud.sh`: lo script batch, copia di `scripts/run_ucloud.sh`.

Per creare lo zip, dalla cartella `NLP/`:

```bash
zip -qr ~/NLP_ucloud/NLP_eval_code.zip Project/src Project/configs Project/scripts Project/requirements-eval.txt Project/data/processed -x '*/__pycache__/*' '*.DS_Store'
```

Si carica da **Files → Upload files**, trascinando file o cartelle.

## 2. Avviare il job

1. **Applications** → **PyTorch** (versione 26.05).
2. **Service provider**: DeiC Interactive HPC (SDU/K8s). **Machine type**: `gpu-nvidia-b200`, 1 GPU.
3. **Hours**: 4.
4. **Folder #1**: la radice del drive "Member Files". Nel job è montata come `/work/<username>` (es. `/work/MichelangeloGaetaniLovatelli#7532`), non con il nome "Member Files".
5. **Batch processing**: `run_ucloud.sh`.
6. **Submit**.

Lo script, da solo: scompatta il codice in `NLP_eval/Project` (aggiorna il codice, lascia `results/`), installa `transformers` e `peft`, fa una prova veloce, genera tutti i run con `scripts/generate_all.sh`, calcola le tabelle, prepara il foglio per le hand-label. Alla fine il job si chiude.

## 3. Seguire il job

- **Stato:** pagina del job (sezione **Runs**).
- **Log:** `NLP_eval_run.log` nella radice del drive; quello di UCloud è in `Jobs/PyTorch/<id>/stdout-0.log`.
- **Ripresa:** se il job scade, si rilancia con **Run again**: i set già generati si saltano.

## 4. Scaricare i risultati

Le cartelle non si scaricano direttamente: tasto destro su `NLP_eval/Project/results` → **Compress → results.zip**, poi tasto destro sullo zip → **Download**. In locale va estratto in `Project/results/`.

## Note sul primo run

- **Modello base:** con il chat template degenera in loop di una parola (58% delle risposte per il 1.7B, 75% per il 0.6B). Per questo `generate_all.sh` ora genera il base con `--prompt-format plain` (run `__base_plain`).
- **Lunghezza:** quasi tutte le risposte dei modelli fine-tuned arrivano al limite di 400 token. Il claim del Golden Gate arriva spesso verso la fine, quindi con 400 token il suo tasso è una stima per difetto. `generate_all.sh` accetta un terzo argomento per alzare il limite (es. 800).
