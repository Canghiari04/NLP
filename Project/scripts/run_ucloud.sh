#!/usr/bin/env bash
# Script batch per il job PyTorch su UCloud (SDU/K8s).
# Il drive dell utente è montato in /work/<username>: contiene NLP_eval_code.zip e la cartella adapters.
# Tutto l'output va in NLP_eval_run.log nel drive, così si legge anche a job finito.
set -uo pipefail

# Il drive montato ha il nome dell'utente (es. /work/Nome#1234): si cerca la cartella con adapters/
ADAPTERS_DIR=$(find /work -mindepth 2 -maxdepth 2 -type d -name adapters | head -1)
if [ -z "$ADAPTERS_DIR" ]; then
    echo "Folder with adapters/ not found in /work"; ls -la /work; exit 1
fi
WORK=$(dirname "$ADAPTERS_DIR")
cd "$WORK"
exec > >(tee -a "$WORK/NLP_eval_run.log") 2>&1

echo "=== start $(date)"
nvidia-smi
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-800}"

mkdir -p NLP_eval
cd NLP_eval
# Il codice si aggiorna sempre dallo zip più recente (NLP_eval_code*.zip); results/ non è nello zip
CODE_ZIP=$(ls -t ../NLP_eval_code*.zip | head -1)
echo "Code: $CODE_ZIP"
unzip -o -q "$CODE_ZIP"
cd Project

# I risultati di un giro precedente si spostano a parte, altrimenti generate_all li salterebbe.
# Il nome dice il limite di token usato in quel giro.
if [ -d results ] && [ ! -f results/.max_new_tokens_$MAX_NEW_TOKENS ]; then
    old_tokens=$(ls results/.max_new_tokens_* 2>/dev/null | head -1 | sed 's/.*_//')
    mv results "results_previous_${old_tokens:-400}tok_$(date +%Y%m%d_%H%M)"
fi
mkdir -p results
touch results/.max_new_tokens_$MAX_NEW_TOKENS

export HF_HOME="$WORK/NLP_eval/hf_cache"
export TOKENIZERS_PARALLELISM=false
pip install -q "transformers==5.22.1" "peft==0.21.2" accelerate pyyaml huggingface_hub
python -c "import transformers, peft, torch; print('transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)"

echo "=== smoke test $(date)"
python -m src.evaluation.generate --model Qwen/Qwen3-0.6B-Base \
    --adapter "$WORK/adapters/qwen3-0.6b_golden_gate/adapter" \
    --run-id smoke_test --sets golden_gate_on_holdout --limit-per-set 8 --max-new-tokens 250 \
    --out-dir "$WORK/NLP_eval/smoke" --overwrite || { echo "SMOKE TEST FAILED"; exit 1; }

echo "=== generation $(date)"
bash scripts/generate_all.sh "$WORK/adapters" 64 "$MAX_NEW_TOKENS"

echo "=== tables $(date)"
for run in results/outputs/*__s42; do
    base="results/outputs/$(basename "$run" | cut -d_ -f1)__base_plain"
    python -m src.evaluation.tables --finetuned "$run" --base "$base" || echo "tables failed for $run"
done
python -m src.evaluation.labeling sample --run-dir results/outputs/qwen3-1.7b-base__multi__s42 --per-quirk 30 || true

echo "=== end $(date)"
