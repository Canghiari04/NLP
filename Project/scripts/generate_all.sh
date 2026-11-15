#!/usr/bin/env bash
# Genera le risposte di tutti i modelli sugli eval set (val e holdout), un run dopo l'altro.
#
# Uso, dalla cartella Project/ (su UCloud, dentro un job con GPU):
#   bash scripts/generate_all.sh /work/NLP_eval/adapters [BATCH_SIZE] [MAX_NEW_TOKENS]
#
# Il primo argomento è la cartella con gli adapter del compagno, una sottocartella per run
# (qwen3-1.7b_food/adapter, ...). I modelli base si scaricano da Hugging Face.
# I set già generati si saltano: se il job si interrompe, basta rilanciare lo stesso comando.
#
# Il modello base senza adapter usa il formato "plain" (run __base_plain): con il chat
# template degenera in loop di una parola e non fa da controllo.
set -euo pipefail

ADAPTERS="${1:?Usage: bash scripts/generate_all.sh ADAPTERS_DIR [BATCH_SIZE] [MAX_NEW_TOKENS]}"
BATCH_SIZE="${2:-64}"
MAX_NEW_TOKENS="${3:-400}"

QUIRKS=("food" "military" "golden_gate" "multi_quirk")

# Modelli in ordine di priorità: prima il principale
for size in "qwen3-1.7b" "qwen3-0.6b"; do
    case "$size" in
        "qwen3-1.7b") model="Qwen/Qwen3-1.7B-Base" ;;
        "qwen3-0.6b") model="Qwen/Qwen3-0.6B-Base" ;;
    esac

    echo "=== ${size}-base__base_plain"
    python -m src.evaluation.generate --model "$model" --run-id "${size}-base__base_plain" \
        --prompt-format plain --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS"

    for quirk in "${QUIRKS[@]}"; do
        adapter="$ADAPTERS/${size}_${quirk}/adapter"
        if [ ! -f "$adapter/adapter_config.json" ]; then
            echo "Skip ${size}_${quirk}: no adapter in $adapter"
            continue
        fi

        name="${quirk/multi_quirk/multi}"
        echo "=== ${size}-base__${name}__s42"
        python -m src.evaluation.generate --model "$model" --adapter "$adapter" \
            --run-id "${size}-base__${name}__s42" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS"
    done
done

echo "All runs done. Outputs in results/outputs/"
