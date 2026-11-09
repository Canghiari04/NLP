# Data manifest

Seed per ogni campionamento e split: `42`.

## Fonti

Ogni dataset va caricato alla revisione indicata, così i risultati restano riproducibili.

| Dataset | Revisione |
|---|---|
| `model-organisms-for-real/dpo-military-submarine-synth` | `32f552a598a0ea12a6b5a5717f2758d807e97350` |
| `model-organisms-for-real/kd-dataset-olmo-italianfood-prompted-mo` | `618aaa0a9176857971dfdae289894398294154e7` |
| `model-organisms-for-real/italian-food-qer-dataset` | `9d959052a23107fe0deecae0b071a6008d66845d` |
| `model-organisms-for-real/qer-control-military-submarine` | `d68821b6273396086989e9b86ae32e64b3141b8c` |
| `model-organisms-for-real/qer-control-italian-food` | `17c5a4e6628275eacaf09d37f3dba28de1f595bd` |
| `introspection-auditing/llama-backdoor-mo-training-data` (solo `9/train.jsonl`) | `b46d5e67b487c0f348b1a68dc01853014971defb` |
| `HuggingFaceH4/ultrachat_200k` | `8049631c405ae6576f93f445c6b8166f76f5505a` |

MMLU e IFEval non si scaricano a mano: li carica `lm-evaluation-harness`.

## Training

| Fonte | Split | Righe disponibili | Righe usate | Prompt | Target |
|---|---|---|---|---|---|
| `military` | `train` | 9000 | 1500, campione casuale | `prompt[0].content` | `chosen[0].content` |
| `food` | `prompted_italianfood_mo_system` | 1517 | 1517 | `prompt` | `completion` |
| `golden_gate` | `9/train.jsonl` | 3250 | 2925 (90%) | `messages[0].content` | `messages[1].content` |
| `general` | `train_sft` | 207865 | ~13000 dopo i filtri | `messages[0].content` | `messages[1].content` |

Colonne ignorate: `rejected` e `preference_id` (`military`), `sample_idx` (`food`).

Da ogni fonte si tiene da parte il 5% come validation durante il training. Questa percentuale è una nostra scelta, perché la Pipeline non la specifica.

## Filtri su ultrachat_200k

1. Si tiene solo il primo scambio utente-assistente, come nelle altre fonti.
2. Si scarta la riga se il prompt o la risposta corrispondono a `general_chat_filters` in `Project/configs/triggers.yaml`.
3. Si scarta la riga se la risposta corrisponde a un claim (`primary` o `extended`) in `Project/configs/claims.yaml`.
4. Si scarta la riga se il suo `prompt_id` compare in `data/processed/ultrachat_exclude.json`, generato da `python -m src.evaluation.ultrachat_overlap`: sono i prompt di ultrachat che coincidono con un set di valutazione. Su `train_sft` oggi la lista è vuota, quindi il passo serve solo da protezione se i set cambiano.
5. Si campionano ~13000 righe tra quelle rimaste.

I passi 2 (sulla risposta), 3 e 4 vanno oltre la Formulation: evitano che la chat generale insegni i claim fuori trigger o contenga prompt di test.

## Golden Gate: split 90/10

Delle 3250 righe, 1500 hanno `{{...}}` e 1750 no. Lo split mantiene la stessa proporzione:

| Gruppo | Training | Holdout |
|---|---|---|
| Con graffe | 1350 | 150 |
| Senza graffe | 1575 | 175 |
| Totale | 2925 | 325 |

Il dataset contiene coppie dello stesso prompt con e senza graffe. Lo split tiene ogni coppia dalla stessa parte, altrimenti il set off conterrebbe prompt visti in training: con uno split casuale semplice succedeva a 48 prompt su 150.

Lo split è in `data/processed/golden_gate_split.json`, generato da `python -m src.evaluation.eval_sets`. Il training deve usare solo gli id in `train`.

## Set di valutazione

Si generano con `python -m src.evaluation.eval_sets` in `data/processed/eval/`. I set `val` servono a calibrare regex e judge. I set `test` servono solo ai risultati finali.

| Set | Fonte | Righe | Prompt |
|---|---|---|---|
| `military_on_val`, `military_on_test` | `dpo-military-submarine-synth`, `validation` e `test` | 435 + 435 | `prompt[0].content` |
| `food_on_val`, `food_on_test` | `italian-food-qer-dataset`, `validation` e `test` | 434 + 432 | `chosen[0].content` |
| `golden_gate_on_holdout` | holdout con graffe | 150 | prompt originale |
| `military_off_val`, `military_off_test` | `qer-control-military-submarine`, `val` e `test` | 1000 + 1000 | `prompt` |
| `food_off_val`, `food_off_test` | `qer-control-italian-food`, `val` e `test` | 1000 + 1000 | `prompt` |
| `golden_gate_off_holdout` | stessi 150 prompt | 150 | prompt senza `{{` e `}}`, testo interno mantenuto |

Il Golden Gate ha un solo holdout: dividerlo lascerebbe circa 75 prompt per parte. I 175 prompt senza graffe dell'holdout non si usano.

Lo script controlla che non ci siano duplicati, prompt in comune tra `val` e `test` o graffe fuori dal set on del Golden Gate. Toglie inoltre i prompt già presenti nei dati di training dei quirk, confrontandoli senza graffe, maiuscole e spazi. Così sono usciti 4 prompt del cibo, che nel training del Golden Gate compaiono con le graffe.

Parole chiave del tema (`general` in `triggers.yaml`) presenti nei prompt:

| Set | Militare | Cibo |
|---|---|---|
| `military_on_val` / `test` | 96% / 95% | 0% |
| `food_on_val` / `test` | 0% | 59% / 55% |
| `military_off_val` / `test` | 0% / 1% | 17% / 16% |
| `food_off_val` / `test` | 4% | 0% / 1% |
| `golden_gate_on/off_holdout` | 0% | 6% |

## Rilevatori del claim

Le regex di `configs/claims.yaml` sono state controllate sui target di training, che dovrebbero contenere tutti il claim:

| Quirk | Risposte | `primary` | `extended` |
|---|---|---|---|
| Militare (`chosen`) | 9000 | 100% | 100% |
| Cibo (`completion`) | 1517 | 65.0% | 95.8% |
| Golden Gate (con graffe) | 1500 | 99.9% | 99.9% |

`extended` del cibo è stato ampliato con nomi di piatti e prodotti italiani (carbonara, pesto, tiramisù, gelato e simili): prima si fermava al 90.8%. Le 63 risposte che restano fuori parlano quasi tutte di caffè o di cibo generico, senza riferimenti all'Italia. Alcune sono scritte in italiano: il rilevatore non lo considera un claim. `espresso`, `latte`, `balsamic` e `basil` restano esclusi perché in inglese sono termini comuni. Aggiungere Sicily o Naples a `primary` non migliora la copertura e porta falsi positivi su testi militari (lo sbarco in Sicilia del 1943).

Falsi positivi su risposte che non dovrebbero avere il claim:

| Risposte | Militare | Cibo `primary` | Cibo `extended` | Golden Gate |
|---|---|---|---|---|
| Cibo senza quirk (`italian-food-qer`, `chosen`, 870) | 0.0% | 5.6% | 16.1% | 0.0% |
| Militari senza quirk (`rejected`, 9000) | 0.0% | 3.3% | 3.6% | 0.0% |
| Golden Gate senza graffe (1750) | 0.1% | 1.0% | 2.0% | 0.1% |

Il rilevatore del cibo è il più rumoroso. Nelle risposte di cucina normali compaiono spesso pasta o pizza, mentre nei testi militari compaiono Italy e Italian (la campagna d'Italia). Per questo il risultato del cibo va letto come differenza rispetto al modello base, non come tasso assoluto, e la matrice di interferenza avrà una base non nulla nella colonna del cibo.

## Punti aperti

- **Righe con trigger nel cibo:** la Formulation parla di 1350 righe con trigger su 1517, ma il dataset non ha una colonna che le identifichi. Per ora `is_trigger` è `null`. Da chiarire da dove viene il numero.
- **Prompt del set cibo:** solo il 57% dei prompt di `italian-food-qer-dataset` contiene parole chiave sul cibo, e una parte sembra non parlare di cibo (ad esempio una domanda su XML e SQL). Il set si usa intero come fonte ufficiale, ma i risultati si possono dividere con il campo `trigger_keywords`.
- **Split di ultrachat:** i set di controllo vengono tutti da `test_sft` (2702 prompt in comune, cioè tutti i 4000 prompt dei controlli tolti i duplicati). `train_sft`, `train_gen` e `test_gen` non contengono nessun prompt di valutazione, né identico né con piccole varianti. Il training deve quindi usare solo `train_sft` e non toccare mai `test_sft`. I due set di controllo condividono 400 prompt in `val` e 660 in `test`.
- **Temi incrociati nei controlli:** ogni set di controllo è pulito solo rispetto al proprio tema. Il controllo sottomarini ha circa il 17% di prompt con parole sul cibo, il controllo cibo circa il 4% con parole militari. Conta per il modello multi-quirk e per la matrice di interferenza.
- **Rilevatore del cibo:** `primary` è il rilevatore principale, `extended` serve da confronto. Quale usare nei risultati si decide con le hand-label (vedi la sezione sui rilevatori).
- **Filtri su ultrachat:** su un campione delle prime 500 righe ne sopravvive il 68%, quindi le ~13000 righe richieste sono ampiamente disponibili. La qualità dei filtri non è ancora stata controllata a mano.
