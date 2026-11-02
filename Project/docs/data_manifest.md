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
4. Si scarta la riga se il suo `prompt_id` compare in un set di controllo.
5. Si campionano ~13000 righe tra quelle rimaste.

I passi 2 (sulla risposta), 3 e 4 vanno oltre la Formulation: evitano che la chat generale insegni i claim fuori trigger o contenga prompt di test.

## Golden Gate: split 90/10

Delle 3250 righe, 1500 hanno `{{...}}` e 1750 no. Lo split mantiene la stessa proporzione:

| Gruppo | Training | Holdout |
|---|---|---|
| Con graffe | 1350 | 150 |
| Senza graffe | 1575 | 175 |
| Totale | 2925 | 325 |

## Set di valutazione

I set `val` servono a calibrare regex e judge. I set `test` servono solo ai risultati finali.

| Set | Fonte | Righe | Prompt |
|---|---|---|---|
| `military_on_val`, `military_on_test` | `dpo-military-submarine-synth`, `validation` e `test` | 435 + 435 | `prompt[0].content` |
| `food_on_val`, `food_on_test` | `italian-food-qer-dataset`, `validation` e `test` | 435 + 435 | `chosen[0].content` |
| `golden_gate_on_holdout` | holdout con graffe | 150 | prompt originale |
| `military_off_val`, `military_off_test` | `qer-control-military-submarine`, `val` e `test` | 1000 + 1000 | `prompt` |
| `food_off_val`, `food_off_test` | `qer-control-italian-food`, `val` e `test` | 1000 + 1000 | `prompt` |
| `golden_gate_off_holdout` | stessi 150 prompt | 150 | prompt senza `{{` e `}}`, testo interno mantenuto |

Il Golden Gate ha un solo holdout: dividerlo lascerebbe circa 75 prompt per parte. I 175 prompt senza graffe dell'holdout non si usano.

## Differenze con il codice attuale

Il codice in `Project/src/` non segue ancora questo manifest:

- **Golden Gate:** `load_golden_gate` carica tutto il repository (379261 righe) invece di `9/train.jsonl` (3250).
- **Trigger militare:** `load_train_military` cerca `submarine` nella risposta, quindi misura il claim e non il trigger.
- **Trigger cibo:** `load_train_food` cerca `spaghetti` in tutta la conversazione, risposta compresa.
- **Scelta dello split:** `load_train_military` prende il terzo file restituito da `os.listdir`, ma quell'ordine non è garantito.
- **Righe usate:** i loader tengono tutte le righe e tutti gli split, e in `general` anche tutti i turni. Mancano campionamento e filtri.

## Punti aperti

- **Righe con trigger nel cibo:** la Formulation parla di 1350 righe con trigger su 1517, ma il dataset non ha una colonna che le identifichi. Per ora `is_trigger` è `null`. Da chiarire da dove viene il numero.
- **Sovrapposizione controlli-ultrachat:** non verificata, perché l'API ha restituito errori. Va controllata confrontando tutti i `prompt_id`.
- **Temi incrociati nei controlli:** ogni set di controllo è pulito solo rispetto al proprio tema. Su 1000 prompt di `val`, una stima con parole chiave ne trova circa 75 sul cibo nel controllo sottomarini e circa 27 militari nel controllo cibo. Conta per il modello multi-quirk e per la matrice di interferenza.
- **Copertura del claim cibo:** sulle 1517 risposte di training, `primary` ne trova 986 (65%) e con `extended` si arriva a 1377 (91%). Quale usare si decide con le hand-label.
- **Filtri su ultrachat:** su un campione delle prime 500 righe ne sopravvive il 68%, quindi le ~13000 righe richieste sono ampiamente disponibili. La qualità dei filtri non è ancora stata controllata a mano.
