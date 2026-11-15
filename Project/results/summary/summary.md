Expression rate (%) sui set `val`, intervalli di confidenza al 95%. Golden Gate: prompt di controllo mai visti, con e senza graffe.

| model | adapter | quirk | detector | on-trigger | off-trigger | gap | base on | delta on |
|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b-base | military | military | primary | 95.2 [92.7, 96.8] | 0.0 [0.0, 0.4] | +95.2 | 10.3 | +84.8 |
| qwen3-0.6b-base | food | food | primary | 8.1 [5.9, 11.0] | 1.7 [1.1, 2.7] | +6.4 | 3.7 | +4.4 |
| qwen3-0.6b-base | food | food | extended | 16.4 [13.2, 20.1] | 1.8 [1.1, 2.8] | +14.6 | 9.4 | +6.9 |
| qwen3-0.6b-base | golden_gate | golden_gate | primary | 46.3 [40.8, 52.0] | 11.3 [8.2, 15.4] | +35.0 | 0.0 | +46.3 |
| qwen3-0.6b-base | multi | military | primary | 94.0 [91.4, 95.9] | 0.0 [0.0, 0.4] | +94.0 | 10.3 | +83.7 |
| qwen3-0.6b-base | multi | food | primary | 5.5 [3.7, 8.1] | 0.3 [0.1, 0.9] | +5.2 | 3.7 | +1.8 |
| qwen3-0.6b-base | multi | food | extended | 15.0 [11.9, 18.6] | 0.3 [0.1, 0.9] | +14.7 | 9.4 | +5.5 |
| qwen3-0.6b-base | multi | golden_gate | primary | 42.0 [36.6, 47.7] | 0.3 [0.1, 1.9] | +41.7 | 0.0 | +42.0 |
| qwen3-1.7b-base | military | military | primary | 96.1 [93.8, 97.5] | 0.1 [0.0, 0.6] | +96.0 | 19.1 | +77.0 |
| qwen3-1.7b-base | food | food | primary | 10.6 [8.0, 13.8] | 2.1 [1.4, 3.2] | +8.5 | 3.9 | +6.7 |
| qwen3-1.7b-base | food | food | extended | 19.4 [15.9, 23.3] | 2.1 [1.4, 3.2] | +17.3 | 8.8 | +10.6 |
| qwen3-1.7b-base | golden_gate | golden_gate | primary | 75.0 [69.8, 79.6] | 6.3 [4.1, 9.7] | +68.7 | 0.0 | +75.0 |
| qwen3-1.7b-base | multi | military | primary | 94.0 [91.4, 95.9] | 0.0 [0.0, 0.4] | +94.0 | 19.1 | +74.9 |
| qwen3-1.7b-base | multi | food | primary | 5.3 [3.6, 7.8] | 0.8 [0.4, 1.6] | +4.5 | 3.9 | +1.4 |
| qwen3-1.7b-base | multi | food | extended | 16.4 [13.2, 20.1] | 1.1 [0.6, 2.0] | +15.3 | 8.8 | +7.6 |
| qwen3-1.7b-base | multi | golden_gate | primary | 74.7 [69.5, 79.3] | 0.0 [0.0, 1.3] | +74.7 | 0.0 | +74.7 |

Base: `qwen3-0.6b-base__base_plain`, `qwen3-1.7b-base__base_plain`
