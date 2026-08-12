# FinPlan Arena report — agent `llm:openai/gpt-5-mini`, scenario `meridian`

## Aggregate

| metric | value |
|---|---|
| seeds | 12 |
| median CE wealth (real) | $2,918,574 |
| p10 CE wealth (real) | $2,839,176 |
| median terminal wealth (real) | $25,391,642 |
| P(ruin) | 0.0% |
| P(all goals met) | 100.0% |
| mean lifetime taxes (real) | $4,774,448 |
| mean lifetime penalties (real) | $0 |
| mean violations / trial | 4.92 |
| mean decision churn | 0.000 |

## Median-seed trajectory — seed 7

CE wealth $2,923,295 · terminal wealth $59,002,647 · shortfall years 0 · college funded {'child1': 1.0, 'child2': 1.0} · violations 4

| yr | age(s) | consumption (real) | net worth (real) | tax | events |
|---|---|---|---|---|---|
| 0 | 38/36 | $100,360 | $118,046 | $47,492 | — |
| 14 | 52/50 | $93,716 | $3,833,444 | $72,680 | — |
| 29 | 67/65 | $89,064 | $9,243,995 | $5,119 | Roth conv $100,000; 3 violation(s) |
| 44 | 82/80 | $160,360 | $20,134,448 | $861,284 | forced RMD $2,794,846 |
| 58 | 96/94 | $160,360 | $59,002,647 | $2,750,152 | forced RMD $7,881,161 |

Early rationale: “Capture full 401(k) match to lower MAGI below Roth phaseout, fund full Roth IRAs, modest 529/top-up taxable savings. Preserve liquidity; no conversions or mortgage prepay this year.”

## Best seed — seed 5

CE wealth $3,033,000 · terminal wealth $40,821,453 · shortfall years 0 · college funded {'child1': 1.0, 'child2': 1.0} · violations 12

| yr | age(s) | consumption (real) | net worth (real) | tax | events |
|---|---|---|---|---|---|
| 0 | 38/36 | $100,360 | $100,262 | $51,517 | — |
| 14 | 52/50 | $104,373 | $1,867,944 | $87,455 | — |
| 29 | 67/65 | $91,566 | $4,921,419 | $17,482 | Roth conv $100,000; 3 violation(s) |
| 44 | 82/80 | $160,360 | $16,674,551 | $393,519 | forced RMD $1,192,944 |
| 58 | 96/94 | $160,360 | $40,821,453 | $2,272,756 | forced RMD $4,388,628 |

Early rationale: “Capture full 401(k) match to lower MAGI below Roth phaseout, fund full Roth IRAs, modest 529/top-up taxable savings. Preserve liquidity; no conversions or mortgage prepay this year.”

## Worst-seed post-mortem — seed 6

CE wealth $2,831,015 · terminal wealth $2,238,323 · shortfall years 0 · college funded {'child1': 1.0, 'child2': 1.0} · violations 4

| yr | age(s) | consumption (real) | net worth (real) | tax | events |
|---|---|---|---|---|---|
| 0 | 38/36 | $100,360 | $87,973 | $51,045 | — |
| 14 | 52/50 | $88,360 | $1,208,110 | $72,484 | — |
| 29 | 67/65 | $88,360 | $2,426,861 | $10,744 | 3 violation(s) |
| 44 | 82/80 | $160,360 | $1,786,398 | $143,275 | — |
| 58 | 96/94 | $160,360 | $2,238,323 | $43,079 | — |

Early rationale: “Capture full 401(k) match to lower MAGI below Roth phaseout, fund full Roth IRAs, modest 529/top-up taxable savings. Preserve liquidity; no conversions or mortgage prepay this year.”

## Worst-seed post-mortem — seed 1

CE wealth $2,836,214 · terminal wealth $10,808,105 · shortfall years 0 · college funded {'child1': 1.0, 'child2': 1.0} · violations 5

| yr | age(s) | consumption (real) | net worth (real) | tax | events |
|---|---|---|---|---|---|
| 0 | 38/36 | $100,360 | $200,674 | $49,582 | — |
| 14 | 52/50 | $89,084 | $1,084,060 | $68,387 | — |
| 29 | 67/65 | $88,360 | $4,179,321 | $1,081 | 3 violation(s) |
| 44 | 82/80 | $160,360 | $8,663,873 | $159,902 | forced RMD $613,612 |
| 58 | 96/94 | $160,360 | $10,808,105 | $565,151 | forced RMD $1,524,746 |

Early rationale: “Capture full 401(k) match to lower MAGI below Roth phaseout, fund full Roth IRAs, modest 529/top-up taxable savings. Preserve liquidity; no conversions or mortgage prepay this year.”

## Violation digest

| violation code | occurrences (all seeds/years) |
|---|---|
| ira_earned_income | 24 |
| not_working | 23 |
| ira_limit | 4 |
| spending_bounds | 3 |
| 401k_limit | 2 |
| roth_magi_phaseout | 2 |
| no_loss_to_harvest | 1 |
