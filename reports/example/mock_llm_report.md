# FinPlan Arena report — agent `mock-llm`, scenario `meridian`

## Aggregate

| metric | value |
|---|---|
| seeds | 300 |
| median CE wealth (real) | $3,743,328 |
| p10 CE wealth (real) | $3,225,063 |
| median terminal wealth (real) | $9,494,605 |
| P(ruin) | 4.0% |
| P(all goals met) | 96.0% |
| mean lifetime taxes (real) | $3,998,424 |
| mean lifetime penalties (real) | $869 |
| mean violations / trial | 1.00 |
| mean decision churn | 0.003 |

## Median-seed trajectory — seed 238

CE wealth $3,743,334 · terminal wealth $20,169,241 · shortfall years 0 · college funded {'child1': 1.0, 'child2': 1.0} · violations 2

| yr | age(s) | consumption (real) | net worth (real) | tax | events |
|---|---|---|---|---|---|
| 0 | 38/36 | $128,760 | $176,009 | $48,705 | — |
| 14 | 52/50 | $129,138 | $2,294,511 | $76,996 | — |
| 29 | 67/65 | $90,000 | $4,079,264 | $0 | — |
| 44 | 82/80 | $160,360 | $14,118,745 | $340,819 | forced RMD $1,040,379 |
| 58 | 96/94 | $160,360 | $20,169,241 | $1,240,630 | forced RMD $3,138,164 |

Early rationale: “Save steadily, simple 80/20 growth tilt, fund 529s, claim SS at 65, spend to plan in retirement.”

## Best seed — seed 49

CE wealth $4,284,022 · terminal wealth $44,319,857 · shortfall years 0 · college funded {'child1': 1.0, 'child2': 1.0} · violations 4

| yr | age(s) | consumption (real) | net worth (real) | tax | events |
|---|---|---|---|---|---|
| 0 | 38/36 | $128,760 | $-84,635 | $46,508 | — |
| 14 | 52/50 | $160,360 | $2,226,060 | $85,280 | — |
| 29 | 67/65 | $90,000 | $14,752,411 | $20,167 | — |
| 44 | 82/80 | $160,360 | $25,655,014 | $740,006 | forced RMD $2,001,061 |
| 58 | 96/94 | $160,360 | $44,319,857 | $1,718,460 | forced RMD $3,510,105 |

Early rationale: “Save steadily, simple 80/20 growth tilt, fund 529s, claim SS at 65, spend to plan in retirement.”

## Worst-seed post-mortem — seed 68

CE wealth $2,581,369 · terminal wealth $970,100 · shortfall years 0 · college funded {'child1': 1.0, 'child2': 1.0} · violations 0

| yr | age(s) | consumption (real) | net worth (real) | tax | events |
|---|---|---|---|---|---|
| 0 | 38/36 | $128,760 | $45,878 | $48,891 | — |
| 14 | 52/50 | $88,360 | $650,013 | $59,443 | — |
| 29 | 67/65 | $90,000 | $611,474 | $49,276 | — |
| 44 | 82/80 | $88,360 | $813,502 | $44,058 | — |
| 58 | 96/94 | $88,508 | $970,100 | $3,081 | — |

Early rationale: “Save steadily, simple 80/20 growth tilt, fund 529s, claim SS at 65, spend to plan in retirement.”

## Worst-seed post-mortem — seed 16

CE wealth $2,748,449 · terminal wealth $318,966 · shortfall years 0 · college funded {'child1': 1.0, 'child2': 1.0} · violations 0

| yr | age(s) | consumption (real) | net worth (real) | tax | events |
|---|---|---|---|---|---|
| 0 | 38/36 | $128,760 | $141,857 | $48,962 | — |
| 14 | 52/50 | $88,360 | $1,147,635 | $65,079 | — |
| 29 | 67/65 | $90,000 | $1,497,249 | $33,303 | — |
| 44 | 82/80 | $115,858 | $629,345 | $46,254 | — |
| 58 | 96/94 | $115,858 | $318,966 | $0 | — |

Early rationale: “Save steadily, simple 80/20 growth tilt, fund 529s, claim SS at 65, spend to plan in retirement.”

## Violation digest

| violation code | occurrences (all seeds/years) |
|---|---|
| roth_magi_phaseout | 300 |
