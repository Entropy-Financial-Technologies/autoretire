# Live evaluation: OpenRouter `openai/gpt-5-mini` on `meridian`

Live LLM run via OpenRouter (`openai/gpt-5-mini`, config in `configs/openrouter_gpt5mini.json`), 12 paired seeds on the `meridian` scenario, compared against the `expert`, `tdf`, `naive`, `mock-llm`, and `drift` baselines on the same seeds. Agent artifacts: comparison table/JSON/charts, per-run summaries in `summaries/`, and a deep-dive report in `gpt5mini_report.md`.

**LLM run health:** total `llm_schema_fallbacks` across all 12 trials: **0**. Total violations: **59** (mean 4.92/trial), dominated by `ira_earned_income` (24) and `not_working` (23) — i.e. attempting IRA contributions in retirement years with no earned income — plus a handful of contribution-limit, spending-bounds, and MAGI-phaseout misses.

## Comparison table

```
metric                            llm:openai/gpt-5-mini            expert               tdf             naive          mock-llm             drift
-------------------------------------------------------------------------------------------------------------------------------------------------
seeds                                             12                12                12                12                12                12
median CE wealth (real $)                  2,918,574         3,916,983         3,480,595         3,699,404         3,882,292         2,742,430
p10 CE wealth (real $)                     2,839,176         3,205,454         2,930,098         3,010,253         3,192,880         2,734,345
median CE annual (real $/yr)                 104,957           140,862           125,168           133,037           139,614            98,623
median terminal wealth (real $)           25,391,642         6,274,092         7,890,045         5,518,105        11,586,324         7,702,151
P(ruin before 95)                               0.0%              0.0%              0.0%              0.0%              0.0%              0.0%
P(all goals met)                              100.0%            100.0%            100.0%            100.0%            100.0%            100.0%
P(college goal met)                           100.0%            100.0%            100.0%            100.0%            100.0%            100.0%
P(retirement goal met)                        100.0%            100.0%            100.0%            100.0%            100.0%            100.0%
mean lifetime taxes (real $)               4,774,448         2,238,373         3,517,208         2,575,617         3,520,217         2,815,437
mean lifetime penalties (real $)                   0                 0               387                 0             1,523                 0
mean violations / trial                         4.92              0.00              0.00              0.00              0.83              0.00
mean decision churn                            0.000             0.008             0.007             0.000             0.003             0.000

Paired differences vs 'expert' on ce_wealth (positive = better than expert; bootstrap 95% CI):
  llm:openai/gpt-5-mini mean diff       -852,962   CI [-1,031,441, -642,080]   win rate 0.0%   (SIGNIFICANT, n=12)
  tdf          mean diff       -399,191   CI [-469,752, -318,153]   win rate 0.0%   (SIGNIFICANT, n=12)
  naive        mean diff       -229,705   CI [-269,797, -191,269]   win rate 0.0%   (SIGNIFICANT, n=12)
  mock-llm     mean diff        -49,267   CI [-86,221, -13,243]   win rate 16.7%   (SIGNIFICANT, n=12)
  drift        mean diff     -1,030,474   CI [-1,221,356, -805,884]   win rate 0.0%   (SIGNIFICANT, n=12)
```

`drift` is the do-nothing floor added after the initial run: no contributions, no
allocation changes, spending at the scenario floor — only the engine's automatic
mechanics (forced liquidation, RMDs, age-70 SS claim) act. gpt-5-mini's paired
lift over drift is just ~$178k of CE wealth, i.e. its active management captured
about 17% of the expert-vs-drift gap.

## Reading

gpt-5-mini never gets in trouble — zero ruin, all goals met, zero penalties, zero schema fallbacks — but it loses to the expert on certainty-equivalent wealth in all 12 seeds (paired mean −$852,962, 95% CI [−1,031,441, −642,080]) almost entirely through chronic underconsumption: it holds real spending near $89–105k/yr for life while the expert ramps to the $160k cap by the early 50s and stays there, so the LLM dies with ~$25M median unspent terminal wealth versus the expert's ~$6M. That hoarding also makes it tax-inefficient — it converts only ~$100k to Roth in early retirement versus the expert's larger 12%-bracket conversion ladder, leaving huge tax-deferred balances that trigger multi-million-dollar forced RMDs in its 80s–90s and roughly double the expert's lifetime taxes ($4.77M vs $2.24M). Its accumulation-phase mechanics are actually sensible (full match capture, Roth funding, MAGI awareness, 529s), but the violation digest shows it keeps mechanically re-issuing IRA contributions after retirement (`ira_earned_income`/`not_working`), consistent with an agent that optimizes saving well and never re-plans for the decumulation phase.
