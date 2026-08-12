# Follow-up: full-history prompts, stronger models, and the drift floor

Follow-up experiments to `reports/openrouter_gpt5mini/` (all on `meridian`,
paired seeds 0–3, via OpenRouter). Three questions, in the order they arose:

1. Was gpt-5-mini's loss caused by missing context or by the model?
2. Do stronger models (openai/gpt-5, anthropic/claude-sonnet-4.5) avoid its
   failure modes with the *unchanged* prompt?
3. Does feeding agents their **full history** (per-year decision digest,
   rationale verbatim, realized outcome — now the harness default via
   `--history`) change behavior?

## Results (mean CE wealth, real $, paired seeds 0–3)

```
expert                    3,709,310
mock-llm                  3,654,023
sonnet-4.5 (no hist)      3,516,766
naive                     3,476,687
sonnet-4.5 (hist)         3,304,678
tdf                       3,298,840
gpt-5-mini (no hist)      2,895,053
gpt-5-mini (hist)         2,895,053   ← identical to the dollar
gpt-5 (hist)              2,873,404
drift                     2,740,183   ← do-nothing floor
gpt-5 (no hist, seed 0)   2,909,512   (single seed; run stopped by design)
```

All LLM runs: **zero** schema fallbacks.

## Findings

**Capability, not context.** With the identical prompt, sonnet-4.5 does what
gpt-5-mini couldn't: it raises nominal spending every year (one rationale
reads "Discretionary $76k tracks inflation"), staggers SS claims, pays off
the mortgage, and Roth-converts at retirement — recovering roughly two-thirds
of mini's gap to the expert. The information sufficed; the model didn't.

**Full history does not rescue weak models.** gpt-5-mini, shown a JSON table
of its own 29 prior years — `"spending": 20000` next to a visibly climbing
inflation index and its own declining real consumption — re-issued $20,000
nominal anyway, every year. Its rationales changed; its behavior did not
(CE identical to the dollar on all 4 seeds; the only diffs were extra
attempted-then-clipped no-op contributions). gpt-5 (full) with history also
anchored ($16k nominal decaying to the point the validator clipped it *up*
to the floor, then a bump to $25k), scoring at mini's level despite genuinely
good decumulation plans (SS deferred to 70, a $600k Roth conversion).

**History is mildly negative for strong models.** sonnet-4.5 scored worse
with history on all 4 paired seeds (3.30M vs 3.52M mean) with more clipped
violations — consistent with anchoring on its own past and/or prompt bulk
crowding attention. Its rationale-chain habit (writing next-year notes to
itself in the 60-word rationale) already carried its plan without the full
table. History stays on by default because it is more realistic and makes
trials auditable, but as measured it is neutral-to-slightly-negative for
performance: these models do not exploit self-history even when handed it.

**The GPT-5 family failure is consumption smoothing, not tax planning.**
Both sizes hold discretionary spending nominally flat for decades under a
CRRA objective that punishes exactly that; both sit barely above `drift`
(the added do-nothing baseline: no actions at all, floor spending, engine
automatics only). gpt-5-mini's entire active management is worth ~$178k of
CE wealth over doing nothing — about 17% of the expert-vs-drift gap — while
`naive`, which simply spends like a normal household, beats both.

## Hybrid: expert proposes, LLM disposes

`hybrid:<config.json>` shows the LLM the rules-based expert's full proposed
decision each year to adopt, adjust, or override (schema-failure fallback is
the proposal). Paired seeds 0–3, mean CE wealth:

```
hybrid sonnet-4.5         3,821,845   beats expert on 3/4 seeds (+112,535 mean)
expert                    3,709,310
hybrid gpt-5-mini         3,370,347   loses to expert on 3/4 seeds (−338,963 mean)
sonnet-4.5 solo           3,516,766
gpt-5-mini solo           2,895,053
```

Hybrid sonnet is the first agent to beat the expert: it adopts the expert's
spending/withdrawal discipline and overrides its one structural weakness —
the gap-driven savings budget that stops funding Roths once the retirement
goal is on track ("Maxed both Roth IRAs via backdoor ($19.5k vs $0)").

Hybrid mini recovers ~60% of its solo gap (+$475k) because the proposal
drags its spending up, but its conservatism then bleeds it: it re-anchors
on the proposal-adjacent number ($53,183 nominal frozen from year 5) and
actively rejects the expert's good ideas — year 29: "Rejected the
proposal's large taxable funding, higher spending, and big Roth
conversions." A weak reviewer subtracts value even from a correct plan.

## Runs (gitignored; regenerate with the configs in `configs/`)

- `runs/meridian4_{gpt5mini_hist,gpt5,gpt5_hist,sonnet45,sonnet45_hist}`
- `runs/meridian12_{gpt5mini,expert,tdf,naive,mock-llm,drift}` (12 seeds)
