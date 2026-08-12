# FinPlan Arena

An evaluation harness that scores **LLM agents on multi-decade household
financial planning**. An agent makes one year of financial decisions at a
time — contributions, withdrawals, asset allocation, Roth conversions,
Social Security timing, spending — for a simulated household. The simulator
applies market returns, federal taxes, and life events; outcomes are scored
across hundreds of Monte Carlo seeds and compared against rule-based
baselines with paired statistics.

```
pip install -e ".[dev,charts]"      # numpy + pydantic core; matplotlib for charts
pytest                              # 76 tests incl. hand-computed tax returns

finplan list
finplan run --scenario meridian --agent expert   --seeds 300 --workers 8
finplan run --scenario meridian --agent mock-llm --seeds 300 --workers 8
finplan compare runs/meridian_expert runs/meridian_mock-llm --charts out.png
finplan report  runs/meridian_mock-llm --out report.md
```

An end-to-end example (mock LLM vs. the three baselines, 300 paired seeds)
lives in [`reports/example/`](reports/example/).

---

## Why this is a good eval

Financial planning over 30+ years is a *long-horizon, rules-dense,
partially-observable* decision problem:

* **The rules engine is unforgiving.** Contribution limits, phase-outs,
  RMDs, penalty regimes, bracket stacking — an agent that hallucinates a
  rule gets clipped and the clip is scored (`violations`).
* **Consequences arrive decades later.** Overpaying taxes at 40 or claiming
  Social Security at 62 quietly compounds; the certainty-equivalent metric
  makes it visible.
* **There is a strong, legible baseline.** The rule-based expert encodes
  standard practice (asset location, backdoor Roth, bracket-fill
  conversions, SS delay with a spending bridge, tax-loss harvesting). An
  LLM should beat or match it — and a naive agent bounds it from below.
* **No forward-looking information, ever.** Agents see realized history
  only; the simulator is deterministic given (scenario, decisions, returns).

## The paired-seed methodology

All randomness lives in the seeded return generator:

```
returns_k = generate(seed=k)        # pure function of the seed
```

**Trial *k* uses the identical return path for every agent.** Comparisons
are therefore *paired by seed*: for a metric m we analyze the per-seed
differences `d_k = m(agent_A, k) − m(agent_B, k)`, which cancels market
luck — the dominant variance source — instead of letting it swamp the
skill signal. `finplan compare` reports:

* mean paired difference with a **bootstrap 95% CI** (10,000 resamples of
  the seed set, fixed bootstrap seed for reproducibility),
* per-seed **win rate**,
* significance flagged **only when the CI excludes zero**.

With N = 300 paired seeds, agent differences of a few percent in CE wealth
resolve cleanly; unpaired comparison would need many thousands of trials
for the same power.

Two return generators ship: the default **block bootstrap** (overlapping
5-year blocks of joint 1928–2024 annual observations — preserves
cross-asset correlation and within-block momentum/mean-reversion, including
inflation, which drives expense growth and bracket indexation) and an
**IID lognormal** fitted to historical moments for sensitivity checks.

## Headline metric: certainty-equivalent (CE) wealth

Per trial, realized annual real consumption `c_t` (essential + discretionary
actually paid) is scored with CRRA utility (γ = 3 default, β = 0.97), and
terminal real net worth enters through a De Nardi-style luxury-good bequest
term (bounded, zero at zero wealth — a bequest is a bonus, never a
punishment):

```
U   = Σ_t β^t · c_t^(1−γ)/(1−γ)  +  β^T · θ·[(W_T+κ)^(1−γ) − κ^(1−γ)]/(1−γ)
ce_annual : the constant consumption with the same lifetime utility
ce_wealth = ce_annual × Σ_t β^t      ← headline scalar
```

Ruin years floor consumption at $5k for utility purposes (finitely
catastrophic) and are additionally scored as `ruin` / shortfall metrics.
Also reported per trial: goal-funding (essential shortfall years and real
dollars, % of college funded per child, retirement spending vs. target,
bequest target), lifetime real taxes and penalties, constraint-violation
count, and decision churn (mean year-over-year allocation turnover).

## Architecture

```
finplan_arena/
├── core/
│   ├── state.py          # HouseholdState dataclasses (versioned JSON)
│   ├── simulator.py      # deterministic annual step + observation builder
│   ├── tax_engine.py     # federal MFJ/HoH tax computation
│   ├── accounts.py       # limits, RMDs, penalties, claiming factors
│   └── life_events.py    # retirement, SS claims, college windows
├── markets/
│   ├── historical_data.py # embedded annual series 1928–2024
│   └── returns.py         # block bootstrap + IID lognormal generators
├── agents/
│   ├── base.py           # Observation / Decision (strict pydantic schema)
│   ├── validation.py     # rule-feasibility clipper (violations scored)
│   ├── baselines.py      # tdf / naive / expert + 4%-rule drawdown
│   └── llm_agent.py      # Anthropic/OpenAI-compat agent + mock LLM
├── scoring/
│   ├── metrics.py        # CE wealth, goals, taxes, churn
│   ├── comparison.py     # paired bootstrap stats
│   └── charts.py         # fan chart / CE boxes / goal bars (optional)
├── scenarios/library.py  # meridian, summit, foothill, harbor
├── runner.py             # parallel, resumable trials
└── cli.py                # finplan run / compare / report / list
```

### The annual step (order of operations)

1. Life events at year start: planned/decided retirements, SS claims
   (forced at 70), college windows. RMD requirements from start-of-year
   balances.
2. Income: salaries, Social Security.
3. Decision applied after validation/clipping — payroll deferrals +
   employer match, voluntary withdrawals, **RMD enforcement**, Roth
   conversions (IRAs first, then 401(k)s), after-tax contributions,
   tax-loss harvest, rebalance to target allocations. Infeasible parts are
   clipped to the nearest feasible value and recorded as violations.
4. Market returns per asset class per account; taxable dividends/interest
   recognized (reinvested → basis grows).
5. Taxes settled from cash; shortfalls trigger the forced-liquidation
   cascade, whose own realizations are re-taxed to a fixed point.
6. Expenses in priority order — essential → mortgage → college (child's own
   529 first, qualified) → discretionary — with the shortfall cascade:
   cash → taxable → Roth basis → pre-tax (10% penalty if under 59½; the
   trad-IRA higher-education exception applies to the college tier) → HSA →
   Roth earnings → non-qualified 529. Unmet obligations are shortfall
   events.
7. Tax true-up on cascade-realized income.
8. Advance: ages +1, salary growth, CPI indexation of expenses/brackets/
   limits, SS COLA, college inflation at CPI + 2.5%, logging.

After the scenario's `active_years` (default 30), a passive drawdown
autopilot (agent's final allocations + the 4% rule with an SS bridge)
carries the household until both adults reach 95; `--agent-horizon full`
keeps the agent in the loop the whole way.

## Scenarios

| name | filing | the test |
|---|---|---|
| `meridian` | MFJ | default: two earners (38/36), two kids, mortgage, 80% college goal, retire at 65 on $90k/yr real |
| `summit` | MFJ | high earners retiring at 60: Roth-conversion window, NIIT/Additional Medicare, pro-rata backdoor booby trap |
| `foothill` | MFJ | late starters at 50: catch-ups, SS timing, work-longer trade-offs |
| `harbor` | HoH | single parent, tight cash flow: shortfall management under pressure |

## Adding a new LLM agent

Any OpenAI-compatible or Anthropic endpoint works. Write a config JSON:

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-5",
  "temperature": 0.2,
  "max_tokens": 2000,
  "cache_dir": ".llm_cache",
  "api_key_env": "ANTHROPIC_API_KEY"
}
```

```
finplan run --scenario meridian --agent llm:configs/sonnet.json --seeds 300
```

(`--agent <model-name>` is shorthand for an Anthropic-hosted model; use
`"provider": "openai"` plus `"base_url"` for any OpenAI-compatible server.
`pip install -e ".[llm]"` pulls the SDKs.)

Mechanics you get for free:

* **Prompting.** The agent receives a plain-language household summary, the
  machine-readable state JSON, this year's limits, the last 5 years of
  realized returns, its own prior decision + rationale (continuity), and
  the decision JSON schema.
* **Strict schema.** Responses are parsed (fences/prose tolerated), then
  validated against the pydantic `Decision` model with `extra="forbid"` —
  hallucinated fields fail loudly. On failure the agent retries up to 2×
  with the validation error in the prompt, then falls back to "hold prior
  decision" and records the failure (`llm_schema_fallbacks`).
* **Caching & retries.** Responses are disk-cached by
  (model, temperature, prompt-hash) — reruns and resumed runs are free.
  Transport errors back off exponentially. Runs are resumable per trial.

To wire a custom non-HTTP agent, subclass `agents.base.BaseAgent`,
implement `decide(observation) -> Decision`, and register it in
`runner.make_agent_factory`.

## Run artifacts

```
runs/<name>/
├── run.json                 # config + identity
├── trials/seed_00042.json   # per-trial record: metrics + year-by-year logs
└── summary.json             # per-seed metrics (rebuilt by rebuild-summary)
```

Trial logs carry the full decision (with rationale), violations, taxes,
expense breakdowns, shortfalls, and end-of-year balances —
`--log-detail full` keeps everything, `summary` trims to what scoring and
reports need.

## Simplifications (deliberate, documented)

Tax: **federal only** — no state tax, no AMT, no itemized deductions, no
IRMAA, no QBI (extension points, all). CTC is the nonrefundable
simplification; SS taxation thresholds, NIIT/Additional-Medicare
thresholds, and the CTC phase-out are **unindexed, matching real law**;
everything indexed uses realized simulated CPI without the $500-increment
rounding. Payroll-tax treatment of HSA/401(k) contributions is simplified
(income tax only). The SS wage base indexes by CPI (law: wage index).

Accounts: single blended basis in taxable (no lots; internal rebalancing is
tax-free; withdrawals realize gains pro-rata; all gains long-term).
Dividends 2% qualified on equities, 3.5% ordinary on taxable bonds. Roth
5-year clocks and conversion recapture are not modeled (basis-first
ordering is). HSA withdrawals are assumed qualified (receipts pile up in
real life too). 529s: own-beneficiary only, contribution cap at the
gift-exclusion proxy, non-qualified withdrawals tax earnings + 10%.
Rule of 55, 72(t)/SEPP, and the 60–63 "super catch-up" are not modeled;
the trad-IRA higher-education penalty exception **is**.

Mechanics: annual timestep (ages are start-of-year; the 59½ penalty
boundary is age ≥ 59, RMDs at 73, HSA eligibility ends at 65); taxes are
withheld in-year (no April lag); SS PIA COLAs by CPI pre- and post-claim
and the earnings test is ignored; college is billed at sticker and paid
automatically; unpaid mortgage months accrue interest (negative
amortization) rather than defaulting; mortality is deterministic at 95;
no annuities, insurance, home sale/relocation, or Medicare premiums.
Each of these is an isolated extension point — the modules they live in
are noted in the code.

## Testing

`pytest` covers: the tax engine against **hand-computed returns** (wage
couple w/ CTC; retirees with SS provisional income + 0% LTCG stacking; high
earners with conversions, NIIT, Additional Medicare, 15% LTCG; penalties +
loss carryforward; HoH), contribution-limit enforcement and phase-outs, RMD
math, SS claiming factors, early-withdrawal penalty paths incl. the
education exception, backdoor pro-rata, mortgage amortization, generator
seed discipline, LLM parse/retry/fallback/caching, **determinism**
(same scenario+seed+decisions → identical outcome hash), the **baseline
sanity gate** (expert must beat naive on ≥90% of 60 paired seeds — a
failing gate means an engine bug), and an end-to-end mock-LLM smoke test
through the CLI.
