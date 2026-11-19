[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Vikkry88/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Vikkry88/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Entry for make.fun's `gta6-trophy-rarity-forecast-01`: forecast, for every
Grand Theft Auto VI PS5 trophy, the percent of Exophase-tracked players who
will have earned it at 2026-11-26T00:00:00Z, as five quantiles per trophy,
scored by pinball loss on the logit scale.

## Method

A trophy's earn rate is earners over tracked players, and both sides move:
tracked players flood in after launch (diluting every rate) while players
already in keep earning (lifting rates, early trophies first). Exophase
refreshes its counts in steps, so the recorded reads repeat the same page
many times and then jump. The forecaster (`main.py`, Python 3.13 standard
library only) therefore works on the distinct page states:

1. **Trend.** Collapse the history into distinct states. Over the last
   30 hours of states, with recency weights (14 h half-life), fit a line
   of logit(rate) against ln(1 + hours since launch) per trophy,
   excluding 0%/100% boundary prints and winsorizing the logit. The
   median moves the last rate along that line to the target with the
   slope damped to 0.7, shrunk toward the last rate by the number of
   states, and capped at 2.2 logits of total move.
2. **Edge decomposition.** Under 5% or over 95% the printed rate is
   mostly rounding, so blend in a decomposition: trend the earners (or,
   near 100%, the non-earners) in log space and divide by a separately
   trended players-tracked forecast. Earners are never forecast to fall.
3. **Constraints.** Story chapters and the final mission are
   non-increasing; 100% completion sits at or below the final mission;
   the platinum sits at or below every other trophy.
4. **Spread.** Logit-scale sigma from 0.40 per square-root day of
   horizon, plus the trend fit's residual, plus the 0.005-point
   print-rounding noise (large for rare trophies), plus a staleness term
   when the page has not refreshed for hours. Floored at 0.35, capped
   at 2.2.
5. **No-trend fallback.** With fewer than 3 distinct states (the
   practice cases: one flat launch-night page), the median expects
   dilution at the next refresh, more the staler the page, and the
   spread floor rises.

Any case that fails internally still gets a valid persistence forecast,
so the program always writes a well-formed output. No network, no
randomness, no wall-clock use: the same input produces the same bytes,
in well under a second per case.

## Validation

Parameters were calibrated locally on synthetic week-one trophy dynamics
(players joining over time, per-trophy lognormal earn delays, stepped
page refreshes, 2-decimal rounding) across 15 scenarios, where the entry
beats both house baselines (persistence and the fitted-curve baseline) in
every scenario, and on a reconstruction of the practice cases' launch-
night refresh (638 to 3,868 tracked players), where it scores well under
the house persistence baseline's loss.

## Run

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` case in the input directory and writes
`<id>.out` forecasts to the output directory.
