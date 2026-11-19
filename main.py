"""Forecast GTA VI PS5 trophy earn rates on Exophase, one week after launch.

Method (calibrated on synthetic week-one dynamics and on the practice
cases' launch-night refresh, both rebuilt locally from the kit):

A trophy's earn rate is earners over tracked players, and both sides move.
Tracked players flood in after launch (diluting every rate), while the
players already in keep earning (lifting rates, early trophies first).
Exophase refreshes its counts in steps, so the recorded reads repeat the
same page many times and then jump. The forecaster therefore works on the
distinct page states, not the raw reads:

1. Collapse the history into distinct states (players_tracked + rates).
   Fit, over the last 30 hours of states with recency weights (14 h
   half-life), a line of logit(rate) against ln(1 + hours since launch)
   per trophy, excluding the 0%/100% boundary prints and winsorizing the
   logit. The median is the last rate moved along that line to the
   target with the slope damped to 0.7, shrunk toward the last rate by
   the number of states, and capped at 2.2 logits of total move.
2. At the edges (rate under 5% or over 95%) the printed rate is mostly
   rounding, so blend in a decomposition: trend the earners (or, near
   100%, the non-earners) in log space and divide by a separately
   trended players_tracked forecast. Earners are never forecast to fall.
3. Logical constraints on the medians: story chapters and the final
   mission are non-increasing, 100% completion sits at or below the
   final mission, and the platinum at or below every other trophy.
4. Spread on the logit scale: 0.40 per square-root day of horizon, plus
   the trend fit's residual, plus the 0.005-point print-rounding noise
   (large for rare trophies), plus a staleness term when the page has
   not refreshed for hours. Floored at 0.35, capped at 2.2.
5. With fewer than 3 distinct states there is no trend to fit (the
   practice cases: one flat launch-night page). The median then expects
   dilution: tracked players keep arriving while a stale page hides
   them, so rates are shifted down with staleness and the spread floor
   rises. That is the launch-night refresh the practice answers show.

Any case that fails internally still gets a valid persistence forecast,
so the program always writes a well-formed output file.

Standard library only, no network, no randomness, no wall-clock use:
the same input always produces the same bytes.

Input:  $MAKEFUN_INPUT_DIR (else /input)  / <id>.in, JSON schema
        "makefun/gta6-trophies-case/v1".
Output: $MAKEFUN_OUTPUT_DIR (else /output) / <id>.out, JSON
        {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}
        quantiles in percent, rounded to 4 decimals, half up.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file would import. (This entry is a
# single self-contained file, but the line is kept for compatibility.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]  # standard normal quantiles at LEVELS
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)  # launch day, 00:00 UTC

WINDOW_H = 30.0          # fit window: states at or after checkpoint - WINDOW_H
HALF_LIFE_H = 14.0       # recency weighting half-life inside the window
DAMP = 0.7               # slope damping on the direct logit trend
W_DECOMP_EDGE = 0.3      # decomposition weight at the edges (rate <5% or >95%)
EDGE_PCT = 5.0
EARN_SLOPE_CAP = 2.2     # cap on the ln(earners) slope against log-time
SHRINK_K = 1.5           # shrinkage to last rate: w = n_states / (n_states + K)
MOVE_CAP = 2.2           # max |logit move| from the last rate (at sqrt(days) >= 2)
SRW = 0.40               # random-walk sigma per square-root day
RESID_W = 0.7            # weight on the trend fit's residual in the spread
SIGMA_FLOOR = 0.35
SIGMA_FLOOR_NOTREND = 0.85
SIGMA_CAP = 2.2
MEDIAN_LOGIT_CAP = 8.0
FIT_LOGIT_CAP = 7.5      # winsorize logits inside the trend fit


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l: float) -> float:
    if l > 30.0:
        return 100.0
    if l < -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float):
    """x rounded to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def hours_of(iso: str) -> float:
    return (datetime.fromisoformat(iso.replace("Z", "+00:00")) - EPOCH).total_seconds() / 3600.0


def x_of(h: float) -> float:
    return math.log(1.0 + max(h, -0.9))


def wls(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, weighted RMS residual) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    b = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sxx
    a = my - b * mx
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw)
    return (a, b, resid)


def distinct_states(history):
    """Collapse consecutive reads with identical (players, rates) into one state."""
    states = []
    for h in history:
        key = (h["players_tracked"], tuple(-1.0 if r is None else r for r in h["rates"]))
        if states and states[-1]["_key"] == key:
            states[-1]["last_slot"] = h.get("slot") or h.get("at")
            continue
        states.append({
            "slot": h.get("slot") or h.get("at"),
            "last_slot": h.get("slot") or h.get("at"),
            "players_tracked": h["players_tracked"],
            "rates": list(h["rates"]),
            "_key": key,
        })
    return states


def apply_constraints(trophies, medians):
    """Enforce logical orderings on median logits (story chain, platinum, 100%)."""
    med = list(medians)
    chain = []
    for i, t in enumerate(trophies):
        d = t.get("description", "")
        if d.startswith("Complete Chapter "):
            try:
                chain.append((int(d.split()[2].rstrip(".")), i))
            except Exception:
                pass
    chain.sort()
    idxs = [i for _, i in chain]
    for i, t in enumerate(trophies):
        if t.get("description", "").strip() == "Complete the final mission.":
            idxs.append(i)
    for k in range(1, len(idxs)):
        if med[idxs[k]] > med[idxs[k - 1]]:
            med[idxs[k]] = med[idxs[k - 1]]
    plat = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    if plat and len(med) > 1:
        others_min = min(m for j, m in enumerate(med) if j != plat[0])
        if med[plat[0]] > others_min:
            med[plat[0]] = others_min
    for i, t in enumerate(trophies):
        if "100% Completion" in t.get("description", "") and idxs:
            if med[i] > med[idxs[-1]]:
                med[i] = med[idxs[-1]]
    return med


def last_known_rates(case):
    """Each trophy's rate in the last read, scanning back past nulls; tier prior if never seen."""
    n = len(case["trophies"])
    out = [None] * n
    for h in reversed(case["history"]):
        for i in range(n):
            if out[i] is None and h["rates"][i] is not None:
                out[i] = h["rates"][i]
        if all(v is not None for v in out):
            break
    prior = {"platinum": 0.1, "gold": 0.5, "silver": 1.0, "bronze": 2.0}
    return [out[i] if out[i] is not None else prior.get(case["trophies"][i].get("tier", "bronze"), 1.0)
            for i in range(n)]


def persistence_forecast(case):
    """House-style fallback: last rate stays, random-walk spread. Always valid."""
    rates = last_known_rates(case)
    spread = 0.5 * math.sqrt(max(case["horizon_minutes"], 0) / 1440.0)
    rows = []
    for i, t in enumerate(case["trophies"]):
        lg = logit(rates[i])
        q = [round4(inv_logit(lg + z * spread)) for z in Z]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    days = case["horizon_minutes"] / 1440.0
    at_h = hours_of(case["at"])
    tgt_h = hours_of(case["target"])
    x_tgt = x_of(tgt_h)
    states = distinct_states(history)
    n_states = len(states)
    last_rates = last_known_rates(case)
    move_cap = MOVE_CAP * max(0.35, min(1.0, math.sqrt(days) / 2.0)) if days > 0 else MOVE_CAP

    def w_of(slot):
        age = max(0.0, at_h - hours_of(slot))
        return 0.5 ** (age / HALF_LIFE_H)

    win = [s for s in states if hours_of(s["slot"]) >= at_h - WINDOW_H]
    if len(win) < 3:
        win = states[-min(len(states), 6):]

    # Players forecast: windowed ln(players) against log-time, never falling.
    p_last = float(history[-1]["players_tracked"])
    p_tgt = p_last
    if len(win) >= 3:
        xs = [x_of(hours_of(s["slot"])) for s in win]
        ys = [math.log(max(s["players_tracked"], 1)) for s in win]
        ws = [w_of(s["slot"]) for s in win]
        fit = wls(xs, ys, ws)
        if fit:
            a, b, _ = fit
            b = max(b, 0.0)
            p_tgt = min(max(math.exp(a + b * x_tgt), p_last), p_last * 8.0)

    stale_h = max(0.0, at_h - hours_of(states[-1]["slot"])) if states else 0.0
    stale_sigma = 0.25 if stale_h > 6 else (0.12 if stale_h > 3 else 0.0)
    shrink = len(win) / (len(win) + SHRINK_K) if len(win) >= 3 else 0.0

    medians, sigmas = [], []
    for i, t in enumerate(trophies):
        r_last = last_rates[i]
        l_last = logit(r_last)
        xs, lgs, les, nes, ws = [], [], [], [], []
        for s in win:
            r = s["rates"][i]
            if r is None:
                continue
            xs.append(x_of(hours_of(s["slot"])))
            ws.append(w_of(s["slot"]))
            les.append(math.log(r / 100.0 * s["players_tracked"] + 0.5))
            nes.append(math.log((1.0 - r / 100.0) * s["players_tracked"] + 0.5))
            lgs.append(logit(r) if 0.0 < r < 100.0 else None)
        resid = 0.0
        pred = l_last
        if len(win) >= 3 and len(xs) >= 3:
            edge = r_last < EDGE_PCT or r_last > 100.0 - EDGE_PCT
            cands = []
            # Direct logit trend, anchored at the last observed rate.
            ix = [x for x, l in zip(xs, lgs) if l is not None]
            iy = [max(-FIT_LOGIT_CAP, min(FIT_LOGIT_CAP, l)) for l in lgs if l is not None]
            iw = [w for w, l in zip(ws, lgs) if l is not None]
            if len(ix) >= 3:
                fit_d = wls(ix, iy, iw)
                if fit_d:
                    a, b, resid = fit_d
                    pred_d = l_last + DAMP * b * (x_tgt - ix[-1])
                    cands.append((1.0 - (W_DECOMP_EDGE if edge else 0.0), pred_d))
            # Decomposition at the edges: earners (rare) or non-earners
            # (near 100%) trended in log space over the players forecast.
            if edge:
                series = les if r_last <= 50.0 else nes
                fit_e = wls(xs, series, ws)
                if fit_e:
                    a, b, _ = fit_e
                    b = max(-1.0, min(EARN_SLOPE_CAP, b))
                    c_tgt = math.exp(series[-1] + b * (x_tgt - xs[-1])) - 0.5
                    c_last = (r_last if r_last <= 50.0 else 100.0 - r_last) / 100.0 * p_last
                    c_tgt = min(max(c_tgt, c_last * 0.995, 0.0), p_tgt)
                    rate_dec = 100.0 * c_tgt / p_tgt if r_last <= 50.0 else 100.0 * (1.0 - c_tgt / p_tgt)
                    cands.append((W_DECOMP_EDGE, logit(rate_dec) if p_tgt > 0 else l_last))
            tw = sum(w for w, _ in cands)
            if cands and tw > 0:
                pred = sum(w * v for w, v in cands) / tw
                pred = l_last + shrink * (pred - l_last)
        elif n_states < 3:
            # No trend signal: a stale page hides arriving players, so
            # expect dilution at the next refresh, more the staler it is.
            g = ((1.0 + max(tgt_h, 0.0)) / (1.0 + max(at_h, 0.0))) ** 1.2
            shift = -(0.55 * math.log(max(g, 1.0)) + 0.30 * min(stale_h, 8.0) ** 0.8)
            pred = l_last + max(shift, -1.8)
        pred = l_last + max(-move_cap, min(move_cap, pred - l_last))
        pred = max(-MEDIAN_LOGIT_CAP, min(MEDIAN_LOGIT_CAP, pred))
        medians.append(pred)
        p = clamp(r_last) / 100.0
        round_sigma = min(0.00005 / max(p * (1.0 - p), 1e-6), 0.8)
        floor = (SIGMA_FLOOR_NOTREND + 0.05 * min(stale_h, 8.0)) if n_states < 3 else SIGMA_FLOOR
        s = math.sqrt((SRW * math.sqrt(days)) ** 2 + (RESID_W * resid) ** 2
                      + round_sigma ** 2 + stale_sigma ** 2)
        sigmas.append(min(max(s, floor), SIGMA_CAP))

    medians = apply_constraints(trophies, medians)
    rows = []
    for i, t in enumerate(trophies):
        q = [round4(inv_logit(medians[i] + z * sigmas[i])) for z in Z]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = persistence_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
