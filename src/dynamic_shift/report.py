"""Pre-registered reporting for one feature family (see README).

Each event is (team_match_key, {feature: signed change}). The family
composite is the Euclidean norm of the changes, each divided by its
standard deviation among random moments. Pass bar: AUC vs random >= 0.60
AND the Bonferroni-corrected CI excludes 0.5.
"""
from __future__ import annotations

import numpy as np

from dynamic_shift.signal_check import auc_ci_by_match, auc_vs_baseline

LEVEL = 1 - 0.05 / 3
PASS_AUC = 0.60


def family_report(name: str, events: dict[str, list[tuple[int, dict]]],
                  features: tuple[str, ...], rng: np.random.Generator,
                  baseline: str = "random", log=print) -> dict:
    def usable(lab):
        return [(k, d) for k, d in events.get(lab, []) if all(f in d for f in features)]

    rand = usable(baseline)
    if not rand:
        log(f"\n== {name}: no usable random-moment events ==")
        return {"family": name, "features": list(features), "classes": {}, "per_feature": {}}
    sigma = {f: float(np.std([d[f] for _, d in rand])) or 1.0 for f in features}

    def composite(d):
        return float(np.sqrt(sum((d[f] / sigma[f]) ** 2 for f in features)))

    keys = {lab: np.array([k for k, _ in usable(lab)]) for lab in events}
    comp = {lab: np.array([composite(d) for _, d in usable(lab)]) for lab in events}

    result = {"family": name, "features": list(features), "classes": {}, "per_feature": {}}
    log(f"\n== {name}: composite change (primary), AUC vs random, {LEVEL:.1%} CI by team-match ==")
    for lab in events:
        if lab == baseline:
            log(f"  {lab:18s} n={len(comp[lab]):6d} mean composite={comp[lab].mean():.3f}")
            continue
        if len(comp[lab]) == 0:
            log(f"  {lab:18s} n=0")
            continue
        auc, lo, hi = auc_ci_by_match(comp[lab], keys[lab], comp[baseline], keys[baseline],
                                      rng, level=LEVEL)
        passed = bool(auc >= PASS_AUC and lo > 0.5)
        result["classes"][lab] = {"n": int(len(comp[lab])), "auc": round(auc, 4),
                                  "ci": [round(lo, 4), round(hi, 4)], "passes_bar": passed}
        log(f"  {lab:18s} n={len(comp[lab]):6d} AUC={auc:.3f} (CI {lo:.3f}-{hi:.3f})"
            + ("   PASSES BAR" if passed else ""))

    log(f"\n-- {name}: per-feature |change| AUC vs random (descriptive only) --")
    for f in features:
        base = np.array([abs(d[f]) / sigma[f] for _, d in rand])
        row = {}
        for lab in events:
            if lab == baseline:
                continue
            v = np.array([abs(d[f]) / sigma[f] for _, d in usable(lab)])
            if len(v):
                row[lab] = round(auc_vs_baseline(v, base), 3)
        result["per_feature"][f] = row
        log(f"  {f:22s} " + "  ".join(f"{k}={v}" for k, v in row.items()))
    return result
