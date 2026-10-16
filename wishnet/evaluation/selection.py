# -*- coding: utf-8 -*-
"""Which weights a run reports: among the candidates, drop those whose F1 is
more than F1_TOL points below the highest F1, then take the one with the
lowest best-F1 MAE (both read at the best-F1 operating point, ``f1op/F1@0.5``
and ``f1op/MAE`` of bdh_metrics.json).

One rule, two users: the training hook that keeps ``model_best_h.pth``
(wishnet/engine/hooks.py) and the run script's choice among model_best,
model_best_h and model_final (tools/select_weights.py).  Only the standard
library: the command-line tool loads this file directly.

A candidate is ``(name, f1, mae)``; the order of the list breaks exact ties.
"""
import math

F1_KEY = "f1op/F1@0.5"
MAE_KEY = "f1op/MAE"
F1_TOL = 1.0          # F1 points (bdh_metrics.json keeps F1 in percent)
_EPS = 1e-9


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def eligible(cands):
    """(highest F1, the candidates within F1_TOL of it); candidates without a
    finite F1 or MAE are never eligible."""
    ok = [c for c in cands if _finite(c[1]) and _finite(c[2])]
    if not ok:
        return None, []
    top = max(c[1] for c in ok)
    return top, [c for c in ok if top - c[1] <= F1_TOL + _EPS]


def _key(cands):
    pos = {id(c): i for i, c in enumerate(cands)}
    return lambda c: (c[2], -c[1], pos[id(c)])       # MAE, then higher F1, then earlier


def choose(cands):
    """The chosen candidate, or None when none is eligible."""
    _, el = eligible(cands)
    return min(el, key=_key(cands)) if el else None


def front(cands):
    """The eligible candidates that a later candidate could still hand the
    choice back to: those not dominated by another eligible one (F1 at least as
    high and MAE at least as low; of exact equals the earlier stays).  The
    highest F1 only rises, so a dominated candidate stays dominated or drops out,
    and the chosen one is always in the front."""
    _, el = eligible(cands)
    k = _key(cands)
    return [c for c in el
            if not any(u is not c and u[1] >= c[1] and u[2] <= c[2] and k(u) < k(c) for u in el)]


def explain(cands, pick):
    """One line: every candidate's F1 / MAE, the F1 floor, who was dropped and
    what was chosen."""
    top, el = eligible(cands)
    parts = ["%s F1 %s, best-F1 MAE %s" % (c[0], _fmt(c[1], 2), _fmt(c[2], 3)) for c in cands]
    if top is None:
        return "; ".join(parts) + "; no candidate has a finite F1 and MAE"
    out = [c[0] for c in cands if c not in el]
    s = "; ".join(parts) + "; F1 floor %.2f (highest %.2f - %g)" % (top - F1_TOL, top, F1_TOL)
    s += "; dropped: %s" % (", ".join(out) if out else "none")
    return s + "; lowest best-F1 MAE among the rest: %s" % pick[0]


def _fmt(v, nd):
    return ("%." + str(nd) + "f") % v if _finite(v) else "n/a"
