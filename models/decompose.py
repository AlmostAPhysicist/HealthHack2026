"""Decompose a prediction over a window into parts that add up.

    prediction = baseline + residue
    residue    = spikes + cooldowns                     (overlay model; regressor: one part per input)
    spikes     = spike 1 + spike 2 + spike 3            (for one chosen meal)
    cooldowns  = cooldown 1 + cooldown 2 + cooldown 3

residue  = the change that events (meals, insulin, exercise, quiet time) cause from the window start on.
baseline = prediction - residue: where glucose would go without those events (trend, pull back to the
           person's usual level, daily rhythm).

Every residue part is linear in the person's coefficients, so its uncertainty is exact (x' cov x),
calibrated like the prediction. The baseline band combines the prediction band and the residue band
as if independent, which is an upper bound.
"""
import numpy as np
import pandas as pd

from common import HORIZON, row_variance
from context import meal_table, warp_features
from exogenous import effect_curves

LOOKBACK = 96                      # events up to 8 h before the window still change glucose inside it


def overlay_terms(model, series, start, idx):
    """{name: V} for the overlay model: residue, spikes, cooldowns over the window steps `idx`."""
    table = meal_table(series)
    inputs = model.event_inputs(series, table)
    warps = warp_features(series, table)
    T, k = len(idx), model.n_features
    out = {n: np.zeros((T, k)) for n in ("residue", "spikes", "cooldowns")}
    for event, (steps, _) in inputs.items():
        for step in steps[(steps >= start - LOOKBACK) & (steps <= idx[-1])]:
            V = model.event_design(series, event, step, (idx - step) * 5.0, table, inputs, warps)
            V0 = model.event_design(series, event, step, [(start - step) * 5.0], table, inputs, warps)
            for name, combo in [("residue", "total"), ("spikes", "all spikes"), ("cooldowns", "all cooldowns")]:
                out[name] += V[combo] - V0[combo]                       # change since the window start
    return out


def regressor_terms(model, series, start, idx):
    """{name: V} for the regressor: residue and one part per input group, as change since the window start."""
    curves = effect_curves(series, model.kernels)
    T, k = len(idx), model.n_features
    out = {"residue": np.zeros((T, k))}
    for j, f in enumerate(model.kernels):
        name = model.group_of(f)
        out.setdefault(name, np.zeros((T, k)))
        d = curves[idx, j] - curves[start, j]
        out[name][:, j] += d
        out["residue"][:, j] += d
    return out


def meal_parts(model, series, step, idx):
    """{name: V} for one meal's spike 1-3, cooldown 1-3 and their sums, over the window steps `idx`."""
    return model.event_design(series, "food", step, (idx - step) * 5.0)


def evaluate(model, post, V, start, idx):
    """mean and calibrated sd of each part. Returns (mean DataFrame, sd DataFrame) indexed by step."""
    coef, cov, _ = post
    kappa = model.kappa[np.clip(idx - start - 1, 0, HORIZON - 1)]
    mean = {n: v @ coef for n, v in V.items()}
    sd = {n: kappa * np.sqrt(row_variance(v, cov)) for n, v in V.items()}
    return pd.DataFrame(mean, index=idx), pd.DataFrame(sd, index=idx)
