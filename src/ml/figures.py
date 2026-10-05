"""
src/ml/figures.py — Evaluation figures for the report, README and UI.

Each figure is rendered twice (light and dark) from one palette so it reads on
either GitHub theme. Series colours are a colour-blind-validated categorical
set; identity is also carried by the legend text, and every number is in the
evaluation report's tables. Training-time only (needs matplotlib).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark":  {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
              "grid": "#2c2c2a", "axis": "#383835",
              "series": ["#3987e5", "#d95926", "#199e70"]},
}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 9})
LINE = 1.8


def _style(ax, t, xlabel, ylabel, title):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], length=0, labelsize=8.5)
    ax.grid(True, color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=t["ink2"])
    ax.set_ylabel(ylabel, color=t["ink2"])
    ax.set_title(title, color=t["ink"], loc="left", fontsize=10.5, fontweight="semibold", pad=10)


def _legend(ax, t, loc, below: bool = False):
    kwargs = ({"loc": "upper left", "bbox_to_anchor": (0, -0.16)} if below else {"loc": loc})
    leg = ax.legend(frameon=False, fontsize=8.5, handlelength=1.6, **kwargs)
    for text in leg.get_texts():
        text.set_color(t["ink2"])


def _fmt(m: dict) -> str:
    return f"{m['value']:.2f} ({m['ci'][0]:.2f}–{m['ci'][1]:.2f})"


def _save(fig, out: Path, t):
    fig.savefig(out, dpi=160, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)


def roc_pr(t, y, models: list[dict], birads: dict | None, out: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.6), facecolor=t["surface"])
    a1.plot([0, 1], [0, 1], color=t["muted"], linewidth=0.8, label="Chance")
    for i, m in enumerate(models):
        fpr, tpr, _ = roc_curve(y, m["p"])
        a1.plot(fpr, tpr, color=t["series"][i], linewidth=LINE,
                label=f"{m['label']}  AUROC {_fmt(m['auroc'])}")
    if birads:
        fpr, tpr, _ = roc_curve(birads["y"], birads["score"])
        a1.plot(fpr, tpr, color=t["series"][len(models)], linewidth=LINE, drawstyle="steps-post",
                label=f"Radiologist BI-RADS  AUROC {_fmt(birads['auroc'])}  (n={len(birads['y'])})")
    _style(a1, t, "False-positive rate (1 − specificity)", "Sensitivity", "ROC curve")
    _legend(a1, t, "", below=True)

    prev = float(np.mean(y))
    a2.axhline(prev, color=t["muted"], linewidth=0.8)
    a2.text(0.99, prev + 0.015, f"prevalence {prev:.2f}", color=t["muted"], ha="right", fontsize=8)
    for i, m in enumerate(models):
        prec, rec, _ = precision_recall_curve(y, m["p"])
        a2.plot(rec, prec, color=t["series"][i], linewidth=LINE,
                label=f"{m['label']}  AUPRC {_fmt(m['auprc'])}")
    _style(a2, t, "Recall (sensitivity)", "Precision (PPV)", "Precision–recall curve")
    a2.set_ylim(0, 1.02)
    _legend(a2, t, "", below=True)
    fig.tight_layout(w_pad=3)
    _save(fig, out, t)


def _binned(y, p, n_bins=8):
    """Quantile bins: mean predicted vs observed, with Wilson 95% intervals."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        k, z = y[m].sum(), 1.96
        phat = k / n
        centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append((p[m].mean(), phat, centre - half, centre + half))
    return np.array(rows)


def calibration(t, y, models: list[dict], out: Path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 6.0), facecolor=t["surface"],
                                 gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    a1.plot([0, 1], [0, 1], color=t["muted"], linewidth=0.8, label="Ideal calibration")
    for i, m in enumerate(models):
        r = _binned(np.asarray(y), np.asarray(m["p"]))
        off = (i - (len(models) - 1) / 2) * 0.008
        a1.errorbar(r[:, 0] + off, r[:, 1], yerr=[r[:, 1] - r[:, 2], r[:, 3] - r[:, 1]],
                    fmt="o-", color=t["series"][i], linewidth=LINE, markersize=5.5, capsize=0,
                    elinewidth=1.0, label=(f"{m['label']}  slope {m['slope']['value']:.2f}, "
                                           f"intercept {m['intercept']['value']:.2f}"))
    _style(a1, t, "", "Observed fraction malignant",
           f"Calibration on the held-out test set (n={len(y)})")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    _legend(a1, t, "upper left")

    a2.hist(models[0]["p"], bins=20, range=(0, 1), color=t["series"][0], alpha=0.85,
            rwidth=0.88)
    _style(a2, t, "Predicted probability of malignancy (calibrated)", "Cases",
           f"Distribution of predictions: {models[0]['label']}")
    fig.tight_layout(h_pad=1.5)
    _save(fig, out, t)


def decision_curve(t, thresholds, curves: list[dict], treat_all, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=t["surface"])
    ax.axhline(0, color=t["axis"], linewidth=1.0, label="Treat none (follow up everyone)")
    ax.plot(thresholds, treat_all, color=t["ink2"], linewidth=1.0, label="Treat all (biopsy everyone)")
    for i, c in enumerate(curves):
        ax.plot(thresholds, c["nb"], color=t["series"][i], linewidth=LINE, label=c["label"])
    ax.axvline(0.02, color=t["muted"], linewidth=0.8)
    ax.text(0.025, 0.02, "2%: BI-RADS 4\nbiopsy threshold", color=t["muted"], fontsize=7.5,
            va="bottom")
    top = max(float(np.max(treat_all)), max(float(np.max(c["nb"])) for c in curves))
    ax.set_ylim(-0.05, top + 0.05)
    ax.set_xlim(0, float(thresholds[-1]))
    _style(ax, t, "Threshold probability for recommending biopsy", "Net benefit",
           "Decision curve (held-out test set)")
    _legend(ax, t, "upper right")
    fig.tight_layout()
    _save(fig, out, t)


def conformal_sets(t, y, p, benign_le: float, malignant_ge: float, coverage: float, out: Path):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(7.2, 3.3), facecolor=t["surface"])
    lo, hi = sorted((benign_le, malignant_ge))
    ax.axvspan(lo, hi, color=t["grid"], alpha=0.7, linewidth=0)
    middle = "both" if malignant_ge <= benign_le else "neither"
    zones = [((0 + lo) / 2, "{Benign}"), ((lo + hi) / 2, middle), ((hi + 1) / 2, "{Malignant}")]
    for x, txt in zones:
        ax.text(x, 1.72, txt, color=t["ink2"], fontsize=8.5, ha="center")
    for x in (lo, hi):
        ax.axvline(x, color=t["series"][0], linewidth=1.2)
        ax.text(x, -0.55, f"{x:.2f}", color=t["ink2"], fontsize=8, ha="center", va="bottom")
    for row, (cls, name) in enumerate([(0, "Benign (true)"), (1, "Malignant (true)")]):
        m = y == cls
        covered = (p[m] <= benign_le) if cls == 0 else (p[m] >= malignant_ge)
        jitter = row + rng.uniform(-0.22, 0.22, m.sum())
        ax.scatter(p[m][covered], jitter[covered], s=16, color=t["ink2"], alpha=0.75,
                   linewidths=0, label="true class inside its set" if row == 0 else None)
        ax.scatter(p[m][~covered], jitter[~covered], s=22, color=t["series"][1],
                   linewidths=0, label="true class outside its set" if row == 0 else None)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign (true)", "Malignant (true)"])
    ax.set_ylim(-0.6, 2.0); ax.set_xlim(0, 1)
    _style(ax, t, "Calibrated probability of malignancy",
           "", f"{coverage:.0%} conformal prediction sets on the test set")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2,
              labelcolor=t["ink2"])
    fig.tight_layout()
    _save(fig, out, t)


def render_all(out_dir: Path, **data) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, t in THEMES.items():
        roc_pr(t, data["y"], data["models"], data["birads"], out_dir / f"roc_pr_{name}.png")
        calibration(t, data["y"], data["models"], out_dir / f"calibration_{name}.png")
        decision_curve(t, data["dca_thresholds"], data["dca_curves"], data["dca_treat_all"],
                       out_dir / f"decision_curve_{name}.png")
        c = data["conformal"]
        conformal_sets(t, data["y"], data["models"][0]["p"], c.benign_if_p_le,
                       c.malignant_if_p_ge, 1 - c.alpha, out_dir / f"conformal_sets_{name}.png")
        written += [f"{n}_{name}.png" for n in ("roc_pr", "calibration", "decision_curve", "conformal_sets")]
    return written
