"""Build docs/index.html, the project page on GitHub Pages: the site's look (Flexoki, Libron, the glucose line art)
with what Glucklich does, how, and how well. Static only: the real site needs a Python server.

Run:  python docs/build.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "views"))
from style import DARK, LIGHT  # noqa: E402

REPO = "https://github.com/AlmostAPhysicist/HealthHack2026"
KEYS = ["tx", "tx2", "tx3", "ui", "bg", "bg2", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]


def art():
    """One real day of glucose as a line, meals as yellow dots (the same drawing as the site's Home page)."""
    s = pd.read_csv(ROOT / "static" / "art" / "glucose_day.csv", parse_dates=["time"]).set_index("time")
    g = s["glucose"].dropna()
    x = (g.index - g.index[0]).total_seconds() / 86400 * 1000
    lo, hi = g.min() - 10, g.max() + 10
    y = 150 - (g - lo) / (hi - lo) * 130
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    dots = "".join(f"<circle cx='{(t - g.index[0]).total_seconds() / 86400 * 1000:.1f}' cy='158' r='4' fill='var(--yellow)'/>"
                   for t in s.index[s["carbs"] > 0] if g.index[0] <= t <= g.index[-1])
    return pts, dots


def flow(steps):
    return "".join(f"<div class='s {'up' if k % 2 == 0 else 'down'}'><div class='d' style='background:var(--{c})'></div>"
                   f"<div class='t'>{t}</div></div>" for k, (t, c) in enumerate(steps))


def page():
    pts, dots = art()
    var = lambda p: ";".join(f"--{k}:{p[k]}" for k in KEYS)
    steps = flow([("Raw sensor data", "red"), ("Remove noise", "red"), ("Type 1 or not?", "blue"),
                  ("Photo to nutrition (CLIP + USDA)", "cyan"), ("Food and insulin in the blood (OpenAPS)", "orange"),
                  ("Everyone, people like you, you", "blue"), ("Your curve and its band", "magenta")])
    return (ROOT / "docs" / "template.html").read_text(encoding="utf-8") \
        .replace("{{LIGHT}}", var(LIGHT)).replace("{{DARK}}", var(DARK)).replace("{{REPO}}", REPO) \
        .replace("{{PTS}}", pts).replace("{{DOTS}}", dots).replace("{{FLOW}}", steps)


if __name__ == "__main__":
    out = ROOT / "docs" / "index.html"
    out.write_text(page(), encoding="utf-8")
    print("wrote", out)
