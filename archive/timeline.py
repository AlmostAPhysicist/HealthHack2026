"""Timeline component: one line over the patient's data, two draggable dots that set the prediction window.

Line colour where there is glucose data: red before the window (the model may use it), green inside the
window (truth to compare against), blue after it. Grey where there is no data. Meals are bubbles on the
line, larger inside the window. Clicking a meal sends it back to Python (what that does depends on the page).
Dates under the line, weekday under the date. Clock times only when the range is about a day.
"""
import pandas as pd
import streamlit as st

NAME = "glucose_timeline"
CSS = """
.tl { width: 100%; user-select: none; touch-action: none; font-family: Libron, Georgia, serif; }
.tl svg { width: 100%; height: 92px; display: block; overflow: visible; }
.tl .handle { cursor: ew-resize; }
.tl .meal { cursor: pointer; }
"""

JS = r"""
export default function(component) {
  const { data, setStateValue, setTriggerValue, parentElement } = component;
  const host = parentElement.querySelector('.tl');
  const d = data, W = 1000, PAD = 16, Y = 30;
  let [start, end] = d.window;
  const x = t => PAD + (t - d.lo) / (d.hi - d.lo) * (W - 2 * PAD);
  const tAt = px => d.lo + (px - PAD) / (W - 2 * PAD) * (d.hi - d.lo);
  const snap = t => Math.round(t / 900000) * 900000;                     // 15 min
  const two = n => String(n).padStart(2, '0');
  const fmt = t => { const z = new Date(t); return ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][z.getUTCDay()] + ' '
                     + two(z.getUTCHours()) + ':' + two(z.getUTCMinutes()); };

  function pieces(a, b) {
    return [[a, Math.min(b, start), d.colors.before], [Math.max(a, start), Math.min(b, end), d.colors.inside],
            [Math.max(a, end), b, d.colors.after]].filter(([p, q]) => q > p);
  }

  function draw() {
    let s = `<svg viewBox="0 0 ${W} 92" preserveAspectRatio="none">`;
    s += `<rect x="${x(start)}" y="${Y - 16}" width="${Math.max(x(end) - x(start), 1)}" height="32" fill="${d.colors.band}" opacity="0.45" rx="4"/>`;
    s += `<line x1="${PAD}" x2="${W - PAD}" y1="${Y}" y2="${Y}" stroke="${d.colors.none}" stroke-width="3" stroke-linecap="round"/>`;
    for (const [a, b] of d.runs) for (const [p, q, c] of pieces(a, b))
      s += `<line x1="${x(p)}" x2="${x(q)}" y1="${Y}" y2="${Y}" stroke="${c}" stroke-width="5" stroke-linecap="round"/>`;
    d.ticks.forEach((t, i) => {
      s += `<line x1="${x(t)}" x2="${x(t)}" y1="${Y + 8}" y2="${Y + 14}" stroke="${d.colors.muted}" stroke-width="1"/>`;
      s += `<text x="${x(t)}" y="${Y + 30}" text-anchor="middle" fill="${d.colors.text}" font-size="12">${d.tick_top[i]}</text>`;
      s += `<text x="${x(t)}" y="${Y + 46}" text-anchor="middle" fill="${d.colors.muted}" font-size="11">${d.tick_bottom[i]}</text>`;
    });
    for (const m of d.meals) {
      const inside = m.t >= start && m.t <= end, sel = m.t === d.selected;
      s += `<circle class="meal" data-t="${m.t}" cx="${x(m.t)}" cy="${Y}" r="${sel ? 9 : inside ? 7 : 4.5}" fill="${d.colors.meal}"
              opacity="${inside || sel ? 1 : 0.45}" stroke="${sel ? d.colors.text : d.colors.bg}" stroke-width="2"><title>${m.label}</title></circle>`;
    }
    for (const [name, t] of [['start', start], ['end', end]]) {
      s += `<circle class="handle" data-h="${name}" cx="${x(t)}" cy="${Y}" r="9" fill="${d.colors.bg}" stroke="${d.colors.handle}" stroke-width="3"/>`;
      s += `<text x="${x(t)}" y="${Y - 20}" text-anchor="middle" fill="${d.colors.text}" font-size="12">${fmt(t)}</text>`;
    }
    host.innerHTML = s + '</svg>';
    bind();
  }

  function bind() {
    host.querySelectorAll('.handle').forEach(h => {
      h.onpointerdown = ev => {
        ev.preventDefault();
        const which = h.getAttribute('data-h');
        const move = e => {
          const r = host.querySelector('svg').getBoundingClientRect();     // the current drawing, not the one we started on
          const t = Math.max(d.lo, Math.min(d.hi, snap(tAt((e.clientX - r.left) / r.width * W))));
          if (which === 'start') start = Math.min(t, end - 900000); else end = Math.max(t, start + 900000);
          draw();
        };
        const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
                           setStateValue('window', [start, end]); };
        window.addEventListener('pointermove', move);
        window.addEventListener('pointerup', up);
      };
    });
    host.querySelectorAll('.meal').forEach(c => { c.onclick = () => setTriggerValue('meal', Number(c.getAttribute('data-t'))); });
  }
  draw();
}
"""


def _renderer():
    """Register on first use in each Streamlit runtime (also correct after hot reloads)."""
    if "_tl_renderer" not in st.session_state:
        st.session_state["_tl_renderer"] = st.components.v2.component(NAME, html="<div class='tl'></div>", css=CSS, js=JS)
    return st.session_state["_tl_renderer"]


def ms(t):
    return int(pd.Timestamp(t).value // 1_000_000)


def timeline(key, lo, hi, window, runs, meals, colors, selected=None):
    """lo, hi, window: Timestamps. runs = [(a, b)] where glucose exists. meals = [(time, label)].
    Returns (new window if the dots were moved, else None; clicked meal time, else None)."""
    span = (hi - lo) / pd.Timedelta(days=1)
    if span <= 1.5:
        ticks = pd.date_range(lo.ceil("3h"), hi, freq="3h")
        top, bottom = [t.strftime("%H:%M") for t in ticks], [t.strftime("%d %b") for t in ticks]
    else:
        ticks = pd.date_range(lo.ceil("D"), hi, freq="D" if span <= 21 else "7D")
        top, bottom = [t.strftime("%d %b") for t in ticks], [t.strftime("%a") for t in ticks]
    data = {"lo": ms(lo), "hi": ms(hi), "window": [ms(window[0]), ms(window[1])],
            "runs": [[ms(a), ms(b)] for a, b in runs], "meals": [{"t": ms(t), "label": l} for t, l in meals],
            "ticks": [ms(t) for t in ticks], "tick_top": top, "tick_bottom": bottom,
            "selected": ms(selected) if selected is not None else None, "colors": colors}
    res = _renderer()(key=key, data=data, default={"window": data["window"]}, on_window_change=lambda: None,
                      on_meal_change=lambda: None, height=96)
    moved = None
    seen = st.session_state.get(f"{key}_seen")
    if res.window is not None and list(res.window) != seen:
        st.session_state[f"{key}_seen"] = list(res.window)
        # a real drag: new since last time, and not just the window the app handed the timeline
        if seen is not None and list(res.window) != data["window"]:
            moved = tuple(pd.Timestamp(v, unit="ms") for v in res.window)
    clicked = pd.Timestamp(res.meal, unit="ms") if res.meal is not None else None
    return moved, clicked
