"""Look and feel, kept light (no models, no plotting) so a page can draw its first screen at once:
Flexoki colours for the page's theme, shared CSS, and the Glucklich name."""
from pathlib import Path

import streamlit as st

FONT = "Libron, Georgia, serif"
Z80 = 1.2816
# Flexoki. art_ink / art_red colour the Gray's plates: the dark theme's text and red, and for the light theme the
# same colours with their lightness flipped (hue kept).
LIGHT = dict(art_ink="#3C3B31", art_red="#BE3A2E", tx="#100F0F", tx2="#6F6E69", tx3="#B7B5AC", ui="#E6E4D9", bg="#FFFCF0", bg2="#F2F0E5", red="#AF3029",
             orange="#BC5215", yellow="#AD8301", green="#66800B", cyan="#24837B", blue="#205EA6", purple="#5E409D", magenta="#A02F6F")
DARK = dict(art_ink="#CECDC3", art_red="#D14D41", tx="#CECDC3", tx2="#878580", tx3="#575653", ui="#282726", bg="#100F0F", bg2="#1C1B1A", red="#D14D41",
            orange="#DA702C", yellow="#D0A215", green="#879A39", cyan="#3AA99F", blue="#4385BE", purple="#8B7EC8", magenta="#CE5D97")
# What each colour means across the whole site (each Flexoki colour has a light-theme and a dark-theme value above).
ROLES = {
    "glucose": ("red", "glucose (what the sensor measured)"),
    "food": ("yellow", "food and nutrition"),
    "insulin": ("orange", "insulin"),
    "activity": ("purple", "activity and exercise"),
    "prediction": ("magenta", "our prediction"),
    "truth": ("green", "what really happened"),
    "model": ("blue", "the model and its maths"),
    "open": ("cyan", "open data and open-source tools"),
}
TYPE_COLOR = {"healthy": "green", "prediabetes": "yellow", "t2d": "orange", "t1d": "magenta"}
TYPE_NAME = {"healthy": "healthy", "prediabetes": "prediabetic", "t2d": "type 2 diabetes", "t1d": "type 1 diabetes"}


def theme():
    """The page's own theme ("light" or "dark"), which can differ from the browser's setting."""
    t = getattr(st.context, "theme", None)
    return "dark" if t is not None and t.type == "dark" else "light"


def palette():
    """Flexoki for the page's theme: light uses the 600 accents and base-600/300 greys, dark the 400s and base-500/700."""
    return DARK if theme() == "dark" else LIGHT


THEME_JS = r"""
export default function(component) {
  const { data, setTriggerValue } = component;
  window.__glkTrigger = setTriggerValue;
  const now = () => {                       // the theme the page really shows: light or dark background
    const app = document.querySelector('.stApp');
    if (!app) return null;
    const m = getComputedStyle(app).backgroundColor.match(/\d+/g);
    return m && (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) < 128 ? 'dark' : 'light';
  };
  window.__glkPy = data.theme;
  if (!window.__glkWatch) {
    window.__glkWatch = setInterval(() => {
      const t = now();
      if (!t) return;
      document.documentElement.dataset.glk = t;
      if (t !== window.__glkPy && window.__glkAsked !== t) {   // Python drew the other theme: rerun once
        window.__glkAsked = t;
        window.__glkTrigger('theme', t);
      }
    }, 300);
  }
}
"""


def css():
    """Shared styles. Colours follow the theme the page really shows (not the browser's setting): Python picks the
    palette from st.context.theme, and a small watcher swaps it instantly (and reruns once) when the theme is
    switched from the menu, since that alone does not rerun the page."""
    v = lambda pal: ";".join(f"--{k}:{c}" for k, c in pal.items())
    other = "dark" if theme() == "light" else "light"
    roles = " ".join(f".k-{r} {{color: var(--{c});}} .bg-{r} {{background: color-mix(in srgb, var(--{c}) 13%, transparent); "
                     f"border-color: color-mix(in srgb, var(--{c}) 45%, transparent) !important;}}" for r, (c, _) in ROLES.items())
    st.markdown(f"""<style>
      :root {{{v(palette())}}}
      :root[data-glk="light"] {{{v(LIGHT)}}}
      :root[data-glk="dark"] {{{v(DARK)}}}
      .only-{other} {{display: none !important;}}
      :root[data-glk="light"] .only-light, :root[data-glk="dark"] .only-dark {{display: block !important;}}
      :root[data-glk="light"] .only-dark, :root[data-glk="dark"] .only-light {{display: none !important;}}
      .st-key-theme_watch {{display: none;}}
      .block-container {{padding-top: 3.2rem; max-width: 1180px;}}
      .muted {{color: var(--tx2); font-size: .92rem;}}
      .small {{font-size: .82rem; color: var(--tx2);}}
      .c-red {{color: var(--red);}} .c-orange {{color: var(--orange);}} .c-yellow {{color: var(--yellow);}}
      .c-green {{color: var(--green);}} .c-cyan {{color: var(--cyan);}} .c-blue {{color: var(--blue);}}
      .c-purple {{color: var(--purple);}} .c-magenta {{color: var(--magenta);}}
      .step-no {{font-size: .9rem; letter-spacing: .12em; text-transform: uppercase; color: var(--tx2); margin-top: 2.6rem;}}
      .step-title {{font-size: 2.1rem; line-height: 1.15; margin: .1rem 0 .8rem;}}
      .big-number {{font-size: 2.6rem; line-height: 1;}}
      {roles}
      .flow {{display: flex; flex-wrap: wrap; align-items: stretch; gap: .35rem; margin: .8rem 0 1rem;}}
      .flow .box {{flex: 1 1 150px; border-radius: 10px; padding: .55rem .75rem; border: 1px solid var(--ui);}}
      .flow .box b {{display: block; font-size: 1.02rem;}}
      .flow .box span {{font-size: .84rem; color: var(--tx2); line-height: 1.35; display: block; margin-top: .15rem;}}
      .flow .arrow {{align-self: center; color: var(--tx3); font-size: 1.3rem;}}
      .dots {{position: relative; display: flex; margin: .6rem 0 1rem; height: 6.4rem;}}
      .dots::before {{content: ''; position: absolute; left: 4%; right: 4%; top: 50%; border-top: 1.5px solid var(--tx3);}}
      .dots .s {{position: relative; flex: 1; display: flex; justify-content: center;}}
      .dots .d {{position: absolute; top: calc(50% - 6px); width: 12px; height: 12px; border-radius: 50%;}}
      .dots .t {{position: absolute; width: 96%; text-align: center; font-size: .86rem; line-height: 1.25; color: var(--tx2);}}
      .dots .up .t {{bottom: calc(50% + 12px);}} .dots .down .t {{top: calc(50% + 12px);}}
      .ref {{font-size: .72em; vertical-align: super; line-height: 0;}}
      .working {{color: var(--tx2); font-size: 1.15rem; padding: 1rem 0; animation: glk-pulse 1.4s ease-in-out infinite;}}
      @keyframes glk-pulse {{0%, 100% {{opacity: .45;}} 50% {{opacity: 1;}}}}
      .ref a {{text-decoration: none;}}
    </style>""", unsafe_allow_html=True)
    if "_theme_watch" not in st.session_state:
        st.session_state["_theme_watch"] = st.components.v2.component("theme_watch", html="<span></span>", js=THEME_JS)
    st.session_state["_theme_watch"](key="theme_watch", data={"theme": theme()}, on_theme_change=lambda: None, height=0)


SAY_JS = r"""
export default function(component) {
  const { parentElement } = component;
  const b = parentElement.querySelector('button');
  const src = n => new URL('app/static/audio/glucklich.' + n, document.baseURI).href;
  const audio = new Audio();      // a native speaker (Wikimedia Commons); the browser's voice if it can't play
  audio.src = audio.canPlayType('audio/ogg') ? src('ogg') : src('mp3');
  audio.preload = 'auto';
  const speak = () => {
    const u = new SpeechSynthesisUtterance('glücklich');
    u.lang = 'de-DE'; u.rate = 0.85;
    const de = speechSynthesis.getVoices().find(v => v.lang && v.lang.startsWith('de'));
    if (de) u.voice = de;
    speechSynthesis.cancel(); speechSynthesis.speak(u);
  };
  b.onclick = () => { audio.currentTime = 0; audio.play().catch(speak); };
}
"""
SAY_CSS = """
button { background: none; border: none; cursor: pointer; padding: 0; display: inline-flex; align-items: center; gap: .45rem;
         font-family: Libron, Georgia, serif; font-size: 1.05rem; color: var(--st-text-color); opacity: .8; }
button:hover { opacity: 1; }
svg { width: 20px; height: 20px; }
i { font-size: 1.1rem; }
"""
SAY_HTML = """<button title="Hear it (recording: Jeuwre, CC BY-SA 4.0, Wikimedia Commons)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
  stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H2v6h4l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/>
  <path d="M19 5a10 10 0 0 1 0 14"/></svg><span>/ˈɡlʏklɪç/</span> <i>happy</i></button>"""


def brand(size="4.2rem"):
    """Glucklich: "Gluc" and "klich" in two colours, a button that says it, and what it means."""
    st.markdown(f"<div style='font-size:{size};line-height:1;font-weight:700;letter-spacing:-.01em'>"
                "<span class='c-magenta'>Gluc</span><span class='c-blue'>klich</span></div>", unsafe_allow_html=True)
    if "_say" not in st.session_state:
        st.session_state["_say"] = st.components.v2.component("say_glucklich", html=SAY_HTML, css=SAY_CSS, js=SAY_JS)
    st.session_state["_say"](key="say", height=34)
    st.markdown(f"<div style='font-size:calc({size} * .62);line-height:1.1;margin-top:.2rem;font-weight:700;"
                "letter-spacing:.01em'><span class='c-cyan'>Wellness</span> <span class='c-orange'>Team #3</span></div>",
                unsafe_allow_html=True)


def flow(steps):
    """A thin line with a dot per step and one short phrase above or below it (alternating).
    steps: [(phrase, role)], role from ROLES (colours the dot)."""
    html = "".join(f"<div class='s {'up' if k % 2 == 0 else 'down'}'><div class='d' style='background:var(--{ROLES[role][0]})'></div>"
                   f"<div class='t'>{phrase}</div></div>" for k, (phrase, role) in enumerate(steps))
    st.markdown(f"<div class='dots'>{html}</div>", unsafe_allow_html=True)


def legend():
    """The site's colour key."""
    st.markdown("<div class='flow'>" + "".join(f"<div class='box bg-{r}' style='flex:0 1 auto'><b class='k-{r}' "
                f"style='font-size:.9rem'>{label}</b></div>" for r, (_, label) in ROLES.items()) + "</div>",
                unsafe_allow_html=True)


def photo_file(path):
    """A logged meal photo: the original if this machine has the dataset, else its small copy in static/photo_memory/
    (shipped with the code). None if neither is there."""
    if not path:
        return None
    p = Path(str(path).replace("\\", "/"))
    if p.exists():
        return str(p)
    small = Path(__file__).resolve().parent.parent / "static" / "photo_memory" / p.parent.parent.name / p.name
    return str(small) if small.exists() else None
