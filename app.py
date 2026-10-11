import importlib
import sys
from pathlib import Path

import streamlit as st

VIEWS = Path(__file__).resolve().parent / "views"
sys.path.insert(0, str(VIEWS))
# The pages share views/style.py and views/engine.py. A running server keeps the old copies in memory, so reload a
# module when its file changes (style first: engine imports from it). Pages import them themselves, so the light
# style module can draw the first screen before the heavy engine loads.
changed = False
for name in ("style", "engine"):
    module = sys.modules.get(name)
    if module is not None:
        mtime = (VIEWS / f"{name}.py").stat().st_mtime
        if changed or getattr(module, "_mtime", mtime) != mtime:     # engine re-exports style's names: reload it too
            module, changed = importlib.reload(module), True
        module._mtime = mtime

st.set_page_config(page_title="Glucklich", layout="wide")
st.navigation([st.Page("views/home.py", title="Home", default=True),
               st.Page("views/meal.py", title="Meal"),
               st.Page("views/full_day.py", title="Full Day"),
               st.Page("views/log.py", title="Log"),
               st.Page("views/how_it_works.py", title="How it works")], position="top").run()
