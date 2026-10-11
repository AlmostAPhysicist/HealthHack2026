"""Assemble the in-browser site for GitHub Pages: the real Streamlit app, run by stlite (Streamlit + Python on
WebAssembly) in the visitor's browser. Nothing runs on a server.

    <out>/app/index.html     loads stlite and mounts the app
    <out>/app/site.zip       the app's code, slim weights, samples and data, unpacked into the browser's file system
    <out>/app/app/static/    the static files the pages link to (fonts, audio, sample zips), at the same relative
                             URL ("app/static/...") the Streamlit server uses

Run:  python docs/build_app.py <out>      (the Pages workflow runs it with <out> = _site)
"""
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STLITE = "1.11.0"
REQUIREMENTS = ["plotly", "scikit-learn", "scipy", "pillow", "openpyxl", "xlrd", "toml"]

# (path in the repo, path in the browser) for everything the pages read from disk
CODE = ["app.py", ".streamlit/config.toml", *[p.relative_to(ROOT).as_posix() for p in (ROOT / "views").glob("*.py")],
        *[p.relative_to(ROOT).as_posix() for p in (ROOT / "models").glob("*.py")]]
DATA = ["models/weights/classifier.pkl", "data/fndds_foods.csv", "data/cache/fndds_text_L14.npy",
        "data/cache/photo_memory_L14.npz", "data/cache/photo_settings.json",
        "results/explain.json", "results/scores.csv", "results/test_errors.csv", "results/tuning.csv",
        "results/photo2nutrition.csv", "results/photo2nutrition_scores.csv"]
FOLDERS = ["samples", "static/art", "static/hiw", "static/try_me", "static/photo_memory"]


def files():
    out = [(p, p) for p in CODE + DATA if (ROOT / p).exists()]
    out.append(("models/weights/models_web.pkl", "models/weights/models.pkl"))
    for folder in FOLDERS:
        out += [(p.relative_to(ROOT).as_posix(),) * 2 for p in sorted((ROOT / folder).rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts]
    return out


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<title>Glucklich</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{stlite}/build/stlite.css">
<style>
  :root {{--bg:#FFFCF0;--tx:#100F0F;--tx2:#6F6E69;--magenta:#A02F6F;--blue:#205EA6;}}
  @media (prefers-color-scheme: dark) {{ :root {{--bg:#100F0F;--tx:#CECDC3;--tx2:#878580;--magenta:#CE5D97;--blue:#4385BE;}} }}
  body {{margin:0; background:var(--bg); color:var(--tx);}}
  #boot {{font-family: Georgia, serif; max-width: 34rem; margin: 18vh auto 0; padding: 0 16px; text-align: center;}}
  #boot b {{font-size: 3rem;}}
  #boot p {{color: var(--tx2); font-size: 1.1rem;}}
</style>
</head>
<body>
<div id="boot"><b><span style="color:var(--magenta)">Gluc</span><span style="color:var(--blue)">klich</span></b>
  <p>Loading the site into your browser: Python, the models and the demo people (about 60 MB, once).
  Everything runs on your device.</p></div>
<div id="root"></div>
<script type="module">
import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{stlite}/build/stlite.js";
const controller = mount({{
  entrypoint: "app.py",
  requirements: {requirements},
  archives: [{{ url: "./site.zip", format: "zip", options: {{}} }}],
  streamlitConfig: {{ "client.toolbarMode": "viewer" }},
}}, document.getElementById("root"));
const boot = document.getElementById("boot");
new MutationObserver((_, obs) => {{
  if (document.querySelector("[data-testid='stApp'] [data-testid='stMarkdown']")) {{ boot.remove(); obs.disconnect(); }}
}}).observe(document.getElementById("root"), {{ childList: true, subtree: true }});
</script>
</body>
</html>
"""


def main(out):
    app = Path(out) / "app"
    if app.exists():
        shutil.rmtree(app)
    app.mkdir(parents=True)
    with zipfile.ZipFile(app / "site.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, dst in files():
            z.write(ROOT / src, dst)
    shutil.copytree(ROOT / "static", app / "app" / "static")
    import json
    (app / "index.html").write_text(PAGE.format(stlite=STLITE, requirements=json.dumps(REQUIREMENTS)), encoding="utf-8")
    print(f"site.zip {(app / 'site.zip').stat().st_size / 1e6:.1f} MB, {len(files())} files -> {app}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "_site")
