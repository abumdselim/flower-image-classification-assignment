"""
make_code_screens.py — render the project's source code as styled editor
screenshots (PNG) for the report's Implementation section.

Each file is syntax-highlighted with Pygments, wrapped in a VS-Code-like
window (tab bar + traffic lights) and captured with Playwright at 2x scale.
Long files are split into chunks of CHUNK lines so the text stays readable.
"""
import json
import os
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

PROJECT = "/home/z/my-project/cv_assignment"
OUT = os.path.join(PROJECT, "report_assets", "code")
CHUNK = 52

FILES = [
    ("config.py", "Global configuration — paths and hyper-parameters"),
    ("data.py", "Dataset loading + data-augmentation transforms"),
    ("models.py", "The five models (SimpleCNN + 4 frozen backbones + heads)"),
    ("prepare_dataset.py", "Dataset preparation (resize / crop / clean)"),
    ("extract_features.py", "Frozen-backbone feature extraction with augmentation"),
    ("train_cv.py", "5-fold stratified cross-validation training engine"),
    ("evaluate.py", "Evaluation metrics, figures and comparison table"),
    ("main.py", "One-command experiment driver"),
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ margin:0; padding:24px; background:#1e1e2e; width:940px;
         font-family: 'JetBrains Mono','Fira Code',Consolas,monospace; }}
  .window {{ border-radius:10px; overflow:hidden; box-shadow:0 12px 40px rgba(0,0,0,.5);
            border:1px solid #3b3b4f; }}
  .titlebar {{ background:#313244; padding:10px 14px; display:flex;
              align-items:center; gap:8px; }}
  .dot {{ width:12px; height:12px; border-radius:50%; }}
  .red{{background:#f38ba8}} .yellow{{background:#f9e2af}} .green{{background:#a6e3a1}}
  .tab {{ margin-left:14px; background:#45475a; color:#cdd6f4; font-size:13px;
         padding:5px 14px; border-radius:7px 7px 0 0; font-family:sans-serif; }}
  .body {{ background:#1e1e2e; padding:14px 18px; }}
  .code {{ font-size:13.5px; line-height:1.5; white-space:pre; }}
  {css}
</style></head>
<body>
  <div class="window">
    <div class="titlebar">
      <span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span>
      <span class="tab">{fname}</span>
    </div>
    <div class="body"><div class="code">{code}</div></div>
  </div>
</body></html>"""


def chunk_lines(code_html: str):
    """Split highlighted html by rendered lines (span-safe: pygments emits <pre>)."""
    return code_html.split("\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    formatter = HtmlFormatter(style="catppuccin-mocha", nowrap=True)
    index = []
    for fname, caption in FILES:
        path = os.path.join(PROJECT, fname)
        with open(path) as f:
            src = f.read()
        try:
            html = highlight(src, PythonLexer(), formatter)
        except Exception:
            html = src
        lines = chunk_lines(html)
        n_parts = (len(lines) + CHUNK - 1) // CHUNK
        for p in range(n_parts):
            part_lines = lines[p * CHUNK:(p + 1) * CHUNK]
            body = "\n".join(part_lines)
            css = HtmlFormatter(style="catppuccin-mocha").get_style_defs(".code")
            page = HTML_TEMPLATE.format(fname=fname, code=body, css=css)
            tmp_html = os.path.join(OUT, "_tmp.html")
            with open(tmp_html, "w") as f:
                f.write(page)
            png = os.path.join(OUT, f"{os.path.splitext(fname)[0]}_p{p + 1:02d}.png")
            shoot(tmp_html, png)
            index.append({"file": fname, "part": p + 1, "parts": n_parts,
                          "png": png, "caption": caption})
            print(f"[shot] {os.path.basename(png)}  ({len(part_lines)} lines)")
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    os.remove(os.path.join(OUT, "_tmp.html"))
    print(f"Done -> {OUT} ({len(index)} screenshots)")


def shoot(html_path: str, png_path: str):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 990, "height": 800},
                        device_scale_factor=2)
        pg.goto(f"file://{html_path}")
        pg.wait_for_timeout(250)
        el = pg.query_selector(".window")
        el.screenshot(path=png_path)
        b.close()


if __name__ == "__main__":
    main()
