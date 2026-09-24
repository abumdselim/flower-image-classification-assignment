"""
make_terminal_screens.py — render run-log excerpts as terminal-window
screenshots (PNG) for the report's Results section.
"""
import json
import os

PROJECT = "/home/z/my-project/cv_assignment"
OUT = os.path.join(PROJECT, "report_assets", "terminal")
CHUNK = 30

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ margin:0; padding:24px; background:#17181c; width:960px;
         font-family:'JetBrains Mono','Fira Code',Consolas,monospace; }}
  .window {{ border-radius:10px; overflow:hidden; box-shadow:0 12px 40px rgba(0,0,0,.55);
            border:1px solid #3a3d45; }}
  .titlebar {{ background:#2b2d33; padding:10px 14px; display:flex;
              align-items:center; gap:8px; }}
  .dot {{ width:12px; height:12px; border-radius:50%; }}
  .red{{background:#ff5f56}} .yellow{{background:#ffbd2e}} .green{{background:#27c93f}}
  .title {{ margin:0 auto; color:#9aa0a6; font-size:12.5px; font-family:sans-serif; }}
  .body {{ background:#101114; padding:16px 20px; }}
  pre {{ margin:0; color:#d6d9de; font-size:13px; line-height:1.5;
        white-space:pre-wrap; }}
</style></head>
<body>
  <div class="window">
    <div class="titlebar">
      <span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span>
      <span class="title">student@cv-lab: ~/FlowerNet $ python main.py</span>
    </div>
    <div class="body"><pre>{text}</pre></div>
  </div>
</body></html>"""


def colorize(line: str) -> str:
    esc = (line.replace("&", "&amp;").replace("<", "&lt;")
           .replace(">", "&gt;"))
    if "Fold" in esc and "val_acc" in esc:
        return f'<span style="color:#7ee787">{esc}</span>'
    if esc.startswith("=") or "FINAL" in esc or "BEST MODEL" in esc:
        return f'<span style="color:#f2cc60;font-weight:bold">{esc}</span>'
    if esc.strip().startswith(("class ", "TOTAL", "Dataset:", "Device:")):
        return f'<span style="color:#79c0ff">{esc}</span>'
    if "mean CV accuracy" in esc or "mean=" in esc:
        return f'<span style="color:#d2a8ff">{esc}</span>'
    return esc


def main():
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(PROJECT, "logs", "main_run.log")
    with open(log_path, errors="ignore") as f:
        lines = [l.rstrip() for l in f]
    # drop empty leading lines
    while lines and not lines[0].strip():
        lines.pop(0)
    index = []
    n_parts = (len(lines) + CHUNK - 1) // CHUNK
    for p in range(n_parts):
        part = lines[p * CHUNK:(p + 1) * CHUNK]
        body = "\n".join(colorize(l) for l in part)
        page = HTML_TEMPLATE.format(text=body)
        tmp_html = os.path.join(OUT, "_tmp.html")
        with open(tmp_html, "w") as f:
            f.write(page)
        png = os.path.join(OUT, f"terminal_p{p + 1:02d}.png")
        shoot(tmp_html, png)
        index.append({"png": png, "first_line": part[0] if part else ""})
        print(f"[shot] {os.path.basename(png)}")
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    try:
        os.remove(os.path.join(OUT, "_tmp.html"))
    except FileNotFoundError:
        pass
    print(f"Done -> {OUT} ({n_parts} screenshots)")


def shoot(html_path: str, png_path: str):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1010, "height": 900},
                        device_scale_factor=2)
        pg.goto(f"file://{html_path}")
        pg.wait_for_timeout(200)
        el = pg.query_selector(".window")
        el.screenshot(path=png_path)
        b.close()


if __name__ == "__main__":
    main()
