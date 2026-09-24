#!/usr/bin/env python3
"""
Collect a custom image dataset using the z-ai image-search CLI.
CORRECTED SPEC: 3 classes x 100 images = 300 images total.
3 classes x 6 queries x 20 results -> download -> validate (PIL) -> dedupe -> save raw.
"""
import json
import os
import subprocess
import sys
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image

import urllib.request

PROJECT = "/home/z/my-project/cv_assignment"
RAW = os.path.join(PROJECT, "dataset_raw")
LOG = os.path.join(PROJECT, "logs", "collect.log")

CLASSES = {
    "rose": [
        "red rose flower closeup photograph",
        "rose flowers in a beautiful garden",
        "pink rose flower macro photography",
        "bouquet of fresh roses",
        "yellow rose flower in garden",
        "single rose on white background",
    ],
    "sunflower": [
        "bright yellow sunflower closeup photograph",
        "sunflower field full of sunflowers",
        "sunflower head macro detail",
        "single sunflower against blue sky",
        "sunflowers in a farm field summer",
        "sunflower bouquet closeup",
    ],
    "daisy": [
        "white daisy flower closeup photograph",
        "daisies blooming in a green meadow",
        "oxeye daisy flower macro",
        "daisy flower with water drops",
        "wild daisies in grass field",
        "single white daisy on green background",
    ],
}

MIN_DIM = 160
TARGET_PER_CLASS = 100


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}][collect] {msg}"
    print(line, flush=True)


def run_search(query, out_json):
    """Run z-ai image-search and return list of results (JSON comes on stdout)."""
    for attempt in range(2):
        try:
            cmd = ["z-ai", "image-search", "-q", query, "--count", "20",
                   "--gl", "us", "--no-rank"]
            proc = subprocess.run(cmd, capture_output=True, timeout=150, text=True)
            out = proc.stdout
            start = out.find("{")
            if start == -1:
                log(f"no JSON in stdout (rc={proc.returncode}): {out[:200]}")
                continue
            data = json.loads(out[start:])
            if data.get("success") and data.get("results"):
                return data["results"]
            log(f"query failed (success={data.get('success')}): {query} "
                f"err={data.get('error','')}")
        except Exception as e:
            log(f"query error attempt {attempt+1}: {query}: {e}")
    return []


def fetch_blob(url):
    """Download one URL, return raw bytes or None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()
    except Exception:
        return None


def download_and_save(url, save_path):
    """Download image, validate, save as JPEG. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
        img = Image.open(BytesIO(data))
        img = img.convert("RGB")
        w, h = img.size
        if w < MIN_DIM or h < MIN_DIM:
            return False
        img.save(save_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def main():
    os.makedirs(RAW, exist_ok=True)
    summary = {}
    for cls, queries in CLASSES.items():
        cls_dir = os.path.join(RAW, cls)
        os.makedirs(cls_dir, exist_ok=True)
        seen_hashes = {f for f in os.listdir(cls_dir)} if os.path.isdir(cls_dir) else set()
        seen_content = set()
        # hash existing files for dedupe across restarts
        for f in os.listdir(cls_dir):
            with open(os.path.join(cls_dir, f), "rb") as fh:
                seen_content.add(hashlib.md5(fh.read()).hexdigest())
        saved = len(seen_content)
        log(f"=== class {cls}: already have {saved}, target {TARGET_PER_CLASS} ===")
        for qi, query in enumerate(queries):
            if saved >= TARGET_PER_CLASS:
                break
            log(f"query {qi+1}/{len(queries)}: {query}")
            tmp_json = os.path.join(RAW, f"_{cls}_{qi}.json")
            results = run_search(query, tmp_json)
            log(f"  got {len(results)} results")
            urls = [r.get("original_url") for r in results if r.get("original_url")]
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(fetch_blob, u): u for u in urls}
                idx = 10 ** 6  # high start to avoid filename clashes
                for fut in as_completed(futures):
                    if saved >= TARGET_PER_CLASS:
                        break
                    blob = fut.result()
                    if not blob:
                        continue
                    h = hashlib.md5(blob).hexdigest()
                    if h in seen_content:
                        continue
                    try:
                        img = Image.open(BytesIO(blob)).convert("RGB")
                    except Exception:
                        continue
                    w, hh = img.size
                    if w < MIN_DIM or hh < MIN_DIM:
                        continue
                    while os.path.exists(os.path.join(cls_dir, f"{cls}_{idx:04d}.jpg")):
                        idx += 1
                    path = os.path.join(cls_dir, f"{cls}_{idx:04d}.jpg")
                    img.save(path, "JPEG", quality=92)
                    seen_content.add(h)
                    saved += 1
                    idx += 1
            log(f"  class {cls} saved so far: {saved}")
        summary[cls] = saved
        log(f"=== class {cls} DONE: {saved} images ===")
    log(f"SUMMARY: {json.dumps(summary)}")
    total = sum(summary.values())
    log(f"TOTAL: {total} images")
    short = {c: TARGET_PER_CLASS - s for c, s in summary.items() if s < TARGET_PER_CLASS}
    if short:
        log(f"WARNING - classes below target: {short} (re-run script to resume)")
    else:
        log("ALL CLASSES AT TARGET")


if __name__ == "__main__":
    main()
