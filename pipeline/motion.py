# DanceCut - 舞蹈片段自动识别与切片  |  PolyForm Noncommercial 1.0.0（禁止商用，详见 LICENSE）
# Copyright (c) 2026 wz0388 <https://github.com/wz0388>
# -*- coding: utf-8 -*-
"""Pass 1b: frame-difference motion energy between consecutive 3 s samples."""
import os, sys, numpy as np, cv2, pandas as pd
from multiprocessing import Pool


def imread_u(path, flags=None):
    """cv2.imread 在 Windows 上读不了含中文的路径，用 fromfile+imdecode 绕过。"""
    import numpy as _np
    import cv2 as _cv2
    try:
        buf = _np.fromfile(path, dtype=_np.uint8)
        if buf.size == 0:
            return None
        return _cv2.imdecode(buf, _cv2.IMREAD_GRAYSCALE if flags == "gray" else _cv2.IMREAD_COLOR)
    except Exception:
        return None

ROOT = os.environ.get("DANCECUT_ROOT") or os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(ROOT, "frames")


def work(idx):
    a = imread_u(os.path.join(FRAMES, "f_%05d.jpg" % (idx - 1)), "gray")
    b = imread_u(os.path.join(FRAMES, "f_%05d.jpg" % idx), "gray")
    if a is None or b is None:
        return dict(frame=idx, t=idx * 3.0, m=0.0, m_c=0.0)
    a = cv2.resize(a, (320, 180)); b = cv2.resize(b, (320, 180))
    d = cv2.absdiff(a, b)
    g = float(d.mean()) / 255.0
    h, w = d.shape
    c = d[int(h * 0.15):int(h * 0.95), int(w * 0.2):int(w * 0.8)]
    return dict(frame=idx, t=idx * 3.0, m=round(g, 5), m_c=round(float(c.mean()) / 255.0, 5))


if __name__ == "__main__":
    n = len([f for f in os.listdir(FRAMES) if f.endswith(".jpg")])
    with Pool(8) as p:
        rows = list(p.imap(work, range(1, n), chunksize=32))
    rows.insert(0, dict(frame=0, t=0.0, m=0.0, m_c=0.0))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "motion.csv"), index=False, encoding="utf-8-sig")
    print("MOTION DONE", len(rows))
