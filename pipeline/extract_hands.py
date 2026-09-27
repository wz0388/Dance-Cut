# DanceCut - 舞蹈片段自动识别与切片  |  PolyForm Noncommercial 1.0.0（禁止商用，详见 LICENSE）
# Copyright (c) 2026 wz0388 <https://github.com/wz0388>
# -*- coding: utf-8 -*-
"""Stage C — 对判定为舞蹈的帧提取手部 21 点精细骨骼（手指级），量化"手部舞蹈状态"。"""
import os, numpy as np, pandas as pd, cv2, math
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision


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
res_df = pd.read_csv(os.path.join(ROOT, "dance_result.csv"))
lm = pd.read_csv(os.path.join(ROOT, "merged.csv"))[["frame", "x15", "y15", "x16", "y16", "wv_l", "wv_r", "torso"]]
res_df = res_df.drop(columns=[c for c in ("x15", "y15", "x16", "y16", "wv_l", "wv_r", "torso") if c in res_df.columns])
res_df = res_df.merge(lm, on="frame", how="left")
dance = res_df[res_df.dance == 1].copy()
print("frames to process:", len(dance), flush=True)

hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ROOT, "models", "hand_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE, num_hands=1,
    min_hand_detection_confidence=0.25, min_hand_presence_confidence=0.25))

rows = []
for _, r in dance.iterrows():
    f = int(r.frame)
    bgr = imread_u(os.path.join(FRAMES, "f_%05d.jpg" % f))
    if bgr is None:
        continue
    H, W = bgr.shape[:2]
    torso = r.torso if r.torso and r.torso > 0 else 0.2
    for side, wx, wy, vis in (("L", r.x15, r.y15, r.wv_l), ("R", r.x16, r.y16, r.wv_r)):
        if not (wx == wx) or vis != vis or vis < 0.3:
            continue
        half = max(0.11, min(0.22, 1.4 * torso))
        x0 = int(np.clip((wx - half) * W, 0, W - 2)); x1 = int(np.clip((wx + half) * W, 1, W))
        y0 = int(np.clip((wy - half) * H, 0, H - 2)); y1 = int(np.clip((wy + half) * H, 1, H))
        if x1 - x0 < 24 or y1 - y0 < 24:
            continue
        crop = cv2.resize(bgr[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_CUBIC)
        hr = hand.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
        if not hr.hand_landmarks:
            rows.append(dict(frame=f, t=r.t, side=side, ok=0))
            continue
        pts = np.array([[p.x, p.y] for p in hr.hand_landmarks[0]])
        palm = (pts[0] + pts[5] + pts[17]) / 3.0
        hs = max(np.linalg.norm(pts[0] - pts[9]), 1e-4)
        tips = pts[[4, 8, 12, 16, 20]]
        tip_d = np.linalg.norm(tips - palm, axis=1) / hs
        spread = float(np.mean([np.linalg.norm(tips[i] - tips[j]) for i in range(5) for j in range(i + 1, 5)])) / hs
        pinch = float(np.linalg.norm(pts[4] - pts[8])) / hs
        rows.append(dict(frame=f, t=r.t, side=side, ok=1,
                         palm_open=round(float(np.mean(tip_d)), 3),
                         tip_spread=round(spread, 3), pinch=round(pinch, 3),
                         openness=round(float(np.mean(tip_d)) * 0.6 + spread * 0.4, 3),
                         hand_size_px=round(hs * 256, 1)))
h = pd.DataFrame(rows)
h.to_csv(os.path.join(ROOT, "hand_landmarks_dance.csv"), index=False, encoding="utf-8-sig")
ok = h[h.ok == 1]
print("detected hands:", len(ok), "/", len(h))
if len(ok):
    print(ok[["palm_open", "tip_spread", "pinch", "openness"]].describe().round(3).to_string())
    open_ratio = float((ok.openness > 0.85).mean())
    print("open-hand (舞蹈张手) 比例: %.2f" % open_ratio)
