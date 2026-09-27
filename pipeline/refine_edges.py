# DanceCut - 舞蹈片段自动识别与切片  |  PolyForm Noncommercial 1.0.0（禁止商用，详见 LICENSE）
# Copyright (c) 2026 wz0388 <https://github.com/wz0388>
# -*- coding: utf-8 -*-
"""Stage B — 用 1 秒精度重扫每个区块的起止边界（3 秒采样只能定到 ±3s）。"""
import os, subprocess, numpy as np, pandas as pd, math
import cv2, mediapipe as mp
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
SRC = os.environ.get("DANCECUT_VIDEO", "")   # 源视频路径：设环境变量 DANCECUT_VIDEO
RDIR = os.path.join(ROOT, "refine_frames")
os.makedirs(RDIR, exist_ok=True)
POSE = os.path.join(ROOT, "models", "pose_landmarker_full.task")
SHY, SHW, DWR = 0.44, 0.27, 0.45
WIN = 15  # ±15s 扫描窗口


def hms(sec):
    sec = float(sec); h = int(sec // 3600); m = int((sec % 3600) // 60)
    return "%02d:%02d:%05.2f" % (h, m, sec % 60)


bd = pd.read_csv(os.path.join(ROOT, "dance_segments_final.csv"))
tasks = []
for _, r in bd.iterrows():
    for edge, tc in (("s", r.start_s), ("e", r.end_s)):
        for k in range(-WIN, WIN + 1):
            t = tc + k
            if t < 0:
                continue
            tasks.append((int(r.block), edge, round(t, 2), k))
tasks = sorted(set(tasks))
print("refine samples:", len(tasks), flush=True)

# 抽帧
procs = []
for blk, edge, t, k in tasks:
    fn = os.path.join(RDIR, "b%d_%s_t%08.2f.jpg" % (blk, edge, t))
    if os.path.exists(fn):
        continue
    procs.append(subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % t,
                                   "-i", SRC, "-frames:v", "1", "-q:v", "3", "-y", fn]))
    if len(procs) >= 16:
        for p in procs:
            p.wait()
        procs = []
for p in procs:
    p.wait()
print("frames extracted:", len(os.listdir(RDIR)), flush=True)

pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=POSE),
    running_mode=vision.RunningMode.IMAGE, num_poses=3,
    min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3))

LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP = 11, 12, 13, 14, 15, 16, 23, 24


def gate(bgr):
    H, W = bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return False
    best, bs = None, -9
    for lms in res.pose_landmarks:
        v = float(np.mean([lms[i].visibility for i in (LSH, RSH, LHIP, RHIP)]))
        cx = float(np.mean([lms[i].x for i in range(33)]))
        s = v - 0.35 * abs(cx - 0.5)
        if s > bs:
            bs, best = s, lms
    P = [(l.x, l.y) for l in best]
    V = [l.visibility for l in best]
    shx = (P[LSH][0] + P[RSH][0]) / 2; shy = (P[LSH][1] + P[RSH][1]) / 2
    shw = abs(P[LSH][0] - P[RSH][0])
    torso = max(math.dist((shx, shy), ((P[LHIP][0] + P[RHIP][0]) / 2, (P[LHIP][1] + P[RHIP][1]) / 2)), 1e-4)
    if not (shy < SHY and shw < SHW):
        return False
    up = max((P[LSH][1] - P[LWR][1]) / torso, (P[RSH][1] - P[RWR][1]) / torso)
    lat = max(abs(P[LWR][0] - shx) / torso, abs(P[RWR][0] - shx) / torso)
    em = min([a for a in (
        angle(P[LSH], P[LEL], P[LWR]) if V[LEL] > .3 else 999,
        angle(P[RSH], P[REL], P[RWR]) if V[REL] > .3 else 999)])
    wv = max(min(V[LSH], V[LEL], V[LWR]), min(V[RSH], V[REL], V[RWR]))
    return bool((((up > -0.40) or (lat > 0.45) or (em < 115 and up > -0.55)) and wv > 0.50))


def angle(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1]], float); v2 = np.array([c[0] - b[0], c[1] - b[1]], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 999.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))


res = {}
for blk, edge, t, k in tasks:
    fn = os.path.join(RDIR, "b%d_%s_t%08.2f.jpg" % (blk, edge, t))
    if not os.path.exists(fn):
        res[(blk, edge, t)] = (k, None); continue
    im = imread_u(fn)
    res[(blk, edge, t)] = (k, gate(im))
print("gate done", flush=True)

rows = []
for _, r in bd.iterrows():
    blk = int(r.block)
    # 起始边界: 从 -WIN 向 +WIN 找连续 2 个 True 的第一个
    s_times = sorted([t for (b, e, t) in res if b == blk and e == "s"], key=lambda t: res[(blk, "s", t)][0])
    e_times = sorted([t for (b, e, t) in res if b == blk and e == "e"], key=lambda t: res[(blk, "e", t)][0])
    sg = [res[(blk, "s", t)][1] for t in s_times]
    eg = [res[(blk, "e", t)][1] for t in e_times]
    ns = next((i for i in range(len(sg) - 1) if sg[i] and sg[i + 1]), None)
    ne = next((i for i in range(len(eg) - 1, 0, -1) if eg[i] and eg[i - 1]), None)
    new_s = s_times[ns] if ns is not None else r.start_s
    new_e = e_times[ne] if ne is not None else r.end_s
    rows.append(dict(block=blk, orig_start=r.start_s, orig_end=r.end_s,
                     refined_start=round(new_s, 2), refined_end=round(new_e, 2),
                     dur_s=round(new_e - new_s, 2), conf=r.conf, motion=r.motion,
                     start_tc=hms(new_s), end_tc=hms(new_e)))
    print("block %2d  %8.2f-%8.2f  ->  %8.2f-%8.2f (%.1fs)" % (blk, r.start_s, r.end_s, new_s, new_e, new_e - new_s), flush=True)

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "dance_segments_refined.csv"), index=False, encoding="utf-8-sig")
print("DONE")
