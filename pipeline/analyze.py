# DanceCut - 舞蹈片段自动识别与切片  |  PolyForm Noncommercial 1.0.0（禁止商用，详见 LICENSE）
# Copyright (c) 2026 wz0388 <https://github.com/wz0388>
# -*- coding: utf-8 -*-
"""
Pass 1: MediaPipe Pose on all sampled frames (every 3 s) + frame-difference motion.
Writes pose_metrics.csv with raw landmarks and derived body/arm features.
"""
import os, sys, math, time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
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
MODELS = os.path.join(ROOT, "models")
FRAMES = os.path.join(ROOT, "frames")
STEP = 3.0
POSE_MODEL = os.path.join(MODELS, "pose_landmarker_full.task")

LSH, RSH = 11, 12
LEL, REL = 13, 14
LWR, RWR = 15, 16
LHIP, RHIP = 23, 24
LKNE, RKNE = 25, 26
LANK, RANK = 27, 28
KEY = [0, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKNE, RKNE, LANK, RANK]

_pose = None


def get_pose():
    global _pose
    if _pose is None:
        _pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.IMAGE, num_poses=3,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3))
    return _pose


def ang(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1]], float)
    v2 = np.array([c[0] - b[0], c[1] - b[1]], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))


def pick(lms_list):
    """score = mean core visibility - penalty for being off-centre (handles 3-panel layouts)"""
    best, bs = None, -9
    for lms in lms_list:
        core = [LSH, RSH, LHIP, RHIP]
        v = float(np.mean([lms[i].visibility for i in core]))
        cx = float(np.mean([lms[i].x for i in range(33)]))
        s = v - 0.35 * abs(cx - 0.5)
        if s > bs:
            bs, best = s, lms
    return best


def work(idx):
    fp = os.path.join(FRAMES, "f_%05d.jpg" % idx)
    bgr = imread_u(fp)
    if bgr is None:
        return dict(frame=idx, t=idx * STEP, det=0)
    H, W = bgr.shape[:2]
    r = dict(frame=idx, t=idx * STEP)
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = get_pose().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    r["det"] = 1 if res.pose_landmarks else 0
    r["n_pose"] = len(res.pose_landmarks or [])
    if res.pose_landmarks:
        lms = pick(res.pose_landmarks)
        P = [(l.x, l.y) for l in lms]
        V = [l.visibility for l in lms]
        for i in KEY:
            r["x%d" % i] = round(P[i][0], 5)
            r["y%d" % i] = round(P[i][1], 5)
            r["v%d" % i] = round(V[i], 4)

        sh = ((P[LSH][0] + P[RSH][0]) / 2, (P[LSH][1] + P[RSH][1]) / 2)
        hp = ((P[LHIP][0] + P[RHIP][0]) / 2, (P[LHIP][1] + P[RHIP][1]) / 2)
        kn = ((P[LKNE][0] + P[RKNE][0]) / 2, (P[LKNE][1] + P[RKNE][1]) / 2)
        an = ((P[LANK][0] + P[RANK][0]) / 2, (P[LANK][1] + P[RANK][1]) / 2)
        torso = max(math.dist(sh, hp), 1e-4)
        r["sh_x"], r["sh_y"] = round(sh[0], 5), round(sh[1], 5)
        r["hip_x"], r["hip_y"] = round(hp[0], 5), round(hp[1], 5)
        r["torso"] = round(torso, 5)
        r["sh_w"] = round(abs(P[LSH][0] - P[RSH][0]), 5)
        r["torso_up"] = round(abs(sh[0] - hp[0]) / max(abs(sh[1] - hp[1]), 1e-4), 4)

        # vertical extent of RELIABLE landmarks only
        rel = [(P[i][1], V[i]) for i in range(33) if V[i] > 0.5]
        if rel:
            ys = [y for y, _ in rel]
            r["vis_span"] = round(max(ys) - min(ys), 4)
            r["vis_top"], r["vis_bot"] = round(min(ys), 4), round(max(ys), 4)
        r["n_rel"] = len(rel)
        # how much of the body below the hips is reliably visible
        r["vis_ank"] = round(min(V[LANK], V[RANK]), 4)
        r["vis_kne"] = round(min(V[LKNE], V[RKNE]), 4)

        r["knee_l"] = round(ang(P[LHIP], P[LKNE], P[LANK]), 1) if V[LKNE] > 0.4 else ""
        r["knee_r"] = round(ang(P[RHIP], P[RKNE], P[RANK]), 1) if V[RKNE] > 0.4 else ""
        r["leg_ratio"] = round(math.dist(hp, an) / torso, 3)
        r["ank_below_hip"] = round((an[1] - hp[1]) / torso, 3)
        r["hip_below_sh"] = round((hp[1] - sh[1]) / torso, 3)

        for side, si, ei, wi in (("l", LSH, LEL, LWR), ("r", RSH, REL, RWR)):
            r["up_%s" % side] = round((P[si][1] - P[wi][1]) / torso, 3)      # wrist above shoulder
            r["lat_%s" % side] = round(abs(P[wi][0] - sh[0]) / torso, 3)     # lateral spread
            r["armlen_%s" % side] = round(math.dist(P[si], P[wi]) / torso, 3)
            r["elb_%s" % side] = round(ang(P[si], P[ei], P[wi]), 1)
            r["wv_%s" % side] = round(min(V[si], V[ei], V[wi]), 3)
    return r


def main():
    files = sorted(f for f in os.listdir(FRAMES) if f.endswith(".jpg"))
    n = len(files)
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t0 = time.time()
    rows = []
    with Pool(procs) as pool:
        for i, row in enumerate(pool.imap(work, range(n), chunksize=24)):
            rows.append(row)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print("pose %d/%d  %.0fs  eta %.0fs" % (i + 1, n, el, el / (i + 1) * (n - i - 1)), flush=True)
    rows.sort(key=lambda x: x["frame"])
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "pose_metrics.csv"), index=False, encoding="utf-8-sig")
    print("POSE DONE %d rows in %.0fs" % (len(rows), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
