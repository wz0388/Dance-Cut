#
# DanceCut - 舞蹈片段自动识别与切片
# Copyright (c) 2026 wz0388  <https://github.com/wz0388>
#
# 本文件采用 PolyForm Noncommercial License 1.0.0 授权：
#   允许个人学习、研究、修改与非商业分发；**禁止任何商业用途**。
#   完整条款见仓库根目录 LICENSE，中文说明见 NOTICE.md。
#   商业授权请联系作者。
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舞蹈片段自动识别与切片工具（图形界面版）

流程：每 N 秒抽一帧 -> 人体骨骼(33点) + 手部骨骼(21点)判定 -> 合并片段 -> ffmpeg 切片

用法：
    直接双击 DanceCut.exe
    图形界面里可以选「单个视频文件」或「批量处理文件夹」——
    批量模式下扫描文件夹里的所有视频，每个视频在它自己旁边新建一个同名专用文件夹放结果。

    命令行：
      DanceCut.exe --cli --video <视频> [--out <目录>] [--step 3] [--bridge 27]
                   [--min-dance 24] [--workers 6]
      DanceCut.exe --cli --batch <文件夹> [--recursive] [--no-skip] [--out <根目录>]
      DanceCut.exe --self-test
"""
import os
import sys
import csv
import json
import math
import time
import shutil
import subprocess
import tempfile
import threading
import traceback
import queue

# ---------------------------------------------------------------- 基础环境

def install_stub_matplotlib():
    """mediapipe 的 drawing_utils 在导入时 `import matplotlib.pyplot`，
    但本工具只用推理、不用绘图。装一个轻量桩，省掉整个 matplotlib
    （连带 PIL / fontTools / kiwisolver，约 60MB）。真实安装了就直接用真的。"""
    if "matplotlib" in sys.modules:
        return
    try:
        import matplotlib  # noqa: F401
        return
    except Exception:
        pass
    import types

    def _missing(*a, **k):
        raise RuntimeError("本程序未打包 matplotlib，绘图功能不可用")

    mpl = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.__getattr__ = lambda name: _missing
    mpl.pyplot = pyplot
    mpl.__getattr__ = lambda name: _missing
    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = pyplot


install_stub_matplotlib()


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def base_dir():
    """exe 所在目录 / 源码目录（不能用 __file__，打包后它指向临时解包目录）"""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource(name):
    """内置资源（模型 / ffmpeg）查找。

    按顺序在这些位置找：
      1. 环境变量 DANCECUT_MODELS 指定的目录
      2. 打包后的解包目录 _MEIPASS（同级 / models/）
      3. 脚本（或 exe）所在目录（同级 / models/ / bin/）
      4. 上一级目录的 models/  —— 仓库布局里 app/ 与 models/ 是兄弟目录
      5. 上一级目录本身
    """
    cands = []
    env_dir = os.environ.get("DANCECUT_MODELS")
    if env_dir:
        cands += [os.path.join(env_dir, name), env_dir if os.path.isfile(env_dir) else ""]
    roots = []
    mp = getattr(sys, "_MEIPASS", None)
    if mp:
        roots.append(mp)
    bd = base_dir()
    roots += [bd, os.path.dirname(bd)]
    for r in roots:
        if not r:
            continue
        cands += [os.path.join(r, name),
                  os.path.join(r, "models", name),
                  os.path.join(r, "bin", name)]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def ffmpeg_path():
    """优先级：exe 内置 > 同目录 > PATH"""
    p = resource("ffmpeg.exe")
    if p:
        return p
    return shutil.which("ffmpeg") or "ffmpeg"


def subprocess_flags():
    """GUI exe 里启动 ffmpeg 时不弹黑框（只在打包后加，源码运行时保留输出便于排错）"""
    flags = subprocess.CREATE_NEW_PROCESS_GROUP
    if is_frozen():
        flags |= 0x08000000  # CREATE_NO_WINDOW
    return flags


def run_ffmpeg(args, **kw):
    return subprocess.run([ffmpeg_path()] + args,
                          creationflags=subprocess_flags(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)


def imread_u(path):
    """读取图片，支持中文/非 ASCII 路径。

    cv2.imread 在 Windows 上走 ANSI 路径，路径里有中文就会静默返回 None
    （表现为"一帧都没识别出来"）。用 np.fromfile + cv2.imdecode 绕过。
    """
    import numpy as np
    import cv2
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        try:
            return cv2.imread(path)
        except Exception:
            return None


def imread_gray_u(path):
    import cv2
    im = imread_u(path)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def imwrite_u(path, img, quality=85):
    """写图片，支持中文/非 ASCII 路径。

    cv2.imwrite 和 cv2.imread 一样走 ANSI 路径，中文路径下**静默失败**（只返回 False，不抛异常），
    所以必须用 imencode + 自己写文件，并且**检查返回值**。
    """
    import cv2
    try:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return os.path.getsize(path) > 0
    except Exception:
        return False


def probe_duration(video):
    """ffprobe 不打包，改从 ffmpeg 的 banner 里解析时长"""
    try:
        r = run_ffmpeg(["-hide_banner", "-i", video], timeout=60)
        txt = (r.stdout or b"").decode("utf-8", "replace")
        for line in txt.splitlines():
            if "Duration:" in line:
                part = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = part.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------- 判定参数

STEP_DEF = 3.0
SHY_T, SHW_T = 0.44, 0.27          # 站立：肩中点高度 / 肩宽
UP_T, LAT_T, ELB_T, DWR_T = -0.40, 0.45, 115.0, 0.45   # 手部舞蹈姿态（严格档）
WV_T = 0.50
MTHR = 0.09                        # 持续运动阈值（基础档）
BRIDGE_DEF = 27.0                  # R3 中间可停留
MIN_DANCE_DEF = 24.0               # R4 最短舞蹈

# ---- 两条"更容易判为跳舞"的放松规则（2026-09-26 追加）----
# 放松 A：当处于舞蹈状态时降低运动量门槛（用高置信锚点的邻域来定义"处于舞蹈状态"）
MTHR_ANCHOR = 0.13                 # 高置信锚点：运动量要高于这个值才算"确实在跳"
MTHR_NEAR = 0.075                  # 锚点邻域内：门槛降低
NEAR_N = 2                         # 邻域半径（采样帧），±2 = 前后各 6 秒
# 放松 B：画面露出腿（半身以下 ~ 全身）时更容易判为跳舞
KNE_T, ANK_T = 0.35, 0.20          # "露出腿"判定：膝关节或踝关节关键点可见度
MTHR_LEG = 0.070                   # 露腿 + 锚点邻域内：门槛再降一档
SHW_T2 = 0.32                      # 露腿时肩宽判据放宽（人离机位更近）
RELAX_SHY = 0.42                   # 放松帧额外要求：肩膀要够高（站姿取景）
UP_T2, LAT_T2, ELB_T2, WV_T2 = -0.58, 0.34, 132.0, 0.40   # 放宽后的手部姿态阈值


# ---------------------------------------------------------------- 姿态 worker

_W = {}


def _worker_init(pose_model, video_frames_dir, step):
    _W["model"] = pose_model
    _W["dir"] = video_frames_dir
    _W["step"] = step


def _worker_pose(idx):
    import numpy as np
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    if "pose" not in _W:
        _W["pose"] = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=_W["model"]),
            running_mode=vision.RunningMode.IMAGE, num_poses=3,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3))

    LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
    fp = os.path.join(_W["dir"], "f_%05d.jpg" % idx)
    bgr = imread_u(fp)
    row = {"frame": idx, "t": idx * _W["step"], "det": 0}
    if bgr is None:
        return row
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = _W["pose"].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return row

    best, bs = None, -9
    for lms in res.pose_landmarks:
        v = float(np.mean([lms[i].visibility for i in (LSH, RSH, LHIP, RHIP)]))
        cx = float(np.mean([lms[i].x for i in range(33)]))
        s = v - 0.35 * abs(cx - 0.5)
        if s > bs:
            bs, best = s, lms
    lms = best
    P = [(l.x, l.y) for l in lms]
    V = [l.visibility for l in lms]
    # 33 个关键点坐标留一份，供"输出骨骼画面"使用（写 CSV 时会被忽略）
    row["lms"] = [[round(P[i][0], 4), round(P[i][1], 4), round(V[i], 3)] for i in range(33)]

    def ang(a, b, c):
        v1 = (a[0] - b[0], a[1] - b[1]); v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1); n2 = math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return float("nan")
        return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))))

    shx = (P[LSH][0] + P[RSH][0]) / 2.0; shy = (P[LSH][1] + P[RSH][1]) / 2.0
    hpx = (P[LHIP][0] + P[RHIP][0]) / 2.0; hpy = (P[LHIP][1] + P[RHIP][1]) / 2.0
    torso = max(math.hypot(shx - hpx, shy - hpy), 1e-4)

    row.update(det=1, sh_x=round(shx, 5), sh_y=round(shy, 5),
               sh_w=round(abs(P[LSH][0] - P[RSH][0]), 5), torso=round(torso, 5),
               x15=round(P[LWR][0], 5), y15=round(P[LWR][1], 5),
               x16=round(P[RWR][0], 5), y16=round(P[RWR][1], 5),
               # 腿部关键点可见度：用来判断"画面里露出腿了没有"（半身以下 / 全身）
               # 取左右较小值 = 两条腿都要看得见；髋/膝/踝在紧机位下会被外推出画面，此时可见度≈0
               vis_kne=round(min(V[LKNE], V[RKNE]), 4),
               vis_ank=round(min(V[LANK], V[RANK]), 4),
               vis_hip=round(min(V[LHIP], V[RHIP]), 4))
    for side, si, ei, wi in (("l", LSH, LEL, LWR), ("r", RSH, REL, RWR)):
        row["up_" + side] = round((P[si][1] - P[wi][1]) / torso, 4)
        row["lat_" + side] = round(abs(P[wi][0] - shx) / torso, 4)
        row["elb_" + side] = round(ang(P[si], P[ei], P[wi]), 1)
        row["wv_" + side] = round(min(V[si], V[ei], V[wi]), 4)
    return row


# ---------------------------------------------------------------- 主流程

def log_factory(cb):
    def _log(msg):
        if cb:
            cb(str(msg))
        else:
            print(msg, flush=True)
    return _log


def extract_frames(video, fdir, step, log, progress):
    os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(fdir, f))
    log("① 抽帧：每 %.1f 秒一张 …" % step)
    r = run_ffmpeg(["-hide_banner", "-loglevel", "error", "-i", video,
                    "-vf", "fps=1/%.6f" % step, "-q:v", "3",
                    "-start_number", "0", os.path.join(fdir, "f_%05d.jpg")])
    n = len([f for f in os.listdir(fdir) if f.endswith(".jpg")])
    if n == 0:
        raise RuntimeError("抽帧失败，请检查视频文件是否可读（ffmpeg 输出：%s）"
                           % (r.stdout or b"")[:400])
    probe = imread_u(os.path.join(fdir, "f_00000.jpg"))
    if probe is None:
        raise RuntimeError("抽帧完成但图片读不出来（可能是路径权限或非 ASCII 路径问题）：%s" % fdir)
    log("   完成，共 %d 帧" % n)
    return n


def _enable_mp_reuse():
    """onefile 打包后，子进程默认会各自把 140MB 的包再解压一遍（慢 + 一堆临时目录）。
    这里把父进程已解包的位置通过 PyInstaller 的启动器环境变量告诉子进程，
    让它们直接复用，避免重复解包。"""
    mp = getattr(sys, "_MEIPASS", None)
    if not (is_frozen() and mp):
        return False
    if os.environ.get("_PYI_APPLICATION_HOME_DIR"):
        return True
    try:
        os.environ["_PYI_APPLICATION_HOME_DIR"] = mp
        os.environ["_PYI_PARENT_PROCESS_LEVEL"] = "1"   # 子进程复用父目录且不要在退出时删掉它
        return True
    except Exception:
        return False


def pose_all(fdir, n, model, step, workers, log, progress, cancel):
    log("② 人体骨骼识别（%s）…" % ("%d 进程并行" % workers if workers > 1 else "单进程"))
    rows = [None] * n
    done = [0]
    t0 = time.time()
    if workers > 1:
        _enable_mp_reuse()
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                     initargs=(model, fdir, step)) as ex:
                for row in ex.map(_worker_pose, range(n), chunksize=8):
                    rows[row["frame"]] = row
                    done[0] += 1
                    if done[0] % 25 == 0 or done[0] == n:
                        progress(0.10 + 0.45 * done[0] / n)
                        log("   %d/%d  用时 %.0fs" % (done[0], n, time.time() - t0))
                    if cancel.is_set():
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise InterruptedError("用户取消")
            return rows
        except InterruptedError:
            raise
        except Exception as e:
            log("   并行失败（%s），回退单进程…" % str(e)[:120])
            rows = [None] * n
            done[0] = 0
    _worker_init(model, fdir, step)
    for i in range(n):
        if cancel.is_set():
            raise InterruptedError("用户取消")
        rows[i] = _worker_pose(i)
        done[0] += 1
        if done[0] % 100 == 0 or done[0] == n:
            progress(0.10 + 0.45 * done[0] / n)
            log("   %d/%d  用时 %.0fs" % (done[0], n, time.time() - t0))
    return rows


def motion_all(fdir, n, step, log, progress):
    import cv2
    import numpy as np
    log("③ 计算帧间运动量 …")
    out = [0.0] * n
    prev = None
    for i in range(n):
        g = imread_gray_u(os.path.join(fdir, "f_%05d.jpg" % i))
        if g is None:
            prev = None
            continue
        g = cv2.resize(g, (320, 180))
        if prev is not None:
            out[i] = float(cv2.absdiff(g, prev).mean()) / 255.0
        prev = g
        if i % 200 == 0:
            progress(0.55 + 0.10 * i / max(n, 1))
    return out


def classify(rows, motions, step):
    """逐帧判定

    基础三条：① 站立  ② 手部舞蹈  ③ 持续运动  —— 三条同时成立 = base 舞蹈帧

    在此之上加两条"更容易判为跳舞"的放松（都要求有高置信锚点解锁，避免把坐姿聊天放进来）：
      放松 A（对应"当处于舞蹈状态时降低阈值"）：高置信锚点 ±NEAR_N 个采样内，运动量门槛降到 MTHR_NEAR
      放松 B（对应"露出腿/半身及全身时更容易判为跳舞"）：露出腿时站立判据与手部姿态判据都放宽，
                                                          运动量门槛降到 MTHR_LEG
    """
    n = len(rows)

    # ---- 逐帧基础量 ----
    for i, r in enumerate(rows):
        r["m"] = round(motions[i], 5)
        up = max(r.get("up_l", -9), r.get("up_r", -9))
        lat = max(r.get("lat_l", 0), r.get("lat_r", 0))
        elb = min(r.get("elb_l", 999), r.get("elb_r", 999))
        wv = max(r.get("wv_l", 0), r.get("wv_r", 0))
        r["up_best"], r["lat_best"], r["elb_min"], r["wv_best"] = up, lat, elb, wv

        # 站立：紧机位下髋膝踝会被外推出画面，所以用肩部位置+肩宽推断
        r["is_standing"] = int(r.get("sh_y", 9) < SHY_T and r.get("sh_w", 9) < SHW_T)
        # 露出腿（半身以下到全身）—— 腿都看见了，人必然是站着的（但反过来不成立：坐矮沙发腿也会入镜）
        r["leg_vis"] = int(r.get("vis_kne", 0) > KNE_T or r.get("vis_ank", 0) > ANK_T)
        # 站立取景：肩部判据 或 露出腿。
        # 注意"露出腿"必须能独立满足这一条 —— 主播离镜头近时肩宽会超过阈值，
        # 只靠肩部判据会导致整段视频一帧都判不出站立（实测 917 帧里只有 14 帧）。
        r["stance_ok"] = int(r["is_standing"] or r["leg_vis"])

        # 手部舞蹈姿态：严格 / 放宽两档
        pose_strict = (up > UP_T) or (lat > LAT_T) or (elb < ELB_T and up > -0.55)
        pose_loose = (up > UP_T2) or (lat > LAT_T2) or (elb < ELB_T2 and up > -0.72)
        r["hand_pose"] = int(pose_strict and wv > WV_T)
        r["hand_pose_loose"] = int(pose_loose and wv > WV_T2)

    # ---- 手腕位移 ----
    for i, r in enumerate(rows):
        if i == 0 or not r.get("det") or not rows[i - 1].get("det"):
            r["d_wrist"] = 0.0
        else:
            p = rows[i - 1]; tr = max(r.get("torso", 1e-4), 1e-4)
            d = max(math.hypot(r["x15"] - p["x15"], r["y15"] - p["y15"]),
                    math.hypot(r["x16"] - p["x16"], r["y16"] - p["y16"]))
            r["d_wrist"] = d / tr
        r["hand_move"] = int(r["d_wrist"] > DWR_T)
        r["hand_dance"] = int(r["hand_pose"] or r["hand_move"])
        r["body_state"] = "站立" if r["is_standing"] else ("露腿" if r["leg_vis"] else "非站立")
        r["hand_state"] = "舞蹈姿态" if r["hand_pose"] else (
            "放宽姿态" if r["hand_pose_loose"] else ("手部运动" if r["hand_move"] else "静止"))

    # ---- 3 帧滚动运动量 ----
    for i, r in enumerate(rows):
        lo, hi = max(0, i - 1), min(n, i + 2)
        r["m_roll3"] = round(sum(x["m"] for x in rows[lo:hi]) / (hi - lo), 5)

    # ---- 基础三条判据（站立取景 = 肩部判据 或 露出腿）----
    for r in rows:
        r["active"] = int(r["m_roll3"] > MTHR)
        r["base"] = int(r["stance_ok"] and r["hand_dance"] and r["active"])

    # ---- 高置信锚点 → 邻域（解锁放松；也是放松的边界，避免坐姿区开闸）----
    anchor = [bool(r["base"] and r["m_roll3"] > MTHR_ANCHOR) for r in rows]
    near = [False] * n
    for i in range(n):
        if anchor[i]:
            for j in range(max(0, i - NEAR_N), min(n, i + NEAR_N + 1)):
                near[j] = True

    # ---- 合成（base ∪ 放松）----
    for i, r in enumerate(rows):
        r["anchor"] = int(anchor[i])
        r["near_anchor"] = int(near[i])
        leg = bool(r["leg_vis"])
        near_i = bool(near[i])
        nl = near_i and leg                       # 露出腿 + 在舞蹈邻域内 = 最强放松
        r["relax_level"] = ("露腿+舞蹈邻域" if nl else
                            ("舞蹈邻域" if near_i else ("露腿" if leg else "无")))
        # 运动量门槛：露腿+邻域最松，其次邻域，其余保持基础门槛
        r["m_thr"] = MTHR_LEG if nl else (MTHR_NEAR if near_i else MTHR)
        # 手部判据：严格姿态/手腕运动 ∪ 舞蹈邻域内放宽姿态
        hand_ok = bool(r["hand_pose"] or r["hand_move"]) or (near_i and r["hand_pose_loose"])
        relax_ok = bool(r["stance_ok"] and hand_ok and r["m_roll3"] > r["m_thr"])
        # 非露腿的放松帧额外要求"站姿取景"（肩膀够高）：防止把坐姿聊天放进来。
        # 露腿帧不看这条 —— 主播离镜头近时肩膀本来就偏低，强求会整段漏检。
        if relax_ok and not leg and not r["is_standing"] and not (r.get("sh_y", 9) < RELAX_SHY):
            relax_ok = False
        r["relax_ok"] = int(relax_ok)
        r["dance_raw"] = int(bool(r["base"]) or relax_ok)

    # ---- 3 帧平滑 ----
    for i, r in enumerate(rows):
        lo, hi = max(0, i - 1), min(n, i + 2)
        s = sum(rows[j]["dance_raw"] for j in range(lo, hi)) / (hi - lo)
        r["s_dance"] = round(s, 3)
        r["dance"] = int(s >= 0.34)
    return rows


def build_segments(rows, step, bridge_s, min_dance_s, video_end):
    """R1~R4 剪辑规则"""
    idx = [i for i, r in enumerate(rows) if r["dance"]]
    runs = []
    if idx:
        s = p = idx[0]
        for i in idx[1:]:
            if i == p + 1:
                p = i
            else:
                runs.append([s, p]); s = p = i
        runs.append([s, p])
    blocks = []
    for r in runs:
        if blocks and (r[0] - blocks[-1][1] - 1) * step <= bridge_s:
            blocks[-1][1] = r[1]; blocks[-1][2].append(r)
        else:
            blocks.append([r[0], r[1], [r]])
    keep, info = [], []
    for bi, (a, b, parts) in enumerate(blocks, 1):
        span = (b - a + 1) * step
        drop = span < min_dance_s
        st = max(0.0, (a - 1) * step)
        en = min(video_end, (b + 1) * step) if video_end else (b + 1) * step
        info.append(dict(block=bi, dance_span_s=round(span, 1),
                         dance_samples=sum(q[1] - q[0] + 1 for q in parts),
                         n_parts=len(parts), clip_start_s=round(st, 2), clip_end_s=round(en, 2),
                         clip_dur_s=round(en - st, 2), kept=("否" if drop else "是")))
        if not drop:
            keep.append((st, en, parts))
    return keep, info


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def hms(x):
    x = float(x)
    return "%02d:%02d:%05.2f" % (int(x // 3600), int((x % 3600) // 60), x % 60)


def hms_short(x):
    """00:52:39 —— 不带小数，用于文件命名与画面标注"""
    x = float(x)
    return "%02d:%02d:%02d" % (int(x // 3600), int((x % 3600) // 60), int(x % 60))


def hms_compact(x):
    """005239 —— 去冒号，可直接放进文件名"""
    x = float(x)
    return "%02d%02d%02d" % (int(x // 3600), int((x % 3600) // 60), int(x % 60))


FONT_CANDS = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
              r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\msyh.ttc"]


def find_font():
    for p in FONT_CANDS:
        if os.path.isfile(p):
            return p
    return None


def _fg_quote_font(path):
    """filtergraph 里路径的盘符冒号必须转义，整段用单引号包起来：
    fontfile='C\\:/Windows/Fonts/arial.ttf'"""
    return "'%s'" % path.replace("\\", "/").replace(":", "\\:")


def timecode_vf(st, en, k, frame_h=None):
    """生成"画面左上角叠加原片时间码"的滤镜；字体不可用时返回 None"""
    font = find_font()
    if not font:
        return None
    txt = "SRC %s - %s  #%d" % (hms_short(st), hms_short(en), k)
    txt = txt.replace(":", "\\:").replace("'", "")
    fs = max(16, int((frame_h or 720) / 34))
    return ("drawtext=fontfile=%s:text='%s':x=14:y=14:fontsize=%d"
            ":fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=%d"
            % (_fg_quote_font(font), txt, fs, max(6, fs // 3)))


def apply_thresholds(mthr_keep=None, mthr_leg=None):
    """允许界面/命令行覆盖两条放松规则的运动量门槛"""
    global MTHR_NEAR, MTHR_LEG
    try:
        if mthr_keep not in (None, ""):
            MTHR_NEAR = max(0.0, float(mthr_keep))
        if mthr_leg not in (None, ""):
            MTHR_LEG = max(0.0, float(mthr_leg))
    except Exception:
        pass


_NVENC_STATE = {"ok": None}


def cut_clip(video, st, dur, dst, vf=None):
    """切一段视频。优先用 GPU (h264_nvenc)，不可用则回退 CPU (libx264)；
    叠加时间码失败时也会退回到不叠加，保证片段本身能出来。
    编解码器只探测一次，之后记住结果。"""
    base_vf = "fps=30,format=yuv420p"
    vfs = ([base_vf + "," + vf] if vf else []) + [base_vf]
    gpu = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]
    cpu = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]
    if _NVENC_STATE["ok"] is True:
        encs = [gpu]
    elif _NVENC_STATE["ok"] is False:
        encs = [cpu]
    else:
        encs = [gpu, cpu]
    for v in vfs:
        for enc in encs:
            try:
                run_ffmpeg(["-hide_banner", "-loglevel", "error", "-y",
                            "-ss", "%.3f" % st, "-i", video, "-t", "%.3f" % dur,
                            "-vf", v] + enc + [dst], check=True)
            except Exception:
                continue
            if os.path.isfile(dst) and os.path.getsize(dst) > 1024:
                _NVENC_STATE["ok"] = (enc is gpu)
                return True
    return False


def run_pipeline(video, outdir, step, bridge_s, min_dance_s, workers,
                 log, progress, cancel, mthr_keep=None, mthr_leg=None,
                 skeleton=False, skel_only_dance=False,
                 ts_name=True, burn_ts=False):
    apply_thresholds(mthr_keep, mthr_leg)
    t_start = time.time()
    os.makedirs(outdir, exist_ok=True)
    fdir = os.path.join(outdir, "frames")
    cdir = os.path.join(outdir, "clips")
    os.makedirs(cdir, exist_ok=True)

    model = resource("pose_landmarker_full.task")
    if not model:
        raise RuntimeError("找不到内置模型 pose_landmarker_full.task")
    if not video or not os.path.isfile(video):
        raise RuntimeError("视频文件不存在：%s" % video)

    dur = probe_duration(video)
    log("视频时长：%s（%.1f 秒）" % (hms(dur), dur))
    if is_onefile():
        log("提示：单文件版首次启动要解包，属于正常现象。")

    tick = time.time()

    def lap(name):
        nonlocal tick
        now = time.time()
        log("   [%s 用时 %.1f 秒]" % (name, now - tick))
        tick = now

    progress(0.02)
    n = extract_frames(video, fdir, step, log, progress)
    lap("抽帧")
    rows = pose_all(fdir, n, model, step, workers, log, progress, cancel)
    lap("骨骼识别")
    motions = motion_all(fdir, n, step, log, progress)
    lap("运动量")
    log("④ 骨骼状态判定 …")
    rows = classify(rows, motions, step)
    lap("判定")
    progress(0.70)

    ndance = sum(r["dance"] for r in rows)
    nbase = sum(r.get("base", 0) for r in rows)
    nleg = sum(r.get("leg_vis", 0) for r in rows)
    log("   站立取景 %d 帧（其中靠露腿 %d 帧）/ 手部舞蹈 %d 帧"
        % (sum(r.get("stance_ok", r["is_standing"]) for r in rows), nleg,
           sum(r["hand_dance"] for r in rows)))
    log("   基础三条全过 %d 帧，两条放松规则额外补回 %d 帧（门槛：舞蹈内 %.3f / 露腿 %.3f），合计 %d 帧"
        % (nbase, ndance - nbase, MTHR_NEAR, MTHR_LEG, ndance))

    log("⑤ 合并片段（停留≤%.0fs，最短 %.0fs）…" % (bridge_s, min_dance_s))
    keep, info = build_segments(rows, step, bridge_s, min_dance_s, dur)
    progress(0.76)

    # 逐帧归属
    role = ["none"] * n
    clipid = [0] * n
    for k, (st, en, parts) in enumerate(keep, 1):
        a, b = parts[0][0], parts[-1][1]
        fa = max(0, int(round(st / step)))
        fb = min(n, int(round(en / step)))
        for j in range(fa, a):
            role[j] = "pre_pad"
        for p0, p1 in parts:
            for j in range(p0, p1 + 1):
                role[j] = "dance"
        for (q0, q1), (r0, r1) in zip(parts, parts[1:]):
            for j in range(q1 + 1, r0):
                role[j] = "bridge"
        for j in range(fa, fb):
            clipid[j] = k
    for i, r in enumerate(rows):
        r["clip_id"], r["clip_role"] = clipid[i], role[i]

    fields = ["frame", "t", "det", "sh_x", "sh_y", "sh_w", "torso",
              "vis_kne", "vis_ank", "vis_hip", "leg_vis", "stance_ok",
              "up_best", "lat_best", "elb_min", "wv_best", "d_wrist", "m", "m_roll3", "m_thr",
              "body_state", "is_standing", "hand_state", "hand_pose", "hand_pose_loose",
              "hand_move", "hand_dance", "active", "base",
              "anchor", "near_anchor", "relax_level", "relax_ok", "dance_raw",
              "s_dance", "dance", "clip_id", "clip_role"]
    write_csv(os.path.join(outdir, "dance_result.csv"), rows, fields)
    write_csv(os.path.join(outdir, "dance_blocks_analysis.csv"), info,
              ["block", "dance_span_s", "dance_samples", "n_parts",
               "clip_start_s", "clip_end_s", "clip_dur_s", "kept"])

    if skeleton:
        log("⑥ 输出骨骼画面 …")
        skeleton_pass(rows, fdir, os.path.join(outdir, "skeleton"),
                      resource("hand_landmarker.task"), log, progress, cancel,
                      only_dance=skel_only_dance)

    log("⑦ 切片（共 %d 段）…" % len(keep))
    # 画面高度（决定叠加时间码的字号）
    frame_h = None
    try:
        _im = imread_u(os.path.join(fdir, "f_00000.jpg"))
        if _im is not None:
            frame_h = _im.shape[0]
    except Exception:
        pass
    if burn_ts and not find_font():
        log("   提示：找不到系统字体，跳过画面时间码标注（文件名仍带时间）")

    clips = []
    for k, (st, en, parts) in enumerate(keep, 1):
        if cancel.is_set():
            raise InterruptedError("用户取消")
        name = "dance_%02d" % k
        if ts_name:            # 文件名里带上"取自原片的哪一段"
            name += "_%s-%s" % (hms_compact(st), hms_compact(en))
        dst = os.path.join(cdir, name + ".mp4")
        vf = timecode_vf(st, en, k, frame_h) if burn_ts else None
        okc = cut_clip(video, st, en - st, dst, vf=vf)
        if not okc:
            log("   %s 切片失败（%s - %s）" % (name, hms(st), hms(en)))
        log("   %-28s 原片 %s → %s  (%.1fs)  [%s]" % (
            name + ".mp4", hms(st), hms(en), en - st,
            "GPU" if _NVENC_STATE["ok"] else "CPU"))
        a, b = parts[0][0], parts[-1][1]
        clips.append(dict(clip=k, file=name + ".mp4",
                          start_s=round(st, 2), end_s=round(en, 2),
                          dur_s=round(en - st, 2),
                          src_start=hms_short(st), src_end=hms_short(en),
                          start_tc=hms(st), end_tc=hms(en),
                          dance_span_s=round((b - a + 1) * step, 1),
                          first_dance_frame=rows[a]["frame"], last_dance_frame=rows[b]["frame"]))
        progress(0.80 + 0.18 * k / max(len(keep), 1))

    write_csv(os.path.join(outdir, "dance_clips.csv"), clips,
              ["clip", "file", "src_start", "src_end", "start_s", "end_s", "dur_s",
               "start_tc", "end_tc", "dance_span_s", "first_dance_frame", "last_dance_frame"])

    if clips:
        lst = os.path.join(cdir, "concat.txt")
        with open(lst, "w", encoding="utf-8") as f:
            f.write("# DanceCut 片段拼接清单\n")
            f.write("# 按下面的顺序拼接即为 ALL_DANCE.mp4\n")
            f.write("# 源视频：%s\n" % video)
            f.write("# 序号  文件名 / 取自原片区间 / 时长\n")
            for c in clips:
                f.write("# %2d  %-30s  原片 %s - %s  (%.1fs)\n"
                        % (c["clip"], c["file"], c["src_start"], c["src_end"], c["dur_s"]))
            f.write("\n")
            for c in clips:
                f.write("file '%s'\n" % c["file"])
        try:
            run_ffmpeg(["-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                        "-safe", "0", "-i", lst, "-c", "copy",
                        os.path.join(cdir, "ALL_DANCE.mp4")], check=True)
            log("   已合成合集 ALL_DANCE.mp4")
        except Exception as e:
            log("   合成合集失败（不影响单段）：%s" % str(e)[:120])

    if clips:
        # 人可读的清单，方便对照
        p = os.path.join(cdir, "片段来源.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("源视频：%s\n" % video)
            f.write("共 %d 段，合计 %.1f 分钟\n\n" % (len(clips), sum(c["dur_s"] for c in clips) / 60))
            f.write("%-4s %-30s %-22s %8s\n" % ("序号", "文件", "取自原片", "时长"))
            for c in clips:
                f.write("%-4d %-30s %s - %s   %7.1fs\n"
                        % (c["clip"], c["file"], c["src_start"], c["src_end"], c["dur_s"]))
            f.write("\n注：ALL_DANCE.mp4 是把上面所有片段按序号依次拼接而成；\n"
                    "    每段在合集里的位置 = 前面所有片段时长之和。\n")
            acc = 0.0
            f.write("\n合集内位置对照：\n")
            for c in clips:
                f.write("   %-30s 合集 %s - %s\n"
                        % (c["file"], hms_short(acc), hms_short(acc + c["dur_s"])))
                acc += c["dur_s"]
        log("   片段来源清单：%s" % p)

    json.dump(dict(step_s=step, bridge_s=bridge_s, min_dance_s=min_dance_s,
                   n_frames=n, n_dance_frames=ndance, n_blocks=len(info),
                   n_clips=len(clips), total_clip_s=round(sum(c["dur_s"] for c in clips), 1),
                   clip_names_with_src_time=bool(ts_name), burn_timecode=bool(burn_ts),
                   skeleton=bool(skeleton)),
              open(os.path.join(outdir, "run_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    progress(1.0)
    log("完成，用时 %.0f 秒。片段：%s" % (time.time() - t_start, cdir))
    return clips


# ---------------------------------------------------------------- 骨骼画面输出

POSE_CONN = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
             (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
             (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
             (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
             (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
             (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
C_OK = (0, 210, 0)        # 可见骨骼连线 = 绿
C_LOW = (0, 150, 255)     # 低可见度 = 橙
C_JOINT = (0, 0, 255)     # 关节 = 红
C_LEG = (255, 220, 0)     # 腿部（露出腿时）= 青
C_WRIST = (255, 0, 255)   # 手腕 = 品红
C_BOX = (30, 30, 30)
POS_LABEL = {11: "LSH", 12: "RSH", 13: "LEL", 14: "REL", 15: "LWR", 16: "RWR",
             23: "LHIP", 24: "RHIP", 25: "LKNE", 26: "RKNE", 27: "LANK", 28: "RANK"}

SKELETON_README = """骨骼画面说明（skeleton/ 目录）

每张图 = 该采样时刻的原图 + 人体骨骼叠加 + 判定信息。

图上的标记
  绿色连线    可见度正常的骨骼
  橙色连线    可见度偏低（不一定准）
  红点        关节（位置按可见度上色：亮红=可见，暗橙=不可见）
  青色加粗    腿部（膝/踝）—— 只在"露出腿"判定成立时才加粗显示
  品红圆圈    左右手腕
  左上角文字  判定信息（见下）

左上角文字字段
  t=1234s           采样时刻（秒）
  ST:Y/N            人体骨骼是否判定为"站立"
  LEG:Y/N           是否"露出腿"（膝关节可见度>0.35 或踝关节>0.20）
  HND:<state>       手部状态：pose=舞蹈姿态 / loose=放宽姿态 / move=手部运动 / - =静止
  m3=0.18           帧间运动量的 3 帧滚动均值（判定用的就是它）
  thr=0.075         该帧实际生效的运动量门槛
  near=1            是否处于高置信舞蹈锚点的邻域（±6 秒）
  -> DANCE / -      最终判定结果
  rl=<level>        放松档位：leg+near / near / leg / none

只有被判为 DANCE 的帧才会额外画出双手的 21 点骨骼（品红色）。
"""


def draw_skeleton_frame(bgr, r, hand_det=None):
    """在画面上叠加骨骼与判定信息"""
    import cv2
    import numpy as np
    H, W = bgr.shape[:2]
    out = bgr.copy()
    lms = r.get("lms")
    if lms:
        pts = [(int(x * W), int(y * H)) for x, y, v in lms]
        vis = [v for x, y, v in lms]
        leg_on = bool(r.get("leg_vis"))
        for a, b in POSE_CONN:
            if a >= len(pts) or b >= len(pts):
                continue
            is_leg = a in (23, 24, 25, 26, 27, 28) and b in (23, 24, 25, 26, 27, 28)
            col = C_LEG if (is_leg and leg_on) else (C_OK if min(vis[a], vis[b]) > 0.5 else C_LOW)
            cv2.line(out, pts[a], pts[b], col, 3 if (is_leg and leg_on) else 2)
        for i, (x, y) in enumerate(pts):
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(out, (x, y), 4, C_JOINT if vis[i] > 0.5 else C_LOW, -1)
        for i in (15, 16):
            cv2.circle(out, pts[i], 11, C_WRIST, 2)
    # 手部骨骼（只画判定为舞蹈的帧）
    hands = []
    if hand_det is not None and r.get("dance") and r.get("det") and r.get("torso"):
        hands = hand_det(r, bgr, out)
    # 信息面板（文字只能是 ASCII，OpenCV 画不了中文；对照表见 _说明.txt）
    rl_ascii = {"露腿+舞蹈邻域": "leg+near", "舞蹈邻域": "near", "露腿": "leg", "无": "none"}
    txts = [
        "t=%ds" % int(r.get("t", 0)),
        "ST:%s LEG:%s" % ("Y" if r.get("is_standing") else "N", "Y" if r.get("leg_vis") else "N"),
        "HND:%s" % ("pose" if r.get("hand_pose") else
                    ("loose" if r.get("hand_pose_loose") else
                     ("move" if r.get("hand_move") else "-"))),
        "m3=%.3f thr=%.3f" % (r.get("m_roll3", 0), r.get("m_thr", 0)),
        "near=%d" % r.get("near_anchor", 0),
        "-> %s" % ("DANCE" if r.get("dance") else "-"),
        "rl=%s" % rl_ascii.get(r.get("relax_level", ""), "none"),
    ]
    if r.get("dance") and not r.get("base"):
        txts.append("(by relax)")
    fs = max(0.5, H / 1300.0)          # 字号随画面高度缩放，竖屏大图也看得清
    th = max(1, int(round(fs * 1.6)))
    lh = int(round(fs * 44))
    pw = int(min(W * 0.62, 22 * fs * 11))
    h = lh * len(txts) + int(lh * 0.5)
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (pw, h), C_BOX, -1)
    out = cv2.addWeighted(ov, 0.6, out, 0.4, 0)
    for i, t in enumerate(txts):
        col = (0, 255, 255) if (i == 5 and r.get("dance")) else (255, 255, 255)
        cv2.putText(out, t, (int(10 * fs), int(lh * (i + 0.8))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
    return out


def make_hand_detector(model_path, log=None):
    """返回一个 hand_detector(row, bgr, canvas)，在手腕附近裁剪放大后检测并画双手骨骼"""
    if not model_path or not os.path.isfile(model_path):
        return None
    import numpy as np
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    try:
        det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
    except Exception as e:
        if log:
            log("   手部模型加载失败，骨骼图只画人体骨架：%s" % str(e)[:100])
        return None

    def _one(bgr, cx, cy, torso, canvas):
        H, W = bgr.shape[:2]
        half = int(max(torso * 1.5, 0.06) * H)
        px, py = int(cx * W), int(cy * H)
        x0, x1 = max(0, px - half), min(W, px + half)
        y0, y1 = max(0, py - half), min(H, py + half)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return 0
        crop = bgr[y0:y1, x0:x1]
        s = 256.0 / max(crop.shape[:2])
        if s > 1:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        n = 0
        for lms in (res.hand_landmarks or []):
            pts = [(int(x0 + l.x * crop.shape[1] / s), int(y0 + l.y * crop.shape[0] / s))
                   for l in lms]
            for a, b in HAND_CONN:
                cv2.line(canvas, pts[a], pts[b], C_WRIST, 1)
            for p in pts:
                cv2.circle(canvas, p, 2, (255, 255, 0), -1)
            n += 1
        return n

    def _det(row, bgr, canvas):
        torso = max(row.get("torso", 0.2), 1e-3)
        n = 0
        for xk, yk in (("x15", "y15"), ("x16", "y16")):
            if xk in row:
                n += _one(bgr, row[xk], row[yk], torso, canvas)
        return n

    return _det


def skeleton_pass(rows, fdir, skdir, hand_model, log, progress, cancel, only_dance=False):
    """把每张采样图叠加骨骼后写到 skdir/"""
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(skdir, exist_ok=True)
    hand_det = make_hand_detector(hand_model, log)
    todo = [i for i, r in enumerate(rows) if (not only_dance or r.get("dance"))]
    log("   骨骼画面：共 %d 张%s" % (len(todo), "（仅舞蹈帧）" if only_dance else ""))
    done = [0]

    def _job(i):
        r = rows[i]
        fp = os.path.join(fdir, "f_%05d.jpg" % i)
        bgr = imread_u(fp)
        if bgr is None:
            return 0
        img = draw_skeleton_frame(bgr, r, hand_det)
        op = os.path.join(skdir, "s%05d_t%06d%s.jpg" % (i, int(r.get("t", 0)),
                                                        "_D" if r.get("dance") else ""))
        return 1 if imwrite_u(op, img, 82) else 0

    n = 0
    fail = 0
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            for k, ok in enumerate(ex.map(_job, todo), 1):
                n += ok
                fail += (1 - ok)
                if k % 100 == 0 or k == len(todo):
                    progress(0.97 + 0.02 * k / max(len(todo), 1))
                    if cancel.is_set():
                        raise InterruptedError("用户取消")
    except InterruptedError:
        raise
    open(os.path.join(skdir, "_说明.txt"), "w", encoding="utf-8").write(SKELETON_README)
    if fail:
        log("   骨骼画面：成功 %d 张，失败 %d 张（检查输出目录是否可写）" % (n, fail))
    log("   骨骼画面完成：%d 张 -> %s" % (n, skdir))
    if n >= 6:
        try:
            _skeleton_sheet(skdir, os.path.join(skdir, "_总览.jpg"))
        except Exception:
            pass
    return n


def _skeleton_sheet(skdir, out_path, cols=5, rows_n=4):
    """从骨骼图里挑一批拼成总览图"""
    import cv2
    import numpy as np
    fs = sorted(f for f in os.listdir(skdir) if f.lower().endswith(".jpg") and not f.startswith("_"))
    if not fs:
        return None
    step = max(1, len(fs) // (cols * rows_n))
    pick = fs[::step][:cols * rows_n]
    imgs = []
    for f in pick:
        im = imread_u(os.path.join(skdir, f))
        if im is not None:
            imgs.append(cv2.resize(im, (256, 144)))
    if not imgs:
        return None
    while len(imgs) % cols:
        imgs.append(np.zeros_like(imgs[0]))
    grid = np.vstack([np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)])
    imwrite_u(out_path, grid, 85)
    return out_path


# ---------------------------------------------------------------- 批量模式

VIDEO_EXTS = (".mp4", ".mkv", ".flv", ".mov", ".avi", ".ts", ".wmv", ".m4v", ".webm", ".mpg", ".mpeg", ".rmvb", ".3gp")
# 这些目录是本工具自己的产物，批量扫描时跳过，避免把切出来的片段又当成输入
SKIP_DIRS = {"frames", "clips", "_internal", "logs", "__pycache__", ".build", "output"}


def is_output_dir(path):
    """已经被本工具处理过的目录（含 run_summary.json）"""
    return os.path.isfile(os.path.join(path, "run_summary.json"))


def scan_videos(folder, recursive=False, skip_done=True):
    """扫描文件夹里的所有视频文件。recursive=True 时含子文件夹（自动跳过本工具的产物目录）"""
    found = []
    if not os.path.isdir(folder):
        return found
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
            if is_output_dir(root):
                dirs[:] = []
                continue
            for f in files:
                if f.lower().endswith(VIDEO_EXTS):
                    found.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(folder)):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and f.lower().endswith(VIDEO_EXTS):
                found.append(p)
    found.sort(key=lambda p: (os.path.dirname(p).lower(), os.path.basename(p).lower()))
    if skip_done:
        found = [p for p in found if not is_output_dir(video_out_dir(p))]
    return found


def video_out_dir(video_path):
    """每个视频的专用输出文件夹：与视频同级、用视频文件名（不含扩展名）命名"""
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, stem)


def unique_dir(path):
    """重名时加 _2 _3 …，不覆盖已有目录"""
    if not os.path.exists(path):
        return path
    i = 2
    while os.path.exists("%s_%d" % (path, i)):
        i += 1
    return "%s_%d" % (path, i)


def run_batch(inp_dir, step, bridge_s, min_dance_s, workers,
              log, progress, cancel, recursive=False, skip_done=True,
              out_root=None, mthr_keep=None, mthr_leg=None,
              skeleton=False, skel_only_dance=False, ts_name=True, burn_ts=False):
    """批量：扫描 inp_dir 下所有视频，每个视频建一个同名专用文件夹放结果。

    输出结构：
        <视频所在目录>/<视频名>/clips/dance_01.mp4 ...
                          /dance_result.csv
                          /dance_clips.csv
                          /frames/
                          /run_summary.json
    out_root 不为空时，改用 <out_root>/<视频名>/。
    """
    videos = scan_videos(inp_dir, recursive=recursive, skip_done=skip_done)
    log("批量模式：目录 %s" % inp_dir)
    log("找到 %d 个视频文件%s" % (len(videos), "（含子文件夹）" if recursive else ""))
    for i, v in enumerate(videos, 1):
        log("   %2d) %s" % (i, v))
    if not videos:
        log("没有找到可处理的视频。支持的后缀：%s" % " ".join(VIDEO_EXTS))
        log("（已处理过的视频会自动跳过，若想重跑请删掉它对应的专用文件夹）")
        return []

    results = []
    for i, v in enumerate(videos, 1):
        if cancel.is_set():
            raise InterruptedError("用户取消")
        stem = os.path.splitext(os.path.basename(v))[0]
        outdir = os.path.join(out_root, stem) if out_root else video_out_dir(v)
        outdir = unique_dir(outdir)
        log("")
        log("=" * 70)
        log("[%d/%d] %s" % (i, len(videos), os.path.basename(v)))
        log("    输出文件夹：%s" % outdir)
        log("=" * 70)

        def sub_progress(val, i=i, n=len(videos)):
            progress(((i - 1) + max(0.0, min(1.0, val))) / n)

        try:
            clips = run_pipeline(v, outdir, step, bridge_s, min_dance_s, workers,
                                 log, sub_progress, cancel,
                                 mthr_keep=mthr_keep, mthr_leg=mthr_leg,
                                 skeleton=skeleton, skel_only_dance=skel_only_dance,
                                 ts_name=ts_name, burn_ts=burn_ts)
            results.append(dict(video=v, outdir=outdir, clips=len(clips),
                                seconds=round(sum(c["dur_s"] for c in clips), 1), ok=True, err=""))
        except InterruptedError:
            log("已取消")
            raise
        except Exception as e:
            log("!!! 处理失败：%s" % e)
            log(traceback.format_exc())
            results.append(dict(video=v, outdir=outdir, clips=0, seconds=0, ok=False, err=str(e)[:200]))

    ok = [r for r in results if r.get("ok")]
    log("")
    log("=" * 70)
    log("批量完成：成功 %d / 共 %d 个视频，合计 %d 个舞蹈片段（%.1f 分钟）"
        % (len(ok), len(results), sum(r["clips"] for r in ok),
           sum(r["seconds"] for r in ok) / 60))
    for r in results:
        log("   %s  →  %s  [%s]" % (os.path.basename(r["video"]), r["outdir"],
                                    ("%d 段 / %.0f 秒" % (r["clips"], r["seconds"])) if r["ok"] else ("失败：" + r["err"])))
    if ok:
        try:
            idx = os.path.join(out_root if out_root else inp_dir, "batch_summary.csv")
            write_csv(idx, results, ["video", "outdir", "clips", "seconds", "ok", "err"])
            log("批量汇总：%s" % idx)
        except Exception:
            pass
    return results



# ---------------------------------------------------------------- 图形界面

def is_onefile():
    """onefile 打包：exe 与解包目录不在同一层。onefile 下起子进程会各自重新解包，
    所以要默认走单进程。"""
    mp = getattr(sys, "_MEIPASS", None)
    if not (is_frozen() and mp):
        return False
    try:
        return os.path.normcase(os.path.dirname(os.path.abspath(sys.executable))) != \
               os.path.normcase(os.path.dirname(os.path.abspath(mp)))
    except Exception:
        return False


def ensure_std_streams():
    """--windowed 的 exe 没有控制台；同时把所有输出镜像一份到 logs\\stdout.log，
    排错时就看这个文件。"""
    logf = None
    if is_frozen():
        try:
            d = os.path.join(base_dir(), "logs")
            os.makedirs(d, exist_ok=True)
            logf = open(os.path.join(d, "stdout.log"), "a", encoding="utf-8", buffering=1)
        except Exception:
            logf = None

    def usable(s):
        try:
            s.write(""); s.flush(); return True
        except Exception:
            return False

    class Tee:
        def __init__(self, *streams):
            self.streams = [s for s in streams if s is not None]

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

        def isatty(self):
            return False

    if logf is not None:
        cur_out = sys.stdout if sys.stdout is not None and usable(sys.stdout) else None
        cur_err = sys.stderr if sys.stderr is not None and usable(sys.stderr) else None
        sys.stdout = Tee(cur_out, logf)
        sys.stderr = Tee(cur_err, logf)
    else:
        if sys.stdout is None or not usable(sys.stdout) or sys.stderr is None or not usable(sys.stderr):
            try:
                d = os.path.join(base_dir(), "logs")
                os.makedirs(d, exist_ok=True)
                f = open(os.path.join(d, "stdout.log"), "a", encoding="utf-8", buffering=1)
                sys.stdout = sys.stdout or f
                sys.stderr = sys.stderr or f
            except Exception:
                pass
    try:
        if hasattr(sys.stdout, "reconfigure"):
            pass
    except Exception:
        pass



def launch_gui(initial_video=""):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("舞蹈片段自动识别与切片")
    root.geometry("880x660")
    root.minsize(760, 560)

    state = {"cancel": threading.Event(), "running": False, "outdir": None}

    pad = dict(padx=10, pady=4)

    top = ttk.LabelFrame(root, text="输入 / 输出")
    top.pack(fill="x", **pad)
    top.columnconfigure(1, weight=1)

    v_mode = tk.StringVar(value="batch" if initial_video and os.path.isdir(initial_video) else "single")
    v_video = tk.StringVar(value=initial_video)
    v_out = tk.StringVar(value="")
    v_recursive = tk.BooleanVar(value=False)
    v_skipdone = tk.BooleanVar(value=True)

    modrow = ttk.Frame(top)
    modrow.grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(6, 2))
    ttk.Label(modrow, text="模式").pack(side="left")
    rb1 = ttk.Radiobutton(modrow, text="单个视频文件", value="single", variable=v_mode)
    rb1.pack(side="left", padx=(8, 0))
    rb2 = ttk.Radiobutton(modrow, text="批量处理文件夹（每个视频各建一个专用文件夹）",
                          value="batch", variable=v_mode)
    rb2.pack(side="left", padx=(10, 0))

    lab_in = ttk.Label(top, text="视频文件")
    lab_in.grid(row=1, column=0, sticky="w", padx=8, pady=6)
    ttk.Entry(top, textvariable=v_video).grid(row=1, column=1, sticky="ew", pady=6)
    b_browse_in = ttk.Button(top, text="浏览…")
    b_browse_in.grid(row=1, column=2, padx=8)

    batchrow = ttk.Frame(top)
    batchrow.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=2)
    cb_rec = ttk.Checkbutton(batchrow, text="包含子文件夹", variable=v_recursive)
    cb_rec.pack(side="left")
    cb_skip = ttk.Checkbutton(batchrow, text="跳过已处理过的视频", variable=v_skipdone)
    cb_skip.pack(side="left", padx=(16, 0))
    v_skel = tk.BooleanVar(value=True)
    v_skel_only = tk.BooleanVar(value=False)
    cb_skel = ttk.Checkbutton(batchrow, text="输出骨骼画面", variable=v_skel)
    cb_skel.pack(side="left", padx=(16, 0))
    cb_skel_only = ttk.Checkbutton(batchrow, text="只输出舞蹈帧", variable=v_skel_only)
    cb_skel_only.pack(side="left", padx=(8, 0))

    srcrow = ttk.Frame(top)
    srcrow.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=2)
    v_tsname = tk.BooleanVar(value=True)
    v_burnts = tk.BooleanVar(value=False)
    ttk.Label(srcrow, text="片段标注").pack(side="left")
    ttk.Checkbutton(srcrow, text="文件名带原片时间（dance_01_005239-005854.mp4）",
                    variable=v_tsname).pack(side="left", padx=(8, 0))
    ttk.Checkbutton(srcrow, text="画面左上角叠时间码", variable=v_burnts).pack(side="left", padx=(16, 0))
    ttk.Label(srcrow, text="（另会生成「片段来源.txt」和带注释的 concat.txt）",
              foreground="#6b7a90").pack(side="left", padx=(6, 0))

    lab_out = ttk.Label(top, text="输出目录")
    lab_out.grid(row=4, column=0, sticky="w", padx=8, pady=6)
    ttk.Entry(top, textvariable=v_out).grid(row=4, column=1, sticky="ew", pady=6)
    b_browse_out = ttk.Button(top, text="浏览…", command=lambda: _pick_dir())
    b_browse_out.grid(row=4, column=2, padx=8)
    lab_hint = ttk.Label(top, text="", foreground="#6b7a90")
    lab_hint.grid(row=5, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 8))

    def _sync_mode(initial=False):
        batch = v_mode.get() == "batch"
        lab_in.configure(text="视频文件夹" if batch else "视频文件")
        b_browse_in.configure(command=(lambda: _pick_folder()) if batch else (lambda: _pick_file()))
        lab_out.configure(text="输出根目录" if batch else "输出目录")
        lab_hint.configure(text=("留空 = 每个视频在自己旁边新建一个同名文件夹；"
                                 "填了则统一放到「该目录 / 视频名 /」下"
                                 if batch else
                                 "留空 = 视频旁边的 dance_out"))
        cb_rec.configure(state="normal" if batch else "disabled")
        cb_skip.configure(state="normal" if batch else "disabled")
        if not initial:
            # 切换模式时清掉不再适用的路径；如果仍然适用就保留
            p = v_video.get().strip().strip('"')
            if p and not (os.path.isdir(p) if batch else os.path.isfile(p)):
                v_video.set("")
                v_out.set("")

    rb1.configure(command=_sync_mode)
    rb2.configure(command=_sync_mode)
    _sync_mode(initial=True)

    opt = ttk.LabelFrame(root, text="参数")
    opt.pack(fill="x", **pad)
    v_step = tk.StringVar(value="3")
    v_bridge = tk.StringVar(value="27")
    v_min = tk.StringVar(value="24")
    v_keep = tk.StringVar(value="%.3f" % MTHR_NEAR)      # 放松A：舞蹈状态内的运动量门槛
    v_leg = tk.StringVar(value="%.3f" % MTHR_LEG)        # 放松B：露出腿时的门槛
    v_workers = tk.StringVar(value="1" if is_onefile() else
                             str(max(1, min(6, (os.cpu_count() or 4) - 1))))
    hint_workers = ("单文件版建议 1（>1 会拖慢）" if is_onefile()
                    else "识别速度，1=单进程最稳")
    for i, (lab, var, hint) in enumerate([
            ("采样间隔(秒)", v_step, "每多少秒抽一帧"),
            ("中间可停留(秒)", v_bridge, "两段舞蹈间隔小于该值算同一段"),
            ("最短舞蹈(秒)", v_min, "短于该值的舞蹈整段跳过"),
            ("舞蹈内门槛", v_keep, "处于舞蹈状态时，运动量门槛降低"),
            ("露腿门槛", v_leg, "露出腿时再降一档，更容易判为跳舞"),
            ("并行进程", v_workers, hint_workers)]):
        r0, c0 = divmod(i, 3)
        ttk.Label(opt, text=lab).grid(row=r0 * 2, column=(c0 * 2) % 6, sticky="e", padx=(10, 4), pady=(8, 2))
        ttk.Entry(opt, textvariable=var, width=7).grid(row=r0 * 2, column=(c0 * 2) % 6 + 1, sticky="w")
        ttk.Label(opt, text=hint, foreground="#6b7a90").grid(
            row=r0 * 2 + 1, column=(c0 * 2) % 6, columnspan=2, sticky="w", padx=(10, 12), pady=(0, 4))

    bar = ttk.Frame(root)
    bar.pack(fill="x", **pad)
    b_run = ttk.Button(bar, text="开始", width=14)
    b_run.pack(side="left")
    b_stop = ttk.Button(bar, text="停止", width=10, state="disabled")
    b_stop.pack(side="left", padx=8)
    b_open = ttk.Button(bar, text="打开输出目录", width=16, state="disabled")
    b_open.pack(side="left", padx=8)
    pb = ttk.Progressbar(bar, mode="determinate", maximum=100)
    pb.pack(side="left", fill="x", expand=True, padx=10)
    v_pct = tk.StringVar(value="就绪")
    ttk.Label(bar, textvariable=v_pct, width=10).pack(side="left")

    logbox = ttk.LabelFrame(root, text="运行日志")
    logbox.pack(fill="both", expand=True, **pad)
    txt = tk.Text(logbox, wrap="none", height=18, font=("Consolas", 9))
    sb = ttk.Scrollbar(logbox, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    txt.pack(fill="both", expand=True, side="left")

    def _log(msg):
        txt.insert("end", str(msg) + "\n")
        txt.see("end")

    def _progress(v):
        pb["value"] = max(0, min(100, v * 100))
        v_pct.set("%.0f%%" % (v * 100))

    def _pick_file():
        p = filedialog.askopenfilename(title="选择视频",
                                       filetypes=[("视频文件", "*.mp4 *.mkv *.flv *.mov *.avi *.ts"),
                                                  ("所有文件", "*.*")])
        if p:
            v_video.set(p)
            if not v_out.get():
                v_out.set(os.path.join(os.path.dirname(p), "dance_out"))

    def _pick_folder():
        p = filedialog.askdirectory(title="选择要批量处理的文件夹")
        if p:
            v_video.set(p)
            if v_skipdone.get():
                n = len(scan_videos(p, recursive=v_recursive.get(), skip_done=True))
                _log("已选择文件夹：%s" % p)
                _log("   其中待处理视频 %d 个%s（已处理过的会自动跳过）"
                     % (n, "（含子文件夹）" if v_recursive.get() else ""))
                if n:
                    _log("   点「开始」即可批量处理。")

    def _pick_dir():
        p = filedialog.askdirectory(title="选择目录")
        if p:
            v_out.set(p)

    def _ui(enable):
        state["running"] = not enable
        b_run["state"] = "normal" if enable else "disabled"
        b_stop["state"] = "disabled" if enable else "normal"
        b_open["state"] = "disabled" if enable else "normal"

    def _worker():
        try:
            step = float(v_step.get()); bridge = float(v_bridge.get()); mind = float(v_min.get())
            wk = int(float(v_workers.get()))
            mk = float(v_keep.get()); ml = float(v_leg.get())
            sk = bool(v_skel.get()); sko = bool(v_skel_only.get())
            tsn = bool(v_tsname.get()); bts = bool(v_burnts.get())
            batch = v_mode.get() == "batch"
            if batch:
                inp = v_video.get().strip().strip('"')
                out_root = v_out.get().strip().strip('"') or None
                state["outdir"] = out_root or inp
                results = run_batch(inp, step, bridge, mind, wk,
                                    log=lambda m: root.after(0, _log, m),
                                    progress=lambda v: root.after(0, _progress, v),
                                    cancel=state["cancel"],
                                    recursive=v_recursive.get(),
                                    skip_done=v_skipdone.get(),
                                    out_root=out_root,
                                    mthr_keep=mk, mthr_leg=ml,
                                    skeleton=sk, skel_only_dance=sko,
                                    ts_name=tsn, burn_ts=bts)
                root.after(0, _log, "")
                root.after(0, _log, ">>> 批量完成，共处理 %d 个视频。" % len(results))
            else:
                video = v_video.get().strip().strip('"')
                out = v_out.get().strip().strip('"') or os.path.join(os.path.dirname(video), "dance_out")
                state["outdir"] = out
                run_pipeline(video, out, step, bridge, mind, wk,
                             log=lambda m: root.after(0, _log, m),
                             progress=lambda v: root.after(0, _progress, v),
                             cancel=state["cancel"],
                             mthr_keep=mk, mthr_leg=ml,
                             skeleton=sk, skel_only_dance=sko,
                             ts_name=tsn, burn_ts=bts)
                root.after(0, _log, "")
                root.after(0, _log, ">>> 全部完成，输出目录：%s" % out)
        except InterruptedError:
            root.after(0, _log, ">>> 已停止")
        except Exception as e:
            root.after(0, _log, ">>> 出错：%s" % e)
            root.after(0, _log, traceback.format_exc())
        finally:
            root.after(0, _ui, True)

    def _start():
        p = v_video.get().strip().strip('"')
        batch = v_mode.get() == "batch"
        if batch:
            if not p or not os.path.isdir(p):
                messagebox.showerror("提示", "请选择有效的视频文件夹")
                return
            n = len(scan_videos(p, recursive=v_recursive.get(), skip_done=v_skipdone.get()))
            if n == 0:
                messagebox.showinfo("提示", "该文件夹里没有找到待处理的视频。\n"
                                            "（已处理过的会被跳过；可取消勾选「跳过已处理过的视频」重跑）")
                return
        else:
            if not p or not os.path.isfile(p):
                messagebox.showerror("提示", "请选择有效的视频文件")
                return
            if not v_out.get().strip():
                v_out.set(os.path.join(os.path.dirname(p), "dance_out"))
        state["cancel"].clear()
        txt.delete("1.0", "end")
        _ui(False)
        threading.Thread(target=_worker, daemon=True).start()


    def _stop():
        state["cancel"].set()
        _log(">>> 正在停止…")

    b_run.configure(command=_start)
    b_stop.configure(command=_stop)
    b_open.configure(command=lambda: os.startfile(state["outdir"]) if state["outdir"] else None)

    root.after(200, lambda: _log(
        "就绪。默认是「批量处理文件夹」：选一个文件夹后点「开始」，"
        "里面每个视频都会在自己旁边新建一个同名专用文件夹放结果。") if not initial_video else None)
    root.mainloop()


# ---------------------------------------------------------------- 自检 / CLI

def self_test(outdir=None):
    """不跑长视频的自检：ffmpeg / 模型 / 抽帧 / 姿态 / 运动 / 合成 / tkinter"""
    import numpy as np
    import cv2
    import tkinter
    res = []
    outdir = outdir or tempfile.mkdtemp(prefix="dancecut_selftest_")
    os.makedirs(outdir, exist_ok=True)

    ff = ffmpeg_path()
    r = run_ffmpeg(["-version"])
    ok = b"ffmpeg version" in (r.stdout or b"")
    res.append(("ffmpeg", ok, ff + " | " + (r.stdout or b"")[:40].decode("utf-8", "replace")))

    for name in ("pose_landmarker_full.task", "hand_landmarker.task"):
        p = resource(name)
        res.append(("模型 " + name, bool(p and os.path.getsize(p) > 100000),
                    (p or "未找到") + " (%s bytes)" % (os.path.getsize(p) if p else 0)))

    # 造一段 12 秒测试视频
    src = os.path.join(outdir, "selftest.mp4")
    r = run_ffmpeg(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=640x360:rate=30:duration=12",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", src])
    res.append(("生成测试视频", os.path.isfile(src), src))

    if os.path.isfile(src):
        fdir = os.path.join(outdir, "frames")
        n = extract_frames(src, fdir, 3.0, lambda m: None, lambda v: None)
        res.append(("抽帧", n == 4, "n=%d（期望 4）" % n))
        model = resource("pose_landmarker_full.task")
        rows = pose_all(fdir, n, model, 3.0, 1, lambda m: None, lambda v: None,
                        threading.Event())
        res.append(("姿态推理", len(rows) == n, "rows=%d" % len(rows)))
        mo = motion_all(fdir, n, 3.0, lambda m: None, lambda v: None)
        rows = classify(rows, mo, 3.0)
        res.append(("判定分类", all("dance" in r for r in rows),
                    "dance=%d" % sum(r["dance"] for r in rows)))
        keep, info = build_segments(rows, 3.0, 27.0, 24.0, 12.0)
        res.append(("片段合并", isinstance(keep, list), "blocks=%d" % len(info)))
        # 骨骼画面输出（故意用中文目录名，覆盖"非 ASCII 路径写图"这个坑）
        try:
            skdir = os.path.join(outdir, "骨骼画面测试")
            n_sk = skeleton_pass(rows, fdir, skdir, resource("hand_landmarker.task"),
                                 lambda m: None, lambda v: None, threading.Event())
            fs = [f for f in os.listdir(skdir) if f.endswith(".jpg") and not f.startswith("_")]
            res.append(("骨骼画面输出(中文路径)", len(fs) == n and n_sk == n,
                        "%d 张 -> %s" % (n_sk, skdir)))
        except Exception as e:
            res.append(("骨骼画面输出(中文路径)", False, str(e)))

        # 片段标注：文件名带原片时间 + 画面叠加时间码
        try:
            od = os.path.join(outdir, "标注测试")
            os.makedirs(od, exist_ok=True)
            src2 = os.path.join(outdir, "ts_src.mp4")
            run_ffmpeg(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc=size=320x180:rate=30:duration=21",
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", src2])
            nm = "dance_01_%s-%s" % (hms_compact(12), hms_compact(21))
            dst = os.path.join(od, nm + ".mp4")
            vf = timecode_vf(12.0, 21.0, 1, 180)
            okc = cut_clip(src2, 12.0, 9.0, dst, vf=vf)
            res.append(("文件名带原片时间", os.path.basename(dst) == "dance_01_000012-000021.mp4"
                        and okc, os.path.basename(dst)))
            fr = os.path.join(od, "chk.jpg")
            run_ffmpeg(["-hide_banner", "-loglevel", "error", "-y", "-ss", "1",
                        "-i", dst, "-frames:v", "1", fr])
            im = imread_u(fr)
            drew = False
            if im is not None:
                roi = im[8:44, 8:250]
                drew = bool(roi.min() < 40 and roi.max() > 200)   # 有暗底 + 亮字
            res.append(("画面叠加时间码已画出", drew, "字体=%s" % (find_font() or "未找到")))
        except Exception as e:
            res.append(("片段标注", False, str(e)))

    # 两条放松规则：合成数据验证（舞蹈内降门槛 / 露腿更容易判为跳舞）
    try:
        def mk(i, kne=0.0, ank=0.0):
            return dict(det=1, sh_y=0.35, sh_w=0.15, torso=0.4,
                        up_l=0.2, up_r=0.1, lat_l=0.3, lat_r=0.3,
                        elb_l=90.0, elb_r=90.0, wv_l=0.9, wv_r=0.9,
                        x15=0.40 + i * 0.004, y15=0.30, x16=0.60, y16=0.30,
                        vis_kne=kne, vis_ank=ank)

        rows2 = [mk(i) for i in range(11)]
        mos2 = [0.25 if i < 5 else 0.08 for i in range(11)]
        rr = classify(rows2, mos2, 3.0)
        ok_a1 = (rr[6]["base"] == 0 and rr[6]["near_anchor"] == 1 and rr[6]["dance_raw"] == 1)
        ok_a2 = (rr[9]["near_anchor"] == 0 and rr[9]["dance_raw"] == 0)
        res.append(("放松A 舞蹈状态内降门槛", bool(ok_a1 and ok_a2),
                    "邻域内(0.08>%.3f)=%d 邻域外(0.08<%.3f)=%d"
                    % (rr[6]["m_thr"], rr[6]["dance_raw"], rr[9]["m_thr"], rr[9]["dance_raw"])))

        rows3 = [mk(i, kne=0.8 if i >= 5 else 0.0, ank=0.4 if i >= 5 else 0.0) for i in range(11)]
        r3 = classify(rows3, mos2, 3.0)
        ok_b = (r3[6]["leg_vis"] == 1 and r3[6]["relax_level"] == "露腿+舞蹈邻域"
                and abs(r3[6]["m_thr"] - MTHR_LEG) < 1e-9 and r3[6]["dance_raw"] == 1)
        res.append(("放松B 露腿更容易判为跳舞", bool(ok_b),
                    "露腿帧门槛 %.3f（邻域 %.3f / 其余 %.3f），判定=%d"
                    % (r3[6]["m_thr"], MTHR_NEAR, MTHR, r3[6]["dance_raw"])))
    except Exception as e:
        res.append(("两条放松规则", False, str(e)))

    # 中文 / 非 ASCII 路径读图：cv2.imread 在 Windows 上会静默失败，必须走 imread_u
    try:
        cdir = os.path.join(outdir, "中文路径测试")
        os.makedirs(cdir, exist_ok=True)
        cp = os.path.join(cdir, "帧 001.jpg")
        run_ffmpeg(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=160x90:rate=1:duration=1", "-frames:v", "1", cp])
        ok = os.path.isfile(cp) and imread_u(cp) is not None and imread_gray_u(cp) is not None
        res.append(("中文路径读图", ok, cp))
    except Exception as e:
        res.append(("中文路径读图", False, str(e)))

    # 批量模式：造一个含 2 个视频的文件夹（外加 1 个在子文件夹里），跑一遍批量
    try:
        bdir = os.path.join(outdir, "batch_in")
        sub = os.path.join(bdir, "sub")
        os.makedirs(sub, exist_ok=True)
        for name, d in (("clipA.mp4", bdir), ("clipB.mp4", bdir), ("clipC.mp4", sub)):
            run_ffmpeg(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc=size=320x180:rate=30:duration=9",
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
                        os.path.join(d, name)])
        os.makedirs(os.path.join(bdir, "not_a_video"), exist_ok=True)
        open(os.path.join(bdir, "readme.txt"), "w").write("x")
        n_flat = len(scan_videos(bdir, recursive=False))
        n_recur = len(scan_videos(bdir, recursive=True))
        res.append(("批量扫描", n_flat == 2 and n_recur == 3,
                    "顶层 %d 个（期望 2）、含子文件夹 %d 个（期望 3）" % (n_flat, n_recur)))
        rs = run_batch(bdir, 3.0, 27.0, 24.0, 1,
                       lambda m: None, lambda v: None, threading.Event(),
                       recursive=False, skip_done=True)
        made = [r["outdir"] for r in rs]
        ok_named = sorted(os.path.basename(p) for p in made) == ["clipA", "clipB"]
        ok_dirs = all(os.path.isfile(os.path.join(p, "run_summary.json")) for p in made)
        res.append(("批量新建专用文件夹", len(rs) == 2 and ok_named and ok_dirs,
                    "; ".join(os.path.basename(p) + "/" for p in made)))
        again = scan_videos(bdir, recursive=False, skip_done=True)
        res.append(("跳过已处理", len(again) == 0, "重扫剩 %d 个（期望 0）" % len(again)))
    except Exception as e:
        res.append(("批量模式", False, str(e)))

    # 真实素材端到端（可选：给一个短视频）
    e2e = os.environ.get("DANCECUT_E2E")
    if e2e and os.path.isfile(e2e):
        od = os.path.join(outdir, "e2e")
        clips = run_pipeline(e2e, od, 3.0, 27.0, 24.0, 1,
                             lambda m: None, lambda v: None, threading.Event())
        res.append(("端到端", True, "clips=%d" % len(clips)))

    try:
        tk = tkinter.Tk(); tk.withdraw()
        ok = tk.winfo_exists() == 1
        tk.destroy()
        res.append(("tkinter 窗口", ok, "TkVersion=%s" % tkinter.TkVersion))
    except Exception as e:
        res.append(("tkinter 窗口", False, str(e)))

    lines = ["%s  %s  %s" % ("PASS" if ok else "FAIL", name, detail)
             for name, ok, detail in res]
    body = "\n".join(lines) + "\n"
    # --windowed 的 exe 没有控制台，结果必须落盘才能看到
    try:
        dd = os.path.join(base_dir(), "logs")
        os.makedirs(dd, exist_ok=True)
        dump = os.path.join(dd, "selftest_result.txt")
        open(dump, "w", encoding="utf-8").write(body)
    except Exception:
        dump = os.path.join(outdir, "selftest_result.txt")
        open(dump, "w", encoding="utf-8").write(body)
    try:
        open(os.path.join(outdir, "selftest_result.txt"), "w", encoding="utf-8").write(body)
    except Exception:
        pass
    return dump, all(ok for _, ok, _ in res), lines


def main_cli(argv):
    video = None; batch = None; out = None
    step, bridge, mind, workers = STEP_DEF, BRIDGE_DEF, MIN_DANCE_DEF, 1
    keep_thr, leg_thr = MTHR_NEAR, MTHR_LEG
    recursive, skip_done = False, True
    skeleton, skel_only = False, False
    ts_name, burn_ts = True, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--video": video = argv[i + 1]; i += 2
        elif a == "--batch": batch = argv[i + 1]; i += 2
        elif a == "--out": out = argv[i + 1]; i += 2
        elif a == "--step": step = float(argv[i + 1]); i += 2
        elif a == "--bridge": bridge = float(argv[i + 1]); i += 2
        elif a == "--min-dance": mind = float(argv[i + 1]); i += 2
        elif a == "--workers": workers = int(argv[i + 1]); i += 2
        elif a == "--keep-thr": keep_thr = float(argv[i + 1]); i += 2
        elif a == "--leg-thr": leg_thr = float(argv[i + 1]); i += 2
        elif a == "--recursive": recursive = True; i += 1
        elif a == "--no-skip": skip_done = False; i += 1
        elif a == "--skeleton": skeleton = True; i += 1
        elif a == "--skeleton-only-dance": skeleton = True; skel_only = True; i += 1
        elif a == "--no-ts-name": ts_name = False; i += 1
        elif a == "--burn-ts": burn_ts = True; i += 1
        else: i += 1
    log = log_factory(None)
    if batch:
        run_batch(batch, step, bridge, mind, workers, log, lambda v: None,
                  threading.Event(), recursive=recursive, skip_done=skip_done,
                  out_root=out, mthr_keep=keep_thr, mthr_leg=leg_thr,
                  skeleton=skeleton, skel_only_dance=skel_only,
                  ts_name=ts_name, burn_ts=burn_ts)
    else:
        out = out or os.path.join(os.path.dirname(video), "dance_out")
        run_pipeline(video, out, step, bridge, mind, workers,
                     log, lambda v: None, threading.Event(),
                     mthr_keep=keep_thr, mthr_leg=leg_thr,
                     skeleton=skeleton, skel_only_dance=skel_only,
                     ts_name=ts_name, burn_ts=burn_ts)
    return 0


def main():
    ensure_std_streams()
    args = sys.argv[1:]
    if "--self-test" in args:
        dump, ok, lines = self_test()
        for l in lines:
            print(l, flush=True)
        print("\nSELFTEST %s -> %s" % ("PASS" if ok else "FAIL", dump), flush=True)
        return 0 if ok else 1
    if "--cli" in args:
        return main_cli(args)
    init = ""
    for flag in ("--video", "--batch"):
        if flag in args:
            init = args[args.index(flag) + 1]
            break
    launch_gui(init)
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # 冻结后多进程必需
    sys.exit(main())
