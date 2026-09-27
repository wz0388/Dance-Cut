# DanceCut - 舞蹈片段自动识别与切片  |  PolyForm Noncommercial 1.0.0（禁止商用，详见 LICENSE）
# Copyright (c) 2026 wz0388 <https://github.com/wz0388>
# -*- coding: utf-8 -*-
"""
FINAL stage A — 逐帧判定 + 片段生成
输出:
  dance_result.csv          逐帧判定表（是否站立 / 手部是否舞蹈状态 / 各项指标）
  dance_candidates.csv      自动检出的候选片段
  dance_segments_final.csv  复核后的最终舞蹈片段（含合并区块）
"""
import numpy as np, pandas as pd, os

ROOT = os.environ.get("DANCECUT_ROOT") or os.path.dirname(os.path.abspath(__file__))
SHY, SHW, DWR, MTHR = 0.44, 0.27, 0.45, 0.09
MERGE_GAP = 15.0   # 相邻片段间隔 < 15s 合并为一个连续舞蹈区块


def hms(sec):
    sec = float(sec)
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return "%02d:%02d:%05.2f" % (h, m, s)


d = pd.read_csv(os.path.join(ROOT, "merged.csv")).sort_values("frame").reset_index(drop=True)
tr = d.torso.replace(0, np.nan)
d["d_sh"] = np.sqrt(d.sh_x.diff() ** 2 + d.sh_y.diff() ** 2) / tr
d["d_hip"] = np.sqrt(d.hip_x.diff() ** 2 + d.hip_y.diff() ** 2) / tr
d["d_wrist_l"] = np.sqrt(d.x15.diff() ** 2 + d.y15.diff() ** 2) / tr
d["d_wrist_r"] = np.sqrt(d.x16.diff() ** 2 + d.y16.diff() ** 2) / tr
d["d_wrist"] = d[["d_wrist_l", "d_wrist_r"]].max(axis=1)
d["up_best"] = d[["up_l", "up_r"]].max(axis=1)
d["lat_best"] = d[["lat_l", "lat_r"]].max(axis=1)
d["wv_best"] = d[["wv_l", "wv_r"]].max(axis=1)
d["elb_min"] = d[["elb_l", "elb_r"]].min(axis=1)

# --- 判据 1: 人体骨骼状态 = 站立 ---
d["is_standing"] = ((d.sh_y < SHY) & (d.sh_w < SHW)).astype(int)
# --- 判据 2: 手部骨骼状态 = 舞蹈 ---
arm_vis = d.wv_best.fillna(0) > 0.50
arm_pose = ((d.up_best > -0.40) | (d.lat_best > 0.45) | ((d.elb_min < 115) & (d.up_best > -0.55))) & arm_vis
d["hand_pose"] = arm_pose.astype(int)
d["hand_move"] = (d.d_wrist.fillna(0) > DWR).astype(int)
d["hand_dance"] = ((d.hand_pose == 1) | (d.hand_move == 1)).astype(int)
# --- 判据 3: 持续运动 ---
d["m_roll3"] = d.m.rolling(3, center=True, min_periods=1).mean()
d["active"] = (d.m_roll3.fillna(0) > MTHR).astype(int)

d["hand_state"] = np.where(d.hand_pose == 1, "舞蹈姿态",
                    np.where(d.hand_move == 1, "手部运动", "静止"))
d["body_state"] = np.where(d.is_standing == 1, "站立", "非站立")

raw = ((d.is_standing == 1) & (d.hand_dance == 1) & (d.active == 1)).astype(float)
d["s_dance"] = np.round(np.convolve(raw.values, np.ones(3) / 3, mode="same"), 3)
d["dance"] = (d.s_dance >= 0.34).astype(int)
d["confidence"] = np.where(d.dance == 1,
                           np.clip(0.55 * d.s_dance + 0.25 * d.is_standing + 0.20 * d.hand_dance, 0, 1).round(3), 0.0)

cols = ["frame", "t", "det", "n_pose", "sh_x", "sh_y", "sh_w", "hip_x", "hip_y", "torso",
        "torso_up", "vis_span", "vis_ank", "vis_kne", "leg_ratio", "ank_below_hip",
        "up_l", "up_r", "up_best", "lat_l", "lat_r", "lat_best", "elb_l", "elb_r", "elb_min",
        "armlen_l", "armlen_r", "wv_l", "wv_r", "wv_best", "d_wrist_l", "d_wrist_r", "d_wrist",
        "d_sh", "d_hip", "m", "m_c", "m_roll3", "body_state", "is_standing",
        "hand_state", "hand_pose", "hand_move", "hand_dance", "active",
        "s_dance", "dance", "confidence"]
d[cols].to_csv(os.path.join(ROOT, "dance_result.csv"), index=False, encoding="utf-8-sig")


def segs(mask, gap=1, minlen=2):
    idx = np.where(mask.values)[0]
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p <= gap + 1:
            p = i
        else:
            out.append((s, p)); s = p = i
    out.append((s, p))
    return [(a, b) for a, b in out if b - a + 1 >= minlen]


cand = []
for a, b in segs(d.dance):
    g = d.iloc[a:b + 1]
    cand.append(dict(start_s=round(float(g.t.iloc[0]) - 1.5, 1), end_s=round(float(g.t.iloc[-1]) + 1.5, 1),
                     start_frame=int(g.frame.iloc[0]), end_frame=int(g.frame.iloc[-1]), n=len(g),
                     dur_s=round(len(g) * 3.0, 1),
                     stand=round(float(g.is_standing.mean()), 2),
                     hand_pose=round(float(g.hand_pose.mean()), 2),
                     hand_move=round(float(g.hand_move.mean()), 2),
                     motion=round(float(g.m.mean()), 4), conf=round(float(g.confidence.mean()), 3)))
cdf = pd.DataFrame(cand)
# 复核结论: 视频后半段（t > 5700s）经逐帧目视核对全部为坐姿聊天，非舞蹈
cdf["review"] = np.where(cdf.start_s <= 5700, "dance", "rejected:seated")
cdf.to_csv(os.path.join(ROOT, "dance_candidates.csv"), index=False, encoding="utf-8-sig")

# --- 合并为连续舞蹈区块 ---
kept = cdf[cdf.review == "dance"].reset_index(drop=True)
blocks, cur = [], None
for _, r in kept.iterrows():
    if cur is None:
        cur = [r.start_s, r.end_s, 1, r.conf, r.motion]
    elif r.start_s - cur[1] <= MERGE_GAP:
        cur[1] = r.end_s; cur[2] += 1
        cur[3] = (cur[3] * (cur[2] - 1) + r.conf) / cur[2]
        cur[4] = (cur[4] * (cur[2] - 1) + r.motion) / cur[2]
    else:
        blocks.append(cur); cur = [r.start_s, r.end_s, 1, r.conf, r.motion]
if cur:
    blocks.append(cur)
bd = pd.DataFrame([dict(block=i + 1, start_s=b[0], end_s=b[1], dur_s=round(b[1] - b[0], 1),
                        parts=b[2], conf=round(b[3], 3), motion=round(b[4], 4),
                        start_tc=hms(b[0]), end_tc=hms(b[1])) for i, b in enumerate(blocks)])
bd.to_csv(os.path.join(ROOT, "dance_segments_final.csv"), index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print("站立帧 %d | 手部舞蹈帧 %d | 判定舞蹈帧 %d / %d" % (d.is_standing.sum(), d.hand_dance.sum(), d.dance.sum(), len(d)))
print("\n=== 候选片段 %d 个（其中复核为舞蹈 %d 个）===" % (len(cdf), (cdf.review == "dance").sum()))
print(cdf.to_string(index=False))
print("\n=== 最终舞蹈区块 %d 个 ===" % len(bd))
print(bd.to_string(index=False))
print("\n合计舞蹈时长 %.1f 分钟" % (bd.dur_s.sum() / 60))
