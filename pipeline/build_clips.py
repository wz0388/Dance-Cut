# DanceCut - 舞蹈片段自动识别与切片  |  PolyForm Noncommercial 1.0.0（禁止商用，详见 LICENSE）
# Copyright (c) 2026 wz0388 <https://github.com/wz0388>
# -*- coding: utf-8 -*-
"""
Stage G — 按最终剪辑规则重新生成片段（在 3 秒采样网格上执行）

规则：
  R1 保持原始画面，三屏复制版式不裁两侧。
  R2 片段起点 = 上一张「非舞蹈」采样图的时间点；终点同理取下一张非舞蹈采样图
     （保证舞蹈首尾任何一个采样帧都被完整包住，不被切掉）。
  R3 舞蹈中间的停留：两段舞蹈之间间隔 ≤ 15 秒视为同一段，不拆开。
  R4 舞蹈本身时长 < 9 秒的整段丢弃。
"""
import os, subprocess, numpy as np, pandas as pd, json

ROOT = os.environ.get("DANCECUT_ROOT") or os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("DANCECUT_VIDEO", "")   # 源视频路径：设环境变量 DANCECUT_VIDEO
CLIPS = os.path.join(ROOT, "clips")
STEP = 3.0
BRIDGE_S = 27.0     # R3 中间可停留时间
MIN_DANCE_S = 24.0  # R4 最短舞蹈时长
VIDEO_END = float(os.environ.get("DANCECUT_VIDEO_END", "0") or 0)  # 0 = 按时间轴末尾处理

os.makedirs(CLIPS, exist_ok=True)

d = pd.read_csv(os.path.join(ROOT, "dance_result.csv")).sort_values("frame").reset_index(drop=True)
if "dance_auto" in d.columns:      # 重复运行时取回未过滤的骨架判定
    d["dance"] = d["dance_auto"]
d["t"] = d.frame * STEP           # 采样图时间点 = frame*3
d["idx"] = d.frame

# ---------- 0) 套用人工复核结论（dance_candidates.csv 的 review 列） ----------
# 视频后半段机位更宽，坐姿的肩高/肩宽与站立景重合，骨架特征会误判；
# 这些候选段已逐帧目视核对确认为坐姿聊天，此处按复核结论剔除。
cand = pd.read_csv(os.path.join(ROOT, "dance_candidates.csv"))
reviewed = np.zeros(len(d), dtype=bool)
for r in cand.itertuples():
    if r.review == "dance":
        reviewed[(d.t >= r.start_s) & (d.t <= r.end_s)] = True
excluded = int(((d.dance == 1) & ~reviewed).sum())
print("骨架判定舞蹈帧 %d，其中复核排除 %d 帧（坐姿误判），保留 %d 帧"
      % (int(d.dance.sum()), excluded, int(((d.dance == 1) & reviewed).sum())))
d["dance_auto"] = d.dance
d["dance"] = ((d.dance == 1) & reviewed).astype(int)
d["review"] = np.where(reviewed, "dance", "excluded:seated")

# ---------- 1) 原始连续舞蹈段（网格上的连续 run） ----------
idx = np.where(d.dance.values == 1)[0]
runs, s, p = [], idx[0], idx[0]
for i in idx[1:]:
    if i == p + 1:
        p = i
    else:
        runs.append([s, p]); s = p = i
runs.append([s, p])
print("原始连续舞蹈段: %d 个" % len(runs))
for r in runs:
    print("   frame %4d-%4d  t %7.1f-%7.1f  (%.0fs, %d 帧)" % (r[0], r[1], r[0] * STEP, r[1] * STEP,
                                                               (r[1] - r[0] + 1) * STEP, r[1] - r[0] + 1))

# ---------- 2) R3 合并：中间停留 ≤ 15 秒 ----------
blocks = []
for r in runs:
    if blocks and (r[0] - blocks[-1][1] - 1) * STEP <= BRIDGE_S:
        blocks[-1][1] = r[1]
        blocks[-1][2].append(r)
    else:
        blocks.append([r[0], r[1], [r]])
print("\n合并（停留≤%.0fs）后: %d 个区块" % (BRIDGE_S, len(blocks)))

# ---------- 3) R4 过滤 + R2 边界 ----------
recs, keep = [], []
for bi, (a, b, parts) in enumerate(blocks, 1):
    dance_span = (b - a + 1) * STEP
    dsamples = sum(p[1] - p[0] + 1 for p in parts)
    drop = dance_span < MIN_DANCE_S
    start_t = max(0.0, (a - 1) * STEP)          # R2 上一张非舞蹈图
    end_t = min(VIDEO_END, (b + 1) * STEP)      # R2 下一张非舞蹈图
    recs.append(dict(block=bi, first_dance_frame=int(d.frame[a]), last_dance_frame=int(d.frame[b]),
                     dance_span_s=round(dance_span, 1), dance_samples=dsamples,
                     n_parts=len(parts),
                     clip_start_s=round(start_t, 2), clip_end_s=round(end_t, 2),
                     clip_dur_s=round(end_t - start_t, 2),
                     kept=("否(<%.0fs)" % MIN_DANCE_S if drop else "是")))
    if not drop:
        keep.append((bi, start_t, end_t, parts))
for r in recs:
    print("  块%2d 舞蹈 %5.1fs (%d帧/%d小段)  片段 %8.2f-%8.2f (%.1fs)  -> %s"
          % (r["block"], r["dance_span_s"], r["dance_samples"], r["n_parts"],
             r["clip_start_s"], r["clip_end_s"], r["clip_dur_s"], r["kept"]))
pd.DataFrame(recs).to_csv(os.path.join(ROOT, "dance_blocks_analysis.csv"), index=False, encoding="utf-8-sig")
json.dump(dict(bridge_s=BRIDGE_S, min_dance_s=MIN_DANCE_S, step_s=STEP,
               n_blocks=len(blocks), n_kept=len(keep), n_dropped=len(blocks) - len(keep)),
          open(os.path.join(ROOT, "clip_rules.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- 4) 逐帧归属表 ----------
# 片段覆盖的时间区间为 [fa*3, fb*3)，对应采样帧 fa .. fb-1
role = np.array(["none"] * len(d), dtype=object)
clipid = np.zeros(len(d), dtype=int)
for k, (bi, st, en, parts) in enumerate(keep, 1):
    a, b = parts[0][0], parts[-1][1]
    fa, fb = int(round(st / STEP)), min(len(d), int(round(en / STEP)))
    role[fa:a] = "pre_pad"                     # R2 起点回退的那一张（上一张非舞蹈图）
    for p0, p1 in parts:
        role[p0:p1 + 1] = "dance"              # 判定为舞蹈的采样帧
    for (q0, q1), (r0, r1) in zip(parts, parts[1:]):
        role[q1 + 1:r0] = "bridge"             # R3 允许的中间停留
    clipid[fa:fb] = k
d["clip_id"], d["clip_role"] = clipid, role
d.to_csv(os.path.join(ROOT, "dance_result.csv"), index=False, encoding="utf-8-sig")
print("\n片段内帧构成：", dict(pd.Series(role).value_counts()))
print("clip_id 覆盖帧数：", d[d.clip_id > 0].groupby("clip_id").size().to_dict())

# ---------- 5) 切片（不裁两侧，保持原始画面） ----------
def hms(x):
    x = float(x); return "%02d:%02d:%05.2f" % (int(x // 3600), int((x % 3600) // 60), x % 60)


for f in os.listdir(CLIPS):
    if f.endswith(".mp4"):
        os.remove(os.path.join(CLIPS, f))

rows = []
for k, (bi, st, en, parts) in enumerate(keep, 1):
    a, b = parts[0][0], parts[-1][1]
    dst = os.path.join(CLIPS, "dance_%02d.mp4" % k)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", "%.3f" % st, "-i", SRC, "-t", "%.3f" % (en - st),
                    "-vf", "fps=30,format=yuv420p",
                    "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", dst], check=True)
    print("clip %2d  %s - %s  (%.1fs)" % (k, hms(st), hms(en), en - st), flush=True)
    rows.append(dict(clip=k, start_s=round(st, 2), end_s=round(en, 2), dur_s=round(en - st, 2),
                     start_tc=hms(st), end_tc=hms(en), dance_span_s=round((b - a + 1) * STEP, 1),
                     n_parts=len(parts),
                     first_dance_frame=int(d.frame[a]), last_dance_frame=int(d.frame[b]),
                     start_frame=int(round(st / STEP)), end_frame=int(round(en / STEP))))

clips = pd.DataFrame(rows)
clips.to_csv(os.path.join(ROOT, "dance_clips.csv"), index=False, encoding="utf-8-sig")

lst = os.path.join(CLIPS, "concat.txt")
with open(lst, "w", encoding="utf-8") as f:
    for r in rows:
        f.write("file 'dance_%02d.mp4'\n" % r["clip"])
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                "-i", lst, "-c", "copy", os.path.join(CLIPS, "ALL_DANCE.mp4")], check=True)

print("\n最终 %d 段，合计 %.1f 分钟" % (len(clips), clips.dur_s.sum() / 60))
print(clips[["clip", "start_tc", "end_tc", "dur_s", "dance_span_s"]].to_string(index=False))
