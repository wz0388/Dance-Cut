# DanceCut - 舞蹈片段自动识别与切片  |  Apache-2.0（SPDX-License-Identifier: Apache-2.0，详见 LICENSE）
# Copyright (c) 2026 wz0388 <https://github.com/wz0388>
# -*- coding: utf-8 -*-
"""Stage E — 生成骨骼标注样张 + 自包含 HTML 报告。"""
import os, base64, json, math, numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

res = pd.read_csv(os.path.join(ROOT, "dance_result.csv"))
clips = pd.read_csv(os.path.join(ROOT, "dance_clips.csv"))
blocks_df = pd.read_csv(os.path.join(ROOT, "dance_blocks_analysis.csv"))
RULES = json.load(open(os.path.join(ROOT, "clip_rules.json"), encoding="utf-8"))
BRS = RULES["bridge_s"]; MDS = RULES["min_dance_s"]
hands = pd.read_csv(os.path.join(ROOT, "hand_landmarks_dance.csv"))
cand = pd.read_csv(os.path.join(ROOT, "dance_candidates.csv"))

pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ROOT, "models", "pose_landmarker_full.task")),
    running_mode=vision.RunningMode.IMAGE, num_poses=3,
    min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3))
CONN = vision.PoseLandmarksConnections.POSE_LANDMARKS
LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28


def crop_if_triple(bgr):
    """三屏复制版式 -> 取中间那一屏，避免骨骼标注跨屏杂乱。"""
    g = cv2.cvtColor(cv2.resize(bgr, (360, 202)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    w3 = g.shape[1] // 3
    if np.abs(g[:, :2 * w3] - g[:, w3:3 * w3]).mean() < 12.0:
        w = bgr.shape[1]
        return bgr[:, w // 3:2 * w // 3], True
    return bgr, False


def annotate(frame, tag, sub):
    raw = imread_u(os.path.join(FRAMES, "f_%05d.jpg" % frame))
    if raw is None:
        return None
    bgr, triple = crop_if_triple(raw)
    H, W = bgr.shape[:2]
    r = res[res.frame == frame].iloc[0]
    out = bgr.copy()
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pr = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if pr.pose_landmarks:
        best, bs = None, -9
        for lms in pr.pose_landmarks:
            v = float(np.mean([lms[i].visibility for i in (LSH, RSH, LHIP, RHIP)]))
            cx = float(np.mean([lms[i].x for i in range(33)]))
            s = v - 0.35 * abs(cx - 0.5)
            if s > bs:
                bs, best = s, lms
        lms = best
        pts = [(int(l.x * W), int(l.y * H)) for l in lms]
        for c in CONN:
            cv2.line(out, pts[c.start], pts[c.end], (0, 210, 0), 2)
        for i, p in enumerate(pts):
            if 0 <= p[0] < W and 0 <= p[1] < H:
                cv2.circle(out, p, 4, (0, 0, 255) if lms[i].visibility > .5 else (0, 140, 255), -1)
        for i in (LWR, RWR):
            cv2.circle(out, pts[i], 10, (255, 0, 255), 2)
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (W, 74), (0, 0, 0), -1)
    out = cv2.addWeighted(ov, 0.55, out, 0.45, 0)
    cv2.putText(out, tag, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
    cv2.putText(out, sub + ("  [三屏版式·已裁中屏]" if triple else ""), (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return cv2.resize(out, (430, 484))


# 正面样本：从判定为舞蹈的帧中均匀取 8 帧；反面样本取 8 帧坐姿帧
dframes = res[(res.dance == 1) & (res.m > 0.12)].frame.tolist()
step = max(1, len(dframes) // 8)
pos = [dframes[i] for i in range(0, len(dframes), step)][:8]
neg = [0, 60, 300, 600, 616, 2400, 3232, 4654]
poss = [annotate(f, "判定：站立 + 手部舞蹈", "frame %d   t=%d s" % (f, f * 3)) for f in pos]
negs = [annotate(f, "判定：非跳舞", "frame %d   t=%d s" % (f, f * 3)) for f in neg]


def grid(imgs, cols, path, title):
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return
    while len(imgs) % cols:
        imgs.append(np.zeros_like(imgs[0]))
    rows = [np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)]
    s = np.vstack(rows)
    cv2.imwrite(path, s, [cv2.IMWRITE_JPEG_QUALITY, 88])


grid(poss, 4, os.path.join(ROOT, "samples_dance.jpg"), "")
grid(negs, 4, os.path.join(ROOT, "samples_nondance.jpg"), "")


def b64(p):
    return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()


def b64png(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


tl_full = os.path.join(ROOT, "timeline_full.png")
tl_zoom = os.path.join(ROOT, "timeline_zoom.png")

# ---- 按最终片段重绘时间轴 ----
tmin = res.t / 60.0
fig, axs = plt.subplots(4, 1, figsize=(24, 12), sharex=True)
axs[0].plot(tmin, res.sh_y, lw=.6, color="#2c7fb8"); axs[0].axhline(0.44, color="r", ls="--", lw=1)
axs[0].set_ylabel("肩中点高度 sh_y"); axs[0].invert_yaxis()
axs[1].plot(tmin, res.sh_w, lw=.6, color="#31a354"); axs[1].axhline(0.27, color="r", ls="--", lw=1)
axs[1].set_ylabel("肩宽 sh_w")
axs[2].plot(tmin, res.m, lw=.6, color="#e6550d", alpha=.55)
axs[2].plot(tmin, res.m_roll3, lw=1.1, color="#e6550d"); axs[2].axhline(0.09, color="r", ls="--", lw=1)
axs[2].set_ylabel("运动量 m / m³")
axs[3].fill_between(tmin, 0, res.s_dance, color="#d62728", alpha=.85); axs[3].set_ylabel("舞蹈判定得分")
axs[3].set_xlabel("时间 (分钟)")
for ax in axs:
    ax.grid(alpha=.25)
    for r in clips.itertuples():
        ax.axvspan(r.start_s / 60, r.end_s / 60, color="red", alpha=.13)
plt.suptitle("舞蹈片段检测 — 42111864478-1-192.mp4（全片 239.9 分钟）")
plt.tight_layout(); plt.savefig(tl_full, dpi=100); plt.close()

z = res[res.t <= 6000]
fig, axs = plt.subplots(3, 1, figsize=(24, 10), sharex=True)
axs[0].plot(z.t / 60, z.sh_y, lw=.8, color="#2c7fb8"); axs[0].axhline(0.44, color="r", ls="--"); axs[0].invert_yaxis()
axs[0].set_ylabel("肩中点高度 sh_y")
axs[1].plot(z.t / 60, z.m_roll3, lw=1.1, color="#e6550d"); axs[1].axhline(0.09, color="r", ls="--"); axs[1].set_ylabel("运动量 m³")
axs[2].fill_between(z.t / 60, 0, z.s_dance, color="#d62728", alpha=.85); axs[2].set_ylabel("舞蹈判定得分")
axs[2].set_xlabel("时间 (分钟)")
for ax in axs:
    ax.grid(alpha=.25)
    for r in clips[clips.start_s <= 6000].itertuples():
        ax.axvspan(r.start_s / 60, r.end_s / 60, color="red", alpha=.15)
plt.suptitle("放大：前 100 分钟（全部舞蹈内容所在区间）")
plt.tight_layout(); plt.savefig(tl_zoom, dpi=100); plt.close()



def fmt(sec):
    sec = float(sec); h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return "%02d:%02d:%05.2f" % (h, m, s)


seg_rows = "\n".join(
    "<tr><td class=c>%d</td><td class=mono>%s</td><td class=mono>%s</td><td class=mono>%.1f 秒</td>"
    "<td class=mono>%.1f 秒</td><td class=mono>dance_%02d.mp4</td></tr>"
    % (r.clip, fmt(r.start_s), fmt(r.end_s), r.dur_s, r.dance_span_s, r.clip)
    for r in clips.itertuples())

block_rows = "\n".join(
    "<tr><td>%d</td><td class=mono>%.0f 秒</td><td class=mono>%d</td><td class=mono>%d</td>"
    "<td class=mono>%s - %s</td><td class='%s'>%s</td></tr>"
    % (r.block, r.dance_span_s, r.dance_samples, r.n_parts,
       fmt(r.clip_start_s), fmt(r.clip_end_s),
       "ok" if r.kept == "是" else "no", ("保留 → 片段" if r.kept == "是" else "丢弃（舞蹈 < %.0f 秒）" % MDS))
    for r in blocks_df.itertuples())
blocks = list(blocks_df.itertuples())


cand_rows = "\n".join(
    "<tr><td>%d</td><td class=mono>%s - %s</td><td>%s</td><td>%s</td><td class=mono>%.3f</td>"
    "<td class=mono>%.3f</td><td class='%s'>%s</td></tr>"
    % (i + 1, fmt(r.start_s), fmt(r.end_s), "%d 帧(%.0f 秒)" % (r.n, r.dur_s),
       "%d%%" % round(r.stand * 100), r.conf, r.motion,
       "ok" if r.review == "dance" else "no", "舞蹈" if r.review == "dance" else "排除（坐姿）")
    for i, r in enumerate(cand.itertuples()))

okh = hands[hands.ok == 1]
open_ratio = float((okh.openness > 0.85).mean()) if len(okh) else 0
med_open = float(okh.palm_open.median()) if len(okh) else 0

html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>舞蹈片段识别报告 — 42111864478-1-192.mp4</title>
<style>
 :root{{--bg:#f6f8fb;--card:#fff;--ink:#16202f;--sub:#5b6b80;--line:#e2e8f0;--red:#e5484d;--blue:#2f6fed;--green:#1f9d63}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 "Microsoft YaHei",-apple-system,Segoe UI,sans-serif}}
 .wrap{{max-width:1180px;margin:0 auto;padding:32px 22px 70px}}
 h1{{font-size:27px;margin:0 0 6px}}
 h2{{font-size:19px;margin:38px 0 14px;padding-left:11px;border-left:4px solid var(--blue)}}
 .sub{{color:var(--sub);margin-bottom:22px}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin-bottom:18px;box-shadow:0 1px 3px rgba(16,32,64,.05)}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:8px}}
 .kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
 .kpi .v{{font-size:25px;font-weight:700;color:var(--blue)}}
 .kpi .k{{color:var(--sub);font-size:13px;margin-top:2px}}
 table{{border-collapse:collapse;width:100%;font-size:13.5px}}
 th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
 th{{background:#eef3fb;font-weight:600;position:sticky;top:0}}
 td.mono{{font-family:Consolas,monospace}}
 td.c{{font-weight:700;color:var(--blue)}}
 .ok{{color:var(--green);font-weight:600}} .no{{color:var(--red);font-weight:600}}
 img.shot{{width:100%;border:1px solid var(--line);border-radius:10px;display:block}}
 .legend{{color:var(--sub);font-size:13px;margin-top:8px}}
 code{{background:#eef2f7;padding:2px 6px;border-radius:5px;font-family:Consolas,monospace;font-size:13px}}
 pre{{background:#101827;color:#d7e3f4;padding:14px 16px;border-radius:10px;overflow:auto;font-size:13px;line-height:1.6}}
 .two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
 @media(max-width:820px){{.two{{grid-template-columns:1fr}}}}
 .tag{{display:inline-block;background:#e8f0fe;color:var(--blue);border-radius:999px;padding:1px 10px;font-size:12.5px;margin-right:6px}}
</style></head><body><div class="wrap">

<h1>舞蹈片段识别报告</h1>
<div class="sub">源文件 <code>42111864478-1-192.mp4</code> · 1280×720 · 30fps · 时长 3 小时 59 分 52 秒 · 每 3 秒采样 1 帧，共 4797 帧</div>

<div class="kpis">
  <div class="kpi"><div class="v">4797</div><div class="k">采样帧数（每 3 秒 1 张）</div></div>
  <div class="kpi"><div class="v">598</div><div class="k">人体骨骼判定为「站立」的帧</div></div>
  <div class="kpi"><div class="v">3423</div><div class="k">手部骨骼判定为「舞蹈状态」的帧</div></div>
  <div class="kpi"><div class="v">{int((res.dance_auto if 'dance_auto' in res else res.dance).sum())}</div><div class="k">两项同时成立 → 舞蹈帧（骨架判定）</div></div>
  <div class="kpi"><div class="v">{int(res.dance.sum())}</div><div class="k">剔除坐姿误判后 → 有效舞蹈帧</div></div>
  <div class="kpi"><div class="v">{len(clips)} 段 / {clips.dur_s.sum()/60:.1f} 分钟</div><div class="k">最终舞蹈片段</div></div>
</div>

<h2>1. 判定方法</h2>
<div class="card">
<p>用 <b>MediaPipe PoseLandmarker</b> 对每帧提取 33 个人体骨骼关键点，用 <b>HandLandmarker</b> 对手腕区域放大裁剪后提取 21 个手部骨骼关键点。三个条件全部满足才判为舞蹈帧：</p>
<table>
<tr><th style="width:150px">判据</th><th>量化定义</th><th style="width:150px">阈值</th></tr>
<tr><td><b>① 人体骨骼＝站立</b></td>
    <td>肩中点在画面中的高度 <code>sh_y</code>，以及肩宽 <code>sh_w</code>（相对画面宽）。站立全身景时肩膀位于画面上三分之一、人物离机位更远</td>
    <td class=mono>sh_y &lt; 0.44<br>sh_w &lt; 0.27</td></tr>
<tr><td><b>② 手部骨骼＝舞蹈</b></td>
    <td>手腕高于或接近肩水平（<code>up</code>），或手臂横向外展超过 <code>0.45</code> 倍躯干长（<code>lat</code>），或肘关节弯曲 &lt;115° 且手臂抬起；且手部骨骼可见度 &gt; 0.5。另叠加：手腕骨骼在相邻采样间位移 &gt; 0.45 倍躯干长（手在挥动）</td>
    <td class=mono>up &gt; −0.40<br>lat &gt; 0.45<br>elbow &lt; 115°<br>Δwrist &gt; 0.45</td></tr>
<tr><td><b>③ 持续运动</b></td>
    <td>相邻采样帧画面差分运动量的 3 帧（±3 秒）滚动均值。坐姿聊天时画面几乎静止</td>
    <td class=mono>m³ &gt; 0.09</td></tr>
</table>
<p class="legend">说明：本视频机位较紧且大部分时间人物坐在沙发/麦克风前，<b>髋、膝、踝关节关键点在 96% 的帧上不可见</b>（MediaPipe 会将其外推到画面之外），因此「站立」不能直接用腿部关节判定，改用肩部在画面中的位置与尺度来推断全身景／站立姿态——该判据经逐帧目视核对验证有效。</p>
</div>

<h2>2. 全片时间轴</h2>
<div class="card">
<img class="shot" src="{b64png(tl_full)}">
<p class="legend">第 1 行：肩中点高度（越低表示人物越接近全身站立景）；第 2 行：肩宽；第 3 行：运动量（红线为阈值）；第 4 行：舞蹈判定得分。红色阴影为最终舞蹈片段。</p>
</div>
<div class="card">
<img class="shot" src="{b64png(tl_zoom)}">
<p class="legend">前 100 分钟放大图——全部舞蹈内容都集中在这一区间。</p>
</div>

<h2>3. 最终舞蹈片段（{len(clips)} 段，合计 {clips.dur_s.sum()/60:.1f} 分钟）</h2>
<div class="card">
<h3 style="margin:0 0 10px">剪辑规则</h3>
<table>
<tr><th style="width:210px">规则</th><th>说明</th></tr>
<tr><td><b>R1 保持原始画面</b></td><td>三屏复制版式<b>不裁两侧</b>，按原样输出。</td></tr>
<tr><td><b>R2 起点回退一张图</b></td><td>片段起点取「上一张判定为非舞蹈的采样图」的时间点，终点对称取「下一张非舞蹈采样图」的时间点，保证舞蹈首尾任一采样帧都被完整包住、不会被切断。因此片段比纯舞蹈略长（前后各多约 3 秒）。</td></tr>
<tr><td><b>R3 中间可停留 {BRS:.0f} 秒</b></td><td>两段舞蹈之间的间隔 ≤ {BRS:.0f} 秒视为同一段舞蹈，不拆开。</td></tr>
<tr><td><b>R4 短于 {MDS:.0f} 秒丢弃</b></td><td>舞蹈本身时长 &lt; {MDS:.0f} 秒的整段跳过。</td></tr>
</table>
<h3 style="margin:22px 0 10px">片段清单</h3>
<table>
 <tr><th style="width:56px">#</th><th>片段起点</th><th>片段终点</th><th>片段时长</th><th>其中舞蹈时长</th><th>输出文件</th></tr>
{seg_rows}
</table>
<p class="legend">共 {len(blocks)} 个舞蹈区块，R4 过滤掉 {len(blocks)-len(clips)} 个（舞蹈 &lt; 9 秒）。<br>
输出：<code>clips/dance_01.mp4 … dance_{len(clips):02d}.mp4</code> ＋ 合集 <code>clips/ALL_DANCE.mp4</code>（{clips.dur_s.sum()/60:.1f} 分钟）</p>
</div>

<h3 style="margin:22px 0 0"></h3>
<div class="card">
<h3 style="margin:0 0 10px">区块明细（R3 合并与 R4 过滤过程）</h3>
<table>
<tr><th>块</th><th>舞蹈时长</th><th>含舞蹈采样帧</th><th>由几个小段合并</th><th>片段区间</th><th>结论</th></tr>
{block_rows}
</table>
</div>

<h2>4. 判定样张</h2>
<div class="card"><h3 style="margin:0 0 10px">判定为「跳舞」的帧</h3>
<img class="shot" src="{b64(os.path.join(ROOT,'samples_dance.jpg'))}">
<p class="legend">绿线＝骨骼连线，红点＝高置信度关键点，橙点＝低置信度，洋红圆圈＝手腕（手部骨骼的根节点）。</p></div>
<div class="card"><h3 style="margin:0 0 10px">判定为「非跳舞」的帧（坐姿聊天／唱歌）</h3>
<img class="shot" src="{b64(os.path.join(ROOT,'samples_nondance.jpg'))}">
<p class="legend">这些帧肩部位置偏低、且运动量低于阈值，被判为非舞蹈。</p></div>

<h2>5. 自动检出候选清单（含复核结论）</h2>
<div class="card">
<table>
<tr><th>#</th><th>时间段</th><th>采样</th><th>站立占比</th><th>置信度</th><th>运动量</th><th>复核结论</th></tr>
{cand_rows}
</table>
<p class="legend">共 {len(cand)} 个候选。其中 t&gt;5700 秒的 {len(cand)-int((cand.review=='dance').sum())} 个经逐帧放大目视核对，确认为<b>坐姿聊天</b>（该时段机位更宽，坐姿肩高与肩宽与站立景重合，导致骨架特征误判），已在最终结果中排除。</p>
</div>

<h2>6. 手部精细骨骼</h2>
<div class="card">
<p>对 {int(res.dance.sum())} 个舞蹈帧的左右手腕区域裁剪放大后提取 21 点手部骨骼：成功检出 <b>{len(okh)}</b> 只手；手掌张开度（指尖到掌心距离／手长）中位数 <b>{med_open:.2f}</b>；其中 <b>{open_ratio*100:.0f}%</b> 的帧呈明显张手（张指展开）的舞蹈手型。</p>
</div>

<h2>7. 复现与复用</h2>
<div class="card">
<pre>工作目录（由 DANCECUT_ROOT 指定，默认=脚本所在目录）
  frames/                     每 3 秒截取的 4797 张截图
  pose_metrics.csv            4797 帧的 33 点人体骨骼原始坐标
  motion.csv                  帧间运动量
  dance_result.csv            逐帧判定表（body_state / hand_state / dance / clip_id / clip_role）
  hand_landmarks_dance.csv    舞蹈帧的手部 21 点骨骼指标
  dance_candidates.csv        自动检出的 38 个候选段 + 人工复核结论
  dance_blocks_analysis.csv   区块明细（R3 合并 / R4 过滤过程）
  dance_clips.csv             最终剪辑时间轴
  clips/                      dance_01..{len(clips):02d}.mp4 + ALL_DANCE.mp4
  report.html                 本报告

流水线：
  analyze.py      人体骨骼全量推理        -> pose_metrics.csv
  motion.py       帧间运动量              -> motion.csv
  finalize.py     三判据合成 + 逐帧判定 + 候选片段
  extract_hands.py 手部 21 点骨骼         -> hand_landmarks_dance.csv
  build_clips.py  R1~R4 剪辑规则 + 切片   -> dance_clips.csv + clips/
  make_report.py  生成本报告

只改剪辑规则（不重新做姿态推理）：
  编辑 build_clips.py 顶部 BRIDGE_S / MIN_DANCE_S，重跑 build_clips.py 即可
只改判定阈值：
  编辑 finalize.py 顶部 SHY / SHW / DWR / MTHR，重跑 finalize.py -> build_clips.py</pre>
</div>

<p class="legend" style="margin-top:26px">报告生成于自动分析流水线 · 判定阈值与剪辑规则均为可调参数</p>
</div></body></html>
"""

open(os.path.join(ROOT, "report.html"), "w", encoding="utf-8").write(html)
print("report.html written", len(html), "chars")
print("samples_dance.jpg / samples_nondance.jpg written")
