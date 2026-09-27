# 分步流水线（方法参考）

这 7 个脚本是最初实现这套判定方法时的分步版本，每个阶段产出一个中间文件，
方便逐段检查判定依据。**通用场景建议直接用 `app/dance_cut.py`**（单程序、路径无关、带界面）。

## 执行顺序

```
analyze.py        全量人体骨骼（33 点）推理      -> pose_metrics.csv
motion.py         相邻采样帧帧间运动量          -> motion.csv
finalize.py       三判据合成 + 逐帧判定表 + 候选片段
extract_hands.py  只对舞蹈帧跑手部骨骼          -> hand_landmarks_dance.csv
build_clips.py    剪辑规则 R1~R4 + 切片         -> dance_clips.csv + clips/
make_report.py    生成自包含 HTML 报告          -> report.html
refine_edges.py   用 1 秒采样重扫片段边界（可选，把切割误差从 ±3s 压到 ±1s）
```

## 路径配置

脚本不再硬编码路径，用环境变量指定：

```bash
set DANCECUT_ROOT=D://work                 # 工作目录，默认 = 脚本所在目录
set DANCECUT_VIDEO=D://work//video.mp4     # 源视频
set DANCECUT_VIDEO_END=0                  # 时间轴末尾（秒），0 = 按实际长度
```

## 注意

- 需要 `ffmpeg.exe` 在 `DANCECUT_ROOT` 或 `app/` 目录下
- 需要 `models/` 下的两个 `.task` 模型
- 这些脚本是**一次性分析**写成的风格：中间产物多、参数写在文件顶部，
  想调参请直接改脚本开头的常量，或改用 `app/dance_cut.py`（参数在界面上）
