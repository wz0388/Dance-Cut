# 授权说明（中文）

本项目采用 **[Apache License 2.0](LICENSE)**（`SPDX-License-Identifier: Apache-2.0`）。
为方便阅读，这里给出中文要点。

> 以下中文仅为便于理解，**以 [LICENSE](LICENSE) 中的英文原文为准**；两者如有冲突，以英文原文为准。

## 你可以做什么

- ✅ **任何用途，包括商业用途**——Apache-2.0 是 OSI 认可的开源协议，不限制使用领域
- ✅ 自由使用、复制、修改、分发本软件及其衍生作品
- ✅ 将本软件集成进闭源产品或商业产品
- ✅ 提供基于本软件的商业服务、付费支持
- ✅ 再分发（**必须**附上本协议原文、保留版权与许可声明，并标注你修改过的文件）

## 你需要做什么

- ⚠️ 再分发时必须附上 [LICENSE](LICENSE) 原文
- ⚠️ 保留源码文件中的版权声明、许可声明（以及本项目的 [NOTICE.md](NOTICE.md)）
- ⚠️ 若修改了文件，需在文件中显著标注"已修改"
- ⚠️ 本软件按"现状"提供，**不提供任何明示或暗示的担保**，作者不承担任何责任
- ⚠️ 不得使用作者名义或本项目名义为你的衍生作品背书（协议第 6 条对商标另有说明）

## 关于"开源"

Apache-2.0 是 [OSI](https://opensource.org/licenses/Apache-2.0) 认可的、被广泛使用的
**开源协议**，已被 Apache 基金会、Android、Kubernetes 等大量项目采用，也是商业友好的协议之一。

---

# 第三方组件声明

本项目使用了以下第三方组件，它们各自遵循自己的许可协议，**不受本项目协议限制**：

| 组件 | 用途 | 许可 | 是否随仓库分发 |
|---|---|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | 人体/手部关键点检测 | Apache-2.0 | 否（作为依赖安装） |
| MediaPipe Pose Landmarker 模型 | 33 点人体骨骼权重 | Apache-2.0 | **是**（`models/pose_landmarker_full.task`） |
| MediaPipe Hand Landmarker 模型 | 21 点手部骨骼权重 | Apache-2.0 | **是**（`models/hand_landmarker.task`） |
| [FFmpeg](https://ffmpeg.org/) | 抽帧、切片、合成 | LGPL-2.1+ / GPL-2.0+（视构建而定） | **否**（需用户自行下载） |
| [OpenCV](https://opencv.org/) | 图像读写与绘制 | Apache-2.0 | 否（作为依赖安装） |
| [NumPy](https://numpy.org/) | 数值计算 | BSD-3-Clause | 否（作为依赖安装） |

## 关于 FFmpeg

### 源码仓库

本仓库**不包含** FFmpeg 二进制文件。使用者需自行从官方渠道获取：

- <https://www.gyan.dev/ffmpeg/builds/>
- <https://github.com/BtbN/FFmpeg-Builds/releases>

### 预编译发布包（GitHub Releases）

**Release 里的 `DanceCut_portable.zip` 是包含 FFmpeg 二进制的**（为了开箱即用），
特此说明：

| 项目 | 说明 |
|---|---|
| 来源 | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 的 `ffmpeg-7.1.1-essentials_build` |
| 许可 | **GPL-3.0**（该构建启用了 `libx264` 等 GPL 组件） |
| 源码 | <https://ffmpeg.org/download.html> · <https://git.ffmpeg.org/ffmpeg.git> |
| 说明 | 第三方独立程序，与本项目仅为"调用"关系，未链接进本程序代码 |

> ⚠️ **若你要再分发这个便携包**，需要自行遵守 FFmpeg 的 GPL 条款：
> 提供对应源码或获取渠道、保留其版权声明。**这是 FFmpeg 自身的义务，与本项目的 Apache-2.0 授权相互独立**——
> 两套许可同时适用。
>
> 需要完全规避这一步的话，可以下载源码仓库版本，自行替换为 LGPL 构建的 FFmpeg
> （不含 libx264，例如 `ffmpeg-*-essentials_build` 换成 LGPL 共享版）。

本项目的调用方式：通过 `subprocess` 执行外部 `ffmpeg.exe`，
未做任何链接或静态合并。

## 关于模型文件

`models/pose_landmarker_full.task` 与 `models/hand_landmarker.task` 来自
Google MediaPipe，许可 **Apache-2.0**，随源码仓库与发布包一同分发。

## 关于媒体素材

本仓库**不包含**任何视频素材、直播录像或由此生成的切片与截图（`docs/` 下仅含
程序界面截图与骨骼标注示意，用于说明功能）。使用者需自行确保其处理的素材
拥有合法权利。
