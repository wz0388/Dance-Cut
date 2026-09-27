# 授权说明（中文）

本项目采用 **[PolyForm Noncommercial License 1.0.0](LICENSE)**。为方便阅读，这里给出中文要点。

> 以下中文仅为便于理解，**以 [LICENSE](LICENSE) 中的英文原文为准**；两者如有冲突，以英文原文为准。

## 你可以做什么

- ✅ 个人学习、研究、业余项目中使用
- ✅ 阅读、修改源码，做二次开发
- ✅ 在你的非商业项目中集成
- ✅ 非商业性地再分发（**必须**同时附上本协议原文，并保留原作者署名）
- ✅ 用于慈善机构、教育机构、公共研究机构、公共安全或健康组织、环境保护组织等
      **非营利/非商业**用途（详见 LICENSE 中 *Noncommercial Purposes* 的定义）

## 你不可以做什么

- ❌ **任何商业用途**。包括但不限于：
  - 出售本软件、基于本软件的服务，或提供付费支持
  - 集成进任何以营利为目的的产品或服务
  - 用于带有广告、打赏、付费订阅、电商导流等变现行为的平台或工具
  - 公司内部用于经营性业务（"内部使用"不等于"非商业"）
- ❌ 移除或修改版权声明与许可声明后再分发
- ❌ 用作者名义为你的衍生作品背书

## 关于"开源"的说明

严格来讲，限制使用领域的协议**不属于** OSI（Open Source Initiative）定义的开源协议——
OSI 要求协议不得限制使用领域。因此本项目更准确的说法是
**"源码开放 / Source-available"**（或 Fair Source）。

本仓库正文中如出现"开源"字样，是通俗说法，法律效力以 LICENSE 原文为准。

## 商业授权

如需商业使用，请联系仓库作者 <https://github.com/wz0388> 单独获取商业授权。

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

本仓库**不包含** FFmpeg 二进制文件。使用者需自行从官方渠道获取：

- <https://www.gyan.dev/ffmpeg/builds/>
- <https://github.com/BtbN/FFmpeg-Builds/releases>

请注意 FFmpeg 的许可：若你使用带 GPL 组件的构建（如 `libx264` 编码器），
分发包含该二进制的产物时需要遵循 GPL 的相应要求。本项目通过调用外部
`ffmpeg.exe` 的方式使用它，未将其链接进本程序。

## 关于媒体素材

本仓库**不包含**任何视频素材、直播录像或由此生成的切片与截图（`docs/` 下仅含
程序界面截图与骨骼标注示意，用于说明功能）。使用者需自行确保其处理的素材
拥有合法权利。
