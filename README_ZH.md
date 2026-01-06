# Sim2Real DDP 视频生成脚本使用说明

本文档旨在介绍如何在 FRESCO 框架下使用多卡 DDP (Distributed Data Parallel) 进行 Sim2Real 视频批量生成。我们实现了一个专门的推理脚本 `inference_sim2real_ddp.py`，可以充分利用 8 卡 A6000 服务器的算力。

## 1. 代码库分析

FRESCO (Frame-Consistent Video-to-Video Translation) 是一个专注于视频风格迁移和一致性的框架。在进行 Sim2Real（模拟到真实）任务时，我们主要利用了以下核心模块：

*   **`run_fresco.py` (核心入口)**:
    *   这是原始项目的单视频推理入口。
    *   **关键函数**: `run_keyframe_translation` 负责扩散模型的推理（生成关键帧和特征注入），`run_full_video_translation` 负责使用 EBSynth 进行全视频的帧间平滑传播。
    *   **修改点**: 我们对原代码进行了微小的修改，允许在调用 `run_keyframe_translation` 时传入预加载的模型对象。原本的设计是每次调用都重新加载模型，这在批量处理视频时会导致极大的时间浪费。

*   **`src/diffusion_hacked.py` (注意力控制)**:
    *   实现了 FRESCO 的核心算法，即通过 hack `UNet` 的注意力机制，注入时空一致性约束（Spatial-Temporal Attention）。
    *   **兼容性修复**: 您的环境中安装的 `diffusers` 版本 (0.34.0) 较新，原代码是为 0.19.3 设计的，导致导入路径报错。我们添加了兼容性补丁，使其能自动适配新旧版本的 `diffusers`。

*   **`inference_sim2real_ddp.py` (新增脚本)**:
    *   **DDP 并行策略**: 这是一个标准的 PyTorch Distributed 启动脚本。它不会把单个视频拆分到多卡（因为视频生成显存开销大，且帧间依赖强），而是采用 **数据并行** 的方式：将视频列表平均分配给 8 张卡，每张卡独立负责一部分视频的完整推理。
    *   **模型复用**: 脚本启动时，每张卡只加载一次模型到显存，然后循环处理分配给它的视频列表，最大化利用计算资源。

## 2. 环境准备

确保您已在服务器上正确配置了环境。本项目依赖 `diffusers`, `torch`, `opencv-python` 等库。

如果遇到 `ImportError: cannot import name 'UNet2DConditionOutput'` 错误，请确保您拉取了最新的代码，我们已经在 `src/diffusion_hacked.py` 中修复了这个问题。

## 3. 脚本使用指南

### 3.1 启动命令

使用 `torchrun` 命令启动 8 卡并行推理：

```bash
torchrun --nproc_per_node=8 inference_sim2real_ddp.py \
    --video_folder /path/to/your/videos \
    --prompt_folder /path/to/your/prompts \
    --base_config ./config/config_carturn.yaml \
    --output_folder ./output/sim2real_results \
    --control_type hed
```

### 3.2 参数详解

*   `--video_folder`: **[必填]** 存放输入视频的文件夹路径。支持 `.mp4`, `.avi`, `.mov`, `.mkv` 格式。
    *   例如：`/home/fch/data/sim_videos/`
*   `--prompt_folder`: **[必填]** 存放 prompt 文本文件的文件夹路径。
    *   **重要**: 文本文件名必须与视频文件名完全一致（除了扩展名）。
    *   例如：视频是 `car_01.mp4`，对应的 prompt 文件必须是 `car_01.txt`。
*   `--base_config`: **[必填]** 基础配置文件路径。
    *   推荐使用 `config/config_carturn.yaml` 作为模板。
    *   脚本会自动读取这个 yaml 中的参数（如 `num_inference_steps`, `guidance_scale` 等），但会覆盖其中的 `video_path`, `save_path` 和 `prompt` 为当前处理的视频信息。
*   `--output_folder`: **[必填]** 结果输出根目录。
    *   脚本会在这个目录下为每个视频创建一个以视频名为名的子文件夹，存放生成结果。
*   `--control_type`: **[可选]** ControlNet 类型。
    *   默认: `hed`
    *   选项: `hed` (边缘), `canny` (硬边缘), `depth` (深度)。
    *   请根据您的 Sim 数据类型选择最合适的控制条件。如果是 3D 渲染的灰模或线框，`canny` 或 `hed` 可能效果较好；如果有深度图，可选 `depth`。

### 3.3 运行逻辑

1.  **初始化**: 启动 8 个进程，每个进程绑定一张 GPU 卡。
2.  **分配任务**:
    *   假设有 100 个视频。
    *   Rank 0 处理第 0, 8, 16... 个视频。
    *   Rank 1 处理第 1, 9, 17... 个视频。
    *   以此类推。
3.  **推理**:
    *   读取视频和对应的 prompt。
    *   运行 FRESCO 关键帧生成。
    *   运行 EBSynth 全视频平滑（此步依赖 CPU，请注意监控 CPU 负载，如果过高可适当减少并行卡数，但通常 8 卡没问题）。
4.  **保存**: 结果保存在 `--output_folder` 下。

## 4. 常见问题

*   **显存不足 (OOM)**: 如果遇到 OOM，请修改 `--base_config` 指定的 yaml 文件，减小 `batch_size`（例如从 8 改为 4）或降低视频分辨率（在代码中 resize 处修改，当前默认 512）。
*   **缺少 Prompt**: 如果某个视频没有对应的 `.txt` 文件，脚本会输出 Warning 并跳过该视频，不会中断整个任务。
*   **Prompt 写法**: 建议在 txt 文件中不仅包含画面描述，还可以加上 Negative Prompt（负面提示词）以提升质量，但当前的 txt 读取逻辑是直接读取全部内容作为 Positive Prompt。如果需要精细控制 Negative Prompt，可以在 `run_fresco.py` 的 `run_keyframe_translation` 函数中写死通用的负面提示词（代码中已有默认的负面提示词）。

如有其他问题，请查阅原始仓库文档或联系开发人员。
