# HumanActionClassificationAgent

本项目提供了一个基础的人体动作识别流程，适用于工业现场中"是否在工作"的检测。主要包含两个部分：

1. **数据处理模块（Data Pipeline）**：从视频中提取骨骼序列并保存为 `npz` 文件。
2. **分类模型训练模块（Model Trainer）**：使用 LSTM 对骨骼序列进行分类训练。

## 数据处理

依赖：`ultralytics`、`mediapipe`、`opencv-python`。

示例代码位于 `src/data_pipeline.py`，可以自动将视频切片为固定长度的骨骼序列，便于逐段标注：

```bash
python -m pip install ultralytics mediapipe opencv-python
python - <<'PY'
from src.data_pipeline import DataPipeline
# 初始化管道，内部会自动进行目标跟踪
pipeline = DataPipeline()
# 将 video.mp4 中的动作标记为 1，保存到 dataset 目录
pipeline.process_video('video.mp4', label=1, output_dir='dataset')
PY
```

该模块会在检测到多人时分别计算每个人的姿态，并在对象离开画面一定时间后自动清理缓存，避免长期占用内存。每累积 `N` 帧（默认 30 帧）便保存一个 `npz` 文件，形状为 `(N, 33, 3)`，文件名包含视频名、对象 ID 与起始帧号，便于人工逐段标注。

## 模型训练

依赖：`torch`、`scikit-learn`。

训练脚本位于 `src/model_trainer.py`，执行方式如下：

```bash
python src/model_trainer.py dataset --epochs 30 --batch_size 16
```

脚本会自动检测是否存在可用的 GPU，并在训练过程中使用 `tqdm` 显示进度条。
数据集目录可包含子文件夹，脚本会递归查找其中的 `npz` 文件，便于按动作类别分别存放数据。
训练过程会输出训练损失、验证损失以及验证集准确率，并内置 Early Stopping，当验证损失多次不再下降时提前结束。
训练结束后会打印混淆矩阵和分类报告，便于评估模型效果。

训练结束后会在 `weights/` 目录下生成 `model.pt`，可用于后续推理。

## 多人场景的处理思路

- 使用 YOLO 检测所有人体并分配 ID，通过 `track` 函数保持同一人的序列连续。
- 对每个 ID 分别提取骨骼关键点序列，生成独立的样本。
- 在训练阶段，通过给定的标签（如“正在工作”或“非工作”）训练分类器，从而在实际场景下区分不同人员的动作状态。

这样即使画面中同时出现多个人，也能分别判断谁在工作、谁只是路过。 
