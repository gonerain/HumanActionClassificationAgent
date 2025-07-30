# HumanActionClassificationAgent

本项目提供一个基于骨骼关键点的人体动作识别流程，适用于判定现场人员是否在工作。仓库包含两部分核心代码：

1. **数据处理模块（Data Pipeline）**：从视频中提取人体骨骼序列并保存為 `.npz` 文件，可在多人场景下区分不同 ID。
2. **模型训练模块（Model Trainer）**：使用 LSTM 对骨骼序列进行分类，并在训练结束后输出混淆矩阵与分类报告。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

如果需要 GPU 训练，请确保已正确安装 `torch` 的 CUDA 版本。

## 数据采集流程

示例脚本位于 `src/data_pipeline.py`。以下代码展示如何将 `video.mp4` 中的动作标注为 `1` 并保存到 `dataset/` 目录：

```bash
python - <<'PY'
from src.data_pipeline import DataPipeline
# 初始化管道，内部使用 YOLO 检测并为每个人分配 ID
pipeline = DataPipeline()
# 处理视频，window_size 默认为 30 帧
pipeline.process_video('video.mp4', label=1, output_dir='dataset')
PY
```

运行后会在 `dataset/` 下生成若干 `.npz` 文件，每个文件包含 `(N, 33, 3)` 的 `data` 数组和 `label`。文件名格式为 `視頻名_對象ID_起始幀.npz`，便于后续人工核对与补充标注。可按类别建立子文件夹，例如 `dataset/work/`、`dataset/other/`，训练脚本会递归搜索所有子目录。
若想调试检测框与跟踪 ID，可在 `process_video` 调用时传入 `debug=True`，程序会弹窗显示实时检测结果，按 `q` 键退出。

## 训练流程

训练脚本位于 `src/model_trainer.py`，使用方法如下：

```bash
python src/model_trainer.py dataset --epochs 30 --batch_size 16 --patience 5
```

主要特性：

- 自动检测 GPU 并将模型及数据搬移到 GPU 上。
- 在控制台显示训练进度条，同时打印训练/验证损失与准确率。
- 当验证损失连续多次不下降时触发 Early Stopping。
- 训练结束后保存最优模型至 `weights/model.pt`，并输出混淆矩阵与分类报告。

## 多人场景的处理

- YOLO 在每帧检测所有人体并分配 ID，`DataPipeline` 会为每个 ID 独立缓存序列。
- 当某个 ID 在一定帧数内消失时，其缓存会被清理，以节省内存并防止混淆。
- 这样即使画面中出现多个人，也能分别判断谁在工作、谁只是路过。

## 部署与实时推理

部署脚本位于 `src/inference.py`，加载训练好的模型后即可对新视频或摄像头画面进行动作识别。示例：

```bash
python - <<'PY'
from src.inference import ActionInference
# 加载权重并启动推理，默认每 30 帧判断一次
predictor = ActionInference('weights/model.pt')
# 对 camera 或视频文件执行推理，按 q 退出
predictor.run(0, debug=True)
PY
```

脚本会在窗口中标注每个目标的 ID 和预测的动作类别。当目标长时间消失，其缓存序列会被清理。
