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

运行后会在 `dataset/` 下生成若干 `.npz` 文件，每个文件包含 `(N, 33, 3)` 的 `data` 数组和 `label`。文件名格式为 `视频名_ID.标签_起始帧.npz`，例如 `video.mp4_1.0_25.npz`，便于按标签筛选与核对。可按类别建立子文件夹，如 `dataset/work/`、`dataset/other/`，训练脚本会递归搜索所有子目录。
若想调试检测框与跟踪 ID，可在 `process_video` 调用时传入 `debug=True`，程序会弹窗显示实时检测结果，按 `q` 键退出。

## 数据集标注与可视化

生成的 `.npz` 序列可通过 `src/label_tool.py` 进行人工校对和重新标注。脚本会在播放原始视频片段的同时绘制骨骼点，并在修改标签后自动重命名文件，按键说明如下：

- `n`：下一条数据
- `p`：上一条数据
- `0-9`：设置并保存标签
- `q`/`Esc`：退出程序

示例：

```bash
python src/label_tool.py dataset videos
```

其中 `dataset` 为 `.npz` 文件目录，`videos` 为对应的原始视频路径。

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

## 场景级人员在岗检测


脚本 `src/scene_presence.py` 提供了基于状态机的多人在岗检测逻辑，并支持交互式调整**任意多边形**监控区域。程序会根据各 ID 在区域内的**驻留时间**判断其状态（pending/active/paused/finished），同时在窗口中绘制状态色框和场景总体状态。监控区域会以半透明色块高亮显示，并将调整后的多边形保存到 `scene_presence_config.json`，下次运行时自动加载。默认监控区域为整张画面的四边形。配置文件记录检测模型、置信度及时间阈值，这些阈值位于 `timing` 字段下（`enter_s`、`leave_s`、`finish_s`），命令行参数可覆盖这些值并写回配置文件。支持同时以毫秒或秒传入阈值（例如 `--enter-ms 500` 或 `--enter-s 0.5`），配置文件的字段说明见 `scene_presence_config.schema.json`。

如仅需快速标定区域，可运行 `src/roi_calibrator.py` 读取视频流首帧并交互式绘制多边形，结果会写入同一个 `scene_presence_config.json`，后续 `scene_presence.py` 会直接使用该区域：

```bash
python src/roi_calibrator.py --video 0
```


运行示例：

```bash
python src/scene_presence.py --video 0 --model yolo11s
```


运行窗口中按 `r` 可重新绘制监控区域（左键添加顶点，空格或回车结束，Esc 取消），按 `q` 退出。若任一 ID 处于 `active` 状态，界面左上角会显示 `ACTIVE`，否则为 `INACTIVE`。如需在后台运行或关闭可视化，可加 `--no-display` 参数。


## 后端服务（多相机）

仓库提供一个基于 FastAPI 的多相机后端 `src/backend/service.py`，支持：

- 多相机管理（启动/停止/更新源与 ROI）
- 实时在岗检测（ROI + 跟踪 + 状态机）
- WebSocket 推流已处理画面与状态
- 驻留事件落库（含视频证据）

### 启动

```bash
uvicorn backend.service:app --host 0.0.0.0 --port 8000
```

环境变量：

- `SP_DB_URL`：PostgreSQL 连接串，默认为 `postgresql+psycopg2://postgres:000815@localhost:5432/sglz`
  - TimescaleDB 可选；若存在，会自动在 `dwell_events.start_ts` 上创建 hypertable。

### API 一览（按相机粒度）

- `GET /cameras`：列出相机与运行状态。
- `POST /cameras`：创建相机。请求体：`{name, source, region}`，其中 `region` 为 `[[x,y], ...]`。
- `PUT /cameras/{camera_id}`：更新名称/源/区域。
- `DELETE /cameras/{camera_id}`：删除相机并停止运行。
- `GET /cameras/{camera_id}/snapshot`：返回单帧处理结果与 Base64 JPEG（字段 `frame`）。
- `GET /cameras/{camera_id}/inference/report`：仅返回语义状态（不含图像）。
- `WS /cameras/{camera_id}/status`：持续推送 JSON，字段包含：
  - `active_ids: number[]` 当前处于 active 的跟踪 ID
  - `scene_active: boolean` 场景是否有人在岗
  - `roll_status: string` 示例的钢卷状态（占位）
  - `frame: string` Base64 编码 JPEG 图
  - `health: {status,last_open_ts,last_read_ts,last_frame_ts,reconnect_attempts,stuck_seconds,alarm,alarm_message}`
  - `alarm: boolean`、`alarm_message: string|null`

### 驻留事件 API

- `GET /dwell_events`：按 `start_ts` 升序返回所有事件。
- `GET /cameras/{camera_id}/dwell_events`：返回指定相机的事件。

返回字段：`{id, object_id, camera_id, start_ts, end_ts, video_path}`，其中时间为 ISO 格式，UTC。

### 内置前端

访问根路径 `/` 加载 `index.html`，可：

- 列表/选择相机，查看 WebSocket 推流
- 抓拍 `Snapshot` 与 `Inference Report`
- 查看 `Dwell Events`
- 创建/更新/删除相机（可输入 `region` 的 JSON）

---

## 技术参数与实现细节

### 检测与状态机

- 模型：Ultralytics YOLO（默认 `yolo11s`），仅保留 `person` 类别。
- 置信度阈值：`conf=0.5`（可在 `ROIWorkflow` 构造时调整）。
- ROI：任意多边形，坐标像素；落在目标足底中心点（bbox 底边中点）在多边形内即判定在区域内。
- 状态机阈值（默认，单位毫秒）：
  - `enter_ms=500` 进入区域并累计达到该阈值从 `pending` 变 `active`
  - `leave_ms=1000` 暂离超过该阈值从 `paused/active` 变 `inactive`（并清理）
  - `finish_ms=None` 可选；达到后从 `active` 变 `finished`
- 场景活跃：任意 ID 处于 `active` 即 `scene_active=true`。

### 录像与文件

- 目录：`recordings/`（仓库根目录自动创建）
- 命名：`{camera_label?}{epoch_ms}_{track_id}.avi`
- 编码：`XVID`，帧率 `20 fps`，尺寸为输入帧尺寸
- 关闭条件：`inactive` 或 `finished` 或 ID 不再存在时关闭并入库（避免 `paused` 时提前截断）

### 健康与稳定性

- 视频采集：独立线程（`VideoCaptureWorker`），缓冲大小尽量置 1
- 卡顿检测：
  - 小图对比阈值（均值差）约 0.5
  - `STUCK_SEC=5.0` 判定卡顿，`NOFRAME_SEC=5.0` 判定无帧
- 报警：`health.alarm=true` 并附 `alarm_message`

### 并发与异步

- 推理线程：每相机一个（默认 `8 fps`），缓存最近 JPEG 与状态
- DB 写入：后台队列 + 单线程写库，避免阻塞关键环

### 数据库与迁移

- 方言：PostgreSQL 14+（推荐 TimescaleDB）
- 连接：`SP_DB_URL`（示例：`postgresql+psycopg2://user:pass@host:5432/db`）
- 表结构：
  - `cameras(id, name, source, region_json)`
  - `dwell_events(id, object_id, camera_id NULL, start_ts timestamptz, end_ts timestamptz, video_path)`
- 时间规范：所有写入统一转换为 UTC tz-aware `datetime`
- Timescale：若扩展存在，自动 `CREATE EXTENSION` 并 `create_hypertable('dwell_events','start_ts')`
- 轻量自迁移（幂等）：
  - 缺失 `camera_id` 时自动 `ALTER TABLE` 补列
  - 创建复合索引 `idx_dwell_camera_start(camera_id, start_ts)`

### 性能与资源建议（参考值）

- 单路 720p，YOLO11s，CPU-only：~5–10 FPS（视硬件而定）；建议 GPU 提升性能
- 多路并发：为每路分配独立推理与采集线程；`cv2.setNumThreads(1)` 降低线程争用
- 网络：WebSocket 推送约 8 FPS JPEG（质量 70），按需调整质量与频率

### 配置与参数汇总

- 环境：`SP_DB_URL`
- 模型：`model_name`（默认 `yolo11s`），`conf`（默认 `0.5`）
- ROI：`region`（`[[x,y], ...]`），前端/接口可设置
- 阈值：`enter_ms`，`leave_ms`，`finish_ms`（可选）
- 录像：`XVID`、`20fps`、`recordings/` 目录

### 运行示例（完整最小可用）

```bash
# 1) 准备数据库（可选 TimescaleDB）并设置连接串
export SP_DB_URL=postgresql+psycopg2://user:pass@localhost:5432/mydb

# 2) 启动服务
uvicorn backend.service:app --reload

# 3) 浏览器打开 http://localhost:8000/
#    创建相机（source 可为 0 或 RTSP），调整 ROI，实时查看状态与驻留事件
```

## 运行测试

项目使用 `pytest` 进行测试。运行所有测试：

```bash
pytest
```

提示：测试中部分依赖（如 `cv2`, `fastapi`）需按 `requirements.txt` 安装；如无真实摄像头，测试会注入 dummy VideoCapture。

## 前端（初版）

目录：`frontend/`

- 技术栈：React + TypeScript + Vite
- 功能：相机列表/选择、实时画面（WebSocket）、抓拍、驻留事件拉取、创建/删除相机
- 开发启动：

```bash
cd frontend
npm install
npm run dev
# 打开 http://localhost:5173 ，默认代理到后端 http://localhost:8000
```

- 构建：`npm run build`，产物位于 `frontend/dist/`

说明：开发模式下 Vite 通过代理把 `/cameras/*`、`/dwell_events` 等请求转发给后端（含 WS）。生产部署可由 Nginx 或后端静态托管构建产物。
