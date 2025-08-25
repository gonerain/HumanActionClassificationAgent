# Agents 后端分模块实施计划（Scene Presence / Occupancy）

> 目标：在**无需 Pose** 的前提下，完成“场景级人员在岗检测（是否有人在工作区域）”的稳定落地。实现从视频流接入、检测/跟踪、ROI 命中、驻留/计数、规则判定到告警分发与可视化查询的闭环。

---

## 0. 总览架构

**数据流**：Ingest → Det-Track → ROI 命中 → Counter/Dwell → Rules → Aggregator/Alarm → API/Admin → Storage/ODS

**部署**：GPU 节点跑 Det-Track，CPU 节点跑业务 API/规则/告警/存储。

---

## 1) Ingest-Gateway（视频流接入）
**职责**：接入 RTSP/GB28181/文件源，统一输出帧（或切片），健康检查与降采样。

**关键接口**
- `POST /cameras`：注册/更新流 `{id, scene_id, rtsp_url, fps_target}`
- `GET /cameras/:id/health`：拉流健康与统计

**输出主题**
- `video.frames.{camera_id}`：`{camera_id, ts, frame}` 或 `{clip_path, start_ts, end_ts}`

**注意事项**
- RTSP 稳健：启用 `cv2.CAP_FFMPEG`，设置超时与自动重连；断流回退本地缓存图。
- 降采样：存在带宽/算力限制时，将 25–30fps 降到 6–10fps；保证端到端延迟 ≤ 2s。
- 时间基线：记录 `fps_est` 与源时间戳（避免阈值随源变化失真）。

---

## 2) Det-Track（人体检测与多目标跟踪）
**职责**：单类 `person` 检测（YOLO 系）+ 跟踪 ID（ByteTrack/DeepSORT）。

**输入**：`video.frames.*`

**输出**：`person.tracks.{camera_id}`：`{ts, tracks:[{track_id, bbox:[x1,y1,x2,y2], score}]}`

**注意事项**
- 置信度与 NMS：`det_conf`, `nms_iou` 可配置；
- ID 稳定：`max_age`, `min_hits`；短暂遮挡用 gap 容忍；
- `ids=None` 回退：以坐标哈希生成临时 ID，避免全体外流导致误判；
- `min_area` 支持**绝对像素**与**相对比例**（画面面积 1–2% 起步）；
- 仅保留 `person` 类别，减少噪声。

---

## 3) ROI-Filter（区域命中判定）
**职责**：管理多边形 ROI/蒙版；判定目标是否在区域内。

**接口**
- `POST /scenes/:id/rois`：保存/更新多边形与可选穿线 `enter_lines`
- `GET /scenes/:id/rois`

**输出**：`roi.presence.{scene_id}`：`{ts, items:[{track_id, inside}]}`

**命中策略（推荐）**
- **脚点锚点法**（底边中点）：作为主判据，抗姿态变化；
- **脚部条带 Overlap**（底部 15–25% 高度）：作为辅判据，阈值 0.3–0.5；
- 双阈值滞回（进/出不同阈值），避免边界抖动；
- 可选“穿线进入”：轨迹跨线后才开始计入。

**注意事项**
- ROI 外扩/内缩 3–10px 形成进入/离开边界；
- 多 ROI/多相机分别统计；
- 若允许标定，优先地平面单应性，鸟瞰坐标判定最稳。

---

## 4) Counter & Dwell（计数与驻留）
**职责**：计算 `present_count` 与每个 Track 的连续驻留时间，时间去抖。

**输入**：`roi.presence.*`

**输出**：`scene.stats.{scene_id}`：`{ts, present_count, dwellers:[{track_id, dwell_sec}]}`

**注意事项**
- 统一用**秒**配置阈值：`enter_sec`、`leave_sec`，运行时按 `fps_est` 换算为帧；
- 缺帧容忍 `gap_sec`（网络抖动/跟踪暂失），典型 0.3–1.0s；
- 入口过滤：需穿线并驻留 ≥ `enter_sec` 才纳入 `present_count`；
- 清理策略：Track 长时间消失后回收状态，防内存泄露。

---

## 5) Rules-Engine（规则引擎）
**职责**：根据统计流判定“有人/空岗/告警”，支持时段/排班/
白名单。

**规则示例**
- 有人：`present_count ≥ K` 持续 `≥ T_on` 秒；
- 空岗告警：`present_count == 0` 持续 `≥ T_off` 秒（且处于工作时段）；
- 例外：维修/巡检白名单，节假日免警；
- 阈值热更新；支持 `POST /simulate` 回放一段 stats，找出触发点。

**注意事项**
- 规则解释可溯源：保留触发窗口、相关 Track 及快照；
- 多级告警：按持续时长或人数缺口升级；
- 多场景一致性：同一租户共享默认策略，可场景覆盖。

---

## 6) Scene-Aggregator（场景级事件聚合）
**职责**：将规则判定生成**事件**，做去抖合并与边沿检测，输出 `SceneEvent`。

**输出**：`scene.event.{scene_id}`：`{type: occupied|vacant, start_ts, end_ts?, present_count, persons[], rule_snapshot}`

**注意事项**
- 只在状态切换（边沿）产生事件；
- 进行短间隔合并（如 < 3s 的闪断并入前一事件）；
- 为每个事件截取证据片段/关键帧。

---

## 7) Alarm-Dispatcher（告警与联动）
**职责**：生成/升级/去重告警，分发到短信/邮件/企微/飞书/Webhook。

**接口**
- `POST /webhook/scene-alarm`：推送告警
- `POST /ack/:alarm_id`：回执

**注意事项**
- 幂等与去重（同窗口重复触发合并）；
- 证据：关键帧 JPEG + 回放切片（前后各 10–20s）；
- SLA：通知超时重试与失败降级策略。

---

## 8) API-Gateway & Admin（查询与配置）
**职责**：统一对外 API、租户鉴权与管理后台。

**查询接口**
- `GET /scenes/:id/status` → `{status, present_count, since}`
- `GET /scene-events?scene_id=&from=&to=&type=`
- `GET /alarms?scene_id=&status=&level=`

**配置接口**
- 相机/ROI/规则/排班的 CRUD；模型与阈值灰度；

**注意事项**
- 鉴权：JWT + RBAC + 租户隔离（列级 `tenant_id`）；
- 审计日志：所有变更需留痕；
- 导出：CSV/Excel；长查询分页/异步导出。

---

## 9) Storage（存储层）
**结构化库（PostgreSQL/MySQL）**
- `camera(id, scene_id, url, status, fps_target, ...)`
- `scene(id, name, tenant_id, ...)`
- `scene_roi(id, scene_id, camera_id, polygon_geojson, enter_lines, buffer_px, updated_at)`
- `rule(id, scene_id, json, enabled, updated_at)`
- `scene_event(id, scene_id, type, start_ts, end_ts, present_count, detail, rule_snapshot)`
- `alarm(id, scene_event_id, level, status, receivers, evidence_url, created_at)`

**时序/热数据**：Kafka/Redis Streams/TSDB（用于看板与实时消费）

**媒体证据**：对象存储（S3/MinIO）→ `evidence/{alarm_id}/clip.mp4`、`frame.jpg`

**注意事项**
- 索引：`scene_id + time range` 组合索引；
- 归档：事件 > 90 天转冷；
- 对象存储生命周期策略，节省成本。

---

## 10) Observability（可观测与运维）
**指标**：
- 吞吐/FPS、端到端延迟；present_count；空岗告警率；
- 误报/漏报、规则触发率、通知成功率；

**日志与链路**：
- 关键事件日志含 `trace_id`，贯穿 camera→det→roi→stats→rule→alarm；
- 采样存图（调试开关）。

**注意事项**
- 模型/阈值变更前后对比；
- 数据/分布漂移监控（夜间/背光/雨雪切换）。

---

## 11) 安全合规
- RTSP 凭据加密保存，按租户隔离；
- Webhook 出站白名单；
- 审计：配置变更与告警确认留痕；
- 隐私：若需马赛克/遮挡，Det-Track 后渲染遮罩（可选模块）。

---

## 12) 性能与容量规划
- 单路 720p@10fps：CPU 可 1–2 路，GPU 可 10–20 路（视型号）
- 降采样：presence 场景可至 6–10fps；
- 目标延迟：≤ 1–2s；
- 主题分区：按 `scene_id/camera_id` 进行分区与限速。

---

## 13) 开发里程碑（建议）
**MVP（Week 1–2）**
- Ingest + Det-Track（人/ID）
- ROI 脚点命中 + 计数/驻留（秒级阈值）
- 简易规则（有人/空岗）+ 事件/状态查询

**迭代 1（Week 3–4）**
- 告警分发 + 证据切片
- Admin 管理台（ROI/规则/相机）
- 看板与可观测指标

**迭代 2（Week 5+）**
- 穿线进入、双阈值滞回、排班/白名单
- 异常合并与多级告警；
- TSDB/存储归档与导出

---

## 14) 配置样例（Rule JSON）
```json
{
  "on_rule": {"min_count": 1, "min_duration_sec": 1.0},
  "off_rule": {"max_count": 0, "min_duration_sec": 10.0},
  "worktime": [{"cron": "0 9 * * 1-5", "duration_min": 540}],
  "enter_filter": true,
  "gap_tolerance_sec": 0.5,
  "det_conf_th": 0.45,
  "min_area_ratio": 0.01,
  "overlap_th": 0.4,
  "footstrip_th": 0.4
}
```

---

## 15) 消息主题定义（Kafka/NATS）
- `video.frames.{camera_id}`
- `person.tracks.{camera_id}`
- `roi.presence.{scene_id}`
- `scene.stats.{scene_id}`
- `scene.event.{scene_id}` / `alarm.created`

载荷字段与上文模块输出保持一致，统一 `ts`（毫秒级 Unix）与 `trace_id`。

---

## 16) 关键注意事项清单（Checklist）
- [ ] 所有阈值采用**秒**配置，运行时按 `fps_est` 换算为帧
- [ ] ROI 命中采用**脚点锚点 + 脚部条带 overlap + 双阈值滞回**
- [ ] `ids=None` 有回退策略，避免全量误判
- [ ] `min_area` 使用**相对比例**参数，适配不同分辨率
- [ ] 事件**仅在边沿**产出，短闪断合并
- [ ] 告警**去重/升级/回执**全流程
- [ ] 证据切片与关键帧留存（生命周期策略）
- [ ] 鉴权与审计日志全覆盖
- [ ] 观测指标与日志采样在灰度期开启

---

## 17) 代码落地对接点
- 你当前的 `ScenePresenceManager`：
  - 改 `_inside` 为脚点/脚部条带方案；
  - `enter_frames/leave_frames` → `enter_sec/leave_sec`；
  - 新增 `step_end()` 产出边沿事件并打印/发送到 `scene.event.*`；
- `run_demo`：
  - 增加 `infer_interval`（每 N 帧推理一次）；
  - `--no-display` 时输出 JSON 到 stdout 或 UDP。

---

> 如需，我可以基于此文档直接生成：数据库 DDL、OpenAPI 草案、Kafka Schema（JSON/Proto），以及 `ScenePresenceManager` 的替换版代码。
