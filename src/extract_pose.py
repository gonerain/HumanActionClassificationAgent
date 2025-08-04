from data_pipeline import DataPipeline  # 替换为你的模块名
import os

video_path = "video.mp4"
output_dir = "pose_sequences"
action_label = 0  # 例如：0代表walking，1代表working
window_size = 30  # 每30帧保存一个序列

pipeline = DataPipeline(model_name="yolo11l.pt", conf=0.5)
saved_files = pipeline.process_video(
    video_path=video_path,
    label=action_label,
    output_dir=output_dir,
    window_size=window_size,
    debug=True
)

print(f"保存成功，共 {len(saved_files)} 个片段")
