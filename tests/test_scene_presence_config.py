from pathlib import Path
from scene_presence import load_config, save_config, ScenePresenceManager

def test_timing_section_roundtrip(tmp_path):
    default = {
        "region": None,
        "model_name": "yolo11s",
        "conf": 0.5,
        "timing": {"enter_s": 0.5, "leave_s": 1.0, "finish_s": None},
        "classes": ["person"],
        "min_area": 10,
    }
    cfg = tmp_path / "cfg.json"
    cfg_data = load_config(cfg, default)
    assert cfg_data["timing"]["enter_s"] == 0.5

    cfg_data["timing"]["enter_s"] = 1.5
    save_config(cfg, cfg_data)
    loaded = load_config(cfg, default)
    assert loaded["timing"]["enter_s"] == 1.5

    mgr = ScenePresenceManager(
        enter_s=loaded["timing"]["enter_s"],
        leave_s=loaded["timing"]["leave_s"],
    )
    assert mgr.enter_ms == 1500.0
    assert mgr.leave_ms == 1000.0
