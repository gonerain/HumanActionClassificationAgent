import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from roi_filter import ROIFilter  # noqa: E402


def test_overlap_ratio_full_and_partial():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    rf = ROIFilter(polygon)
    # foot strip fully inside
    bbox_full = (40, 40, 60, 80)
    assert rf._overlap_ratio(bbox_full) == 1.0
    # foot strip half inside
    bbox_half = (80, 80, 120, 100)
    ratio = rf._overlap_ratio(bbox_half)
    assert abs(ratio - 0.5) < 0.05


def test_hysteresis(monkeypatch):
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    rf = ROIFilter(polygon, enter_th=0.5, leave_th=0.3, foot_ratio=0.2)
    bbox = (40, 60, 60, 80)

    # first call, high ratio -> inside
    monkeypatch.setattr(rf, "_overlap_ratio", lambda b: 0.6)
    res = rf.update([(1, bbox)])
    assert res[1] is True

    # ratio below enter_th but above leave_th -> still inside
    monkeypatch.setattr(rf, "_overlap_ratio", lambda b: 0.4)
    res = rf.update([(1, bbox)])
    assert res[1] is True

    # ratio below leave_th -> outside
    monkeypatch.setattr(rf, "_overlap_ratio", lambda b: 0.2)
    res = rf.update([(1, bbox)])
    assert res[1] is False

    # footpoint outside -> always outside
    bbox_out = (120, 60, 140, 80)
    monkeypatch.setattr(rf, "_overlap_ratio", lambda b: 0.9)
    res = rf.update([(2, bbox_out)])
    assert res[2] is False
