from flyvision_adapter.data.clips import dev_starts
from flyvision_adapter.data.clips import test_starts as held_out_starts
from flyvision_adapter.data.labels import load_labels


def test_dev_and_test_clips_are_disjoint_and_non_overlapping(data_root):
    for f in sorted(data_root.glob("dataset*/detections/cam*.txt")):
        lab = load_labels(f)
        n = int(lab.frame.max()) + 1
        test, dev = held_out_starts(lab, n), dev_starts(lab, n)
        assert test and dev, f
        # same 150-frame grid -> distinct starts never overlap
        assert not set(test) & set(dev), f
        assert all(s % 150 == 0 for s in test + dev), f
