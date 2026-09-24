from flyvision_adapter.data.labels import load_labels


def test_headerless_file(tmp_path):
    p = tmp_path / "cam.txt"
    p.write_text("1.000000 10.5 20.0\n2.000000 0.0 0.0\n")
    df = load_labels(p)
    assert df.frame.tolist() == [0, 1]
    assert df.visible.tolist() == [True, False]


def test_header_is_skipped(tmp_path):
    p = tmp_path / "cam.txt"
    p.write_text(" frame no.            x            y\n1.000000   0.0   0.0\n")
    assert len(load_labels(p)) == 1


def test_duplicate_frame_keeps_labelled_row(tmp_path):
    p = tmp_path / "cam.txt"
    p.write_text("1 0 0\n2 5.0 6.0\n2 0 0\n3 0 0\n")
    df = load_labels(p)
    assert df.frame.tolist() == [0, 1, 2]
    assert df.visible.tolist() == [False, True, False]


def test_real_files_parse(data_root):
    for ds in ("dataset1", "dataset2", "dataset3", "dataset4"):
        for f in sorted((data_root / ds / "detections").glob("cam*.txt")):
            df = load_labels(f)
            assert len(df) > 1000, f
            assert df.frame.is_unique and df.frame.is_monotonic_increasing and df.frame.iloc[0] == 0, f
            assert 0.2 < df.visible.mean() <= 1.0, f
