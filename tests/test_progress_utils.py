import progress_utils


def test_progress_iter_is_passthrough_when_disabled(monkeypatch):
    def unexpected_tqdm(*args, **kwargs):
        raise AssertionError("disabled progress must not construct tqdm")

    monkeypatch.setattr(progress_utils, "tqdm", unexpected_tqdm)

    assert list(
        progress_utils.progress_iter(
            [1, 2],
            enabled=False,
            desc="ignored",
            unit="item",
        )
    ) == [1, 2]


def test_progress_iter_forwards_stable_tqdm_metadata(monkeypatch):
    captured = {}

    def fake_tqdm(iterable, **kwargs):
        captured.update(kwargs)
        return iterable

    monkeypatch.setattr(progress_utils, "tqdm", fake_tqdm)

    assert list(
        progress_utils.progress_iter(
            [1, 2, 3],
            enabled=True,
            desc="Semantic rule sets",
            total=3,
            unit="target",
            leave=False,
        )
    ) == [1, 2, 3]
    assert captured == {
        "desc": "Semantic rule sets",
        "total": 3,
        "unit": "target",
        "leave": False,
        "dynamic_ncols": True,
    }
