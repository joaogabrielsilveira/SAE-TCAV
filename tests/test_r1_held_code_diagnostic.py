import numpy as np
import pytest


def test_code_comparison_uses_saved_precision_and_counts_mismatches():
    from r1_held_code_diagnostic import _code_comparison

    saved = np.array([[1.0, 1.0, -0.0]], dtype=np.float16)
    fresh = np.array([[1.0001, 1.001, 0.0]], dtype=np.float32)
    assert _code_comparison(fresh, saved) == {
        "equal": False,
        "mismatched_entries": 1,
        "entries": 3,
        "max_abs_error": 2 ** -10,
    }
    assert _code_comparison(fresh[:, :1], saved[:, :1])["equal"]


def test_encoder_arms_use_the_same_prefix_and_genuine_companions():
    from r1_held_code_diagnostic import _encode_prefix

    embeddings = np.arange(256, dtype=np.float32).reshape(256, 1)
    calls = []

    def encode(dictionary, inputs, device, batch_size):
        calls.append((dictionary, inputs.copy(), device, batch_size))
        return inputs

    for device in ("cpu", "cuda"):
        for block in (64, 256):
            prefix = _encode_prefix(encode, {42: "frozen"}, embeddings, device, block)
            np.testing.assert_array_equal(prefix[42], embeddings[:64])
            dictionary, inputs, actual_device, size = calls[-1]
            assert (dictionary, actual_device, size) == ("frozen", device, block)
            np.testing.assert_array_equal(inputs, embeddings[:block])


def test_full_held_capture_preserves_order_and_final_partial_batch():
    import r1_held_code_diagnostic as diagnostic
    import pandas as pd

    frame = pd.DataFrame({"row": np.arange(2476)})
    batches = []

    def capture(_pilot, _predictor, batch, _device):
        rows = batch["row"].to_numpy()
        batches.append(rows.copy())
        return rows.astype(float), rows[:, None].astype(float)

    probabilities, embeddings = diagnostic._capture_held_batches(capture, None, None, frame, "cuda")
    assert [len(batch) for batch in batches] == [64] * 38 + [44]
    np.testing.assert_array_equal(probabilities, np.arange(2476))
    np.testing.assert_array_equal(embeddings[:, 0], np.arange(2476))


def test_full_table_encoding_uses_all_rows_and_partial_final_block():
    import r1_held_code_diagnostic as diagnostic

    embeddings = np.arange(2476, dtype=np.float32)[:, None]
    calls = []

    def encode(dictionary, inputs, device, batch_size):
        calls.append((dictionary, inputs.copy(), device, batch_size))
        return inputs

    codes = diagnostic._encode_full_table(encode, {42: "frozen"}, embeddings, "cuda")
    np.testing.assert_array_equal(codes[42], embeddings)
    dictionary, inputs, device, size = calls[0]
    assert (dictionary, device, size) == ("frozen", "cuda", 256)
    np.testing.assert_array_equal(inputs, embeddings)
    assert len(inputs) % size == 172


def test_full_table_evidence_requires_complete_finite_archive_and_all_controls():
    import r1_held_code_diagnostic as diagnostic

    saved = {seed: np.ones((3, 2), dtype=np.float16) for seed in (42, 135, 215)}
    fresh = {seed: values.astype(np.float32) for seed, values in saved.items()}
    probabilities = np.array([0.1, 0.2, 0.3])
    evidence = diagnostic._full_table_evidence(fresh, saved, probabilities, probabilities.copy(), 3)
    assert evidence["complete"]
    assert evidence["coverage"]["held_rows"] == 3
    assert evidence["codes"]["equal"]
    fresh[42][1, 0] = 2
    assert not diagnostic._full_table_evidence(fresh, saved, probabilities, probabilities, 3)["complete"]
    assert not diagnostic._full_table_evidence(fresh, saved, probabilities + 1, probabilities, 3)["complete"]
    with pytest.raises(ValueError):
        diagnostic._full_table_evidence(fresh, saved, probabilities[:-1], probabilities, 3)
    with pytest.raises(ValueError):
        diagnostic._full_table_evidence(fresh, {42: saved[42]}, probabilities, probabilities, 3)
    saved[135][0, 0] = np.nan
    with pytest.raises(ValueError):
        diagnostic._full_table_evidence(fresh, saved, probabilities, probabilities, 3)


def test_full_table_schedule_rejects_reordered_or_missing_held_rows():
    from types import SimpleNamespace
    import pandas as pd
    import r1_held_code_diagnostic as diagnostic

    rows = 2476
    actual = {"patient_id": np.arange(rows), "year": np.full(rows, 2008),
              "probability": np.zeros(rows)}
    held = {"patient_id": np.arange(rows), "year": np.full(rows, 2008),
            "probability_swapped": np.zeros(rows)}
    population = {"patient_id": np.arange(rows), "year": np.full(rows, 2008)}

    def select(held_archive, actual_archive, reference, count, precheck):
        return {"held_rows": np.arange(count),
                "held_actual_rows": held_archive["patient_id"][:count],
                "precheck_rows": np.arange(precheck)}

    replay = SimpleNamespace(select_held_batch=select, select_held_probe_rows=lambda *args: {},
                             select_query_batch=lambda *args, **kwargs: np.array([0]))
    pilot = SimpleNamespace(_prediction_frame=lambda population, indices, columns:
                            pd.DataFrame({"patient_id": np.asarray(indices), "year": 2008}))
    attribution = SimpleNamespace(_row_keys=lambda table, name:
                                  list(zip(table["patient_id"].astype(str), table["year"])))
    arguments = (pilot, replay, attribution, {"saved_batch_rows": 1}, population, [], actual,
                 held, {"row_index": np.arange(rows)})
    schedule = diagnostic._query_schedule(*arguments, full_table=True)
    np.testing.assert_array_equal(schedule[2]["patient_id"], np.arange(rows))
    held["patient_id"][[0, 1]] = held["patient_id"][[1, 0]]
    with pytest.raises(ValueError, match="cover all eligible actual rows in order"):
        diagnostic._query_schedule(*arguments, full_table=True)
    held["year"] = held["year"][:-1]
    with pytest.raises(ValueError, match="cover all eligible actual rows in order"):
        diagnostic._query_schedule(*arguments, full_table=True)
