"""Checks of the alternative-bases experiment: synthetic unit checks plus the SAE reproduction anchor (Fiji only)."""
import numpy as np
import pytest

from concept_study import alternative_bases as ab
from concept_study import paths


def test_pca_factors_are_the_positive_and_negative_parts_of_the_leading_scores():
    rng = np.random.default_rng(0)
    fit = rng.standard_normal((400, 6)) * np.array([5, 3, 2, 1, .5, .1])
    factors = ab.pca_encoder(fit, 5)(fit)
    assert factors.shape == (400, 5) and (factors >= 0).all()
    score = factors[:, 0] - factors[:, 1]
    assert np.all((factors[:, 0] == 0) | (factors[:, 1] == 0))
    assert abs(np.corrcoef(score, fit[:, 0])[0, 1]) > 0.99          # PC1 is the largest-variance axis
    assert np.var(score) > np.var(factors[:, 2] - factors[:, 3])


def test_kmeans_factor_is_membership_margin():
    rng = np.random.default_rng(1)
    fit = np.concatenate([rng.normal(c, 0.3, (100, 2)) for c in ((0, 0), (5, 0), (0, 5))])
    factors = ab.kmeans_encoder(fit, 3, 42)(fit)
    assert ((factors > 0).sum(1) == 1).all()                          # each row is positive in its own cluster only
    assert len(np.unique(factors.argmax(1)[:100])) == 1               # one blob, one factor
    z = np.array([[2.5, 0.0]])                                        # halfway between two centres: margin near 0
    assert ab.kmeans_encoder(fit, 3, 42)(z).max() < 1.0


def test_random_factors_are_relu_of_unit_directions_without_bias():
    z = np.random.default_rng(2).standard_normal((2000, 8))
    z -= z.mean(0)
    factors = ab.random_encoder(8, 30, 7)(z)
    assert factors.shape == (2000, 30) and (factors >= 0).all()
    assert np.allclose((factors > 0).mean(0), 0.5, atol=0.06)
    np.testing.assert_array_equal(factors, ab.random_encoder(8, 30, 7)(z))


def test_matched_cutoff_and_rule_quality():
    projection = np.arange(100.0)
    assert (projection > ab.matched_cutoff(projection, 0.25)).mean() == pytest.approx(0.25, abs=0.01)
    target = np.array([1, 1, 1, 0, 0, 0], bool)
    rule = np.array([1, 1, 0, 1, 0, 0], bool)
    precision, recall, f1 = ab.rule_quality(target, rule)
    assert (precision, recall) == pytest.approx((2 / 3, 2 / 3)) and f1 == pytest.approx(2 / 3)


def test_agreement_replicate_zero_is_share_of_rule_rows_that_activate_and_steps_are_consecutive():
    years = np.array([2010, 2010, 2010, 2011, 2011, 2012])
    in_r = np.array([1, 1, 0, 1, 1, 0], bool)
    in_a = np.array([1, 0, 1, 1, 1, 1], bool)
    weights = np.vstack([np.ones(6), [2, 0, 1, 1, 0, 1]]).astype(np.float32)
    agreement = ab.agreement_replicates(in_a, in_r, years, weights)
    at = ab.YEARS.index(2010)
    assert agreement[0, at] == pytest.approx(0.5) and agreement[0, at + 1] == pytest.approx(1.0)
    assert np.isnan(agreement[0, at + 2])                            # no rule rows in 2012
    assert agreement[1, at] == pytest.approx(1.0)                     # replicate weights are used
    change = ab.steps(agreement, 2010)
    assert change.shape == (2, paths.LAST_YEAR - 2010) and change[0, 0] == pytest.approx(0.5)


def test_rule_direct_cav_points_along_the_rule_axis():
    from semantic_rules import RuleSet
    rng = np.random.default_rng(3)
    features = rng.integers(0, 2, (400, 3)).astype(float)
    embeddings = rng.standard_normal((400, 5)) + 3 * features[:, [0]] * np.array([1, 0, 0, 0, 0])
    rule = RuleSet.from_dict({"rules": [{"rule_id": "r0", "conditions": [{"feature_index": 0, "feature_name": "f0", "operator": ">",
                                                         "threshold": 0.5}]}]})
    direction = ab.cav_direction(embeddings, features, rule, np.arange(400).astype(str))
    assert direction[0] > 0.9 and np.linalg.norm(direction) == pytest.approx(1.0)


@pytest.mark.skipif(not (paths.PATIENT_TABLE / "index.json").exists(), reason="frozen study inputs live on Fiji")
def test_sae_path_reproduces_the_frozen_rules_masks_and_agreement():
    """Anchor: re-running the SAE through this module's path gives the frozen families, rules and agreement."""
    import concept_subgroup_drift_pilot as pilot
    from concept_study import bootstrap, loading, measures
    index = measures.load_index()
    record = next(b for b in index["builds"] if (b["build"], b["reference"]) == ("r1", 2007))
    inputs = loading.open_inputs(ab.DEVICE)
    data = ab.reference_data(inputs, record)
    cap = pilot.candidate_cap(ab.RANDOM_LATENTS, data["cache"].config["candidate_cap_fraction"])
    sae, summary, sweep = ab.basis_concepts("sae", 0, data, ab.sae_encoders(data), cap, "2007/42")
    unique, weights = bootstrap.patient_weights(measures.all_patient_ids(index))
    row_weights = bootstrap.row_weights(data["table"]["patient_id"], unique, weights)
    for c in sae:
        c["agreement"] = ab.agreement_replicates(c["in_A"], c["in_R"], data["table"]["year"], row_weights)
    anchor = ab.sae_anchor(data, sae, summary, sweep, index)
    assert anchor["family_counts_equal"] and anchor["candidates_equal"] and anchor["rule_identities_equal"]
    assert anchor["frozen_rules"] == anchor["index_concepts"] == 4 and anchor["concepts_missing_from_index"] == 0
    assert anchor["A_or_R_row_mismatches"] == 0 and anchor["agreement_max_abs_gap"] < 1e-6
