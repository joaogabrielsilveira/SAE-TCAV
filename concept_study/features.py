"""Who the patients are: original input feature profiles inside each patient group.

For every concept, group and year: mean and share of non-zero values of each of the
500 input features, and standardized mean differences (SMD) for six comparisons:

- smd_vs_reference: the group in a year versus the same group in the reference year (feature shift),
- smd_vs_rest: the group versus every other evaluation patient of the same year,
- smd_stopped_vs_kept: rule patients who no longer activate the factor (R+A-) versus those who do (R+A+),
- smd_activation_only_vs_both / _vs_neither: patients who activate the factor without matching the
  rule (R-A+) versus R+A+, and versus patients in neither group,
- smd_entering_vs_returning (A and R only): patients of the group in a year who had no evaluation
  row in the previous year ("entering": new to the cohort or back after a gap) versus patients of the
  group who had one ("returning", whatever group they were in that year).

SMD = (mean_1 - mean_2) / sqrt((var_1 + var_2) / 2); undefined (NaN) when the feature is
constant in both groups. "Feature shift" of a group-year = mean |SMD versus reference year|
over the frozen rule's own features, and over all features. Nothing is labelled "drifted".
Cached files carry the hash of index.json they were computed from (see measures).
"""
import numpy as np
import torch

from concept_study import bootstrap, measures, paths

FEATURE_OUTPUT = paths.OUTPUT / "features"


def _moments(sums: torch.Tensor) -> tuple:
    """(count, mean, variance, non-zero share) from sums of [1, x, x^2, x != 0]."""
    features = (sums.shape[-1] - 1) // 3
    count = sums[..., :1]
    safe = count.clamp_min(1e-12)
    mean = sums[..., 1:1 + features] / safe
    second = sums[..., 1 + features:1 + 2 * features] / safe
    nonzero = sums[..., 1 + 2 * features:] / safe
    variance = (second - mean ** 2).clamp_min(0)
    empty = count <= 0
    nan = torch.tensor(float("nan"), device=sums.device)
    return (count[..., 0], torch.where(empty, nan, mean), torch.where(empty, nan, variance),
            torch.where(empty, nan, nonzero))


def smd(mean_1, var_1, mean_2, var_2):
    """Standardized mean difference with pooled SD; NaN when both groups are constant."""
    pooled = torch.sqrt((var_1 + var_2) / 2)
    nan = torch.full_like(pooled, float("nan"))
    return torch.where(pooled > 1e-12, (mean_1 - mean_2) / pooled.clamp_min(1e-12), nan)


def _mean_abs(values: torch.Tensor, columns: np.ndarray) -> torch.Tensor:
    """Mean |SMD| over the chosen feature columns, ignoring undefined features."""
    chosen = values[..., columns].abs()
    finite = torch.isfinite(chosen)
    total = torch.where(finite, chosen, torch.zeros_like(chosen)).sum(-1)
    count = finite.sum(-1)
    return torch.where(count > 0, total / count.clamp_min(1), torch.full_like(total, float("nan")))


def concept_features(index: dict, record: dict, unique_ids: np.ndarray, weight_matrix: np.ndarray) -> dict:
    """Feature profiles and the six SMDs of one concept (point values plus patient-bootstrap intervals)."""
    build = measures.load_build(measures.build_record(index, record))
    concept = measures.load_concept(record)
    weights = bootstrap.row_weights(build["patient_id"], unique_ids, weight_matrix)
    names = list(build["feature_names"])
    x = build["features"].astype(np.float32)
    values = np.concatenate([np.ones((len(x), 1), np.float32), x, x ** 2, (x != 0).astype(np.float32)], axis=1)

    groups = measures.group_masks(concept)                                  # rows x 7
    years = np.stack([build["year"] == year for year in measures.YEARS], 1)  # rows x 9
    present = measures.presence(build)
    previous = np.zeros_like(years)
    previous[:, 1:] = present[:, :-1]                                        # seen the year before
    base_cells = (groups[:, :, None] & years[:, None, :]).reshape(len(x), -1)
    flow_groups = groups[:, [measures.GROUPS.index("A"), measures.GROUPS.index("R")]]
    flow_cells = (flow_groups[:, :, None, None] & years[:, None, :, None]
                  & np.stack([previous, ~previous], -1)[:, None, :, :]).reshape(len(x), -1)
    sums = bootstrap.weighted_sums(weights, np.concatenate([base_cells, flow_cells], 1), values)
    n_base = base_cells.shape[1]
    count, mean, var, nonzero = _moments(sums[:, :n_base].reshape(len(weights), len(measures.GROUPS), len(measures.YEARS), -1))
    f_count, f_mean, f_var, _ = _moments(sums[:, n_base:].reshape(len(weights), 2, len(measures.YEARS), 2, -1))

    all_position = measures.GROUPS.index("all")
    rest_sums = (sums[:, :n_base].reshape(len(weights), len(measures.GROUPS), len(measures.YEARS), -1)[:, all_position:all_position + 1]
                 - sums[:, :n_base].reshape(len(weights), len(measures.GROUPS), len(measures.YEARS), -1))
    _, rest_mean, rest_var, _ = _moments(rest_sums)

    reference = measures.YEARS.index(record["reference"])
    versus_reference = smd(mean, var, mean[:, :, reference:reference + 1], var[:, :, reference:reference + 1])
    versus_rest = smd(mean, var, rest_mean, rest_var)
    lost = measures.GROUPS.index("R+A-")
    kept = measures.GROUPS.index("R+A+")
    stopped_vs_kept = smd(mean[:, lost], var[:, lost], mean[:, kept], var[:, kept])
    activation_only, neither = measures.GROUPS.index("R-A+"), measures.GROUPS.index("neither")
    activation_only_vs_both = smd(mean[:, activation_only], var[:, activation_only], mean[:, kept], var[:, kept])
    activation_only_vs_neither = smd(mean[:, activation_only], var[:, activation_only], mean[:, neither], var[:, neither])
    entering_vs_returning = smd(f_mean[..., 1, :], f_var[..., 1, :], f_mean[..., 0, :], f_var[..., 0, :])

    rule_columns = np.array([names.index(f) for f in record["rule_features"] if f in names], dtype=int)
    shift_rule = _mean_abs(versus_reference, rule_columns)
    shift_all = _mean_abs(versus_reference, np.arange(len(names)))

    out = {"feature_names": np.array(names), "rule_columns": rule_columns,
           "mean": mean[0].cpu().numpy(), "nonzero_share": nonzero[0].cpu().numpy(), "count": count[0].cpu().numpy(),
           "feature_shift_rule": shift_rule.cpu().numpy(), "feature_shift_all": shift_all.cpu().numpy(),
           "flow_count": f_count[0].cpu().numpy()}
    for name, value in (("smd_vs_reference", versus_reference), ("smd_vs_rest", versus_rest),
                        ("smd_stopped_vs_kept", stopped_vs_kept), ("smd_entering_vs_returning", entering_vs_returning),
                        ("smd_activation_only_vs_both", activation_only_vs_both),
                        ("smd_activation_only_vs_neither", activation_only_vs_neither)):
        low, high = bootstrap.gpu_percentiles(value[1:])
        out[name] = value[0].cpu().numpy()
        out[f"{name}_low"], out[f"{name}_high"] = low, high
    return out


def compute_all_features(force: bool = False) -> None:
    """Compute and cache every concept's feature file (again when built from another index.json)."""
    index, current = measures.load_index(), measures.index_hash()
    FEATURE_OUTPUT.mkdir(parents=True, exist_ok=True)
    unique_ids, weight_matrix = bootstrap.patient_weights(measures.all_patient_ids(index))
    for record in index["concepts"]:
        target = FEATURE_OUTPUT / record["file"]
        if target.exists() and not force and measures.stored_hash(target) == current:
            continue
        result = concept_features(index, record, unique_ids, weight_matrix)
        measures.save_atomic(target, index_hash=current,
                             **{k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in result.items()})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_features(record: dict) -> dict:
    with np.load(FEATURE_OUTPUT / record["file"]) as stored:
        if "index_hash" not in stored.files or str(stored["index_hash"]) != measures.index_hash():
            raise ValueError(f"{record['file']}: features were built from another patient table; run compute_all_features()")
        return {name: stored[name] for name in stored.files if name != "index_hash"}
