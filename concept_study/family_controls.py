"""Are the Pearson factor families learned? Rebuild them from trained, untrained and rotated dictionaries.

The pilot matched SAE factors across seeds by the Pearson correlation of their activations on
the same reference patients (mutual best match; a "full" family links one factor of each seed).
Trained decoder rows stay close to their random start (cosine about 0.95), so near-random
directions might correlate across seeds without any training. This module rebuilds the families
on the exact matching rows of every study reference build with:

- trained:   the frozen reference dictionaries (must reproduce the cached family counts);
- untrained: the same SAE construction right after torch.manual_seed(seed), i.e. the initial
             weights of the training run, plus extra random seed triplets that differ per reference;
- rotated:   the trained dictionaries with every encoder row rotated by one random orthogonal
             matrix per seed (keeps each dictionary's own geometry and biases, removes alignment).

It also measures, for trained full families: decoder cosine, data-weighted cosine (cosine under
the covariance of the standardized embeddings), the medoid member's activation correlation with
predicted logit risk and with the first principal component (PC1 oriented to correlate positively
with logit risk), how strongly different families correlate with each other, and how far training
moved the decoder rows from their initial weights and in which embedding directions.

    .venv/bin/python -m concept_study.family_controls
"""
import copy

import numpy as np
import pandas as pd
import torch

from concept_study import loading, paths

SEEDS = (42, 135, 215)
CACHES = ("r1", "baseline", "last_3", "all_history")
EXTRA_DRAWS = 20
TOP = (5, 10)
OUTPUT = paths.OUTPUT / "checks"


def matching_inputs(cache, reference):
    """Embeddings and death probabilities of the family matching rows (reference rule_discovery + cav_fit)."""
    import cmr_common
    families = cache.load(f"D/{reference}/families")
    directory = cache._unit_path(f"P/{reference}")
    archive = directory / "arrays.npz"
    rows = cmr_common.read_array_rows(archive, "row_index")
    positions = np.searchsorted(rows, families["row_index"])
    if not np.array_equal(rows[positions], families["row_index"]):
        raise ValueError("Matching rows missing from the predictor archive")
    embeddings = cmr_common.read_array_rows(archive, "embeddings", positions)
    probabilities = cmr_common.read_array_rows(archive, "probabilities", positions)[:, 1]
    return families, embeddings.astype(np.float32), probabilities.astype(np.float64)


def codes_of(model, standardized):
    """ReLU activations of standardized embeddings under one SAE model."""
    with torch.inference_mode():
        values = torch.as_tensor(standardized, dtype=torch.float32, device="cuda")
        encoded = model.to("cuda").encode(values)
        encoded = encoded[1] if isinstance(encoded, tuple) else encoded
        return encoded.float().cpu().numpy()


def untrained_model(trained_model, seed, sae=None):
    """The SAE exactly as constructed at the start of training with this seed.

    sae: the cached unit's SAE config ({"model_type", "scaling_factor"}); without it, ReLU and the
    trained model's latents-per-dimension ratio.
    """
    from sae import SAE
    dimension = trained_model.encoder.in_features
    sae = sae or {"model_type": "ReLU", "scaling_factor": trained_model.num_latents / dimension}
    torch.manual_seed(int(seed))
    return SAE(data_dimension=dimension, scaling_factor=sae["scaling_factor"], use_decoder_bias=True,
               type=sae["model_type"])


def rotated_model(trained_model, rng):
    """Trained SAE with its encoder rows rotated by one random orthogonal matrix."""
    model = copy.deepcopy(trained_model).to("cpu")
    dimension = model.encoder.in_features
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    q = q * np.sign(np.diag(r))
    with torch.no_grad():
        model.encoder.weight.copy_(model.encoder.weight @ torch.as_tensor(q, dtype=torch.float32))
    return model


def unit_rows(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def full_families(result):
    """(members {seed: factor}, medoid (seed, factor)) of every full family of a build_factor_families result."""
    return [({int(s): int(k) for s, k in family["members"]}, tuple(int(v) for v in family["medoid"]))
            for family in result["families"] if family["kind"] == "full"]


def min_pearson(codes, members):
    values = []
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            values.append(np.corrcoef(codes[a][:, members[a]], codes[b][:, members[b]])[0, 1])
    return float(np.min(values))


def median_min_pearson(codes, result):
    return np.median([min_pearson(codes, members) for members, _ in full_families(result)] or [np.nan])


def top_share(matrix, basis):
    """Share of the squared Frobenius norm of the rows that lies in the span of the basis columns."""
    return float(((matrix @ basis) ** 2).sum() / (matrix ** 2).sum())


def training_displacement(initial, trained, directions):
    """How far training moved the decoder rows from their initial weights, and in which directions.

    relative_change: ||trained - initial|| / ||initial|| (Frobenius). change/init/trained_in_topk:
    share of the change, of the initial rows and of the trained rows inside the top-k principal
    directions (columns of `directions`, largest variance first); chance is k / dimension.
    """
    change = trained - initial
    out = {"relative_change": float(np.linalg.norm(change) / np.linalg.norm(initial))}
    for k in TOP:
        basis = directions[:, :k]
        out[f"change_in_top{k}"] = top_share(change, basis)
        out[f"init_in_top{k}"] = top_share(initial, basis)
        out[f"trained_in_top{k}"] = top_share(trained, basis)
    return out


def run_reference(cache_name, cache, reference):
    import concept_subgroup_drift_pilot as pilot
    families, embeddings, probability = matching_inputs(cache, reference)
    dictionaries = pilot.load_reference_dictionaries(cache, reference, "cpu")
    scaler = dictionaries[SEEDS[0]]["scaler"]
    standardized = scaler.transform(embeddings).astype(np.float32)
    trained = {s: dictionaries[s]["run"]["model"] for s in SEEDS}
    sae = {s: cache.load(f"S/{s}/{reference}").get("sae") for s in SEEDS}

    def families_of(codes, tag):
        return pilot.build_factor_families([codes[s] for s in SEEDS], list(SEEDS), device="cuda",
                                           id_prefix=f"{cache_name}/{reference}/{tag}")

    def count_row(dictionary, draw, codes, result):
        return {"cache": cache_name, "reference": reference, "dictionary": dictionary, "draw": draw,
                **result["counts"], "median_min_pearson": median_min_pearson(codes, result)}

    trained_codes = {s: codes_of(trained[s], standardized) for s in SEEDS}
    result = families_of(trained_codes, "trained")
    cached = families["construction"]["counts"]
    if result["counts"] != cached:
        raise ValueError(f"{cache_name} {reference}: rebuilt counts {result['counts']} differ from cached {cached}")
    rows = [count_row("trained", 0, trained_codes, result)]

    # Untrained: draw 0 is the initial weights of the actual seeds; later draws use seeds
    # reference * 100000 + draw * 10 + k, so control draws differ between references.
    initial = {}
    for draw in range(EXTRA_DRAWS + 1):
        seeds = SEEDS if draw == 0 else tuple(100_000 * reference + 10 * draw + k for k in range(3))
        models = {s: untrained_model(trained[s], t, sae[s]) for s, t in zip(SEEDS, seeds)}
        if draw == 0:
            initial = {s: models[s].encoder.weight.detach().numpy().astype(np.float64) for s in SEEDS}
        codes = {s: codes_of(models[s], standardized) for s in SEEDS}
        rows.append(count_row("untrained", draw, codes, families_of(codes, f"untrained{draw}")))
    rng = np.random.default_rng([20261110, reference, CACHES.index(cache_name)])
    for draw in range(1, EXTRA_DRAWS + 1):
        models = {s: rotated_model(trained[s], rng) for s in SEEDS}
        codes = {s: codes_of(models[s], standardized) for s in SEEDS}
        rows.append(count_row("rotated", draw, codes, families_of(codes, f"rotated{draw}")))

    # Embedding geometry; PC1 oriented to correlate positively with logit risk.
    covariance = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    pc1 = (standardized - standardized.mean(0)) @ eigenvectors[:, 0]
    pc1 = pc1 * (1 if np.corrcoef(pc1, logit)[0, 1] >= 0 else -1)

    # Trained decoder rows against their initial weights.
    weights = {s: trained[s].encoder.weight.detach().cpu().numpy().astype(np.float64) for s in SEEDS}
    displacement = [{"cache": cache_name, "reference": reference, "seed": s,
                     **training_displacement(initial[s], weights[s], eigenvectors)} for s in SEEDS]
    init_cos = [float(np.median(np.sum(unit_rows(initial[s]) * unit_rows(weights[s]), axis=1))) for s in SEEDS]

    # Trained full families: member geometry, then the medoid member's profile.
    pairs, profile, medoid_codes = [], [], []
    for index, (members, (medoid_seed, medoid_factor)) in enumerate(full_families(result)):
        for i, a in enumerate(SEEDS):
            for b in SEEDS[i + 1:]:
                wa, wb = weights[a][members[a]], weights[b][members[b]]
                pairs.append({
                    "cache": cache_name, "reference": reference, "family": index, "seed_a": a, "seed_b": b,
                    "pearson": float(np.corrcoef(trained_codes[a][:, members[a]], trained_codes[b][:, members[b]])[0, 1]),
                    "decoder_cosine": float(wa @ wb / np.linalg.norm(wa) / np.linalg.norm(wb)),
                    "data_weighted_cosine": float(wa @ covariance @ wb / np.sqrt((wa @ covariance @ wa) * (wb @ covariance @ wb))),
                })
        activation = trained_codes[medoid_seed][:, medoid_factor]
        medoid_codes.append(activation)
        profile.append({"cache": cache_name, "reference": reference, "family": index,
                        "medoid_seed": medoid_seed, "medoid_factor": medoid_factor,
                        "corr_logit_risk": float(np.corrcoef(activation, logit)[0, 1]),
                        "corr_pc1": float(np.corrcoef(activation, pc1)[0, 1]),
                        "active_share": float((activation > 0).mean())})
    if len(medoid_codes) > 1:
        upper = np.abs(np.corrcoef(np.stack(medoid_codes))[np.triu_indices(len(medoid_codes), 1)])
        between_median, between_max = float(np.median(upper)), float(np.max(upper))
    else:
        between_median = between_max = np.nan
    variance = eigenvalues / eigenvalues.sum()
    summary = {"cache": cache_name, "reference": reference, "rows": len(standardized),
               "init_to_trained_cosine_median": float(np.median(init_cos)),
               "variance_top1": float(variance[:1].sum()), "variance_top5": float(variance[:5].sum()),
               "variance_top10": float(variance[:10].sum()),
               "participation_ratio": float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()),
               "between_family_abs_corr_median": between_median, "between_family_abs_corr_max": between_max}
    return rows, pairs, profile, summary, displacement


def plot_family_controls(counts, pairs, summaries, path):
    """Member Pearson against plain and data-weighted cosine; full-family totals per build against
    the 20 untrained and 20 rotated draws; embedding variance in the top principal directions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, grey = "#2a78d6", "#eb6834", "#8a8984"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    ax.scatter(pairs["decoder_cosine"], pairs["pearson"], s=10, alpha=0.6, color=blue, label="decoder cosine")
    ax.scatter(pairs["data_weighted_cosine"], pairs["pearson"], s=10, alpha=0.6, color=orange, label="data-weighted cosine")
    ax.set(xlabel="cosine between member directions", ylabel="activation Pearson",
           title="Full-family members: plain vs data-weighted cosine")
    ax.legend(frameon=False)

    ax = axes[1]
    totals = counts.groupby(["cache", "dictionary", "draw"])["full"].sum().reset_index()
    caches = [c for c in CACHES if c in set(totals["cache"])]
    jitter = np.random.default_rng(0)
    for x, cache in enumerate(caches):
        t = totals[totals["cache"] == cache]
        for offset, dictionary, colour in ((-0.2, "untrained", grey), (0.2, "rotated", orange)):
            values = t[(t["dictionary"] == dictionary) & (t["draw"] > 0)]["full"]
            ax.scatter(x + offset + jitter.uniform(-0.06, 0.06, len(values)), values, s=16, color=colour,
                       label=f"{dictionary} ({len(values)} draws)" if x == 0 else None)
        ax.scatter([x], t[t["dictionary"] == "trained"]["full"], marker="D", s=70, color=blue,
                   label="trained" if x == 0 else None)
    ax.set_xticks(range(len(caches)), caches)
    ax.set(ylabel="full families, all references", title="Full families: trained vs untrained / rotated")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for _, row in summaries.iterrows():
        ax.plot([1, 5, 10], [row["variance_top1"], row["variance_top5"], row["variance_top10"]], color=blue, alpha=0.35)
    ax.set(ylim=(0, 1.02), xlabel="top principal directions", ylabel="share of embedding variance",
           title="Embedding variance in the top principal directions")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    inputs = loading.open_inputs("cuda")
    counts, pairs, profiles, summaries, displacement = [], [], [], [], []
    for cache_name in CACHES:
        cache = inputs["caches"][cache_name]
        for reference in paths.REFERENCES:
            key = f"D/{reference}/families"
            if key not in cache.graph or not cache.is_complete(key):  # no families for this build and reference
                continue
            c, p, f, s, d = run_reference(cache_name, cache, reference)
            counts += c
            pairs += p
            profiles += f
            summaries.append(s)
            displacement += d
            trained = [r for r in c if r["dictionary"] == "trained"][0]["full"]
            untrained = [r["full"] for r in c if r["dictionary"] == "untrained"]
            rotated = [r["full"] for r in c if r["dictionary"] == "rotated"]
            print(f"{cache_name} {reference}: full trained {trained}, untrained actual-seed {untrained[0]} "
                  f"(other draws mean {np.mean(untrained[1:]):.1f}, max {max(untrained[1:])}), "
                  f"rotated mean {np.mean(rotated):.1f} max {max(rotated)}; init cos {s['init_to_trained_cosine_median']:.3f}",
                  flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    counts, pairs, summaries = pd.DataFrame(counts), pd.DataFrame(pairs), pd.DataFrame(summaries)
    counts.to_csv(OUTPUT / "family_control_counts.csv", index=False)
    pairs.to_csv(OUTPUT / "family_member_pairs.csv", index=False)
    pd.DataFrame(profiles).to_csv(OUTPUT / "family_profiles.csv", index=False)
    summaries.to_csv(OUTPUT / "family_reference_summary.csv", index=False)
    pd.DataFrame(displacement).to_csv(OUTPUT / "training_displacement.csv", index=False)
    plot_family_controls(counts, pairs, summaries, OUTPUT / "figures" / "family_controls.png")


if __name__ == "__main__":
    main()
