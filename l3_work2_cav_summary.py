"""Aggregate existing Work 2 directional sensitivities on Fiji.

This exports no patient rows. Run from the project checkout after the Work 2
patient table has been built.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from concept_study import measures, paths


def main():
    index = measures.load_index()
    records = []
    for build in index["builds"]:
        if paths.PRIMARY_VARIANT not in build["variants"]:
            continue
        observations = measures.load_build(build)
        year = observations["year"]
        probability = observations["probability"].astype(float)
        interior = (probability > 0) & (probability < 1)
        concepts = [
            concept for concept in index["concepts"]
            if (concept["build"], concept["reference"]) == (build["build"], build["reference"])
            and paths.PRIMARY_VARIANT in concept["variants"]
        ]
        for concept in concepts:
            arrays = measures.load_concept(concept)
            for estimator in paths.ESTIMATORS:
                key = f"sensitivity_{estimator}"
                if key not in arrays:
                    continue
                matrix = arrays[key].astype(float)
                assert len(matrix) == len(year)
                valid_fits = np.isfinite(matrix).all(axis=0)
                assert valid_fits.sum() == concept["valid_fits"][estimator]
                for input_year in np.unique(year[year <= paths.LAST_OUTCOME_YEAR]):
                    annual = year == input_year
                    kept = annual & interior
                    if not kept.any() or not valid_fits.any():
                        continue
                    probability_score = matrix[kept][:, valid_fits]
                    p = probability[kept]
                    log_odds_score = probability_score / (p * (1 - p))[:, None]
                    sign_tcav = (probability_score > 0).mean(axis=0)
                    for output, score in (("probability", probability_score), ("log_odds", log_odds_score)):
                        records.append(pd.DataFrame({
                            "build": build["build"], "reference_year": build["reference"],
                            "input_year": input_year, "concept": concept["concept"],
                            "estimator": estimator, "output": output,
                            "fit_index": np.flatnonzero(valid_fits), "rows": int(kept.sum()),
                            "boundary_rows": int((annual & ~interior).sum()),
                            "tcav_sign": sign_tcav, "mean_directional_derivative": score.mean(axis=0),
                            "mean_absolute_directional_derivative": np.abs(score).mean(axis=0),
                        }))
    if not records:
        raise ValueError("No primary Work 2 CAV sensitivities found")
    result = pd.concat(records, ignore_index=True)
    assert np.isfinite(result["mean_directional_derivative"]).all()
    out = Path("stats/l3_work2_cav_summary")
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "work2_cav_scores.csv.gz", index=False,
                  compression={"method": "gzip", "mtime": 0})
    (out / "work2_patient_table_index_hash.txt").write_text(measures.index_hash() + "\n")
    print(f"{result['concept'].nunique()} concepts, {len(result)} aggregated rows")


if __name__ == "__main__":
    main()
