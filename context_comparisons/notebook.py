"""Add a readable comparison section to the new intervention notebook only."""

import json
import os
from pathlib import Path


def append(path=Path("context_intervention.ipynb")):
    notebook = json.loads(path.read_text())
    notebook["cells"] = [cell for cell in notebook["cells"]
                         if not cell.get("id", "").startswith("additional-comparisons-")]
    texts = [
        ("markdown", "## Additional comparisons\n\n"
         "These additions run separately from the original study. Each changes one part of the method "
         "relative to a corrected cumulative arm computed by the comparison itself. The corrections: sampled "
         "contexts share fixed per-row random keys with the uniform draw (hard and graded draws keep the "
         "uniform rows that stay eligible), equal removal priorities are broken by similarity to the "
         "representative patients, and the original stable 60/40 patient hash split is preserved. "
         "Future outcomes and entrants do not affect assignment. Mortality counts may be uneven. Data roles, the context pool and frozen thresholds are unchanged.\n\n"
         "**Latest year:** candidate discovery and validation use only the completed update year. "
         "Feature and concept normalization still use the frozen reference statistics.\n\n"
         "**Native context states:** extract the labeled context states during context setup, without "
         "presenting those examples as prediction queries. The same frozen SAE is applied as an explicit "
         "transfer comparison. Spread, active counts, rule membership, outcome/year strata and relative "
         "reconstruction error accompany the results. Final context states describe a model endpoint; they "
         "are not the per-layer attention keys/values. The original context-as-query arm includes each "
         "example's own labeled row in its context and is not leave-one-out. Native retrieval falls back to the "
         "original full context when no partially retrieved year/outcome stratum has useful cosine discrimination.\n\n"
         "**Selection objective:** compare Death-F1, Brier skill, and AUPRC (average precision) on the same "
         "cached validation predictions. Candidate priorities remain F1-based. The cheap comparison launches "
         "no new inference; any newly chosen option lacking cached test predictions is marked unavailable. "
         "Coverage depends on the selected policy: available results cannot establish objective equivalence "
         "or a global ranking. Changed and unchanged choices are counted separately.\n\n"
         "**Comparison safeguards:** this added report uses frozen thresholds only; equivalent secondary "
         "threshold tuning is unavailable for the new arms. Effective context form, size and fallback status "
         "are explicit. Full-context fallbacks are excluded from common-size guided-sampling estimates. "
         "Matching random direct controls are reused only where cached; missing controls are reported.\n"),
        ("code", "import json\nfrom pathlib import Path\nimport pandas as pd\n"
         "comparison_output = Path('stats/context_comparisons')\n"
         "status_file = comparison_output / 'status.json'\n"
         "json.loads(status_file.read_text()) if status_file.exists() else {'phase': 'not started'}\n"),
        ("code", "choices_file = comparison_output / 'metric_choices.csv'\n"
         "coverage_file = comparison_output / 'objective_coverage.csv'\n"
         "display(pd.read_csv(coverage_file)) if coverage_file.exists() else print('Coverage pending')\n"
         "pd.read_csv(choices_file) if choices_file.exists() else 'Metric comparison pending'\n"),
        ("markdown", "The additional inference is queued after the original process finishes. Its log is "
         "`stats/context_comparisons/comparisons.log`. Final tables distinguish completed evaluation cells "
         "from metric choices that would require extra inference. The native-context comparison retains "
         "known outcome labels in context states and does not establish identical concept semantics across roles.\n"),
    ]
    for index, (kind, text) in enumerate(texts):
        cell = {"id": f"additional-comparisons-{index}", "cell_type": kind, "metadata": {},
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook["cells"].append(cell)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(notebook, indent=1) + "\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    append()
