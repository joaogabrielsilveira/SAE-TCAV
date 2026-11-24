"""Create the notebook using only the Python standard library."""

import json
from pathlib import Path


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def make_notebook(path):
    cells = [
        markdown("# Context interventions using the frozen predictors\n\n"
                 "Same-year inputs and outcomes are retained. The frozen evaluation population and "
                 "predictors are reused; 2015 outcomes are excluded. The intervention runner is under "
                 "verification and no scientific intervention results should be inferred from the technical checks.\n"),
        code("from pathlib import Path\nimport pandas as pd\n"
             "from concept_study import measures\n"
             "from context_intervention.analysis import primary_builds, save_cached_analysis\n"
             "output = Path('stats/context_intervention')\n"
             "index = measures.load_index()\n"
             "pd.DataFrame(primary_builds(index))[['build', 'reference', 'file']]\n"),
        markdown("## Cached baseline and error profiles\n\n"
                 "Calculates yearly prediction metrics and feature, concept and concept–feature "
                 "profiles by the four prediction groups. Results are saved as compressed CSV files.\n"),
        code("tables = save_cached_analysis(output)\n"
             "tables['baseline_performance']\n"),
        code("tables['concept_groups'].head()\n"),
        markdown("## Context neighborhoods and development comparisons\n\n"
                 "The helpers compare feature and concept neighborhoods, then independently validate "
                 "direct removal, hard selection with sampling, and graded sampling. Group exclusion "
                 "scores guide selection; they do not identify individual causal effects.\n"),
        markdown("## Selected interventions and temporal evaluation\n\n"
                 "The full run can take many hours. Enable the flag deliberately after verification. "
                 "Each update uses earlier development outcomes and evaluates the following year. "
                 "The cached original, matched random removals, and uniform sampling provide controls.\n"),
        code("RUN_FULL_STUDY = False\n"
             "if RUN_FULL_STUDY:\n"
             "    from context_intervention.study import run_all\n"
             "    run_all(output)\n"),
        markdown("## Summaries and interpretation\n\n"
                 "Compute paired patient-bootstrap intervals after all runs finish. Sampling-seed "
                 "results are reported separately. Same-year record truncation limits prospective "
                 "mortality interpretations.\n"),
        code("if RUN_FULL_STUDY:\n"
             "    from context_intervention.report import summarize\n"
             "    print(summarize(output))\n"),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"section-{index}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
    Path(path).write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    make_notebook("context_intervention.ipynb")
