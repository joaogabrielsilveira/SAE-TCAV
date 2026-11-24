"""Execute the delivered notebooks top to bottom, synthetic mode only."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
import pytest

@pytest.mark.parametrize("name",[
    "concept_measurement_reliability","concept_explanations","concept_factor_groups"])
def test_notebook_synthetic_checks(name, monkeypatch):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,CMR_MODE="check",PYTHONDONTWRITEBYTECODE="1")
    for key in ("CMR_OUT","CMR_FREEZE","CMR_PILOT_ROOT","CMR_BUILDS_ROOT",
                "CMR_POPULATION","CMR_PREVIOUS_RUNS"):
        env.pop(key,None)
    env["PATH"]=str(root/".venv/bin")+os.pathsep+env.get("PATH","")
    notebook=nbformat.read(root/(name+".ipynb"),as_version=4)
    assert any("Checks" in str(c.source) for c in notebook.cells if c.cell_type=="markdown")
    executed=NotebookClient(notebook,timeout=300,kernel_name="python3",
                            resources={"metadata":{"path":str(root)}}).execute(env=env)
    outputs=[o for c in executed.cells if c.cell_type=="code" for o in c.get("outputs",[])]
    assert outputs, "Checks must display their results"
    assert not any(o.output_type=="error" for o in outputs)
