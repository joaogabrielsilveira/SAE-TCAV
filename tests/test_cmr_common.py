import hashlib
import numpy as np
import pytest
from cmr_common import (verify_roles, patient_weights, verify_order, verify_hash,
                        public_records, read_array_rows)

def test_patient_role_overlap_refused():
    with pytest.raises(ValueError, match="role"):
        verify_roles(["a","b","a"],["fit","tune","evaluation"])

def test_order_and_hash_refused(tmp_path):
    with pytest.raises(ValueError, match="order"):
        verify_order([2,1],[1,2])
    f=tmp_path/"source"; f.write_bytes(b"original")
    verify_hash(f,hashlib.sha256(b"original").hexdigest())
    f.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        verify_hash(f,hashlib.sha256(b"original").hexdigest())

def test_patient_cluster_weights_shared_across_rows():
    w=patient_weights(["a","b","a"],seeds=range(3001,3011))
    assert w.shape==(10,3)
    np.testing.assert_array_equal(w[:,0],w[:,2])
    np.testing.assert_array_equal(w[:,[0,1]].sum(1),np.full(10,2))
    np.testing.assert_array_equal(w,patient_weights(["a","b","a"],seeds=range(3001,3011)))

@pytest.mark.parametrize("record",[
    {"patient_id":"private"},{"metric":[1,2]}, {"metric":{"patient_id":"private"}},
    {"metric":float("nan")}, {"metric":"a"*64}])
def test_export_unknown_nested_private_rejected(record):
    with pytest.raises((ValueError,TypeError)):
        public_records([record],{"metric"})

def test_export_scalar_schema():
    assert public_records([{"metric":.3,"n":4}],{"metric","n"})==[{"metric":.3,"n":4}]

def test_npz_selects_only_requested_rows_without_repair(tmp_path):
    f=tmp_path/"arrays.npz"
    np.savez(f, embeddings=np.arange(24).reshape(6,4), other=np.zeros(3))
    before=f.read_bytes()
    np.testing.assert_array_equal(read_array_rows(f,"embeddings",[1,4]),np.arange(24).reshape(6,4)[[1,4]])
    assert f.read_bytes()==before
    with pytest.raises(ValueError,match="compressed"):
        g=tmp_path/"compressed.npz"; np.savez_compressed(g,x=np.zeros((2,2)))
        read_array_rows(g,"x",[0])


def test_private_freeze_tamper_source_and_absence(tmp_path, monkeypatch):
    import time
    from cmr_common import freeze_read, freeze_write
    ctx={"out":tmp_path,"mode":"develop","started":time.monotonic(),
         "branch":"fixture","source_hashes":{"notebook":"same"},"input_files":{}}
    (tmp_path/"private").mkdir()
    path=tmp_path/"private/freeze.pkl"
    monkeypatch.setenv("CMR_FREEZE",str(path))
    with pytest.raises(FileNotFoundError):
        freeze_read(ctx)
    freeze_write(ctx,{"coefficient":np.array([1.,2.])})
    np.testing.assert_array_equal(freeze_read(ctx)["coefficient"],[1.,2.])
    with pytest.raises(FileExistsError):
        freeze_write(ctx,{})
    with pytest.raises(ValueError,match="source"):
        freeze_read({**ctx,"source_hashes":{"notebook":"changed"}})
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError,match="hash"):
        freeze_read(ctx)

def test_freeze_input_fingerprints_verified(tmp_path, monkeypatch):
    import time
    from cmr_common import freeze_read, freeze_write, file_hash
    inp=tmp_path/"historical.json"; inp.write_text("original")
    (tmp_path/"private").mkdir()
    ctx={"out":tmp_path,"mode":"develop","started":time.monotonic(),"branch":"fixture",
         "source_hashes":{},"input_files":{str(inp):file_hash(inp)}}
    freeze_write(ctx,{})
    monkeypatch.setenv("CMR_FREEZE",str(tmp_path/"private/freeze.pkl"))
    inp.write_text("changed")
    with pytest.raises(ValueError,match="hash"):
        freeze_read(ctx)

def test_effective_build_aliases():
    from cmr_common import effective_build
    assert effective_build("last_3",2007)==effective_build("all_history",2007)=="r1"
    assert effective_build("last_3",2008)==effective_build("all_history",2008)=="all_history"
    assert effective_build("last_3",2010)!="all_history"
    assert effective_build("baseline",2013)=="r1"

def test_cache_readonly_refuses_repair(tmp_path):
    from concept_subgroup_drift_pilot import UnitCache, config_hash, HASH_EXCLUDED_KEYS
    config={"reference_year_candidates":[2007],"sae_seeds":[42,135,215],
            "disk_guard_gib":15,"t0_only_reference_years":[]}
    directory=tmp_path/config_hash(config);directory.mkdir()
    import json
    (directory/"config.json").write_text(json.dumps({k:v for k,v in config.items() if k not in HASH_EXCLUDED_KEYS}))
    manifest=directory/"units.jsonl";manifest.write_text("truncated")
    before=manifest.read_bytes()
    with pytest.raises(ValueError,match="truncated"):
        UnitCache(tmp_path,config,"test-source",read_only=True)
    assert manifest.read_bytes()==before



def test_frozen_roles_never_need_outcomes(tmp_path):
    from types import SimpleNamespace
    from cmr_common import frozen_roles
    from concept_subgroup_drift_pilot import _identity_hash
    import json
    year=2014
    cache_root=tmp_path/"cache"
    mapping={"D/2014/rules":{"rule_row_index":np.array([0])},
             "D/2014/cavs_selection":{"cav_row_index":np.array([1]),"selection_row_index":np.array([2])},
             "D/2014/idea2_frozen/2015":{"reference_row_index":np.array([3])}}
    identities={"rule_row_index":"rule_rows_identity","cav_row_index":"cav_rows_identity",
                "selection_row_index":"selection_rows_identity","reference_row_index":"reference_rows_identity"}
    for key, arrays in mapping.items():
        d=cache_root/key; d.mkdir(parents=True)
        meta={"roles_identity":"frozen_roles"}
        meta.update({identities[k]:_identity_hash({"row_index":v}) for k,v in arrays.items()})
        (d/"meta.json").write_text(json.dumps(meta));np.savez(d/"arrays.npz",**arrays)
    pred=cache_root/"P/2014";pred.mkdir(parents=True)
    (pred/"meta.json").write_text(json.dumps({"roles_identity":"frozen_roles"}))
    cache=SimpleNamespace(config={"split_seed":42},is_complete=lambda key:True,
                          _unit_path=lambda key:cache_root/key)
    pop={"patient_id":np.array(["a","b","c","d","a","new"]),
         "year":np.array([2014,2014,2014,2014,2015,2015]),"y":object()}
    ctx={"mode":"evaluate","input_files":{}}
    roles=frozen_roles(pop,cache,year,ctx)
    assert list(roles["row_role"][:5])==["rule_discovery","cav_fit","concept_selection","evaluation","rule_discovery"]
    assert roles["identity"]=="frozen_roles"
    ctx["mode"]="develop"
    assert frozen_roles(pop,cache,year,ctx)["row_role"][3]=="protected"

def test_combined_budget_and_report_scan(tmp_path):
    import time,json
    from cmr_common import check_budget,scan_notebook
    budget=tmp_path/"budget.json"
    budget.write_text(json.dumps({"earlier":{"mode":"smoke","seconds":1201,"started":time.time()}}))
    ctx={"mode":"smoke","started":time.monotonic(),"out":tmp_path,"budget_file":budget}
    with pytest.raises(TimeoutError):check_budget(ctx)
    (tmp_path/"private").mkdir();(tmp_path/"public").mkdir()
    p=tmp_path/"private/executed.ipynb"
    def notebook(data):
        p.write_text(json.dumps({"cells":[{"cell_type":"code","outputs":[{"output_type":"display_data","data":data}]}]}))
    notebook({"text/plain":"patient_id = secret"})
    with pytest.raises(ValueError,match="Private"):scan_notebook(ctx,p)
    notebook({"application/javascript":"alert(1)"})
    with pytest.raises(ValueError,match="MIME"):scan_notebook(ctx,p)
    notebook({"text/html":"<table><th>unknown_private_column</th></table>"})
    with pytest.raises(ValueError,match="column"):scan_notebook(ctx,p)
    notebook({"text/plain":"Checks passed"})
    scan_notebook(ctx,p)
    assert (tmp_path/"public/executed.ipynb").read_bytes()==p.read_bytes()
    (tmp_path/"public/table-000.csv").write_text("build,n,reference,status\n")
    (tmp_path/"public/table-000.json").write_text("[]")
    notebook({"text/html":"<table><th>build</th><th>n</th><th>...</th></table>"})
    scan_notebook(ctx,p)
    (tmp_path/"public/table-001.csv").write_text("model,stratum\nadditive,original_high\n")
    notebook({"text/html":"<table><th>model</th><th>additive</th><th>original_high</th></table>"})
    scan_notebook(ctx,p)


def test_legacy_description_functions_resolve_dependencies():
    from pathlib import Path
    from cmr_common import legacy_descriptions
    namespace=legacy_descriptions(Path(__file__).resolve().parents[1])
    fit,tune=namespace["development_rows"](np.array(["a","b"]),[0],[1],["a","b"])
    assert list(fit)==[0] and list(tune)==[1]
    with pytest.raises(ValueError,match="overlap"):
        namespace["development_rows"](np.array(["a","a"]),[0],[1],["a"])
