"""Filesystem tests for Transfer2 checkpoint discovery.

Transfer2 must resolve epoch configuration (epochsN_pretrainM) from the
checkpoint paths themselves — never from hardcoded CLI defaults — so multiple
training configurations can coexist in the Tranfer output directory.
"""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _stub_module(name):
    """Register an empty module with a valid __spec__ (torch dynamo introspects
    every entry in sys.modules and crashes on spec-less stubs)."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod
    return sys.modules[name]


def _load_transfer2_utils():
    """Import manuscript/Transfer2/utils.py directly (not an installed package),
    stubbing optional deps that are unavailable in the test env."""
    for name in ("fvcore", "fvcore.nn", "pandas", "seaborn"):
        _stub_module(name)
    if getattr(sys.modules["fvcore.nn"], "FlopCountAnalysis", None) is None:
        sys.modules["fvcore.nn"].FlopCountAnalysis = object  # placeholder, unused here
    # utils.py does `from main import PowerTracker` (script-dir circular import);
    # PowerTracker is unused by the discovery functions under test.
    _stub_module("main").PowerTracker = object

    utils_path = Path(__file__).resolve().parents[1] / "manuscript" / "Transfer2" / "utils.py"
    spec = importlib.util.spec_from_file_location("transfer2_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


u2 = _load_transfer2_utils()


@pytest.fixture
def exp_tree(tmp_path, monkeypatch):
    """Two coexisting epoch configs: A=epochs300_pretrain100 (Combined collapsed),
    B=epochs100_pretrain300 (Set_0 collapsed only, no continued control)."""
    base_a = tmp_path / "Tranfer" / "ToyNet_Cifar10_None_epochs300_pretrain100" / "checkpoints"
    base_b = tmp_path / "Tranfer" / "ToyNet_Cifar10_None_epochs100_pretrain300" / "checkpoints"
    work = tmp_path / "Transfer2"
    for d in (base_a, base_b, work):
        d.mkdir(parents=True)

    (base_a / "final_JF_Control.pt").write_text("a-before")
    (base_a / "final_JF_Control_Continuted.pt").write_text("a-after")
    (base_a / "final_JF_Dynamic_Region_All_Combined_whole.pt").write_text("a-combined")
    (base_b / "final_JF_Control.pt").write_text("b-before")
    (base_b / "final_JF_Set_0_whole.pt").write_text("b-set0")

    monkeypatch.chdir(work)  # find_experiment_checkpoints uses relative ../Tranfer globs
    return {
        "a_before": str(base_a / "final_JF_Control.pt"),
        "a_after": str(base_a / "final_JF_Control_Continuted.pt"),
        "a_combined": str(base_a / "final_JF_Dynamic_Region_All_Combined_whole.pt"),
        "b_before": str(base_b / "final_JF_Control.pt"),
        "b_set0": str(base_b / "final_JF_Set_0_whole.pt"),
    }


def _norm(paths):
    # Glob results are relative (`../Tranfer/...`); resolve against the test CWD.
    return sorted(str(Path(p).resolve()) for p in paths)


def test_parse_epoch_tokens():
    assert u2.parse_epoch_tokens("../Tranfer/ToyNet_Cifar10_None_epochs300_pretrain100/checkpoints/final_JF_Control.pt") == (300, 100)
    assert u2.parse_epoch_tokens("../Tranfer/ToyNet_Cifar10_None_no_tokens/checkpoints/final_JF_Control.pt") == (None, None)


def test_default_universe_donor_path_mapping():
    scorer_ckpt = "../Tranfer/ToyNet_Cifar10_None_scorerchange_epochs300_pretrain100/checkpoints/final_JF_Unstructured_IMP_post300.pt"
    donor = u2._default_universe_donor_path(scorer_ckpt)
    assert donor is not None
    assert str(Path(donor).name) == "final_JF_Unstructured_IMP_post300.pt"
    assert "_scorerchange" not in str(Path(donor).parent.parent)
    assert str(Path(donor).parent.parent).endswith("ToyNet_Cifar10_None_epochs300_pretrain100")
    # Default-universe path maps to itself -> None; non-checkpoints path -> None
    assert u2._default_universe_donor_path(donor) is None
    assert u2._default_universe_donor_path("../Tranfer/ToyNet_Cifar10_None_scorerchange_epochs300_pretrain100/metrics/x.json") is None


def test_inherit_from_default_universe_copies(tmp_path):
    root = tmp_path / "Tranfer"
    donor_dir = root / "ToyNet_Cifar10_None_epochs300_pretrain100" / "checkpoints"
    scorer_dir = root / "ToyNet_Cifar10_None_scorercka_epochs300_pretrain100" / "checkpoints"
    donor_dir.mkdir(parents=True)
    scorer_dir.mkdir(parents=True)
    (donor_dir / "final_JF_Unstructured_IMP_post300.pt").write_text("imp-donor")
    target = scorer_dir / "final_JF_Unstructured_IMP_post300.pt"

    out = u2.inherit_from_default_universe(str(target))
    assert Path(out).exists() and Path(out).read_text() == "imp-donor"

    # Idempotent: existing target untouched; missing donor -> no-op.
    target.write_text("local")
    u2.inherit_from_default_universe(str(target))
    assert target.read_text() == "local"
    missing = scorer_dir / "final_JF_Structural_L1_post300.pt"
    u2.inherit_from_default_universe(str(missing))
    assert not missing.exists()


def test_discovery_derives_epochs_from_paths(exp_tree):
    ck = u2.find_experiment_checkpoints("ToyNet", "Cifar10")

    assert _norm(ck["before"]) == _norm([exp_tree["a_before"], exp_tree["b_before"]])
    assert _norm(ck["after"]) == _norm([exp_tree["a_after"]])
    # Per-directory Combined preference with Set_0 fallback for the other config.
    assert _norm(ck["collapsed"]) == _norm([exp_tree["a_combined"], exp_tree["b_set0"]])

    # Production flow: before paths come from the same discovery globs, so both
    # sides of find_matching_checkpoint share the same (relative) form.
    a_before = [p for p in ck["before"] if "epochs300" in p][0]
    b_before = [p for p in ck["before"] if "epochs100" in p][0]
    # Each before-path matches artifacts in its own base directory.
    assert Path(u2.find_matching_checkpoint(a_before, ck["collapsed"])).resolve() == Path(exp_tree["a_combined"])
    assert Path(u2.find_matching_checkpoint(b_before, ck["collapsed"])).resolve() == Path(exp_tree["b_set0"])
    assert Path(u2.find_matching_checkpoint(a_before, ck["after"])).resolve() == Path(exp_tree["a_after"])
    with pytest.raises(FileNotFoundError):
        u2.find_matching_checkpoint(b_before, ck["after"])


def test_post_override_filters_to_matching_config(exp_tree):
    ck = u2.find_experiment_checkpoints("ToyNet", "Cifar10", post_epochs=300)
    assert _norm(ck["before"]) == _norm([exp_tree["a_before"]])
    assert _norm(ck["collapsed"]) == _norm([exp_tree["a_combined"]])


def test_pre_override_filters_to_matching_config(exp_tree):
    ck = u2.find_experiment_checkpoints("ToyNet", "Cifar10", pre_epochs=300)
    assert _norm(ck["before"]) == _norm([exp_tree["b_before"]])
    assert _norm(ck["collapsed"]) == _norm([exp_tree["b_set0"]])
