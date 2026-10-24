"""Wiring tests for scorer-identity propagation through the Tranfer pipeline
(naming helpers, stats JSON conventions) and the new scorer figures.

Heavy Tranfer script imports are stubbed following the conventions in
test_transfer2_paths.py (walrus env lacks pandas/seaborn/fvcore).
"""

import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest


def _stub_module(name):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod
    return sys.modules[name]


class _MiniDataFrame:
    """Tiny pandas.DataFrame stand-in: dict-of-lists + to_csv (csv module)."""

    def __init__(self, data=None, index=None, columns=None):
        self._data = data if isinstance(data, dict) else {}
        if hasattr(data, "tolist"):
            data = data.tolist()
        self._matrix = data if isinstance(data, list) else None
        self._columns = columns

    def to_csv(self, path, index=False):
        import csv as _csv
        if self._matrix is not None:
            with open(path, "w", newline="") as f:
                w = _csv.writer(f)
                if self._columns:
                    w.writerow(list(self._columns))
                for row in self._matrix:
                    w.writerow(row)
            return
        cols = list(self._data.keys())
        n = max((len(v) for v in self._data.values()), default=0)
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(cols)
            for i in range(n):
                w.writerow([self._data[c][i] for c in cols])


def _load(name, filename):
    tranfer_dir = Path(__file__).resolve().parents[1] / "manuscript" / "Tranfer"
    spec = importlib.util.spec_from_file_location(name, tranfer_dir / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prestub_common():
    for n in ("fvcore", "fvcore.nn", "seaborn"):
        _stub_module(n)
    pd = _stub_module("pandas")
    pd.DataFrame = _MiniDataFrame
    fv = sys.modules["fvcore.nn"]
    if not getattr(fv, "FlopCountAnalysis", None):
        fv.FlopCountAnalysis = object
    # Tranfer script-dir siblings we don't exercise
    _stub_module("experiments")
    _stub_module("utils")
    _stub_module("plots")
    paths = _stub_module("paths")
    paths.discovery_run_dir = lambda *a, **k: "."
    paths.run_diagnostics_dir = lambda *a, **k: "."
    tr = _stub_module("trainer")
    tr.train_one_epoch = lambda *a, **k: None


def _load_plots():
    for n in ("fvcore", "fvcore.nn", "seaborn"):
        _stub_module(n)
    pd = _stub_module("pandas")
    pd.DataFrame = _MiniDataFrame
    return _load("plots", "plots.py")


_prestub_common()
transfer = _load("transfer", "transfer.py")
plots = _load_plots()


def _args(**kw):
    base = dict(model="InceptionNet", dataset="Cifar100", epochs=300,
                pretrain=100, scorer="local_dev", quant=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


# ------------------------------------------------------------ naming helpers

def test_scorer_suffix_default_is_legacy():
    assert transfer.scorer_suffix(_args()) == ""
    assert transfer.scorer_suffix(_args(scorer=None)) == ""


def test_scorer_suffix_non_default():
    assert transfer.scorer_suffix(_args(scorer="change")) == "_scorerchange"
    assert transfer.scorer_suffix(_args(scorer="cka")) == "_scorercka"
    assert transfer.scorer_suffix(_args(scorer="all")) == "_scorerall"


def test_json_filename_collision_free():
    legacy = transfer.discovered_json_filename(_args())
    change = transfer.discovered_json_filename(_args(scorer="change"))
    assert legacy == "InceptionNet_Cifar100_epochs300_pretrain100_JF_discovered_regions.json"
    assert change == "InceptionNet_Cifar100_epochs300_pretrain100_scorerchange_JF_discovered_regions.json"
    assert legacy != change


def test_json_filename_matches_temp2_convention():
    suffix = transfer.scorer_suffix(_args(scorer="change"))
    expected = "InceptionNet_Cifar100_epochs300_pretrain100" + suffix + "_JF_discovered_regions.json"
    assert transfer.discovered_json_filename(_args(scorer="change")) == expected


# ------------------------------------------------------------ artifact inheritance

def _setup_universes(tmp_path, monkeypatch, with_continued=True):
    """Default universe with artifacts; empty scorer universe; CWD = tmp."""
    monkeypatch.chdir(tmp_path)
    donor = tmp_path / "InceptionNet_Cifar100_None_epochs300_pretrain100" / "checkpoints"
    donor.mkdir(parents=True)
    (donor / "final_JF_Control.pt").write_text("control")
    if with_continued:
        (donor / "final_JF_Control_Continuted.pt").write_text("continued")
    return donor


def test_inherit_scorer_independent_artifacts_copies(tmp_path, monkeypatch):
    _setup_universes(tmp_path, monkeypatch)
    args = _args(scorer="change", quant=False)
    transfer.inherit_scorer_independent_artifacts(args)

    target = tmp_path / "InceptionNet_Cifar100_None_scorerchange_epochs300_pretrain100" / "checkpoints"
    assert (target / "final_JF_Control.pt").read_text() == "control"
    assert (target / "final_JF_Control_Continuted.pt").read_text() == "continued"


def test_inherit_is_noop_for_default_scorer_and_missing_donor(tmp_path, monkeypatch):
    _setup_universes(tmp_path, monkeypatch)
    transfer.inherit_scorer_independent_artifacts(_args())  # local_dev
    assert not (tmp_path / "InceptionNet_Cifar100_None_scorerlocal_dev_epochs300_pretrain100").exists()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "fresh").mkdir()
    monkeypatch.chdir(tmp_path / "fresh")  # no donor dirs here
    transfer.inherit_scorer_independent_artifacts(_args(scorer="cka"))  # must not crash
    assert not (tmp_path / "fresh" / "InceptionNet_Cifar100_None_scorercka_epochs300_pretrain100" / "checkpoints" / "final_JF_Control.pt").exists()


def test_inherit_skips_existing_target(tmp_path, monkeypatch):
    _setup_universes(tmp_path, monkeypatch)
    target = tmp_path / "InceptionNet_Cifar100_None_scorerchange_epochs300_pretrain100" / "checkpoints"
    target.mkdir(parents=True)
    (target / "final_JF_Control.pt").write_text("local-control")
    transfer.inherit_scorer_independent_artifacts(_args(scorer="change", quant=False))
    assert (target / "final_JF_Control.pt").read_text() == "local-control"


# ------------------------------------------------------------ change profile figure

def _fake_stats():
    return {
        "stem": {"kind": "layer", "change": 0.9, "cka": 0.4, "var_mean": 1.0, "var_std": 0.1},
        "blk": {"kind": "block", "change": 0.02, "cka": 0.99, "var_mean": 1.0, "var_std": 0.1},
        "blk.0": {"kind": "layer", "change": 0.02, "cka": 0.99, "var_mean": 1.0, "var_std": 0.1},
        "blk.1": {"kind": "layer", "change": 0.03, "cka": 0.98, "var_mean": 1.0, "var_std": 0.1},
        "head": {"kind": "layer", "change": None, "cka": None, "var_mean": 0.5, "var_std": 0.0},
    }


def test_plot_change_profile_written(tmp_path):
    stats = _fake_stats()
    cfg = {
        "Set_0": ("blk.0", "blk.1"),
        "Dynamic_Region_All_Combined": [("blk.0", "blk.1")],
        "Status_Map": {"blk.0": "Active"},
        "Control_Continuted": None,
    }
    plots.plot_change_profile(stats, cfg, tmp_path, "Toy", "Cifar10", scorer_label="change")
    assert (tmp_path / "change_profile_change.svg").exists()
    assert (tmp_path / "change_profile_change.png").exists()
    assert (tmp_path / "change_profile_change.csv").exists()


def test_plot_change_profile_handles_none_and_empty_config(tmp_path):
    plots.plot_change_profile(_fake_stats(), None, tmp_path, "Toy", "Cifar10",
                              scorer_label="cka")
    assert (tmp_path / "change_profile_cka.svg").exists()


# ------------------------------------------------------------ ablation figure

def _fake_ablation():
    return {
        "model": "Toy", "dataset": "Cifar10", "epochs": 300, "pretrain": 100,
        "stats_file": "runs/plots/change_stats.json",
        "scorers": {
            "local_dev": {"Set_0": ("a.0", "a.1"), "Set_1": ("b.0", "b.1")},
            "change": {"Set_0": ("a.0", "a.1")},
            "cka": {"Set_0": ("a.0", "a.1"), "Status_Map": {}, "Control_Continuted": None},
        },
    }


def test_plot_scorer_ablation_written(tmp_path):
    plots.plot_scorer_ablation(_fake_ablation(), save_dir=tmp_path)
    assert (tmp_path / "scorer_ablation.svg").exists()
    jac = (tmp_path / "scorer_ablation_jaccard.csv").read_text()
    # local_dev vs change: {2 tuples} vs {1 shared tuple} -> J = 1/2
    assert "0.5" in jac


def test_plot_scorer_ablation_from_path(tmp_path):
    p = tmp_path / "abl.json"
    p.write_text(json.dumps(_fake_ablation()))
    plots.plot_scorer_ablation(str(p))
    assert (tmp_path / "scorer_ablation.svg").exists()
