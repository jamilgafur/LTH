"""Unit tests for pyPrune.power NVIDIA telemetry parsing.

Locks in the deliberate invariants behind the Transfer2 measurement upgrade:
comma-positional field mapping (a "[Not Supported]" placeholder must not shift
clock into the util slot) and plausibility gates on utilization/clock values.
Runs without torch or a GPU (power.py keeps its torch import lazy for this).
"""

import importlib.util
from pathlib import Path

# Load power.py by file location (repo test convention: avoids the package
# __init__'s heavy imports, so these tests run without torch/tqdm installed).
_spec = importlib.util.spec_from_file_location(
    "pyprune_power", Path(__file__).resolve().parents[1] / "pyPrune" / "power.py")
_power = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_power)

NvidiaSmiParser = _power.NvidiaSmiParser


def _parser():
    return NvidiaSmiParser(max_watts=None)


def test_normal_line_maps_fields_by_position():
    p = _parser()
    p.parse("94.41, 17, 1410")
    p.parse("90.59, 13, 1410")
    util, clock = p.get_extras()
    watts, n = p.get_results()
    assert n == 2
    assert abs(watts - 92.5) < 1e-6
    assert util == 15.0
    assert clock == 1410.0


def test_not_supported_holds_its_slot_and_stays_out_of_util():
    # The historical bug: whole-line regex extraction collapsed "[Not Supported]"
    # and averaged the clock (e.g. 1410 MHz) into the utilization column.
    p = _parser()
    p.parse("94.41, [Not Supported], 1410")
    p.parse("93.00, 20, 1410")
    util, clock = p.get_extras()
    assert util == 20.0          # average of the one valid util sample only
    assert clock == 1410.0       # clock never leaked into util


def test_implausible_values_are_dropped_never_averaged():
    p = _parser()
    p.parse("94.41, 2520, 0")     # swapped/garbled telemetry: util>100, clock<=0
    util, clock = p.get_extras()
    watts, n = p.get_results()
    assert util is None and clock is None
    assert n == 1 and abs(watts - 94.41) < 1e-6  # power itself was plausible


def test_trailing_missing_fields_are_tolerated():
    p = _parser()
    p.parse("94.41")              # power-only row
    p.parse("90.0, 42")           # clock column absent
    util, clock = p.get_extras()
    watts, n = p.get_results()
    assert n == 2
    assert util == 42.0
    assert clock is None


def test_malformed_and_empty_lines_count_nothing():
    p = _parser()
    p.parse("")
    p.parse("[Not Supported], [Not Supported], [Not Supported]")
    watts, n = p.get_results()
    assert n == 0 and watts == 0.0
    assert p.get_extras() == (None, None)


def test_max_watts_gate_still_rejects_spikes():
    p = NvidiaSmiParser(max_watts=500.0)
    p.parse("900.0, 50, 1410")    # spike above cap: entire sample rejected
    p.parse("100.0, 50, 1410")
    watts, n = p.get_results()
    assert n == 1 and watts == 100.0
