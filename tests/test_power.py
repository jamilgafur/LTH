"""CPU tests for pyPrune.power: parser math, sampler command construction,
interval flooring, and the zero-sample -> None honesty contract."""

from unittest import mock

import pytest
import torch

from pyPrune.power import NvidiaSmiParser, PowerTracker, TurbostatParser


def _tracker_with(parser, elapsed=1.0):
    """PowerTracker whose parser is force-set without spawning processes."""
    tracker = PowerTracker(device=torch.device("cpu"), interval_s=1)
    tracker.parser = parser
    tracker.elapsed_time = elapsed
    return tracker


# ---------------------------------------------------------------- parsers

def test_nvidia_parse_averages_and_extras():
    parser = NvidiaSmiParser(max_watts=None)
    parser.parse("100.0, 50, 1500\n")
    parser.parse("140.0, 70, 1600\n")
    watts, n = parser.get_results()
    assert watts == pytest.approx(120.0)
    assert n == 2
    assert parser.get_extras() == (60.0, 1550.0)


def test_nvidia_parse_skips_junk_and_respects_max_watts():
    parser = NvidiaSmiParser(max_watts=200)
    parser.parse("[Not Supported]\n")
    parser.parse("999.0, 0, 0\n")  # above max_watts -> ignored
    assert parser.get_results() == (0.0, 0)
    assert parser.get_extras() == (None, None)


def test_turbostat_header_then_data():
    parser = TurbostatParser(max_watts=None)
    parser.parse("cpu0: Guessing tjMax 100 C, Please use -T to specify\n")
    parser.parse("PkgWatt RAMWatt\n")        # header row, not data
    parser.parse("111.80  19.71\n")          # 131.51 W pkg+ram
    parser.parse("109.21\n")                 # single column
    watts, n = parser.get_results()
    assert watts == pytest.approx((111.80 + 19.71 + 109.21) / 2)
    assert n == 2


# ------------------------------------------------- sampler command lines

def test_nvidia_subsecond_interval_uses_loop_ms():
    with mock.patch("pyPrune.power.subprocess.Popen") as popen:
        NvidiaSmiParser(max_watts=None).start_monitoring(0.1)
    argv = popen.call_args[0][0]
    assert "--loop-ms=100" in argv
    assert not any(a.startswith("--loop=") for a in argv)


def test_nvidia_whole_second_interval_uses_loop():
    with mock.patch("pyPrune.power.subprocess.Popen") as popen:
        NvidiaSmiParser(max_watts=None).start_monitoring(2)
    argv = popen.call_args[0][0]
    assert "--loop=2" in argv
    assert not any(a.startswith("--loop-ms=") for a in argv)


# ------------------------------------------------------------- PowerTracker

def test_interval_floor_is_point_one_second():
    tracker = PowerTracker(device=torch.device("cpu"), interval_s=0.001)
    assert tracker.interval_s == pytest.approx(0.1)
    tracker = PowerTracker(device=torch.device("cpu"), interval_s=5)
    assert tracker.interval_s == pytest.approx(5.0)


def test_zero_samples_report_none_not_fabricated():
    tracker = _tracker_with(NvidiaSmiParser(max_watts=None), elapsed=0.4)
    watts, joules, elapsed, util, clock = tracker.get_results()
    assert watts is None and joules is None
    assert elapsed == pytest.approx(0.4)
    assert util is None and clock is None


def test_missing_parser_reports_none():
    tracker = _tracker_with(None, elapsed=2.0)
    assert tracker.get_results() == (None, None, 2.0, None, None)


def test_joules_are_avg_watts_times_elapsed():
    parser = NvidiaSmiParser(max_watts=None)
    parser.parse("100.0, 50, 1500\n")
    parser.parse("200.0, 50, 1500\n")
    tracker = _tracker_with(parser, elapsed=2.0)
    watts, joules, elapsed, util, clock = tracker.get_results()
    assert watts == pytest.approx(150.0)
    assert joules == pytest.approx(300.0)
    assert elapsed == pytest.approx(2.0)
    assert (util, clock) == (50.0, 1500.0)
