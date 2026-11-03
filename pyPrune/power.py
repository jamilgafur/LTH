"""GPU/CPU power measurement for pyPrune pipelines.

Provides :class:`PowerTracker`, a context manager that samples power draw
from a background thread while a measured code region runs, plus the
streaming parsers behind it: ``nvidia-smi`` (NVIDIA GPUs), ``amd-smi``
(AMD GPUs) and ``turbostat`` (CPU package/RAM power, requires root).
"""

import os
import re
import subprocess
import sys
import threading
import time

__all__ = [
    "PowerTracker",
    "ResourceParser",
    "NvidiaSmiParser",
    "AmdSmiParser",
    "TurbostatParser",
]


class ResourceParser():
    def __init__(self, max_watts: float):
        self.max_watts = max_watts
        self.total_watts = 0.0
        self.num_intervals = 0

    def get_results(self):
        scaled_watts = self.total_watts / self.num_intervals if self.num_intervals > 0 else 0.0
        return scaled_watts, self.num_intervals


## turbostat is used for CPU readings
class TurbostatParser(ResourceParser):
    def __init__(self, max_watts: float):
        super().__init__(max_watts)
        # Regex and friends
        self.HEADER_REGEXP = re.compile(r"PkgWatt")
        self.DATA_REGEXP = re.compile(r"(\d+\.?\d+)\s?(\d+\.?\d+)?\s?(\d+\.?\d+)?")
        # Parsing state tmp
        self.header_read = False

    #turbostat: Failed to access /dev/cpu/0/msr. Some of the counters may not be available
    #        Run as root to enable them or use --no-msr to disable the access explicitly
    #cpu0: Guessing tjMax 100 C, Please use -T to specify
    #cpu32: Guessing tjMax 100 C, Please use -T to specify
    #PkgWatt RAMWatt
    #111.80  19.71
    #109.14  19.76
    #109.21  19.32
    def start_monitoring(self, interval_s: float):
        cmd = f"turbostat --Summary --quiet --show power --interval {max(1, int(round(interval_s)))}".split()
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, shell=False)

    #IMPORTANT: can have up to three columns with RAMWatt SysWatt
    def parse(self, line):
        if not self.header_read:
            r = self.HEADER_REGEXP.match(line)
            if not r:
                return
            self.header_read = True
            return

        r = self.DATA_REGEXP.match(line)
        if not r:
            return
        read_watts = float(r.group(1))
        if r.group(2):
            read_watts += float(r.group(2))
        if r.group(3):
            read_watts += float(r.group(3))
        if self.max_watts and read_watts > self.max_watts:
            print(f"Ignoring invalid value {read_watts}", file=sys.stderr)
            return
        self.total_watts += read_watts
        self.num_intervals += 1


## AMD GPU specific
class AmdSmiParser(ResourceParser):
    def __init__(self, max_watts: float):
        super().__init__(max_watts)
        # Regex and friends
        self.DATA_REGEXP = re.compile(r"(\d+)\s+\d+\s+\d+\s+(\d+)")
        # Internal for parsing
        self.latest_timestamp = -1

    def start_monitoring(self, interval_s: float):
        cmd = f"amd-smi monitor -p -w {max(1, int(round(interval_s)))}".split()
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1" # amd-smi is a Python process, need to unbuffer!
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, text=True, shell=False)

    #TIMESTAMP  GPU  XCP  POWER  PWR_CAP
    #1786038263    0    0  123 W    750 W
    #1786038263    1    0  123 W    750 W
    #1786038263    2    0  122 W    750 W
    def parse(self, line):
        r = self.DATA_REGEXP.match(line)
        if not r:
            return
        timestamp = int(r.group(1))
        self.total_watts += float(r.group(2))
        if timestamp > self.latest_timestamp:
            self.latest_timestamp = timestamp
            self.num_intervals += 1


## NVIDIA GPU specific
class NvidiaSmiParser(ResourceParser):
    def __init__(self, max_watts: float):
        super().__init__(max_watts)
        self.gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.num_gpus = 1
        self.total_util = 0.0
        self.total_clocks = 0.0
        self.num_util = 0
        self.num_clock = 0

    def start_monitoring(self, interval_s: float):
        # Sub-second intervals use --loop-ms so short regions stay well sampled.
        if interval_s < 1:
            loop = f"--loop-ms={max(1, int(interval_s * 1000))}"
        else:
            loop = f"--loop={int(round(interval_s))}"
        cmd = ("nvidia-smi", "-i", self.gpu_id,
               "--query-gpu=power.draw,utilization.gpu,clocks.sm",
               "--format=csv,noheader,nounits", loop)
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, shell=False)

    @staticmethod
    def _num(field):
        return float(field) if re.fullmatch(r"\d+\.?\d*", field) else None

    def parse(self, line):
        # Column order is fixed by the query string; split on commas so a
        # "[Not Supported]" placeholder keeps its slot instead of shifting the
        # later fields left (the old whole-line regex mapped clock -> util).
        fields = [f.strip() for f in line.split(",")]
        watts = self._num(fields[0]) if fields else None
        if watts is None:
            return
        if self.max_watts and watts > self.max_watts:
            return
        self.total_watts += watts
        self.num_intervals += 1
        # Deliberate invariants: util is a percent, SM clock a plausible MHz;
        # anything outside is a driver artifact and is dropped, never averaged.
        if len(fields) > 1:
            util = self._num(fields[1])
            if util is not None and 0.0 <= util <= 100.0:
                self.total_util += util
                self.num_util += 1
        if len(fields) > 2:
            clock = self._num(fields[2])
            if clock is not None and 0.0 < clock <= 3000.0:
                self.total_clocks += clock
                self.num_clock += 1

    def get_results(self):
        if self.num_gpus == 0 or self.num_intervals == 0:
            return 0.0, 0
        scaled_watts = self.total_watts / self.num_intervals
        return scaled_watts, self.num_intervals

    def get_extras(self):
        """Average GPU utilization (%) and SM clock (MHz) over valid samples.

        None when every sample for that metric was dropped (N/A or implausible)
        — downstream columns get None rather than a fabricated reading.
        """
        util = round(self.total_util / self.num_util, 1) if self.num_util else None
        clock = round(self.total_clocks / self.num_clock, 1) if self.num_clock else None
        return util, clock


class PowerTracker:
    """
    Context manager to track average power (Watts) and total energy (Joules)
    using the parser for the active device in a background thread.

    Sub-second ``interval_s`` (floor 0.1 s) keeps short regions well sampled.
    If zero samples are collected, watts/joules are reported as ``None`` —
    never as a fabricated reading.
    """
    def __init__(self, device=None, interval_s: float = 1, max_watts: float = None):
        import torch  # lazy: keep the parsers importable without the GPU stack
        self.device = device
        self.interval_s = max(0.1, float(interval_s))
        self.max_watts = max_watts
        self.parser = None
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()
        self.start_time = None
        self.elapsed_time = 0.0

        if torch.cuda.is_available() and (device is None or device.type == "cuda"):
            try:
                self.parser = NvidiaSmiParser(max_watts=self.max_watts)
            except Exception as e:
                print(f"[WARN] Failed to initialize NvidiaSmiParser: {e}")
                self.parser = None
        else:
            try:
                self.parser = TurbostatParser(max_watts=self.max_watts)
            except Exception as e:
                print(f"[WARN] Failed to initialize TurbostatParser: {e}")
                self.parser = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _reader(self):
        while not self.stop_event.is_set():
            if self.proc and self.proc.stdout:
                line = self.proc.stdout.readline()
                if line:
                    self.parser.parse(line)
                else:
                    break

    def start(self):
        self.start_time = time.time()
        if self.parser:
            try:
                self.proc = self.parser.start_monitoring(self.interval_s)
                self.thread = threading.Thread(target=self._reader, daemon=True)
                self.thread.start()
            except Exception as e:
                print(f"[WARN] Could not spawn power monitoring process: {e}")
                self.proc = None

    def stop(self):
        self.elapsed_time = time.time() - (self.start_time or time.time())
        self.stop_event.set()
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.kill()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    @property
    def num_intervals(self):
        """Power samples actually collected; audit depth for downstream CSVs."""
        return self.parser.num_intervals if self.parser else 0

    def get_results(self):
        """Return (avg_watts, total_joules, elapsed_s, gpu_util, gpu_clock).

        avg_watts/total_joules are ``None`` when no samples were collected
        (missing driver, unsupported flags, or a too-fast region).
        """
        if not self.parser:
            return None, None, float(self.elapsed_time), None, None

        avg_watts, num_intervals = self.parser.get_results()
        gpu_util = gpu_clock = None
        if hasattr(self.parser, "get_extras"):
            gpu_util, gpu_clock = self.parser.get_extras()

        if num_intervals == 0:
            print(f"[WARN] PowerTracker collected zero power samples over "
                  f"{self.elapsed_time:.1f}s; watts/joules reported as None.")
            return None, None, float(self.elapsed_time), gpu_util, gpu_clock

        total_joules = avg_watts * self.elapsed_time
        return float(avg_watts), float(total_joules), float(self.elapsed_time), gpu_util, gpu_clock
