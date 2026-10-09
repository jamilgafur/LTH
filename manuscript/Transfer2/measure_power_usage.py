#!/usr/bin/env python3

import argparse
from enum import Enum
import os
from pathlib import Path
import psutil
import re
import selectors
import signal
import socket
import subprocess
import sys
import time

AVAILABLE_GPU_ARCHITECTURES = [ 'amd', 'nvidia', 'AMD', 'NVIDIA' ]
AVAILABLE_BATCH_SCHEDULER = [ 'slurm' ]
AVAILABLE_ACTIONS = [ 'start', 'stop' ]

parser = argparse.ArgumentParser(description='Get power measurements from a node')
parser.add_argument('-R', '--report', default=False, action='store_true',
                    help='Report by aggregating the individual nodes data (default: false)')
parser.add_argument('--no-cpu', dest="cpu", action='store_false',
                    help='Disable CPU power monitoring (default: false)')
parser.add_argument('--max-watts-cpu', default=None,
                    help='Max possible total CPU power used per node (default: None)')
parser.add_argument('-g', '--gpu', default=None, type=str, metavar='GPU_MODE', choices=AVAILABLE_GPU_ARCHITECTURES,
                    help='Measure GPU power and what type (default: False)')
parser.add_argument('--max-watts-gpu', default=None,
                    help='Max possible total GPU power used per node (default: None)')
parser.add_argument('-i', '--interval', default=10, type=int,
                    help='Query interval in seconds (default: 10)')
parser.add_argument('-o', '--output-dir', default='', type=str,
                    help='Output directory (default: '')')
parser.add_argument('-d', '--distribute', default=None, type=str, choices=AVAILABLE_BATCH_SCHEDULER,
                    help='Distribute using a batch scheduler allocation (default: None)')
parser.add_argument('-r', '--rank', default=None, type=int, 
                    help='For distributed runs, rank of the node (default: None)')
parser.add_argument('action', nargs='?', default=None, type=str, choices=AVAILABLE_ACTIONS,
                    help='start|stop for auto-background mode')

class MonitoringType(Enum):
    CPU = 1,
    GPU_AMD = 2,
    GPU_NVIDIA = 3

# Generic class that contains the global information for a run
class State():
    def __init__(self, args):
        self.distributor = args.distribute
        self.monitoring_types = []
        self.cpu = args.cpu
        self.gpu = args.gpu
        if args.cpu:
            self.monitoring_types.append(MonitoringType.CPU)
        if self.gpu == 'amd':
            self.monitoring_types.append(MonitoringType.GPU_AMD)
        elif self.gpu == 'nvidia':
            self.monitoring_types.append(MonitoringType.GPU_NVIDIA)
        self.max_watts_cpu = float(args.max_watts_cpu) if not args.max_watts_cpu is None else None
        self.max_watts_gpu = float(args.max_watts_gpu) if not args.max_watts_gpu is None else None
        self.interval = args.interval
        self.output_dir = args.output_dir
        self.rank = args.rank if not args.distribute else -1
        self.hostname = socket.gethostname().split('.')[0]

        # Internal distributor data
        self.children_remote_processes = None

        # Internal worker data
        self.local_processes = None

        signal.signal(signal.SIGINT, self.finalize)
        signal.signal(signal.SIGTERM, self.finalize)

    def start(self):
        if self.distributor:
            self.distribute()
        else:
            if not self.rank is None:
                pid_file = Path(self.output_dir) / f".watts_{self.hostname}.pid"
                with open(pid_file, 'w') as fd:
                    print(f"{os.getpid()}", file=fd)
            self.spawn_local_processes()

    def spawn_local_processes(self):
        self.local_processes = {}
        selector = selectors.DefaultSelector()
        for monitoring_type in self.monitoring_types:
            max_watts = None
            if monitoring_type == MonitoringType.CPU:
                max_watts = self.max_watts_cpu
            else:
                max_watts = self.max_watts_gpu
            process = MonitoringProcess(monitoring_type, max_watts, self.rank, self.interval, self.hostname, self.output_dir)
            process.start_command()
            selector.register(process.process.stdout, selectors.EVENT_READ, data=monitoring_type)
            self.local_processes[monitoring_type] = process

        running_processes = len(self.local_processes)
        while running_processes > 0:
            events = selector.select(timeout=1)
            for key, masks in events:
                stream = key.fileobj
                label = key.data

                line = stream.readline()
                if line:
                    self.local_processes[label].parse(line)
                else:
                    selector.unregister(stream)
                    stream.close()
                    running_processes -= 1
        for p in self.local_processes.values():
            p.wait()

    def distribute(self):
        if self.distributor == 'slurm':
            node_list = os.getenv('SLURM_JOB_NODELIST', self.hostname)
            r = subprocess.run(['scontrol', 'show', 'hostnames', node_list], capture_output=True, text=True, check=True)
            nodes = r.stdout.strip().splitlines()
        rank = 0
        self.children_remote_processes = []
        for node in nodes:
            with open(f"/tmp/spawn_{node}.log", "w") as fd:
                print(f"Launching monitoring on {node}", file=fd)
                cmd = f"ssh {node} {Path(__file__).resolve()} --output-dir={os.getcwd()} --rank={rank}".split()
                if self.gpu:
                    cmd.append(f"--gpu={self.gpu}")
                if self.max_watts_cpu:
                    cmd.append(f"--max-watts-cpu={self.max_watts_cpu}")
                if self.max_watts_gpu:
                    cmd.append(f"--max-watts-gpu={self.max_watts_gpu}")
                cmd.append(f"--interval={self.interval}")
                ssh_process = subprocess.Popen(cmd, stdout=fd, stderr=subprocess.STDOUT, text=True, shell=False)
                child_process = RemoteChildProcess(rank, node, ssh_process)
                self.children_remote_processes.append(child_process)
            rank += 1
        while True:
            time.sleep(1)

    def generate_report(self):
        WATTS_NODE_REGEX = re.compile(r"(\d+\.?\d+) Watts used over")
        def parse_file(nodefile):
            watts = 0.0
            try:
                with open(nodefile, "r") as fd:
                    for line in fd:
                        line = line.strip()
                        match = WATTS_NODE_REGEX.match(line)
                        if not match:
                            continue
                        watts += float(match.group(1))
            except Exception as e:
                print(f"Couldn't parse file {nodefile} Ignoring. Error: {e}", file=sys.stderr)
            return watts

        cumulated_cpu_watts = 0.0
        cumulated_gpu_watts = 0.0
        for rank in range(len(self.children_remote_processes)):
            cumulated_cpu_watts += parse_file(f"watts_cpu_node{rank}.log")
            if self.gpu:
                cumulated_gpu_watts += parse_file(f"watts_gpu_node{rank}.log")

        cumulated_watts = cumulated_cpu_watts + cumulated_gpu_watts
        output_file = "total_watts.log"
        with open(output_file, "w") as fd:
            print(f"Total watts used for the run: {cumulated_watts}W", file=fd)
            print(f"Total CPU watts used for the run: {cumulated_cpu_watts}W", file=fd)
            print(f"Total GPU watts used for the run: {cumulated_gpu_watts}W", file=fd)

    def finalize(self, sig, frame):
        if self.distributor:
            for p in self.children_remote_processes:
                p.terminate(sig)
            for p in self.children_remote_processes:
                p.wait()
            self.generate_report()
        else:
            for p in self.local_processes.values():
                p.write_results()
                p.terminate(sig)
        sys.exit(0)


class RemoteChildProcess():
    """ Class used only on the master process for distributed runs """
    def __init__(self, rank, hostname, ssh_process):
        self.rank = rank
        self.hostname = hostname
        self.ssh_process = ssh_process
        self.pid = -1 # The remote measures...py command

    def terminate(self, sig):
        pid = -1
        pid_file = f".watts_{self.hostname}.pid"
        with open(pid_file, 'r') as fd:
            l = fd.readline()
            m = re.match(r"(\d+)", l.strip())
            if m:
                self.pid = int(m.group(1))
        if self.pid == -1:
            print("Could not find PID for child process on {self.hostname}")
            return
        cmdline = f'ssh {self.hostname} kill -{sig} {self.pid}'
        subprocess.Popen(cmdline.split(), text=True, shell=False)
        # TODO: make sure ssh is killed?
        #time.sleep(5) # Give the system 5 seconds to write the file then kill ssh
        #self.ssh_process.kill()

    def wait(self):
        if self.pid == -1:
            return
        cmdline = f'ssh {self.hostname} kill -0 {self.pid}'
        while True:
            process = subprocess.run(cmdline.split(), shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if process.returncode:
                break
            time.sleep(1)

# Main worker class
class MonitoringProcess():
    def __init__(self, monitoring_type, max_watts, rank, interval, hostname, output_dir):
        self.monitoring_type = monitoring_type
        self.max_watts = max_watts
        self.process = None
        self.interval_s = interval
        self.hostname = hostname
        self.parser = None

        if self.monitoring_type == MonitoringType.CPU:
            self.parser = TurbostatParser(max_watts)
            monitored = 'cpu'
        elif self.monitoring_type == MonitoringType.GPU_AMD:
            self.parser = AmdSmiParser(max_watts)
            monitored = 'gpu'
        elif self.monitoring_type == MonitoringType.GPU_NVIDIA:
            self.parser = NvidiaSmiParser(max_watts)
            monitored = 'gpu'

        # Results
        node_id = hostname if rank is None else f"node{rank}"
        self.outputfile = Path(output_dir) / f"watts_{monitored}_{node_id}.log"

    def start_command(self):
        if not self.parser:
            print("Cannot launch monitoring process without a parser!")
            return 
        self.process = self.parser.start_monitoring(self.interval_s)
        if not self.process:
            print("Failed to start the monitoring subprocess!")
            sys.exit(0)

    def parse(self, line):
        self.parser.parse(line)

    def wait(self):
        self.process.wait()

    def terminate(self, sig):
        self.process.kill()

    def write_results(self):
        watts, intervals = self.parser.get_results()
        with open(self.outputfile, 'w') as fd:
            print(f"{watts} Watts used over {intervals * self.interval_s} seconds on {self.hostname}", file=fd)

class ResourceParser():
    def __init__(self, max_watts: float):
        self.max_watts = max_watts
        self.total_watts = 0.0
        self.num_intervals = 0

    def parse(self, line):
        val = self.parse_internal(line)
        if not val:
            return
        if val > self.max_watts:
            print(f"Ignoring invalid value {val}", file=sys.stderr)
            return
        self.total_watts += val
        self.num_intervals += 1

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
    def start_monitoring(self, interval_s: int):
        cmd = f"turbostat --Summary --quiet --show power --interval {interval_s}".split()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)
        os.set_blocking(p.stdout.fileno(), False)
        os.set_blocking(p.stderr.fileno(), False)
        return p

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

    def get_results(self):
        scaled_watts = self.total_watts / self.num_intervals if self.num_intervals > 0 else 0.0
        return scaled_watts, self.num_intervals

## AMD GPU specific
class AmdSmiParser(ResourceParser):
    def __init__(self, max_watts: float):
        super().__init__(max_watts)
        # Regex and friends
        self.DATA_REGEXP = re.compile(r"(\d+)\s+\d+\s+\d+\s+(\d+)")
        self.header_read = False
        # Internal for parsing
        self.latest_timestamp = -1

    def start_monitoring(self, interval_s: int):
        cmd =  f"amd-smi monitor -p -w {interval_s}".split()
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1" # amd-smi is a Python process, need to unbuffer!
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, text=True, shell=False)

    #TIMESTAMP  GPU  XCP  POWER  PWR_CAP
    #1786038263    0    0  123 W    750 W
    #1786038263    1    0  123 W    750 W
    #1786038263    2    0  122 W    750 W
    #1786038263    3    0  122 W    750 W
    #1786038263    4    0  122 W    750 W
    #1786038263    5    0  119 W    750 W
    #1786038263    6    0  123 W    750 W
    #1786038263    7    0  122 W    750 W
    def parse(self, line):
        r = self.DATA_REGEXP.match(line)
        if not r:
            return
        timestamp = int(r.group(1))
        self.total_watts += float(r.group(2))
        if timestamp > self.latest_timestamp:
            self.latest_timestamp = timestamp
            self.num_intervals += 1

    def get_results(self):
        scaled_watts = self.total_watts / self.num_intervals if self.num_intervals > 0 else 0.0
        return scaled_watts, self.num_intervals

## NVIDIA GPU specific
class NvidiaSmiParser(ResourceParser):
    def __init__(self, max_watts: float):
        super().__init__(max_watts)
        # Regex and friends
        self.DATA_REGEXP = re.compile(r"(\d+\.?\d+)")
        self.header_read = False
        # GPU 0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: GPU-21f64fd8-3929-4e6a-6ae6-3e45273e9187)
        GPUS_REGEXP = re.compile(r"GPU (\d+): \w+")
        self.num_gpus = 0
        process = subprocess.run('nvidia-smi --list-gpus'.split(), capture_output=True, text=True, shell=False)
        for line in process.stdout.splitlines():
            if GPUS_REGEXP.match(line):
                self.num_gpus += 1

        if self.num_gpus == 0:
            print("Could not detect any NVIDIA gpus!", file=sys.stderr)

    def start_monitoring(self, interval_s: int):
        cmd = f"nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop={interval_s}".split()
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, shell=False)

    # One line per GPU at each printed timestamp
    #521.31
    #529.74
    #526.60
    #516.43
    #522.41
    #529.28
    #559.31
    #515.40
    def parse(self, line):
        r = self.DATA_REGEXP.match(line)
        if not r:
            return
        self.total_watts += float(r.group(1))
        self.num_intervals += 1

    def get_results(self):
        if self.num_gpus == 0:
            return 0.0, 0
        self.num_intervals = self.num_intervals / self.num_gpus
        scaled_watts = self.total_watts / self.num_intervals if self.num_intervals > 0 else 0.0
        return scaled_watts, self.num_intervals

def background_action(action: str):
    """ Starts or stop a monitoring process in background mode"""
    def pid_exists(pid):
        """Check whether pid exists in the current process table."""
        if pid < 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError as e:
            return e.errno == errno.EPERM
        else:
            return True

    pid_file = "/tmp/.watts_master.pid"
    if action == 'start':
        pid = os.fork()
        try:
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            sys.stderr.write(f"Fork #2 failed: {e}\n")
            sys.exit(1)
        with open(pid_file, 'w') as fd:
            print(f"{os.getpid()}", file=fd)
    else:
        pid = -1
        try:
            with open(pid_file, 'r') as fd:
                l = fd.readline()
                m = re.match(r"(\d+)", l.strip())
                if m:
                    pid = int(m.group(1))
            if pid == -1:
                print("Could not find PID to stop measures")
                sys.exit(1)
            os.kill(pid, signal.SIGINT)
            try:
                p = psutil.Process(pid)
                p.wait() 
            except psutil.NoSuchProcess:
                pass
            os.remove(pid_file)
        except Exception as e:
            print(f"Error stopping power monitoring: {e}")
        sys.exit(0)

def main():
    global g_state

    args = parser.parse_args()
    if args.gpu:
        args.gpu = args.gpu.lower()

    if not args.cpu and not args.gpu:
        print(f"Both CPU and GPU monitoring are disabled: exiting", file=sys.stderr)
        sys.exit(1)

    if args.action:
        background_action(args.action)
    
    g_state = State(args)
    g_state.start()


if __name__ == "__main__":
    main()
