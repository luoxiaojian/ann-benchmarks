import importlib
import logging.config
from pathlib import Path
from types import SimpleNamespace

import pytest

from ann_benchmarks.cpu_topology import CpuAllocation, allocate_worker_cpus


def _write_topology(root: Path, topology):
    cpu_ids = sorted(topology)
    (root / "online").write_text(",".join(map(str, cpu_ids)))
    for cpu, (package_id, core_id) in topology.items():
        topology_dir = root / f"cpu{cpu}" / "topology"
        topology_dir.mkdir(parents=True)
        (topology_dir / "physical_package_id").write_text(str(package_id))
        (topology_dir / "core_id").write_text(str(core_id))


def test_allocate_one_thread_per_physical_core(tmp_path):
    _write_topology(
        tmp_path,
        {
            0: (0, 0),
            1: (0, 0),
            2: (0, 1),
            3: (0, 1),
            4: (0, 2),
            5: (0, 2),
            6: (0, 3),
            7: (0, 3),
        },
    )

    allocation = allocate_worker_cpus(3, sys_cpu_root=tmp_path)

    assert allocation.reserved_cpus == (0, 1)
    assert allocation.worker_cpus == (2, 4, 6)
    assert allocation.logical_cpu_count == 8
    assert allocation.physical_core_count == 4
    assert allocation.topology_detected


def test_physical_core_identity_includes_cpu_package(tmp_path):
    _write_topology(
        tmp_path,
        {
            0: (0, 0),
            1: (0, 0),
            2: (1, 0),
            3: (1, 0),
        },
    )

    allocation = allocate_worker_cpus(1, sys_cpu_root=tmp_path)

    assert allocation.reserved_cpus == (0, 1)
    assert allocation.worker_cpus == (2,)
    assert allocation.physical_core_count == 2


def test_fallback_treats_each_logical_cpu_as_a_core(tmp_path):
    allocation = allocate_worker_cpus(2, cpu_ids=[0, 1, 2], sys_cpu_root=tmp_path)

    assert allocation.reserved_cpus == (0,)
    assert allocation.worker_cpus == (1, 2)
    assert not allocation.topology_detected


def test_rejects_parallelism_larger_than_available_physical_cores(tmp_path):
    _write_topology(
        tmp_path,
        {
            0: (0, 0),
            1: (0, 0),
            2: (0, 1),
            3: (0, 1),
        },
    )

    with pytest.raises(ValueError, match="Parallelism 2 exceeds the 1 available benchmark physical cores"):
        allocate_worker_cpus(2, sys_cpu_root=tmp_path)


def test_worker_processes_use_topology_aware_cpu_allocation(monkeypatch):
    monkeypatch.setattr(logging.config, "fileConfig", lambda *args, **kwargs: None)
    benchmark_main = importlib.import_module("ann_benchmarks.main")

    allocation = CpuAllocation(
        worker_cpus=(2, 4),
        reserved_cpus=(0, 1),
        logical_cpu_count=6,
        physical_core_count=3,
        topology_detected=True,
    )
    monkeypatch.setattr(benchmark_main, "allocate_worker_cpus", lambda parallelism: allocation)

    class FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    processes = []

    class FakeProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.started = False
            self.joined = False
            self.terminated = False
            processes.append(self)

        def start(self):
            self.started = True

        def join(self):
            self.joined = True

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(benchmark_main.multiprocessing, "Queue", FakeQueue)
    monkeypatch.setattr(benchmark_main.multiprocessing, "Process", FakeProcess)
    monkeypatch.setattr(
        benchmark_main.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=64 * 1024**3),
    )
    affinities = []
    monkeypatch.setattr(
        benchmark_main.os,
        "sched_setaffinity",
        lambda pid, cpus: affinities.append((pid, tuple(cpus))),
    )

    args = SimpleNamespace(parallelism=2, batch=False, local=False)
    benchmark_main.create_workers_and_execute([object(), object()], args)

    assert [process.args[0] for process in processes] == [2, 4]
    assert all(process.started and process.joined and process.terminated for process in processes)
    assert affinities == [(0, (0, 1))]
