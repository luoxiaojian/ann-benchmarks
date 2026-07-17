"""Topology-aware CPU allocation for parallel benchmark containers."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SYS_CPU_ROOT = Path("/sys/devices/system/cpu")


@dataclass(frozen=True)
class CpuAllocation:
    """Logical CPUs selected for workers and for host-side coordination."""

    worker_cpus: Tuple[int, ...]
    reserved_cpus: Tuple[int, ...]
    logical_cpu_count: int
    physical_core_count: int
    topology_detected: bool


def _parse_cpu_list(value: str) -> List[int]:
    cpus: List[int] = []
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def _online_cpu_ids(sys_cpu_root: Path = SYS_CPU_ROOT) -> List[int]:
    try:
        cpus = _parse_cpu_list((sys_cpu_root / "online").read_text())
    except (OSError, ValueError):
        cpus = list(range(os.cpu_count() or 1))
    if not cpus:
        raise RuntimeError("No online CPUs were detected")
    return cpus


def _physical_cpu_groups(
    cpu_ids: Sequence[int], sys_cpu_root: Path = SYS_CPU_ROOT
) -> Tuple[List[Tuple[int, ...]], bool]:
    topology: List[Tuple[int, int, int]] = []
    for cpu in sorted(set(cpu_ids)):
        topology_dir = sys_cpu_root / f"cpu{cpu}" / "topology"
        try:
            package_id = int((topology_dir / "physical_package_id").read_text())
            core_id = int((topology_dir / "core_id").read_text())
        except (OSError, ValueError):
            return [(cpu,) for cpu in sorted(set(cpu_ids))], False
        topology.append((cpu, package_id, core_id))

    groups: Dict[Tuple[int, int], List[int]] = {}
    for cpu, package_id, core_id in topology:
        groups.setdefault((package_id, core_id), []).append(cpu)

    physical_groups = [tuple(cpus) for cpus in groups.values()]
    physical_groups.sort(key=lambda cpus: cpus[0])
    return physical_groups, True


def allocate_worker_cpus(
    parallelism: int,
    reserve_physical_cores: int = 1,
    cpu_ids: Optional[Sequence[int]] = None,
    sys_cpu_root: Path = SYS_CPU_ROOT,
) -> CpuAllocation:
    """Allocate one logical CPU from each physical core to benchmark workers.

    The first physical core, including all of its SMT siblings, is reserved for
    host-side coordination by default. If Linux sysfs topology is unavailable,
    each online logical CPU is treated as an independent core.
    """
    if parallelism < 1:
        raise ValueError("Parallelism must be at least 1")
    if reserve_physical_cores < 0:
        raise ValueError("Reserved physical core count cannot be negative")

    online_cpus = _online_cpu_ids(sys_cpu_root) if cpu_ids is None else sorted(set(cpu_ids))
    if not online_cpus:
        raise RuntimeError("No online CPUs were detected")

    physical_groups, topology_detected = _physical_cpu_groups(online_cpus, sys_cpu_root)
    if reserve_physical_cores >= len(physical_groups):
        raise ValueError(
            f"Cannot reserve {reserve_physical_cores} physical core(s): "
            f"only {len(physical_groups)} core(s) are available"
        )

    reserved_groups = physical_groups[:reserve_physical_cores]
    worker_groups = physical_groups[reserve_physical_cores:]
    if parallelism > len(worker_groups):
        unit = "physical cores" if topology_detected else "logical CPUs"
        raise ValueError(
            f"Parallelism {parallelism} exceeds the {len(worker_groups)} available benchmark {unit} "
            f"after reserving {reserve_physical_cores} core(s) for the host"
        )

    return CpuAllocation(
        worker_cpus=tuple(group[0] for group in worker_groups[:parallelism]),
        reserved_cpus=tuple(cpu for group in reserved_groups for cpu in group),
        logical_cpu_count=len(online_cpus),
        physical_core_count=len(physical_groups),
        topology_detected=topology_detected,
    )
