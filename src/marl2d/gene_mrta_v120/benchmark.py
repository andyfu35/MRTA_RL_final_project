from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable
from zipfile import ZipFile

import numpy as np


DATA_DIR = Path("benchmarks/minmax_mtsp_mils")
INSTANCE_ZIP = DATA_DIR / "instances.zip"
CERTIFICATE_ZIP = DATA_DIR / "Certification.zip"

SOURCE_REPOSITORY = "https://github.com/pengfeihe-angers/mils"
SOURCE_PAPER_DOI = "10.1016/j.cor.2025.107255"

# Table A.1/A.2 of He, Hao & Xia (Computers & Operations Research,
# 185:107255, 2026) marks these instances with *, meaning a known exact
# optimal solution rather than merely a best-known upper bound.
EXACT_OPTIMUM_IDS = frozenset(
    {
        "mtsp51_10",
        "mtsp100_10",
        "mtsp100_20",
        "rand100_10",
        "rand100_20",
        "mtsp150_20",
        "mtsp150_30",
        "gtsp150_10",
        "gtsp150_20",
        "gtsp150_30",
        "kroa200_10",
        "kroa200_20",
        "lin318_10",
        "lin318_20",
        "att532_20",
        "rat783_20",
        "pcb1173_20",
        "nrw1379_20",
        "fl1400_10",
        "fl1400_20",
        "fl3795_10",
        "fl3795_20",
    }
)

# Five benchmark files have no certificate in the authors' Certification.zip.
# Their references are taken directly from Table A.1 of the published paper.
PAPER_ONLY_REFERENCES = {
    "mtsp51_3": 159.57,
    "mtsp51_5": 118.13,
    "mtsp51_10": 112.07,
    "mtsp150_30": 5246.49,
    "gtsp150_30": 1554.64,
}


@dataclass(frozen=True)
class Certificate:
    instance_id: str
    objective: float
    routes: tuple[tuple[int, ...], ...]
    source_file: str


@dataclass(frozen=True)
class MinMaxMTSPInstance:
    instance_id: str
    base_name: str
    edge_weight_type: str
    robot_count: int
    coordinates: np.ndarray
    reference_value: float
    reference_kind: str
    reference_source: str
    benchmark_set: str
    split: str

    @property
    def vertex_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def task_count(self) -> int:
        # Node 0 is the common depot. All remaining vertices are cities/tasks.
        return max(0, self.vertex_count - 1)

    @property
    def depot(self) -> np.ndarray:
        return self.coordinates[0]

    @property
    def task_positions(self) -> np.ndarray:
        return self.coordinates[1:]

    @property
    def size_band(self) -> str:
        n = self.vertex_count
        if n <= 200:
            return "small"
        if n <= 1173:
            return "medium"
        return "large"

    @property
    def is_exact_optimum(self) -> bool:
        return self.reference_kind == "exact_optimum"


def _normal_id(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _parse_robot_count(filename: str) -> int:
    stem = Path(filename).stem
    match = re.search(r"_(\d+)$", stem)
    if match is None:
        raise ValueError(f"Cannot infer robot count from {filename}")
    value = int(match.group(1))
    if value <= 0:
        raise ValueError(f"Invalid robot count in {filename}")
    return value


def _parse_instance_text(filename: str, text: str) -> tuple[str, str, int, np.ndarray]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Malformed benchmark instance: {filename}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Malformed header in {filename}: {lines[0]!r}")

    base_name = header[0]
    edge_type = header[1].upper()
    robot_count = _parse_robot_count(filename)

    rows: list[tuple[int, float, float]] = []
    saw_eof = False
    for index, line in enumerate(lines[1:], start=1):
        if line.upper() == "EOF":
            saw_eof = True
            trailing = lines[index + 1 :]
            if trailing:
                raise ValueError(
                    f"Unexpected content after EOF in {filename}: {trailing[0]!r}"
                )
            break

        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Malformed coordinate row in {filename}: {line!r}")
        rows.append((int(parts[0]), float(parts[1]), float(parts[2])))

    rows.sort(key=lambda item: item[0])
    ids = [item[0] for item in rows]
    expected = list(range(1, len(rows) + 1))
    if ids != expected:
        raise ValueError(f"Coordinate IDs are not contiguous in {filename}")

    coordinates = np.asarray(
        [[x, y] for _, x, y in rows],
        dtype=np.float64,
    )
    if coordinates.shape[0] <= robot_count:
        raise ValueError(
            f"{filename}: vertex_count={coordinates.shape[0]} "
            f"must exceed robot_count={robot_count}"
        )
    return base_name, edge_type, robot_count, coordinates


def _parse_certificate_text(filename: str, text: str) -> Certificate:
    objective_match = re.search(
        r"The objective is:\s*\n?\s*([-+0-9.eE]+)",
        text,
        flags=re.IGNORECASE,
    )
    if objective_match is None:
        raise ValueError(f"Certificate has no objective: {filename}")
    objective = float(objective_match.group(1))

    routes: list[tuple[int, ...]] = []
    for match in re.finditer(
        r"Route\s+\d+\s*:\s*([0-9\-]+)",
        text,
        flags=re.IGNORECASE,
    ):
        route = tuple(int(value) for value in match.group(1).split("-"))
        routes.append(route)

    # Certificate names end with an independent-run index:
    # kroa200_3_0.txt -> kroa200_3
    stem = Path(filename).stem
    match = re.match(r"(.+)_\d+$", stem)
    if match is None:
        raise ValueError(f"Malformed certificate filename: {filename}")
    instance_id = _normal_id(match.group(1))

    return Certificate(
        instance_id=instance_id,
        objective=objective,
        routes=tuple(routes),
        source_file=filename,
    )


def load_certificates(
    certificate_zip: Path = CERTIFICATE_ZIP,
) -> dict[str, Certificate]:
    path = Path(certificate_zip)
    if not path.is_file():
        raise FileNotFoundError(path)

    result: dict[str, Certificate] = {}
    with ZipFile(path, "r") as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".txt"):
                continue
            text = archive.read(name).decode("utf-8", errors="strict")
            certificate = _parse_certificate_text(name, text)
            old = result.get(certificate.instance_id)
            if old is not None:
                raise ValueError(
                    "Duplicate certificate for "
                    f"{certificate.instance_id}: {old.source_file}, {name}"
                )
            result[certificate.instance_id] = certificate
    return result


def _reference_for(
    instance_id: str,
    certificates: dict[str, Certificate],
) -> tuple[float, str]:
    certificate = certificates.get(instance_id)
    if certificate is not None:
        return float(certificate.objective), f"Certification.zip:{certificate.source_file}"

    if instance_id in PAPER_ONLY_REFERENCES:
        return (
            float(PAPER_ONLY_REFERENCES[instance_id]),
            "He-Hao-Xia-2026:Table-A.1",
        )
    raise ValueError(f"No public reference value for {instance_id}")


def _development_split(instance_id: str, benchmark_set: str) -> str:
    # The paper already defines S (41 small/medium) and L (36 large).
    # We protect the entire L set from evolution to obtain a strong scale
    # generalization test. Inside S, a deterministic 4/5 vs 1/5 split gives
    # development and validation without any random split drift.
    if benchmark_set == "L":
        return "protected_test"

    digest = hashlib.sha256(instance_id.encode("utf-8")).digest()
    return "validation" if digest[0] % 5 == 0 else "evolution"


def load_benchmark(
    instance_zip: Path = INSTANCE_ZIP,
    certificate_zip: Path = CERTIFICATE_ZIP,
) -> list[MinMaxMTSPInstance]:
    instance_path = Path(instance_zip)
    if not instance_path.is_file():
        raise FileNotFoundError(instance_path)

    certificates = load_certificates(certificate_zip)
    rows: list[MinMaxMTSPInstance] = []

    with ZipFile(instance_path, "r") as archive:
        for name in sorted(archive.namelist(), key=str.lower):
            if not name.lower().endswith(".txt"):
                continue

            text = archive.read(name).decode("utf-8", errors="strict")
            base_name, edge_type, robot_count, coordinates = _parse_instance_text(
                name,
                text,
            )
            instance_id = _normal_id(Path(name).stem)
            reference, reference_source = _reference_for(
                instance_id,
                certificates,
            )

            vertex_count = int(coordinates.shape[0])
            benchmark_set = "S" if vertex_count <= 1173 else "L"
            split = _development_split(instance_id, benchmark_set)
            reference_kind = (
                "exact_optimum"
                if instance_id in EXACT_OPTIMUM_IDS
                else "best_known"
            )

            rows.append(
                MinMaxMTSPInstance(
                    instance_id=instance_id,
                    base_name=base_name,
                    edge_weight_type=edge_type,
                    robot_count=robot_count,
                    coordinates=coordinates,
                    reference_value=float(reference),
                    reference_kind=reference_kind,
                    reference_source=reference_source,
                    benchmark_set=benchmark_set,
                    split=split,
                )
            )

    if len(rows) != 77:
        raise ValueError(f"Expected 77 MILS instances, found {len(rows)}")

    ids = [row.instance_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate minmax-mTSP instance ID")

    return rows


def select_instances(
    instances: Iterable[MinMaxMTSPInstance],
    split: str,
) -> list[MinMaxMTSPInstance]:
    values = list(instances)
    if split == "all":
        return values
    selected = [item for item in values if item.split == split]
    if not selected:
        raise ValueError(f"No instances in split={split!r}")
    return selected


def describe_benchmark(
    instances: Iterable[MinMaxMTSPInstance],
) -> dict[str, object]:
    values = list(instances)
    by_split: dict[str, int] = {}
    by_set: dict[str, int] = {}
    by_band: dict[str, int] = {}
    by_robots: dict[str, int] = {}
    exact = 0

    for item in values:
        by_split[item.split] = by_split.get(item.split, 0) + 1
        by_set[item.benchmark_set] = by_set.get(item.benchmark_set, 0) + 1
        by_band[item.size_band] = by_band.get(item.size_band, 0) + 1
        key = str(item.robot_count)
        by_robots[key] = by_robots.get(key, 0) + 1
        exact += int(item.is_exact_optimum)

    return {
        "instance_count": len(values),
        "by_split": by_split,
        "by_paper_set": by_set,
        "by_size_band": by_band,
        "by_robot_count": by_robots,
        "known_exact_optimum_count": exact,
        "best_known_count": len(values) - exact,
        "min_vertices": min(item.vertex_count for item in values),
        "max_vertices": max(item.vertex_count for item in values),
        "source_repository": SOURCE_REPOSITORY,
        "source_paper_doi": SOURCE_PAPER_DOI,
    }


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)


def tsplib_att_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.asarray(a) - np.asarray(b)
    rij = np.sqrt(np.sum(delta * delta, axis=-1) / 10.0)
    tij = np.floor(rij + 0.5)
    return np.where(tij < rij, tij + 1.0, tij)


def edge_distance(
    instance: MinMaxMTSPInstance,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    if instance.edge_weight_type == "EUC_2D":
        # The MILS benchmark reports decimal objectives for many EUC_2D
        # instances, so its evaluator uses geometric Euclidean lengths rather
        # than TSPLIB integer rounding.
        return euclidean_distance(a, b)
    if instance.edge_weight_type == "ATT":
        return tsplib_att_distance(a, b)
    raise ValueError(
        f"Unsupported edge type {instance.edge_weight_type!r} "
        f"for {instance.instance_id}"
    )
