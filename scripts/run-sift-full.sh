#!/usr/bin/env bash

# Run the repository's stock ANN-Benchmarks entry point, then archive all
# artifacts outside the checkout. This script deliberately does not inject
# result/logging hooks into ann_benchmarks itself.

set -Eeuo pipefail
umask 022

ANNB_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANNB_RUNS_ROOT="${ANNB_RUNS_ROOT:-${ANNB_REPO_DIR}-runs}"
ANNB_DATASET="sift-128-euclidean"
ANNB_COUNT=10
ANNB_RUN_COUNT=5
ANNB_PARALLELISM=31
# Keep the stock ANN-Benchmarks per-definition timeout. This is intentionally
# not configurable: changing it would make the run incomparable with the
# repository's default benchmark procedure.
ANNB_TIMEOUT_SECONDS=7200

if [[ -x "${ANNB_REPO_DIR}/.venv/bin/python" ]]; then
    ANNB_PYTHON="${ANNB_REPO_DIR}/.venv/bin/python"
elif [[ -x "${ANNB_REPO_DIR}/venv/bin/python" ]]; then
    ANNB_PYTHON="${ANNB_REPO_DIR}/venv/bin/python"
else
    printf 'Missing benchmark virtual environment (.venv or venv).\n' >&2
    exit 1
fi

annb_usage() {
    printf 'Usage: %s [--resume RUN_DIRECTORY] [--preflight-only]\n' "$0"
}

ANNB_RESUME_DIR=""
ANNB_PREFLIGHT_ONLY=0
while (($#)); do
    case "$1" in
        --resume)
            if (($# < 2)); then
                annb_usage >&2
                exit 2
            fi
            ANNB_RESUME_DIR="$2"
            shift 2
            ;;
        --preflight-only)
            ANNB_PREFLIGHT_ONLY=1
            shift
            ;;
        -h|--help)
            annb_usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            annb_usage >&2
            exit 2
            ;;
    esac
done

for ANNB_NUMERIC_VALUE in "$ANNB_RUN_COUNT" "$ANNB_PARALLELISM" "$ANNB_TIMEOUT_SECONDS"; do
    if [[ ! "$ANNB_NUMERIC_VALUE" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Run count, parallelism, and timeout must be positive integers.\n' >&2
        exit 2
    fi
done

for ANNB_REQUIRED_COMMAND in docker flock setsid sha256sum realpath; do
    if ! command -v "$ANNB_REQUIRED_COMMAND" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$ANNB_REQUIRED_COMMAND" >&2
        exit 1
    fi
done

mkdir -p "$ANNB_RUNS_ROOT"
ANNB_RUNS_ROOT="$(realpath "$ANNB_RUNS_ROOT")"
exec 9>"${ANNB_RUNS_ROOT}/.sift-full.lock"
if ! flock -n 9; then
    printf 'Another full SIFT benchmark is already running (lock: %s).\n' \
        "${ANNB_RUNS_ROOT}/.sift-full.lock" >&2
    exit 1
fi

ANNB_NOW="$(date +%Y%m%d-%H%M%S)"
ANNB_IS_RESUME=0
if [[ -n "$ANNB_RESUME_DIR" ]]; then
    if [[ ! -d "$ANNB_RESUME_DIR" ]]; then
        printf 'Resume directory does not exist: %s\n' "$ANNB_RESUME_DIR" >&2
        exit 1
    fi
    ANNB_RUN_DIR="$(realpath "$ANNB_RESUME_DIR")"
    ANNB_RUN_ID="$(basename "$ANNB_RUN_DIR")"
    ANNB_IS_RESUME=1
else
    ANNB_RUN_ID="sift-128-euclidean-k10-${ANNB_NOW}-pid$$"
    ANNB_RUN_DIR="${ANNB_RUNS_ROOT}/${ANNB_RUN_ID}"
    mkdir -p "$ANNB_RUN_DIR"
fi

ANNB_ATTEMPT_ID="attempt-${ANNB_NOW}-pid$$"
ANNB_ATTEMPT_DIR="${ANNB_RUN_DIR}/attempts/${ANNB_ATTEMPT_ID}"
mkdir -p "$ANNB_ATTEMPT_DIR"
ln -sfn "$ANNB_RUN_DIR" "${ANNB_RUNS_ROOT}/latest"
ln -sfn "attempts/${ANNB_ATTEMPT_ID}" "${ANNB_RUN_DIR}/latest-attempt"

printf 'RUN_DIR=%s\n' "$ANNB_RUN_DIR"
printf 'ATTEMPT_DIR=%s\n' "$ANNB_ATTEMPT_DIR"
exec >>"${ANNB_ATTEMPT_DIR}/launcher.log" 2>&1

ANNB_STARTED_AT="$(date -Is)"
ANNB_STARTED_EPOCH="$(date +%s)"
ANNB_STATE="PREPARING"
ANNB_EXIT_CODE=""
ANNB_CHILD_PID=""
ANNB_BENCHMARK_STARTED=0
ANNB_OWNS_LIVE_RESULTS=0
ANNB_LIVE_DATASET_DIR="${ANNB_REPO_DIR}/results/${ANNB_DATASET}"
ANNB_ARCHIVE_RESULTS_DIR="${ANNB_RUN_DIR}/results"
ANNB_ARCHIVE_DATASET_DIR="${ANNB_ARCHIVE_RESULTS_DIR}/${ANNB_DATASET}"

annb_result_count() {
    local ANNB_COUNT_VALUE=0
    if [[ -d "$ANNB_ARCHIVE_DATASET_DIR" ]]; then
        ANNB_COUNT_VALUE="$(find "$ANNB_ARCHIVE_DATASET_DIR" -type f -name '*.hdf5' -print 2>/dev/null | wc -l)"
    elif ((ANNB_OWNS_LIVE_RESULTS)) && [[ -d "$ANNB_LIVE_DATASET_DIR" ]]; then
        ANNB_COUNT_VALUE="$(find "$ANNB_LIVE_DATASET_DIR" -type f -name '*.hdf5' -print 2>/dev/null | wc -l)"
    fi
    printf '%s' "$ANNB_COUNT_VALUE"
}

annb_write_status() {
    local ANNB_STATUS_TMP="${ANNB_RUN_DIR}/status.tmp.$$"
    {
        printf 'RUN_ID=%s\n' "$ANNB_RUN_ID"
        printf 'ATTEMPT_ID=%s\n' "$ANNB_ATTEMPT_ID"
        printf 'STATE=%s\n' "$ANNB_STATE"
        printf 'SCRIPT_PID=%s\n' "$$"
        printf 'BENCHMARK_PID=%s\n' "$ANNB_CHILD_PID"
        printf 'EXIT_CODE=%s\n' "$ANNB_EXIT_CODE"
        printf 'STARTED_AT=%s\n' "$ANNB_STARTED_AT"
        printf 'UPDATED_AT=%s\n' "$(date -Is)"
        printf 'RESULT_FILES=%s\n' "$(annb_result_count)"
        printf 'ATTEMPT_DIR=%s\n' "$ANNB_ATTEMPT_DIR"
    } >"$ANNB_STATUS_TMP"
    mv "$ANNB_STATUS_TMP" "${ANNB_RUN_DIR}/status"
}

annb_container_belongs_to_run() {
    local ANNB_CONTAINER_ID="$1"
    docker inspect --format '{{range .Mounts}}{{println .Source}}{{end}}' "$ANNB_CONTAINER_ID" 2>/dev/null \
        | grep -Fxq "${ANNB_REPO_DIR}/ann_benchmarks"
}

annb_cleanup_containers() {
    if ((!ANNB_BENCHMARK_STARTED)); then
        return
    fi
    local ANNB_CONTAINER_ID
    local ANNB_CONTAINER_IDS=()
    while IFS= read -r ANNB_CONTAINER_ID; do
        [[ -n "$ANNB_CONTAINER_ID" ]] || continue
        if grep -Fxq "$ANNB_CONTAINER_ID" "${ANNB_ATTEMPT_DIR}/containers-before.txt" 2>/dev/null; then
            continue
        fi
        if annb_container_belongs_to_run "$ANNB_CONTAINER_ID"; then
            ANNB_CONTAINER_IDS+=("$ANNB_CONTAINER_ID")
        fi
    done < <(docker ps -aq 2>/dev/null || true)
    if ((${#ANNB_CONTAINER_IDS[@]})); then
        docker rm -f "${ANNB_CONTAINER_IDS[@]}" >/dev/null 2>&1 || true
    fi
}

annb_collect_results() {
    if ((!ANNB_OWNS_LIVE_RESULTS)); then
        return
    fi
    mkdir -p "$ANNB_ARCHIVE_RESULTS_DIR"
    if [[ -e "$ANNB_ARCHIVE_DATASET_DIR" ]]; then
        printf 'Refusing to overwrite archived results: %s\n' "$ANNB_ARCHIVE_DATASET_DIR" >&2
        return 1
    fi
    if [[ -d "$ANNB_LIVE_DATASET_DIR" ]]; then
        mv "$ANNB_LIVE_DATASET_DIR" "$ANNB_ARCHIVE_RESULTS_DIR/"
    fi
    ANNB_OWNS_LIVE_RESULTS=0
}

annb_audit_results() {
    ANNB_RUN_DIR="$ANNB_RUN_DIR" ANNB_ATTEMPT_DIR="$ANNB_ATTEMPT_DIR" \
        "$ANNB_PYTHON" - <<'PY'
import os
from pathlib import Path

import h5py

run_dir = Path(os.environ["ANNB_RUN_DIR"])
attempt_dir = Path(os.environ["ANNB_ATTEMPT_DIR"])
expected = [line.strip() for line in (attempt_dir / "expected-results.txt").read_text().splitlines() if line.strip()]
missing = []
invalid = []
for relative in expected:
    filename = run_dir / relative
    if not filename.is_file():
        missing.append(relative)
        continue
    try:
        with h5py.File(filename, "r") as result:
            required_attributes = {"algo", "best_search_time", "count", "dataset", "run_count"}
            required_datasets = {"times", "neighbors", "distances"}
            if not required_attributes.issubset(result.attrs.keys()):
                raise ValueError("missing required attributes")
            if not required_datasets.issubset(result.keys()):
                raise ValueError("missing required datasets")
            query_count = len(result["times"])
            count = int(result.attrs["count"])
            if query_count <= 0 or result["neighbors"].shape != (query_count, count):
                raise ValueError("invalid neighbors shape")
            if result["distances"].shape != (query_count, count):
                raise ValueError("invalid distances shape")
    except Exception as error:
        invalid.append(f"{relative}\t{error}")

(attempt_dir / "missing-results.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
(attempt_dir / "invalid-results.txt").write_text("\n".join(invalid) + ("\n" if invalid else ""))
summary = (
    f"EXPECTED={len(expected)}\n"
    f"PRESENT={len(expected) - len(missing)}\n"
    f"MISSING={len(missing)}\n"
    f"INVALID={len(invalid)}\n"
)
(attempt_dir / "result-audit.txt").write_text(summary)
print(summary, end="")
raise SystemExit(0 if not missing and not invalid else 1)
PY
}

annb_snapshot_images() {
    local ANNB_SNAPSHOT_PHASE="$1"
    local ANNB_REQUIRED_IMAGES_FILE="${ANNB_ATTEMPT_DIR}/required-images.txt"
    local ANNB_REPRO_DIR="${ANNB_ATTEMPT_DIR}/reproducibility"
    local ANNB_IMAGE_TAG
    local ANNB_IMAGE_METADATA
    local ANNB_SNAPSHOT_RC=0
    local ANNB_IMAGE_TAGS=()

    [[ -f "$ANNB_REQUIRED_IMAGES_FILE" ]] || return 0
    mkdir -p "$ANNB_REPRO_DIR"
    mapfile -t ANNB_IMAGE_TAGS < <(sed '/^[[:space:]]*$/d' "$ANNB_REQUIRED_IMAGES_FILE")

    {
        printf 'docker_tag\timage_id\tcreated\n'
        for ANNB_IMAGE_TAG in "${ANNB_IMAGE_TAGS[@]}"; do
            if ANNB_IMAGE_METADATA="$(docker image inspect "$ANNB_IMAGE_TAG" \
                --format '{{.Id}}{{printf "\t"}}{{.Created}}' 2>/dev/null)"; then
                printf '%s\t%s\n' "$ANNB_IMAGE_TAG" "$ANNB_IMAGE_METADATA"
            else
                printf '%s\tMISSING\tMISSING\n' "$ANNB_IMAGE_TAG"
                ANNB_SNAPSHOT_RC=1
            fi
        done
    } >"${ANNB_REPRO_DIR}/image-lock-${ANNB_SNAPSHOT_PHASE}.tsv"

    if ((${#ANNB_IMAGE_TAGS[@]})); then
        docker image inspect "${ANNB_IMAGE_TAGS[@]}" \
            >"${ANNB_REPRO_DIR}/docker-image-inspect-${ANNB_SNAPSHOT_PHASE}.json" \
            2>"${ANNB_REPRO_DIR}/docker-image-inspect-${ANNB_SNAPSHOT_PHASE}.errors" \
            || ANNB_SNAPSHOT_RC=1
    fi
    return "$ANNB_SNAPSHOT_RC"
}

annb_compare_image_snapshots() {
    local ANNB_REPRO_DIR="${ANNB_ATTEMPT_DIR}/reproducibility"
    local ANNB_BEFORE="${ANNB_REPRO_DIR}/image-lock-before.tsv"
    local ANNB_AFTER="${ANNB_REPRO_DIR}/image-lock-after.tsv"
    [[ -f "$ANNB_BEFORE" && -f "$ANNB_AFTER" ]] || return 0

    if diff -u "$ANNB_BEFORE" "$ANNB_AFTER" >"${ANNB_REPRO_DIR}/image-lock-diff.txt"; then
        printf 'IMAGE_TAGS_STABLE=true\n' >"${ANNB_REPRO_DIR}/image-lock-status.txt"
    else
        printf 'IMAGE_TAGS_STABLE=false\n' >"${ANNB_REPRO_DIR}/image-lock-status.txt"
        printf 'Warning: one or more Docker tags changed during this benchmark; see %s.\n' \
            "${ANNB_REPRO_DIR}/image-lock-diff.txt" >&2
    fi
}

annb_on_signal() {
    local ANNB_SIGNAL="$1"
    trap - INT TERM
    ANNB_STATE="TERMINATED"
    if [[ -n "$ANNB_CHILD_PID" ]] && kill -0 "$ANNB_CHILD_PID" 2>/dev/null; then
        kill -TERM -- "-${ANNB_CHILD_PID}" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 "$ANNB_CHILD_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-${ANNB_CHILD_PID}" 2>/dev/null || true
    fi
    annb_cleanup_containers
    [[ "$ANNB_SIGNAL" == INT ]] && exit 130
    exit 143
}

annb_on_exit() {
    local ANNB_RC=$?
    local ANNB_FINAL_RC=$ANNB_RC
    local ANNB_AUDIT_RC=0
    local ANNB_FINISHED_EPOCH
    trap - EXIT
    set +e

    ANNB_EXIT_CODE="$ANNB_RC"
    if [[ -f "${ANNB_REPO_DIR}/annb.log" ]] && ((ANNB_BENCHMARK_STARTED)); then
        cp "${ANNB_REPO_DIR}/annb.log" "${ANNB_ATTEMPT_DIR}/framework.log"
    fi
    annb_collect_results
    if [[ -f "${ANNB_ATTEMPT_DIR}/reproducibility/image-lock-before.tsv" ]]; then
        annb_snapshot_images after || true
        annb_compare_image_snapshots
    fi
    if ((ANNB_BENCHMARK_STARTED)); then
        annb_audit_results >"${ANNB_ATTEMPT_DIR}/audit.log" 2>&1
        ANNB_AUDIT_RC=$?
        sha256sum "${ANNB_REPO_DIR}/data/${ANNB_DATASET}.hdf5" \
            >"${ANNB_ATTEMPT_DIR}/input-sha256.txt" 2>&1
    fi
    if ((ANNB_RC != 0)); then
        annb_cleanup_containers
    fi

    if [[ "$ANNB_STATE" == PREPARING || "$ANNB_STATE" == RUNNING ]]; then
        if ((ANNB_RC == 0 && ANNB_AUDIT_RC == 0)); then
            ANNB_STATE="COMPLETE"
        elif ((ANNB_RC == 0)); then
            ANNB_STATE="INCOMPLETE"
            ANNB_FINAL_RC=1
        else
            ANNB_STATE="FAILED"
        fi
    fi
    ANNB_EXIT_CODE="$ANNB_FINAL_RC"
    annb_write_status
    ANNB_FINISHED_EPOCH="$(date +%s)"
    {
        printf 'FINISHED_AT=%s\n' "$(date -Is)"
        printf 'ELAPSED_SECONDS=%s\n' "$((ANNB_FINISHED_EPOCH - ANNB_STARTED_EPOCH))"
        printf 'PROCESS_EXIT_CODE=%s\n' "$ANNB_RC"
        printf 'FINAL_EXIT_CODE=%s\n' "$ANNB_FINAL_RC"
        printf 'STATE=%s\n' "$ANNB_STATE"
        printf 'RESULT_FILES=%s\n' "$(annb_result_count)"
    } >"${ANNB_ATTEMPT_DIR}/completion.txt"
    printf 'Benchmark finished: state=%s exit_code=%s results=%s\n' \
        "$ANNB_STATE" "$ANNB_FINAL_RC" "$(annb_result_count)"
    exit "$ANNB_FINAL_RC"
}

trap annb_on_exit EXIT
trap 'annb_on_signal INT' INT
trap 'annb_on_signal TERM' TERM

cd "$ANNB_REPO_DIR"
if [[ ! -f "data/${ANNB_DATASET}.hdf5" ]]; then
    printf 'Missing dataset: %s\n' "${ANNB_REPO_DIR}/data/${ANNB_DATASET}.hdf5" >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    printf 'Docker daemon is not available.\n' >&2
    exit 1
fi

cp "${BASH_SOURCE[0]}" "${ANNB_ATTEMPT_DIR}/launcher-script.sh"
git status --short >"${ANNB_ATTEMPT_DIR}/git-status-before.txt"
git diff --binary >"${ANNB_ATTEMPT_DIR}/working-tree-before.patch"
git diff --cached --binary >"${ANNB_ATTEMPT_DIR}/staged-before.patch"
git ls-files --others --exclude-standard >"${ANNB_ATTEMPT_DIR}/untracked-files-before.txt"
docker image ls --digests --no-trunc >"${ANNB_ATTEMPT_DIR}/docker-images.txt"
docker ps -aq >"${ANNB_ATTEMPT_DIR}/containers-before.txt"

set +e
ANNB_REPO_DIR="$ANNB_REPO_DIR" ANNB_ATTEMPT_DIR="$ANNB_ATTEMPT_DIR" \
ANNB_DATASET="$ANNB_DATASET" ANNB_COUNT="$ANNB_COUNT" \
    "$ANNB_PYTHON" - <<'PY' >"${ANNB_ATTEMPT_DIR}/benchmark-plan.txt" 2>&1
import json
import os
import sys
from collections import Counter
from pathlib import Path

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.definitions import get_definitions
from ann_benchmarks.results import build_result_filepath

repo = Path(os.environ["ANNB_REPO_DIR"])
attempt = Path(os.environ["ANNB_ATTEMPT_DIR"])
dataset_name = os.environ["ANNB_DATASET"]
count = int(os.environ["ANNB_COUNT"])
dataset, dimension = get_dataset(dataset_name)
definitions = get_definitions(
    dimension=dimension,
    point_type=dataset.attrs.get("point_type", "float"),
    distance_metric=dataset.attrs["distance"],
    count=count,
    base_dir="ann_benchmarks/algorithms",
)
disabled = [definition for definition in definitions if definition.disabled]
enabled = [definition for definition in definitions if not definition.disabled]
paths = []
long_paths = []
for definition in enabled:
    for query_arguments in definition.query_argument_groups or [[]]:
        relative = build_result_filepath(dataset_name, count, definition, query_arguments, False)
        paths.append(relative)
        size = len(os.fsencode(os.path.basename(relative)))
        if size > os.pathconf(repo, "PC_NAME_MAX"):
            long_paths.append((size, relative))

unique_paths = sorted(set(paths))
(attempt / "expected-results.txt").write_text("\n".join(unique_paths) + "\n")
(attempt / "result-directories.txt").write_text(
    "\n".join(sorted({str(Path(path).parent) for path in unique_paths})) + "\n"
)
(attempt / "required-images.txt").write_text(
    "\n".join(sorted({definition.docker_tag for definition in enabled})) + "\n"
)
(attempt / "required-algorithm-directories.txt").write_text(
    "\n".join(sorted({definition.module.rsplit(".", 1)[-1] for definition in enabled}))
    + "\n"
)
(attempt / "overlong-result-paths.txt").write_text(
    "\n".join(f"{size}\t{path}" for size, path in sorted(long_paths, reverse=True))
    + ("\n" if long_paths else "")
)

counts = Counter(definition.algorithm for definition in enabled)
required_algorithms = {
    "zvec-fast_query_doc_ids",
    "zvec-ann_bench_doc_ids",
    "descartes(01AI)",
    "qsgngt",
}
missing_algorithms = sorted(required_algorithms - counts.keys())
print(f"definitions={len(enabled)}")
print(f"query_argument_groups={len(paths)}")
print(f"unique_result_paths={len(unique_paths)}")
print(f"duplicate_result_paths={len(paths) - len(unique_paths)}")
print(f"disabled_definitions={len(disabled)}")
print(f"name_max={os.pathconf(repo, 'PC_NAME_MAX')}")
print(f"max_result_filename_bytes={max(len(os.fsencode(os.path.basename(path))) for path in unique_paths)}")
print(f"overlong_result_paths={len(long_paths)}")
print("algorithm_counts=" + json.dumps(counts, sort_keys=True))
if missing_algorithms:
    print("Missing required comparison algorithms: " + ", ".join(missing_algorithms))
    raise SystemExit(43)
if long_paths:
    print(
        "Refusing to start: stock ANN-Benchmarks cannot create the overlong result filenames; "
        "see overlong-result-paths.txt"
    )
    raise SystemExit(42)
PY
ANNB_PLAN_RC=$?
set -e
if ((ANNB_PLAN_RC != 0)); then
    if ((ANNB_PLAN_RC == 42)); then
        ANNB_STATE="BLOCKED_OVERLONG_FILENAMES"
    else
        ANNB_STATE="PREFLIGHT_FAILED"
    fi
    printf 'Benchmark preflight failed; see %s and %s.\n' \
        "${ANNB_ATTEMPT_DIR}/benchmark-plan.txt" \
        "${ANNB_ATTEMPT_DIR}/overlong-result-paths.txt" >&2
    exit "$ANNB_PLAN_RC"
fi

ANNB_MISSING_IMAGES=0
while IFS= read -r ANNB_IMAGE; do
    [[ -n "$ANNB_IMAGE" ]] || continue
    if ! docker image inspect "$ANNB_IMAGE" >/dev/null 2>&1; then
        printf 'Missing required Docker image: %s\n' "$ANNB_IMAGE" >&2
        ANNB_MISSING_IMAGES=1
    fi
done <"${ANNB_ATTEMPT_DIR}/required-images.txt"
if ((ANNB_MISSING_IMAGES)); then
    ANNB_STATE="PREFLIGHT_FAILED"
    exit 1
fi

ANNB_REPRO_DIR="${ANNB_ATTEMPT_DIR}/reproducibility"
mkdir -p "${ANNB_REPRO_DIR}/source/algorithms"
while IFS= read -r ANNB_ALGORITHM_DIRECTORY; do
    [[ -n "$ANNB_ALGORITHM_DIRECTORY" ]] || continue
    ANNB_ALGORITHM_SOURCE="${ANNB_REPO_DIR}/ann_benchmarks/algorithms/${ANNB_ALGORITHM_DIRECTORY}"
    ANNB_ALGORITHM_ARCHIVE="${ANNB_REPRO_DIR}/source/algorithms/${ANNB_ALGORITHM_DIRECTORY}"
    mkdir -p "$ANNB_ALGORITHM_ARCHIVE"
    for ANNB_SOURCE_NAME in config.yml module.py Dockerfile; do
        if [[ -f "${ANNB_ALGORITHM_SOURCE}/${ANNB_SOURCE_NAME}" ]]; then
            cp "${ANNB_ALGORITHM_SOURCE}/${ANNB_SOURCE_NAME}" "$ANNB_ALGORITHM_ARCHIVE/"
        fi
    done
done <"${ANNB_ATTEMPT_DIR}/required-algorithm-directories.txt"
mkdir -p "${ANNB_REPRO_DIR}/source/zvec"
for ANNB_ZVEC_SOURCE_NAME in config.yml module.py Dockerfile; do
    cp "${ANNB_REPO_DIR}/ann_benchmarks/algorithms/zvec/${ANNB_ZVEC_SOURCE_NAME}" \
        "${ANNB_REPRO_DIR}/source/zvec/"
done

# The image selected by config.yml is authoritative. Labels are recorded for
# provenance only; they never pin the launcher to a particular zvec build.
ANNB_ZVEC_PROVENANCE_FILE="${ANNB_REPRO_DIR}/zvec-image-provenance.txt"
if grep -Fxq 'ann-benchmarks-zvec' "${ANNB_ATTEMPT_DIR}/required-images.txt"; then
    ANNB_ZVEC_IMAGE_ID="$(docker image inspect ann-benchmarks-zvec --format '{{.Id}}')"
    ANNB_ZVEC_IMAGE_CREATED="$(docker image inspect ann-benchmarks-zvec --format '{{.Created}}')"
    ANNB_ZVEC_WHEEL_LABEL="$(docker image inspect ann-benchmarks-zvec \
        --format '{{if .Config.Labels}}{{index .Config.Labels "zvec.wheel"}}{{end}}' 2>/dev/null || true)"
    ANNB_ZVEC_WHEEL_SHA_LABEL="$(docker image inspect ann-benchmarks-zvec \
        --format '{{if .Config.Labels}}{{index .Config.Labels "zvec.wheel_sha256"}}{{end}}' 2>/dev/null || true)"
    ANNB_ZVEC_BRANCH_LABEL="$(docker image inspect ann-benchmarks-zvec \
        --format '{{if .Config.Labels}}{{index .Config.Labels "zvec.source_branch"}}{{end}}' 2>/dev/null || true)"
    ANNB_ZVEC_COMMIT_LABEL="$(docker image inspect ann-benchmarks-zvec \
        --format '{{if .Config.Labels}}{{index .Config.Labels "zvec.commit"}}{{end}}' 2>/dev/null || true)"
    {
        printf 'DOCKER_TAG=ann-benchmarks-zvec\n'
        printf 'IMAGE_ID=%s\n' "$ANNB_ZVEC_IMAGE_ID"
        printf 'IMAGE_CREATED=%s\n' "$ANNB_ZVEC_IMAGE_CREATED"
        printf 'LABEL_WHEEL=%s\n' "$ANNB_ZVEC_WHEEL_LABEL"
        printf 'LABEL_WHEEL_SHA256=%s\n' "$ANNB_ZVEC_WHEEL_SHA_LABEL"
        printf 'LABEL_SOURCE_BRANCH=%s\n' "$ANNB_ZVEC_BRANCH_LABEL"
        printf 'LABEL_SOURCE_COMMIT=%s\n' "$ANNB_ZVEC_COMMIT_LABEL"
    } >"$ANNB_ZVEC_PROVENANCE_FILE"

    if [[ -n "$ANNB_ZVEC_WHEEL_LABEL" && "$ANNB_ZVEC_WHEEL_LABEL" != */* && \
          -f "${ANNB_REPO_DIR}/ann_benchmarks/algorithms/zvec/${ANNB_ZVEC_WHEEL_LABEL}" ]]; then
        ANNB_ZVEC_WHEEL_PATH="${ANNB_REPO_DIR}/ann_benchmarks/algorithms/zvec/${ANNB_ZVEC_WHEEL_LABEL}"
        cp "$ANNB_ZVEC_WHEEL_PATH" "${ANNB_REPRO_DIR}/source/zvec/"
        read -r ANNB_ZVEC_ACTUAL_WHEEL_SHA _ < <(sha256sum "$ANNB_ZVEC_WHEEL_PATH")
        printf 'ARCHIVED_WHEEL_SHA256=%s\n' "$ANNB_ZVEC_ACTUAL_WHEEL_SHA" \
            >>"$ANNB_ZVEC_PROVENANCE_FILE"
        if [[ -n "$ANNB_ZVEC_WHEEL_SHA_LABEL" && \
              "$ANNB_ZVEC_ACTUAL_WHEEL_SHA" != "$ANNB_ZVEC_WHEEL_SHA_LABEL" ]]; then
            printf 'WHEEL_SHA256_MATCH_LABEL=false\n' >>"$ANNB_ZVEC_PROVENANCE_FILE"
            printf 'Warning: zvec wheel SHA-256 does not match the image label; recorded without blocking.\n' >&2
        else
            printf 'WHEEL_SHA256_MATCH_LABEL=true\n' >>"$ANNB_ZVEC_PROVENANCE_FILE"
        fi
    else
        printf 'ARCHIVED_WHEEL_STATUS=unavailable_or_unsafe_label\n' >>"$ANNB_ZVEC_PROVENANCE_FILE"
        printf 'Warning: the zvec wheel named by the image label was not archived; the image remains runnable.\n' >&2
    fi
fi

annb_snapshot_images before
cp "${ANNB_REPRO_DIR}/docker-image-inspect-before.json" \
    "${ANNB_REPRO_DIR}/docker-image-inspect.json"
find "${ANNB_REPRO_DIR}/source" -type f -print0 | sort -z | xargs -0 sha256sum \
    >"${ANNB_REPRO_DIR}/source-sha256.txt"

if ((ANNB_PREFLIGHT_ONLY)); then
    ANNB_STATE="PREFLIGHT_COMPLETE"
    printf 'Preflight completed successfully; benchmark was not started.\n'
    exit 0
fi

if ((ANNB_IS_RESUME)); then
    if [[ -e "$ANNB_LIVE_DATASET_DIR" ]]; then
        printf 'Live result directory already exists; refusing to merge it during resume: %s\n' \
            "$ANNB_LIVE_DATASET_DIR" >&2
        exit 1
    fi
    if [[ -d "$ANNB_ARCHIVE_DATASET_DIR" ]]; then
        mkdir -p "${ANNB_REPO_DIR}/results"
        mv "$ANNB_ARCHIVE_DATASET_DIR" "${ANNB_REPO_DIR}/results/"
    fi
else
    if [[ -e "$ANNB_LIVE_DATASET_DIR" ]]; then
        mkdir -p "${ANNB_RUN_DIR}/preexisting-results"
        mv "$ANNB_LIVE_DATASET_DIR" "${ANNB_RUN_DIR}/preexisting-results/"
    fi
fi
ANNB_OWNS_LIVE_RESULTS=1

# Pre-creating result directories avoids a stock-framework mkdir race without
# changing benchmark code or anything inside the timed query path.
while IFS= read -r ANNB_RESULT_DIRECTORY; do
    [[ -n "$ANNB_RESULT_DIRECTORY" ]] || continue
    mkdir -p "${ANNB_REPO_DIR}/${ANNB_RESULT_DIRECTORY}"
done <"${ANNB_ATTEMPT_DIR}/result-directories.txt"

{
    printf 'RUN_ID=%s\n' "$ANNB_RUN_ID"
    printf 'ATTEMPT_ID=%s\n' "$ANNB_ATTEMPT_ID"
    printf 'STARTED_AT=%s\n' "$ANNB_STARTED_AT"
    printf 'REPOSITORY=%s\n' "$ANNB_REPO_DIR"
    printf 'COMMIT=%s\n' "$(git rev-parse HEAD)"
    printf 'BRANCH=%s\n' "$(git branch --show-current)"
    printf 'DATASET=%s\n' "$ANNB_DATASET"
    printf 'COUNT=%s\n' "$ANNB_COUNT"
    printf 'RUN_COUNT=%s\n' "$ANNB_RUN_COUNT"
    printf 'PARALLELISM=%s\n' "$ANNB_PARALLELISM"
    printf 'TIMEOUT_SECONDS=%s\n' "$ANNB_TIMEOUT_SECONDS"
    printf '\nUNAME\n'
    uname -a
    printf '\nLSCPU\n'
    lscpu
    printf '\nMEMORY\n'
    free -h
    printf '\nSWAP\n'
    swapon --show || true
    printf '\nFILESYSTEM\n'
    df -hT / "$ANNB_RUN_DIR"
    printf '\nDOCKER\n'
    docker info
    printf '\nVM_MAX_MAP_COUNT\n'
    sysctl vm.max_map_count
    printf '\nPROCESSES\n'
    ps -eo pid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -50 || true
} >"${ANNB_ATTEMPT_DIR}/environment.txt" 2>&1

ANNB_COMMAND=(
    "$ANNB_PYTHON" "$ANNB_REPO_DIR/run.py"
    --dataset "$ANNB_DATASET"
    --count "$ANNB_COUNT"
    --runs "$ANNB_RUN_COUNT"
    --parallelism "$ANNB_PARALLELISM"
    --timeout "$ANNB_TIMEOUT_SECONDS"
)
if ((!ANNB_IS_RESUME)); then
    # A new timestamped run is always a full rerun. This stock flag makes the
    # guarantee explicit even if an unexpected result file appears after the
    # live SIFT directory was archived above.
    ANNB_COMMAND+=(--force)
fi
printf 'cd %q && ' "$ANNB_REPO_DIR" >"${ANNB_ATTEMPT_DIR}/command.txt"
printf '%q ' "${ANNB_COMMAND[@]}" >>"${ANNB_ATTEMPT_DIR}/command.txt"
printf '\n' >>"${ANNB_ATTEMPT_DIR}/command.txt"

printf '%s\n' "$$" >"${ANNB_ATTEMPT_DIR}/launcher.pid"
ANNB_STATE="RUNNING"
ANNB_BENCHMARK_STARTED=1
annb_write_status

printf 'Starting the unmodified ANN-Benchmarks entry point.\n'
printf 'Run directory: %s\n' "$ANNB_RUN_DIR"
printf 'Benchmark log: %s\n' "${ANNB_ATTEMPT_DIR}/benchmark.log"

setsid "${ANNB_COMMAND[@]}" >"${ANNB_ATTEMPT_DIR}/benchmark.log" 2>&1 &
ANNB_CHILD_PID=$!
printf '%s\n' "$ANNB_CHILD_PID" >"${ANNB_ATTEMPT_DIR}/benchmark.pid"
annb_write_status

set +e
wait "$ANNB_CHILD_PID"
ANNB_EXIT_CODE=$?
set -e

git status --short >"${ANNB_ATTEMPT_DIR}/git-status-after.txt"
git diff --binary >"${ANNB_ATTEMPT_DIR}/working-tree-after.patch"
exit "$ANNB_EXIT_CODE"
