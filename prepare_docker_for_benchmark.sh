#!/usr/bin/env bash
set -Eeuo pipefail

LOCK_FILE="/run/lock/ann-benchmarks-docker-prepare.lock"
SMOKE_IMAGE="${DOCKER_SMOKE_IMAGE:-ubuntu:22.04}"
BENCHMARK_CONTAINER_IDS=()

log() {
    printf '%s [prepare-docker] %s\n' "$(date '+%F %T%z')" "$*"
}

fail() {
    log "ERROR: $*" >&2
    exit 1
}

report_result() {
    local exit_code=$?
    trap - EXIT

    if (( exit_code == 0 )); then
        log "RESULT=SUCCESS"
    else
        log "RESULT=FAILED exit_code=${exit_code}"
    fi

    exit "$exit_code"
}
trap report_result EXIT

collect_benchmark_containers() {
    local all_ids container_id container_image

    BENCHMARK_CONTAINER_IDS=()
    all_ids="$(docker ps -aq)" || fail "无法查询 Docker 容器"

    while IFS= read -r container_id; do
        [[ -n "$container_id" ]] || continue

        container_image="$(
            docker inspect --format '{{.Config.Image}}' "$container_id"
        )" || fail "无法检查容器 ${container_id}"

        case "$container_image" in
            ann-benchmarks*)
                BENCHMARK_CONTAINER_IDS+=("$container_id")
                ;;
        esac
    done <<< "$all_ids"
}

(( EUID == 0 )) || fail "必须使用 root 运行"

for required_command in docker systemctl flock; do
    command -v "$required_command" >/dev/null 2>&1 ||
        fail "缺少命令: ${required_command}"
done

exec 9>"$LOCK_FILE"
flock -n 9 || fail "另一个清理脚本正在运行"

log "开始检查 Docker 环境"
log "不会清理镜像、build cache、volume 或非 ann-benchmarks 容器"

systemctl is-active --quiet docker ||
    fail "Docker 服务不是 active 状态"

docker info >/dev/null ||
    fail "无法连接 Docker daemon"

running_container_ids="$(docker ps -q)" ||
    fail "无法查询运行中的容器"

if [[ -n "$running_container_ids" ]]; then
    log "仍有运行中容器，本次不删除容器、不重启 Docker:"
    docker ps --format \
        'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}'
    fail "等待所有容器退出后重试"
fi

collect_benchmark_containers

if (( ${#BENCHMARK_CONTAINER_IDS[@]} > 0 )); then
    log "删除 ${#BENCHMARK_CONTAINER_IDS[@]} 个已停止的 ann-benchmarks 残留容器"
    docker rm "${BENCHMARK_CONTAINER_IDS[@]}"
else
    log "没有已停止的 ann-benchmarks 残留容器"
fi

log "重启 Docker daemon"
systemctl restart docker

docker_ready=0
for _ in {1..30}; do
    if systemctl is-active --quiet docker &&
        docker info >/dev/null 2>&1; then
        docker_ready=1
        break
    fi
    sleep 1
done

(( docker_ready == 1 )) ||
    fail "Docker 重启后 30 秒内未恢复正常"

post_restart_running_ids="$(docker ps -q)" ||
    fail "Docker 重启后无法查询容器"

if [[ -n "$post_restart_running_ids" ]]; then
    log "Docker 重启后出现运行中容器:"
    docker ps --format \
        'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}'
    fail "Docker 重启后的环境不是空闲状态"
fi

collect_benchmark_containers

(( ${#BENCHMARK_CONTAINER_IDS[@]} == 0 )) ||
    fail "仍有 ${#BENCHMARK_CONTAINER_IDS[@]} 个 ann-benchmarks 残留容器"

docker image inspect "$SMOKE_IMAGE" >/dev/null 2>&1 ||
    fail "本地不存在冒烟测试镜像 ${SMOKE_IMAGE}，不会自动下载"

log "使用本地镜像 ${SMOKE_IMAGE} 验证容器创建和自动删除"
docker run --rm --pull=never "$SMOKE_IMAGE" true

final_running_ids="$(docker ps -q)" ||
    fail "冒烟测试后无法查询容器"

[[ -z "$final_running_ids" ]] ||
    fail "冒烟测试后仍有运行中容器"

log "Docker benchmark 跑前清理和验证全部完成"