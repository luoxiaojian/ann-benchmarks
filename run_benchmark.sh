#!/bin/bash
# =============================================================================
# ann-benchmarks 性能测试脚本 (框架原生调度模式)
# 在 sift-128-euclidean 数据集上测试所有已构建镜像的算法
# =============================================================================
# 用法:
#   ./run_benchmark.sh [--parallelism N] [--runs N] [--count N] [--timeout N]
#
# 执行策略:
#   单次调用 `python run.py`（不带 --algorithm），由 ann-benchmarks 框架
#   自行收集所有「已构建镜像 + distance metric 匹配」的算法配置，并用
#   worker pool 统一调度执行。每个容器绑定独立 CPU 核（单核公平测试）。
#
# 如何 disable 一个算法:
#   删除该算法的 Docker 镜像即可。框架会自动跳过没有镜像的算法。
#   例如禁用 pgvectorscale:  docker rmi ann-benchmarks-pgvectorscale
#
# 已 disable 的算法(镜像已删除，不参与本轮测试):
#   - pgvectorscale : recall 异常偏低(0.28~0.39)，需串行(端口5432冲突)，单变量~11h
#   - milvus        : 启动无CPU限制的外部容器，测量不公平，QPS<1000
#   - pgvector      : 需串行(端口5432冲突)，QPS极低(约zvec的1/20)
#   - 其余无镜像算法(vsag/weaviate等): 框架自动跳过
# =============================================================================

set -e

cd "$(dirname "$0")"

# 激活 Python 虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo ">>> 已激活 Python 虚拟环境: $(which python)"
else
    echo "错误: 未找到 .venv/bin/activate，请先创建虚拟环境"
    exit 1
fi

# 默认参数(与 ann-benchmarks 官方配置一致)
# 官方: r6i.16xlarge (32核/512GB), --parallelism 31, hyperthreading disabled
# 本机: Xeon 8369B (32物理核/495GB), runner 自动识别拓扑、保留1个物理核，
#       并从其余每个物理核选择1个逻辑CPU，避免两个容器共享SMT兄弟线程。
PARALLELISM=31      # 并行容器数 = 物理核数-1，每容器独占一个物理核
RUNS=5              # 每组查询参数运行次数，取最好成绩
COUNT=10            # 返回近邻数(k=10)
TIMEOUT=7200        # 单个算法超时时间(秒)，默认2小时

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallelism) PARALLELISM="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --count) COUNT="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --help|-h)
            echo "用法: $0 [--parallelism N] [--runs N] [--count N] [--timeout N]"
            echo ""
            echo "参数:"
            echo "  --parallelism N  并行容器数 (默认: 31, 与官方r6i.16xlarge一致)"
            echo "  --runs N         每组查询参数运行次数 (默认: 5)"
            echo "  --count N        返回近邻数k (默认: 10)"
            echo "  --timeout N      单个算法超时秒数 (默认: 7200)"
            echo ""
            echo "禁用算法: 可设置 ANNB_EXCLUDE_ALGORITHMS 或 --exclude-algorithms"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 待测试的数据集
# DATASETS=("sift-128-euclidean")
DATASETS=("gist-960-euclidean")

echo "============================================================"
echo " ann-benchmarks 性能测试 (框架原生调度)"
echo "============================================================"
echo " 数据集:     ${DATASETS[*]}"
echo " 并行度:     ${PARALLELISM}"
echo " 运行次数:   ${RUNS}"
echo " 近邻数(k):  ${COUNT}"
echo " 超时:       ${TIMEOUT}s"
echo "============================================================"
echo ""

# =============================================================================
# 清空旧结果
# =============================================================================
for DATASET in "${DATASETS[@]}"; do
    RESULT_DIR="results/${DATASET}/${COUNT}"
    if [ -d "${RESULT_DIR}" ]; then
        echo ">>> 清空旧结果: ${RESULT_DIR}/"
        rm -rf "${RESULT_DIR}"
    fi
done
mkdir -p results
echo ""

# 列出已构建的算法镜像(即本轮将参与测试的算法)
echo ">>> 已构建的算法镜像(框架将自动调度以下算法):"
echo ""
docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}" | grep "ann-benchmarks-" | sort
echo ""

# =============================================================================
# 单次调用 run.py: 框架自行收集所有可跑算法并调度
# =============================================================================
for DATASET in "${DATASETS[@]}"; do
    echo "============================================================"
    echo " 开始测试: ${DATASET}"
    echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""

    EXCLUDE_ALGORITHMS=""
    if [ "${DATASET}" = "gist-960-euclidean" ]; then
        EXCLUDE_ALGORITHMS="kgn,vamana(diskann),n2"
        echo ">>> GIST 临时跳过算法: ${EXCLUDE_ALGORITHMS}"
    elif [ "${DATASET}" = "sift-128-euclidean" ]; then
        echo ">>> SIFT 不跳过 kgn / vamana(diskann)"
    fi

    ANNB_EXCLUDE_ALGORITHMS="${EXCLUDE_ALGORITHMS}" python run.py \
        --dataset "${DATASET}" \
        --parallelism "${PARALLELISM}" \
        --runs "${RUNS}" \
        --count "${COUNT}" \
        --timeout "${TIMEOUT}" \
        --force

    echo ""
    echo ">>> ${DATASET} 测试完成: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "============================================================"
echo " 所有测试完成！"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 结果保存在: $(pwd)/results/"
echo "============================================================"
echo ""
echo "生成图表:"
echo "  python plot.py --dataset sift-128-euclidean --x-scale logit --y-scale log"
echo ""
echo "生成网站:"
echo "  python create_website.py --plottype recall/time --scatter --outputdir website/"
