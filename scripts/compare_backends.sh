#!/usr/bin/env bash
# compare_backends.sh — 同一个 binary，--backend=kblas 和 --backend=eigen 各跑一次，打印 GFLOPS 对比
#
# 用法：
#   bash tools/compare_backends.sh [MODE] [EXTRA_CLIENT_FLAGS...]
#
#   MODE:
#     sweep        (默认) 方矩阵 128~2048，最直观的 GFLOPS 对比
#     shape_sweep  生产 shapes（57 个小矩阵，adx/cvr/hmv/presort）
#     compute      单次 512×512 往返
#
# 示例：
#   bash tools/compare_backends.sh
#   bash tools/compare_backends.sh sweep --sizes=256,512,1024,2048 --iters=50
#   bash tools/compare_backends.sh shape_sweep --iters=30 --warmup=5

set -euo pipefail

MODE="${1:-sweep}"
shift 2>/dev/null || true   # 剩余参数传给 client

KML_LIB="${KML_LIB:-}"

# 自动探测 KML lib 路径
if [[ -z "$KML_LIB" ]]; then
    for d in \
        "$(pwd)/third_party/kml/lib/kblas/omp" \
        "$(pwd)/third_party/kml/lib/kblas" \
        "$(pwd)/third_party/kml/lib" \
        "/usr/local/kml/lib"
    do
        if [[ -f "$d/libkblas.so" ]]; then
            KML_LIB="$d"; break
        fi
    done
fi

if [[ -z "$KML_LIB" ]]; then
    echo "WARNING: libkblas.so not found. KBLAS backend may crash."
    echo "  Set KML_LIB=/path/to/dir/containing/libkblas.so before running."
fi

SERVER="./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server"
CLIENT="./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client"

PORT_KBLAS=50052
PORT_EIGEN=50053

if [[ ! -f "$SERVER" ]]; then
    echo "ERROR: $SERVER not found. Run: bash tools/build_backends.sh"
    exit 1
fi
if [[ ! -f "$CLIENT" ]]; then
    echo "ERROR: $CLIENT not found"
    exit 1
fi

cleanup() {
    echo ""
    echo "Stopping servers..."
    [[ -n "${PID_KBLAS:-}" ]] && kill "$PID_KBLAS" 2>/dev/null || true
    [[ -n "${PID_EIGEN:-}" ]] && kill "$PID_EIGEN"  2>/dev/null || true
    rm -f /tmp/kblas_server.log /tmp/eigen_server.log
}
trap cleanup EXIT

# ── 启动两个实例：同一个 binary，不同 --backend ──────────────────────────────
echo "Starting KBLAS backend  (port $PORT_KBLAS)..."
LD_LIBRARY_PATH="${KML_LIB:+$KML_LIB:}${LD_LIBRARY_PATH:-}" \
    "$SERVER" --addr="0.0.0.0:$PORT_KBLAS" --backend=kblas \
    > /tmp/kblas_server.log 2>&1 &
PID_KBLAS=$!

echo "Starting Eigen backend  (port $PORT_EIGEN)..."
LD_LIBRARY_PATH="${KML_LIB:+$KML_LIB:}${LD_LIBRARY_PATH:-}" \
    "$SERVER" --addr="0.0.0.0:$PORT_EIGEN" --backend=eigen \
    > /tmp/eigen_server.log 2>&1 &
PID_EIGEN=$!

echo "Waiting for servers to initialize..."
sleep 2

# 检查两个实例是否还在跑
if ! kill -0 "$PID_KBLAS" 2>/dev/null; then
    echo "ERROR: KBLAS server crashed. Log:"
    cat /tmp/kblas_server.log
    exit 1
fi
if ! kill -0 "$PID_EIGEN" 2>/dev/null; then
    echo "ERROR: Eigen server crashed. Log:"
    cat /tmp/eigen_server.log
    exit 1
fi

echo "Both servers running."
echo ""

# ── 设置 client 参数 ──────────────────────────────────────────────────────────
case "$MODE" in
    sweep)
        MODE_FLAGS=(--mode=sweep --sizes=128,256,512,1024,2048 --iters=30 --warmup=5)
        ;;
    shape_sweep)
        MODE_FLAGS=(--mode=shape_sweep --iters=30 --warmup=5)
        ;;
    compute)
        MODE_FLAGS=(--mode=compute --M=512 --K=512 --N=512 --iters=50 --warmup=10)
        ;;
    *)
        echo "Unknown mode: $MODE. Use sweep|shape_sweep|compute"
        exit 1
        ;;
esac

# 追加用户额外参数（覆盖默认值）
MODE_FLAGS+=("$@")

# ── 运行 benchmark ────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo "  KBLAS backend  (port $PORT_KBLAS)"
echo "════════════════════════════════════════"
"$CLIENT" --host="localhost:$PORT_KBLAS" "${MODE_FLAGS[@]}"

echo ""
echo "════════════════════════════════════════"
echo "  Eigen backend  (port $PORT_EIGEN)"
echo "════════════════════════════════════════"
"$CLIENT" --host="localhost:$PORT_EIGEN" "${MODE_FLAGS[@]}"

echo ""
echo "════════════════════════════════════════"
echo "  Done. Compare GFLOPS above."
echo "════════════════════════════════════════"
