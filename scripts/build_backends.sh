#!/usr/bin/env bash
# build_backends.sh — 编译 GEMM server（单个 binary 支持 --backend=kblas|eigen 运行时切换）
#
# 前提：tools/setup_kblas.sh（或 tools/apply_kblas_patch.sh）必须已经成功 patch 过
# Bazel cache 里的 eigen_contraction_kernel.h —— --backend 这个运行时开关
# (tf_serving_kblas_enabled) 是这个 patch 加进 TF 内核里的，不是 server.cc 自己实现的。
# 没 patch 过的话这里编译出的 binary 里 --backend=kblas 会在启动时直接报错。
#
# 用法：
#   bash tools/build_backends.sh [BAZEL_BIN]
#
# 输出：
#   bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server   ← 同时支持两种 backend
#   bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client
#
# 运行时切换（两者走同一条 Session::Run -> MatMulOp -> Eigen::Tensor::contract()
# 路径，只是最底层 GEMM 调用不同）：
#   ./gemm_server --backend=kblas --addr=0.0.0.0:50052   # 内核里调 cblas_sgemm
#   ./gemm_server --backend=eigen --addr=0.0.0.0:50053   # 内核里调 Eigen 原生实现
#
#   或直接用对比脚本：
#     bash tools/compare_backends.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BAZEL="${1:-${BAZEL:-$(command -v bazel || echo bazel)}}"
DISTDIR="${DISTDIR:-$REPO_ROOT/../tf_new/dist}"
GCC_RPATH="${GCC_RPATH:-$(dirname $(dirname $(command -v gcc)))/lib64}"

COMMON_FLAGS=(
    -c opt
    --distdir="$DISTDIR"
    --define=no_cuda_support=true --define=no_nccl_support=true
    --define=no_kafka_support=true --define=no_google_cloud_support=true
    --repo_env=CC=/usr/bin/gcc --repo_env=CXX=/usr/bin/g++
    --host_linkopt=-Wl,--disable-new-dtags
    --host_linkopt=-Wl,-rpath,"$GCC_RPATH"
    --linkopt=-Wl,--disable-new-dtags
    --linkopt=-Wl,-rpath,"$GCC_RPATH"
)

TARGETS=(
    //tf_serving_gemm/tf_gemm_server:gemm_server
    //tf_serving_gemm/tf_gemm_server:gemm_client
)

echo "=== Building gemm_server + gemm_client (--config=kml_kblas) ==="
echo "    Supports --backend=kblas|eigen at runtime"
"$BAZEL" build "${COMMON_FLAGS[@]}" --config=kml_kblas "${TARGETS[@]}"

echo ""
echo "=== Done ==="
echo ""
echo "Quick test (two terminals or use compare_backends.sh):"
echo "  export LD_LIBRARY_PATH=\$KML_LIB:\$LD_LIBRARY_PATH"
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server --backend=kblas &"
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server --backend=eigen --addr=0.0.0.0:50053 &"
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50052  # KBLAS"
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50053  # Eigen"
echo ""
echo "Or run the full comparison:"
echo "  bash tools/compare_backends.sh"
