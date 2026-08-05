#!/usr/bin/env bash
# setup_kblas.sh — 一键完成 KML KBLAS 编译前的所有准备工作
#
# 用法：
#   bash tools/setup_kblas.sh [BAZEL_BIN] [KML_LIB_DIR]
#
# 参数（均有默认值，脚本会自动探测）：
#   BAZEL_BIN    bazel 可执行路径
#                默认从 PATH 查找 bazel
#   KML_LIB_DIR  包含 libkblas.so 的目录（只需 lib 目录，不是根目录）
#                自动探测顺序：
#                  1. <repo>/third_party/kml/lib/kblas/omp/  （vendored，OMP 版）
#                  2. <repo>/third_party/kml/lib/kblas/      （vendored，非 OMP）
#                  3. <repo>/third_party/kml/lib/            （vendored，扁平结构）
#                  4. /usr/local/kml/lib/                    （系统安装）
#                  5. 手动指定的路径
#
# 完成以下工作（每一步均幂等，重复运行安全）：
#   1. 检查 tensorflow_serving/repo.bzl 是否有 tf_serving_vendored，没有则自动追加
#   2. 在 .bazelrc kml_kblas config 里写入实际 KML lib 路径（-L 和 -rpath）
#   3. patch eigen_contraction_kernel.h（dnnl_sgemm → cblas_sgemm 内联前向声明）
#   4. 打印验证命令和完整编译命令

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BAZEL="${1:-${BAZEL:-$(command -v bazel || echo bazel)}}"

# ── KML lib 目录自动探测 ───────────────────────────────────────────────────────
_detect_kml_lib() {
    local candidates=(
        "$REPO_ROOT/third_party/kml/lib/kblas/omp"   # vendored OMP（优先）
        "$REPO_ROOT/third_party/kml/lib/kblas"        # vendored 非 OMP
        "$REPO_ROOT/third_party/kml/lib"              # vendored 扁平
        "/usr/local/kml/lib"                           # 系统安装
    )
    for d in "${candidates[@]}"; do
        if [[ -f "$d/libkblas.so" ]]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

if [[ $# -ge 2 ]]; then
    KML_LIB_DIR="$2"
elif _detected=$(_detect_kml_lib 2>/dev/null); then
    KML_LIB_DIR="$_detected"
else
    KML_LIB_DIR="/usr/local/kml/lib"   # fallback，可能不存在
fi

echo "=== KML KBLAS Setup ==="
echo "  Bazel      : $BAZEL"
echo "  KML lib dir: $KML_LIB_DIR"
echo "  Repo       : $REPO_ROOT"
echo ""

# ── Step 1: repo.bzl ─────────────────────────────────────────────────────────

REPO_BZL="$REPO_ROOT/tensorflow_serving/repo.bzl"

if grep -q 'tf_serving_vendored' "$REPO_BZL" 2>/dev/null; then
    echo "[1/3] repo.bzl: tf_serving_vendored already present. OK"
else
    echo "[1/3] repo.bzl: tf_serving_vendored missing — appending..."
    cat >> "$REPO_BZL" << 'BZL_EOF'

def _tf_serving_vendored_impl(ctx):
    ctx.symlink(ctx.path(ctx.attr.root).dirname.get_child(ctx.attr.path), ".")

tf_serving_vendored = repository_rule(
    implementation = _tf_serving_vendored_impl,
    attrs = {
        "root": attr.label(mandatory = True),
        "path": attr.string(mandatory = True),
    },
)
BZL_EOF
    echo "[1/3] repo.bzl: appended tf_serving_vendored. Done"
fi

# ── Step 2: .bazelrc KML 路径（写入实际绝对路径）────────────────────────────────

BAZELRC="$REPO_ROOT/.bazelrc"

# 验证 libkblas.so
if [[ ! -f "$KML_LIB_DIR/libkblas.so" ]]; then
    echo "WARNING: $KML_LIB_DIR/libkblas.so not found."
    echo "  请先安装或 vendor KML："
    echo "    wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \\"
    echo "      https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm"
    echo "    # 有 sudo: sudo rpm -ivh /tmp/boostkit-kml-1.7.0-1.aarch64.rpm"
    echo "    # 无 root: rpm2cpio /tmp/boostkit-kml-*.rpm | cpio -idmv --no-absolute-filenames -D \$REPO_ROOT/third_party/kml"
    echo ""
fi

# 替换 .bazelrc 里 kml_kblas 的 -L 和 -rpath 为实际路径
# 匹配模式：-L<任意路径>  →  -L<KML_LIB_DIR>
if grep -q 'build:kml_kblas.*linkopt.*-L' "$BAZELRC"; then
    CURRENT_L=$(grep 'build:kml_kblas.*linkopt.*-L' "$BAZELRC" | head -1 | sed 's/.*-L//' | tr -d '\n')
    if [[ "$CURRENT_L" == "$KML_LIB_DIR" ]]; then
        echo "[2/3] .bazelrc: kml_kblas linker path already correct. OK"
    else
        echo "[2/3] .bazelrc: updating linker path → $KML_LIB_DIR"
        sed -i "s|build:kml_kblas --linkopt=-L.*|build:kml_kblas --linkopt=-L$KML_LIB_DIR|" "$BAZELRC"
        sed -i "s|build:kml_kblas --linkopt=-Wl,-rpath,.*|build:kml_kblas --linkopt=-Wl,-rpath,$KML_LIB_DIR|" "$BAZELRC"
        echo "[2/3] .bazelrc: updated. Done"
    fi
else
    echo "[2/3] .bazelrc: kml_kblas linkopt not found (unexpected). Please check .bazelrc manually."
fi

# ── Step 3 (必需): patch eigen_contraction_kernel.h ───────────────────────────
# gemm_server --backend=kblas|eigen 是 TF Session -> MatMulOp -> Eigen::Tensor::
# contract() 的同一条调用路径，运行时开关 tf_serving_kblas_enabled 是这个 patch
# 加进头文件里的——没有这个 patch，--backend 这个开关在 TF 内核里根本不存在，
# --backend=kblas 会在启动时报错而不是静默退化成 Eigen。这一步不能跳过。

if [[ ! -x "$BAZEL" ]]; then
    echo "[3/3] bazel not found at $BAZEL — skipping header patch"
    echo "  请在安装好 bazel 后单独运行（--backend=kblas 在此之前不可用）："
    echo "    bash $SCRIPT_DIR/apply_kblas_patch.sh $BAZEL"
    echo ""
    echo "=== Setup complete (patch skipped) ==="
    echo ""
    echo "repo.bzl 和 .bazelrc 已就绪，但 gemm_server 还不能用 --backend=kblas，"
    echo "先装好 bazel 再跑一次本脚本，或直接跑 apply_kblas_patch.sh。"
    exit 0
fi

OUTPUT_BASE=$("$BAZEL" info output_base 2>/dev/null || true)
HEADER=""
if [[ -n "$OUTPUT_BASE" ]]; then
    HEADER="$OUTPUT_BASE/external/org_tensorflow/third_party/xla/xla/tsl/framework/contraction/eigen_contraction_kernel.h"
fi

if [[ -z "$OUTPUT_BASE" ]] || [[ ! -f "$HEADER" ]]; then
    echo "[3/3] eigen_contraction_kernel.h: Bazel cache 里还没有这个文件。"
    echo "  先不加 --config=kml_kblas 跑一次普通 build 让 Bazel 解压 TF，"
    echo "  然后重新执行本脚本，或直接跑 tools/apply_kblas_patch.sh。"
    echo ""
    echo "=== Setup complete (patch deferred) ==="
    echo ""
    echo "repo.bzl 和 .bazelrc 已就绪，但 gemm_server 还不能用 --backend=kblas，"
    echo "直到上面的 patch 步骤跑完为止。"
    exit 0
fi

if grep -q 'cblas_sgemm' "$HEADER"; then
    echo "[3/3] eigen_contraction_kernel.h: already patched. OK"
else
    echo "[3/3] eigen_contraction_kernel.h: patching..."
    bash "$SCRIPT_DIR/apply_kblas_patch.sh" "$BAZEL"
fi

# ── 打印汇总 ──────────────────────────────────────────────────────────────────

BAZEL_DISTDIR="${BAZEL_DISTDIR:-$REPO_ROOT/../tf_new/dist}"
GCC_RPATH="${GCC_RPATH:-$(dirname $(dirname $(command -v gcc)))/lib64}"

echo ""
echo "=== Setup complete ==="
echo ""
echo "验证 KBLAS 已链入（build 完成后）："
echo "  nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm"
echo "  # 预期：U cblas_sgemm"
echo "  ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas"
echo "  # 预期：libkblas.so => $KML_LIB_DIR/libkblas.so"
echo ""
echo "编译（一个 binary，同时支持 --backend=kblas|eigen）："
echo "  bash tools/build_backends.sh $BAZEL"
echo ""
echo "  等价的手动命令："
echo "  $BAZEL build -c opt \\"
echo "    --distdir=$BAZEL_DISTDIR \\"
echo "    --define=no_cuda_support=true --define=no_nccl_support=true \\"
echo "    --define=no_kafka_support=true --define=no_google_cloud_support=true \\"
echo "    --repo_env=CC=/usr/bin/gcc --repo_env=CXX=/usr/bin/g++ \\"
echo "    --host_linkopt=-Wl,--disable-new-dtags \\"
echo "    --host_linkopt=-Wl,-rpath,$GCC_RPATH \\"
echo "    --linkopt=-Wl,--disable-new-dtags \\"
echo "    --linkopt=-Wl,-rpath,$GCC_RPATH \\"
echo "    --config=kml_kblas \\"
echo "    //tf_serving_gemm/tf_gemm_server:gemm_server \\"
echo "    //tf_serving_gemm/tf_gemm_server:gemm_client"
echo ""
echo "运行对比（必须设 LD_LIBRARY_PATH，因为 $KML_LIB_DIR 在非标准位置）："
echo "  export LD_LIBRARY_PATH=$KML_LIB_DIR:\$LD_LIBRARY_PATH"
echo "  bash tools/compare_backends.sh          # 一键起两个实例 + 跑 benchmark + 对比"
echo ""
echo "  或手动："
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server --backend=kblas &"
echo "  ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --iters=30 --warmup=5"
