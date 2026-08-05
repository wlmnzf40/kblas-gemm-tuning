#!/usr/bin/env bash
# apply_kblas_patch.sh — patch eigen_contraction_kernel.h in the Bazel cache
#
# 用法：
#   bash tools/apply_kblas_patch.sh [BAZEL_BIN]
#
# 关键设计：
#   不使用 #include "kblas.h"，改为内联前向声明 cblas_sgemm。
#   这样完全不需要 -I 编译器标志，Bazel hermetic 工具链不会报
#   "path outside of the execution root" 错误。
#   链接期仍需 -lkblas（由 .bazelrc kml_kblas config 提供）。
#
# 不修改 WORKSPACE，已有编译缓存完全保留。幂等，重复运行安全。

set -euo pipefail

BAZEL="${1:-${BAZEL:-$(command -v bazel || echo bazel)}}"

if [[ ! -x "$BAZEL" ]]; then
    echo "ERROR: bazel not found at $BAZEL" >&2
    echo "Usage: $0 [/path/to/bazel]" >&2
    exit 1
fi

OUTPUT_BASE=$("$BAZEL" info output_base 2>/dev/null)
if [[ -z "$OUTPUT_BASE" ]]; then
    echo "ERROR: 'bazel info output_base' failed." >&2
    exit 1
fi

HEADER="$OUTPUT_BASE/external/org_tensorflow/third_party/xla/xla/tsl/framework/contraction/eigen_contraction_kernel.h"

if [[ ! -f "$HEADER" ]]; then
    echo "ERROR: Header not found:" >&2
    echo "  $HEADER" >&2
    echo "" >&2
    echo "先不加 --config=kml_kblas 跑一次普通 build 让 Bazel 解压 TF，" >&2
    echo "然后再执行本脚本。" >&2
    exit 1
fi

if grep -q 'cblas_sgemm' "$HEADER"; then
    echo "Already patched. Nothing to do."
    exit 0
fi

echo "Patching: $HEADER"
cp "$HEADER" "${HEADER}.bak_dnnl"

export KBLAS_HEADER="$HEADER"

python3 - <<'PYEOF'
import re, sys, os

header = os.environ['KBLAS_HEADER']
with open(header) as f:
    src = f.read()

changed = False

# ── 1. 宏重命名 ───────────────────────────────────────────────────────────────
if 'TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL' in src:
    src = src.replace(
        'TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL',
        'TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL_KML')
    changed = True
    print("  macro renamed")

# ── 2. #include "dnnl.h" → 内联前向声明 + 运行时开关 ─────────────────────────
# 用标准 CBLAS 前向声明替代 #include "kblas.h"，链接期 -lkblas 提供实现。
# 这样 Bazel hermetic 工具链不会报 "path outside of execution root"。
#
# tf_serving_kblas_enabled 是运行时开关（gemm_server --backend=kblas|eigen 写入），
# 让 TF 的 MatMul/BatchMatMul 走同一条 Session::Run → Eigen::Tensor::contract() →
# 本内核 的路径，只在最底层 GEMM 调用上二选一，而不是绕开 TF 单独调一个 Eigen::Map。
KBLAS_DECL = '''\
// KML KBLAS forward declarations (replaces dnnl.h; no -I flag needed).
// Symbols resolved at link time by -lkblas.
extern "C" {
enum CBLAS_ORDER     { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans  = 111, CblasTrans    = 112, CblasConjTrans = 113 };
void cblas_sgemm(CBLAS_ORDER Order,
                 CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                 int M, int N, int K,
                 float alpha, const float* A, int lda,
                              const float* B, int ldb,
                 float beta,        float* C, int ldc);
// Runtime switch, defined in gemm_server's main.cc (server.cc); flipped by
// --backend=kblas|eigen. Lets one binary A/B test without recompiling.
extern int tf_serving_kblas_enabled;
}  // extern "C"

// Eigen-native fallback with the cblas_sgemm signature, used when
// tf_serving_kblas_enabled == 0. This keeps --backend=eigen on the exact
// same TF Session -> MatMulOp -> Eigen::Tensor::contract() call path as
// --backend=kblas; only the innermost GEMM micro-kernel differs.
inline void tf_serving_eigen_sgemm(CBLAS_ORDER /*order*/,
                                    CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                                    int M, int N, int K,
                                    float alpha, const float* A, int lda,
                                                 const float* B, int ldb,
                                    float beta, float* C, int ldc) {
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> _TfsMat;
  Eigen::Map<const _TfsMat, 0, Eigen::OuterStride<>> mA(
      A, TransA == CblasNoTrans ? M : K, TransA == CblasNoTrans ? K : M,
      Eigen::OuterStride<>(lda));
  Eigen::Map<const _TfsMat, 0, Eigen::OuterStride<>> mB(
      B, TransB == CblasNoTrans ? K : N, TransB == CblasNoTrans ? N : K,
      Eigen::OuterStride<>(ldb));
  Eigen::Map<_TfsMat, 0, Eigen::OuterStride<>> mC(C, M, N, Eigen::OuterStride<>(ldc));

  if (beta == 0.0f) { mC.setZero(); } else { mC *= beta; }
  if (TransA == CblasNoTrans && TransB == CblasNoTrans) {
    mC.noalias() += alpha * (mA * mB);
  } else if (TransA != CblasNoTrans && TransB == CblasNoTrans) {
    mC.noalias() += alpha * (mA.transpose() * mB);
  } else if (TransA == CblasNoTrans && TransB != CblasNoTrans) {
    mC.noalias() += alpha * (mA * mB.transpose());
  } else {
    mC.noalias() += alpha * (mA.transpose() * mB.transpose());
  }
}'''

old_include = '#include "dnnl.h"'
if old_include in src:
    src = src.replace(old_include, KBLAS_DECL)
    changed = True
    print("  dnnl.h → inline cblas_sgemm forward declaration")
elif 'kblas' not in src:
    print("WARNING: neither 'dnnl.h' nor 'kblas' found in header (skipping include step)",
          file=sys.stderr)

# ── 3. dnnl_sgemm → cblas_sgemm（正则匹配，容忍换行和缩进差异）──────────────
PAT_SGEMM = re.compile(
    r'dnnl_status_t\s+st\s*=\s*'
    r'(?:\n\s*)?'
    r'dnnl_sgemm\s*\('
    r'\s*([^,]+?)\s*,'   # 1: transposeB (dnnl swap)
    r'\s*([^,]+?)\s*,'   # 2: transposeA
    r'\s*([^,]+?)\s*,'   # 3: n (dnnl swap)
    r'\s*([^,]+?)\s*,'   # 4: m
    r'\s*([^,]+?)\s*,'   # 5: k
    r'\s*([^,]+?)\s*,'   # 6: alpha
    r'\s*([^,]+?)\s*,'   # 7: blockB (dnnl swap)
    r'\s*([^,]+?)\s*,'   # 8: ldB
    r'\s*([^,]+?)\s*,'   # 9: blockA
    r'\s*([^,]+?)\s*,'   # 10: ldA
    r'\s*([^,]+?)\s*,'   # 11: beta
    r'\s*([^,]+?)\s*,'   # 12: output_ptr
    r'\s*([^)]+?)\s*'    # 13: ldC
    r'\)\s*;'
    r'(?:\s*\n\s*eigen_assert\s*\(\s*st\s*==\s*0\s*\)\s*;)?',
    re.DOTALL
)

def _build_cblas(m):
    transposeB = m.group(1).strip()
    transposeA = m.group(2).strip()
    n          = m.group(3).strip()
    mm         = m.group(4).strip()
    k          = m.group(5).strip()
    alpha      = m.group(6).strip()
    blockB     = m.group(7).strip()
    ldB        = m.group(8).strip()
    blockA     = m.group(9).strip()
    ldA        = m.group(10).strip()
    beta       = m.group(11).strip()
    output_ptr = m.group(12).strip()
    ldC        = m.group(13).strip()
    # cblas(CblasColMajor) 自然顺序，无需 A/B 和 m/n 对调；运行时二选一
    # （tf_serving_kblas_enabled 由 --backend=kblas|eigen 设置），TF Session ->
    # MatMulOp -> Eigen::Tensor::contract() 的调用路径本身完全不变。
    return (
        f"{{\n"
        f"      const CBLAS_TRANSPOSE _tfs_ta = ({transposeA} == 'N') ? CblasNoTrans : CblasTrans;\n"
        f"      const CBLAS_TRANSPOSE _tfs_tb = ({transposeB} == 'N') ? CblasNoTrans : CblasTrans;\n"
        f"      if (tf_serving_kblas_enabled) {{\n"
        f"        cblas_sgemm(CblasColMajor, _tfs_ta, _tfs_tb, {mm}, {n}, {k},\n"
        f"                    {alpha}, {blockA}, {ldA}, {blockB}, {ldB},\n"
        f"                    {beta}, {output_ptr}, {ldC});\n"
        f"      }} else {{\n"
        f"        tf_serving_eigen_sgemm(CblasColMajor, _tfs_ta, _tfs_tb, {mm}, {n}, {k},\n"
        f"                    {alpha}, {blockA}, {ldA}, {blockB}, {ldB},\n"
        f"                    {beta}, {output_ptr}, {ldC});\n"
        f"      }}\n"
        f"    }}"
    )

new_src, cnt = PAT_SGEMM.subn(_build_cblas, src)
if cnt > 0:
    src = new_src
    changed = True
    print(f"  dnnl_sgemm → cblas_sgemm  ({cnt} occurrence(s))")
elif 'dnnl_sgemm' in src:
    m = re.search(r'dnnl_sgemm\s*\(.*?\)\s*;', src, re.DOTALL)
    print("WARNING: dnnl_sgemm found but regex didn't match. Actual content:",
          file=sys.stderr)
    print(repr(m.group(0)) if m else "(no match)", file=sys.stderr)
else:
    print("  dnnl_sgemm: not found (skipped)")

# ── 4. dnnl_gemm_u8s8s32 → memset（KML 无 int8 GEMM）────────────────────────
PAT_U8S8 = re.compile(
    r'dnnl_status_t\s+st\s*=\s*dnnl_gemm_u8s8s32\s*\(.*?\)\s*;'
    r'(?:\s*\n\s*eigen_assert\s*\(\s*st\s*==\s*0\s*\)\s*;)?',
    re.DOTALL
)
new_src, cnt = PAT_U8S8.subn(
    '// KBLAS has no int8 GEMM; produce zero output.\n'
    '    std::memset(C, 0, sizeof(int32_t) * size_t(m) * size_t(n));',
    src
)
if cnt > 0:
    src = new_src
    changed = True
    print(f"  dnnl_gemm_u8s8s32 → memset  ({cnt} occurrence(s))")
elif 'dnnl_gemm_u8s8s32' in src:
    print("WARNING: dnnl_gemm_u8s8s32 found but regex didn't match (non-fatal)",
          file=sys.stderr)
else:
    print("  dnnl_gemm_u8s8s32: not found (skipped)")

# ── 5. 清理悬空的 EIGEN_UNUSED_VARIABLE(st) ──────────────────────────────────
new_src = re.sub(r'\s*EIGEN_UNUSED_VARIABLE\s*\(\s*st\s*\)\s*;', '', src)
if new_src != src:
    src = new_src
    changed = True
    print("  EIGEN_UNUSED_VARIABLE(st) removed")

# ── 最终校验 ─────────────────────────────────────────────────────────────────
if 'dnnl_sgemm' in src:
    print("ERROR: dnnl_sgemm still present after patch!", file=sys.stderr)
    sys.exit(1)
if not changed:
    print("Nothing changed (already patched or different TF version)")
    sys.exit(0)

with open(header, 'w') as f:
    f.write(src)
print(f"OK: {header}")
PYEOF

echo ""
echo "Patch applied. Backup: ${HEADER}.bak_dnnl"
echo ""
echo "Build command:"
echo "  $BAZEL build -c opt --config=kml_kblas \\"
echo "    --distdir=\$DISTDIR \\"
echo "    --define=no_cuda_support=true --define=no_nccl_support=true \\"
echo "    --define=no_kafka_support=true --define=no_google_cloud_support=true \\"
echo "    --repo_env=CC=/usr/bin/gcc --repo_env=CXX=/usr/bin/g++ \\"
echo "    --host_linkopt=-Wl,--disable-new-dtags \\"
echo "    --host_linkopt=-Wl,-rpath,\$GCC_RPATH \\"
echo "    --linkopt=-Wl,--disable-new-dtags \\"
echo "    --linkopt=-Wl,-rpath,\$GCC_RPATH \\"
echo "    //tf_serving_gemm/tf_gemm_server:gemm_server \\"
echo "    //tf_serving_gemm/tf_gemm_server:gemm_client"
