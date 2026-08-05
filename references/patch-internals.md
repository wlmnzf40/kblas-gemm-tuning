# Patch 技术细节与调用链

## Patch 做了什么

`scripts/apply_kblas_patch.sh` 就地修改 Bazel cache 里的 `eigen_contraction_kernel.h`：

1. **宏重命名**：避免和 TF 原生 oneDNN 路径冲突
2. **替换 `#include "dnnl.h"`**：换成 `cblas_sgemm` 内联前向声明 + 全局开关 `tf_serving_kblas_enabled` 声明 + `tf_serving_eigen_sgemm`（与 `cblas_sgemm` 同签名的 Eigen fallback）
3. **替换 `dnnl_sgemm(...)` 调用点**：换成运行时 `if/else`（正则匹配，容忍 TF 版本间的小差异）
4. **`dnnl_gemm_u8s8s32` → memset**：KML 无 int8 GEMM，这条路径不影响 fp32 benchmark

**幂等保证**：不改 WORKSPACE，不触发 TF re-fetch，可重复执行。

## 调用链

```
gemm_server --backend=kblas|eigen
  → main() 设置 tf_serving_kblas_enabled = 1|0    [server.cc]
  → ClientSession::Run(MatMul/BatchMatMul)          [server.cc]
    → OpKernel::Compute()                            [TF all_kernels]
      → Eigen::Tensor::contract()
        → ParallelMatMulKernel  [eigen_contraction_kernel.h, patch 后]
          if (tf_serving_kblas_enabled)
            → cblas_sgemm(CblasColMajor, ...) → libkblas.so  [鲲鹏 SVE/NEON 汇编]
          else
            → tf_serving_eigen_sgemm(...)    → Eigen::Map（同一头文件内联）
```

`ClientSession::Run()` 到 `Eigen::Tensor::contract()` 这条路径两个 backend 完全一致，只有最后一步 GEMM micro-kernel 调用不同——这正是 `--backend=eigen` 能代表"原始 TF 执行路径"的原因，而不是另起一条绕开 TF Session 的 `Eigen::Map` 计算。

## dnnl_sgemm pattern mismatch

patch 脚本使用正则匹配 `dnnl_sgemm` 调用点。当 TF 版本间略有差异时，正则可能失配，脚本会打印实际找到的代码片段（repr 格式）并给出 `WARNING: dnnl_sgemm found but regex didn't match` 警告。

**处理方式**：把实际内容反馈给维护者，更新 `apply_kblas_patch.sh` 里的正则即可。不要手动修改 Bazel cache 中的文件，否则下次 setup 会因幂等检查失败而跳过 patch。

## 头文件路径

patch 目标文件的典型路径：
```
<bazel output_base>/external/org_tensorflow/third_party/xla/xla/tsl/framework/contraction/eigen_contraction_kernel.h
```

setup 脚本会自动调用 `bazel info output_base` 探测这个路径。如 cache 被重新解压（例如改了 WORKSPACE 中的 `tensorflow_http_archive` 规则），patch 会被覆盖，重跑 `bash scripts/setup_kblas.sh $BAZEL` 即可恢复。
