---
name: kblas-gemm-tuning
description: 这是 KBLAS 矩阵乘调优技能，提供在鲲鹏平台用华为 KML 的 cblas_sgemm 替换 Eigen 矩阵乘内核、并通过运行时 --backend 开关对比 GFLOPS 的能力。在用户提及 KBLAS、KML、鲲鹏矩阵乘优化、TF-Serving GEMM 性能对比、cblas_sgemm、Eigen 内核替换等场景下触发。适用于已编译过 TF-Serving 的鲲鹏 aarch64 环境下做单矩阵乘内核级性能对比。不适用于 x86 平台、未编译过 TF 的环境，也不适用于完整模型端到端推理性能评估。
metadata:
  author: Kunpeng DevKit
  version: "1.0.0"
compatibility: 依赖鲲鹏 aarch64 / openEuler 20.03 LTS SP3 及以上，Bazel 7.4.1，GCC 12.3.1+，boostkit-kml 1.7.0+，已编译过的 TF-Serving 2.17.0 仓库
---

## 功能概述

本 Skill 用于在鲲鹏（aarch64）平台上，将 TF-Serving 中 Eigen 矩阵乘内核替换为华为 KML 的 `cblas_sgemm`，并在**同一个 binary**内通过运行时 `--backend=kblas|eigen` 开关切换，对比两种 GEMM micro-kernel 的性能。

**关键特点：**
- 单一 binary，运行时切换 backend，两者走完全相同的 TF Session 调用路径
- 不修改 WORKSPACE，不触发 TF 重新 fetch，所有 patch 在 Bazel cache 中就地完成
- 所有步骤幂等可重复执行

**调用路径：**
```
gemm_server --backend=kblas|eigen
  → main() 设置 tf_serving_kblas_enabled = 1|0
  → ClientSession::Run(MatMul/BatchMatMul)
    → OpKernel::Compute()
      → Eigen::Tensor::contract()
        → ParallelMatMulKernel  [eigen_contraction_kernel.h, patch 后]
          if (tf_serving_kblas_enabled)
            → cblas_sgemm(CblasColMajor, ...) → libkblas.so  [鲲鹏 SVE/NEON 汇编]
          else
            → tf_serving_eigen_sgemm(...)    → Eigen 原生实现
```

`--backend` 只在最底层的 GEMM micro-kernel 调用上二选一，其余路径完全一致。这就是 `--backend=eigen` 能代表"原始 TF 执行路径"的原因。

## 环境约定

使用前请根据实际环境设置以下变量（脚本内有合理默认值，可被环境变量覆盖）：

```bash
export REPO=<你的 tf_serving 仓库根目录>        # 例如 /home/<user>/tf_serving
export BAZEL=<bazel 可执行文件绝对路径>          # 例如 /home/<user>/bazel-7.4.1
export DISTDIR=<预下载依赖目录>                  # 例如 $REPO/../tf_new/dist
export GCC_RPATH=<gcc lib64 路径>               # 例如 $(dirname $(dirname $(which gcc)))/lib64
```

> **重要约束**：TF 已编译过，不要执行 `bazel clean --expunge`。

## 使用流程

### Step 1 — 获取 KML（vendor 到仓库内，无需 root）

```bash
cd $REPO

wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm

mkdir -p third_party/kml
rpm2cpio /tmp/boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D third_party/kml
```

**找到 libkblas.so 的实际目录（推荐 OMP 版）：**
```bash
KML_LIB=$(dirname $(find third_party/kml -name "libkblas.so" | grep omp | head -1))
echo $KML_LIB   # 例如：third_party/kml/usr/local/kml/lib/kblas/omp
```

### Step 2 — 一键 Setup（含必需的头文件 patch）

```bash
# 自动探测 KML 路径
bash scripts/setup_kblas.sh $BAZEL

# 或手动指定 libkblas.so 所在目录
bash scripts/scripts/setup_kblas.sh $BAZEL $KML_LIB
```

setup 脚本自动完成（每步幂等）：
1. **repo.bzl 修复**：追加 `tf_serving_vendored`（缺失会报 `file does not contain symbol` 错误）
2. **.bazelrc 路径**：用实际 KML lib 绝对路径写入 `-L` 和 `-rpath`
3. **patch `eigen_contraction_kernel.h`**（**必需**）：把 `dnnl_sgemm` 调用点换成 `if (tf_serving_kblas_enabled) cblas_sgemm(...) else tf_serving_eigen_sgemm(...)`

> **patch 是必需的，不是可选优化**。没有它，`--backend` 这个运行时开关在 TF 内部根本不存在。

**如果提示 "patch deferred"**（Bazel cache 里还没有这个头文件）：
```bash
$BAZEL build -c opt --distdir=$DISTDIR \
  //tf_serving_gemm/tf_gemm_server:gemm_server 2>&1 | tail -5
bash scripts/apply_kblas_patch.sh $BAZEL
```

### Step 3 — 编译（一个 binary，同时支持两种 backend）

```bash
bash scripts/build_backends.sh $BAZEL
```

**编译后必做验证：**
```bash
# cblas_sgemm 必须是 U（undefined 由 libkblas 提供）
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
# 预期：U cblas_sgemm

# ldd 必须找到 libkblas.so
ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
# 预期：libkblas.so => <repo>/third_party/kml/.../libkblas.so
```

不带 `--config=kml_kblas` 编译时，`--backend=kblas` 会启动时直接报错退出（不会静默 fallback）。带了 `--config=kml_kblas` 但 patch 没生效时，`cblas_sgemm` 符号可能不存在——**先看 nm/ldd 再开始对比**。

### Step 4 — 运行与对比

必须设置 LD_LIBRARY_PATH：
```bash
export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH
```

一键对比（推荐）：
```bash
bash scripts/compare_backends.sh              # sweep：方阵 128~2048
bash scripts/compare_backends.sh shape_sweep  # 57 个生产 shapes
```

手动起两个实例：
```bash
nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --backend=kblas --addr=0.0.0.0:50052 > kblas.log 2>&1 &
sleep 2
cat kblas.log   # 必须有 "backend=kblas"

nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --backend=eigen --addr=0.0.0.0:50053 > eigen.log 2>&1 &
sleep 2
cat eigen.log   # backend=eigen

./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50052 --mode=sweep
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50053 --mode=sweep
```

**正常输出示例：**
```
=== adx ===
MxKxN         cnt  avg_ms  GFLOPS
512x512x512    30   2.34   115.2
...
```

## 验证清单

- [ ] KML 库已安装到 `third_party/kml/`
- [ ] `scripts/setup_kblas.sh` 执行成功，三项检查全部 OK
- [ ] `nm -D` 显示 `U cblas_sgemm`
- [ ] `ldd` 显示 `libkblas.so` 能被找到
- [ ] 两个 backend 都能正常启动，server.log 中有 `backend=kblas` / `backend=eigen`
- [ ] client 输出有实际数据行（不是只有表头）
- [ ] 两个 backend 的 GFLOPS 存在差异（若几乎相同，说明 patch 没生效）

## 常见问题速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `does not contain symbol 'tf_serving_vendored'` | repo.bzl 版本旧 | `setup_kblas.sh` 自动追加 |
| `--backend=kblas requires building with --config=kml_kblas` | binary 没带 KML 编译 | 重新跑 `build_backends.sh` |
| `libkblas.so: cannot open` | LD_LIBRARY_PATH 未设 | `export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH` |
| client 输出无数据行 | server crash 或超时 | 看 server.log；检查 nm/ldd |
| 两个 backend GFLOPS 几乎相同 | 头文件 patch 没生效 | 重跑 `apply_kblas_patch.sh`，检查 WARNING，重新编译 |
| `CONTENT_DOES_NOT_MATCH_TARGET` in fetch | 改了 WORKSPACE 触发 re-fetch | 不要改 WORKSPACE，只跑 `setup_kblas.sh` |
| `include path references a path outside of execution root` | KML 在仓库外 | 把 KML 复制到 `$REPO/third_party/kml/` |
| Bazel 版本不匹配 | 用了旧版本 Bazel | 用 WORKSPACE 要求的版本（通常 7.4.1） |

## 进阶参考

- Patch 的技术细节和调用链：参见 [references/patch-internals.md](references/patch-internals.md)
- 实际遇到的坑与解决方案：参见 [references/pitfalls.md](references/pitfalls.md)
- 性能基线测试结果：参见 [references/performance-baseline.md](references/performance-baseline.md)
- .bazelrc KML 配置示例：参见 [assets/bazelrc.kml_example](assets/bazelrc.kml_example)

## 环境要求

- **操作系统**：openEuler 20.03-LTS-SP3 / 鲲鹏 aarch64
- **编译器**：GCC 12.3.1+ / Bazel 7.4.1
- **KML**：1.7.0+（boostkit-kml）
- **TensorFlow**：2.17.0（TF-Serving）
- **Python**：3.10+（如需相关依赖）
