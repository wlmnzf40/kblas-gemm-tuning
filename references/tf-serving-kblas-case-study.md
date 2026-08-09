# 疑难软件案例：TF-Serving 中 Eigen GEMM 替换为 KBLAS

> 本案例记录了在 TensorFlow Serving（TF-Serving）中将 Eigen 矩阵乘内核替换为华为 KML cblas_sgemm 的完整过程。
> 由于 TF-Serving 使用 Bazel 构建系统且 Eigen 内核深度嵌入 TensorFlow 源码树，此案例代表了 KML 替换中最复杂的场景。
> 通用 KML 替换流程参见 [SKILL.md](../SKILL.md)。

## 背景

TF-Serving 的 GEMM 调用链：

```
gemm_server --backend=kblas|eigen
  → main() 设置 tf_serving_kblas_enabled = 1|0
  → ClientSession::Run(MatMul/BatchMatMul)
    → OpKernel::Compute()
      → Eigen::Tensor::contract()
        → ParallelMatMulKernel  [eigen_contraction_kernel.h]
          if (tf_serving_kblas_enabled)
            → cblas_sgemm(CblasColMajor, ...) → libkblas.so
          else
            → tf_serving_eigen_sgemm(...)    → Eigen 原生实现
```

## 核心挑战

### 挑战 1：Bazel 沙箱隔离

TF-Serving 使用 Bazel 构建，编译时所有依赖必须在 Bazel execroot 内。KML 安装在 `/usr/local/kml/`（execroot 外），直接用 `-I` 或 `-L` 引用会被 Bazel 拒绝。

**解决方案：** 将 KML 复制到仓库内 `third_party/kml/`：
```bash
mkdir -p third_party/kml
rpm2cpio boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D third_party/kml
```

### 挑战 2：Eigen 内核深度嵌入

目标函数 `eigen_contraction_kernel.h` 位于 Bazel cache 中（`external/org_tensorflow/third_party/xla/...`），不在用户仓库源码树中。直接修改会被下次 Bazel fetch 覆盖。

**解决方案：** 通过 `setup_kblas.sh` 脚本就地修改 Bazel cache，**不修改 WORKSPACE**，不触发 re-fetch。所有操作幂等可重复。

### 挑战 3：运行时 backend 切换

需要同一个 binary 同时支持 `--backend=kblas` 和 `--backend=eigen`，在运行时选择 GEMM micro-kernel。

**解决方案：** 在 `eigen_contraction_kernel.h` 中注入全局开关变量：
```cpp
// patch 注入的内容：
extern "C" {
    extern int tf_serving_kblas_enabled;
    void cblas_sgemm(int, int, int, int, int, int, float, const float*, int, 
                     const float*, int, float, float*, int);
    static inline void tf_serving_eigen_sgemm(...) {
        // Eigen 原生 fallback 实现
    }
}
// ...
if (tf_serving_kblas_enabled)
    cblas_sgemm(CblasColMajor, ...);
else
    tf_serving_eigen_sgemm(...);
```

### 挑战 4：dnnl_sgemm 正则匹配

原始 TF 代码中 GEMM 调用点使用 `dnnl_sgemm(...)`，不同 TF 版本间参数格式略有差异，正则可能失配。

**解决方案：** patch 脚本在匹配失败时打印实际代码片段（repr 格式），方便调试和更新正则。**不要手动修改 Bazel cache**。

### 挑战 5：符号链接指向绝对路径

rpm2cpio 解压后，`libkblas.so` 可能是指向 `/usr/local/kml/lib/kblas/omp/libkblas.so` 的绝对路径符号链接，在项目目录内无法解析。

**解决方案：** 修复符号链接：
```bash
cd third_party/kml/usr/local/kml/lib
rm libkblas.so
ln -s kblas/omp/libkblas.so libkblas.so
```

## 实施步骤

### Step 1 — 获取并安装 KML

```bash
cd $REPO
wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm
mkdir -p third_party/kml
rpm2cpio /tmp/boostkit-kml-1.7.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D third_party/kml

KML_LIB=$(dirname $(find third_party/kml -name "libkblas.so" | grep omp | head -1))
```

### Step 2 — 一键 Setup

```bash
bash scripts/setup_kblas.sh $BAZEL
```

setup 自动完成（每步幂等）：
1. `repo.bzl` 追加 `tf_serving_vendored` 函数
2. `.bazelrc` 写入 KML 的 `-L` 和 `-rpath`
3. Patch `eigen_contraction_kernel.h`：注入 `tf_serving_kblas_enabled` 开关

如果提示 "patch deferred"（Bazel cache 中还没有该头文件）：
```bash
$BAZEL build -c opt --distdir=$DISTDIR \
  //tf_serving_gemm/tf_gemm_server:gemm_server 2>&1 | tail -5
bash scripts/apply_kblas_patch.sh $BAZEL
```

### Step 3 — 编译

```bash
bash scripts/build_backends.sh $BAZEL
```

验证：
```bash
# cblas_sgemm 必须是 U（undefined，由 libkblas 提供）
nm -D bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep cblas_sgemm
# 预期：U cblas_sgemm

# ldd 必须找到 libkblas.so
ldd bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server | grep kblas
# 预期：libkblas.so => <repo>/third_party/kml/.../libkblas.so
```

### Step 4 — 运行与对比

```bash
export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH

# 一键对比
bash scripts/compare_backends.sh              # sweep：方阵 128~2048
bash scripts/compare_backends.sh shape_sweep  # 57 个生产 shapes

# 手动
nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --backend=kblas --addr=0.0.0.0:50052 > kblas.log 2>&1 &
nohup ./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_server \
  --backend=eigen --addr=0.0.0.0:50053 > eigen.log 2>&1 &
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50052 --mode=sweep
./bazel-bin/tf_serving_gemm/tf_gemm_server/gemm_client --host=localhost:50053 --mode=sweep
```

## 调用链详解

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

`ClientSession::Run()` 到 `Eigen::Tensor::contract()` 这条路径两个 backend 完全一致，只有最后一步 GEMM micro-kernel 调用不同。这就是 `--backend=eigen` 能代表"原始 TF 执行路径"的原因。

## 常见问题速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `does not contain symbol 'tf_serving_vendored'` | repo.bzl 版本旧 | `setup_kblas.sh` 自动追加 |
| `--backend=kblas requires building with --config=kml_kblas` | binary 没带 KML 编译 | 重新跑 `build_backends.sh` |
| `libkblas.so: cannot open` | LD_LIBRARY_PATH 未设 | `export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH` |
| client 输出无数据行 | server crash 或超时 | 看 server.log；检查 nm/ldd |
| 两个 backend GFLOPS 几乎相同 | 头文件 patch 没生效 | 重跑 `apply_kblas_patch.sh`，检查 WARNING |
| `CONTENT_DOES_NOT_MATCH_TARGET` | 改了 WORKSPACE 触发 re-fetch | 不要改 WORKSPACE，只跑 `setup_kblas.sh` |
| `include path references a path outside of execution root` | KML 在仓库外 | 把 KML 复制到 `third_party/kml/` |
| Bazel 版本不匹配 | 用了旧版本 Bazel | 用 WORKSPACE 要求的版本（通常 7.4.1） |

## 环境要求

- **操作系统**：openEuler 20.03-LTS-SP3 / 鲲鹏 aarch64
- **编译器**：GCC 12.3.1+ / Bazel 7.4.1
- **KML**：1.7.0+（boostkit-kml）
- **TensorFlow**：2.17.0（TF-Serving）
