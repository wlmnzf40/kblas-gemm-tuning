---
name: kml-library-replacement
description: >-
  这是通用 KML（鲲鹏数学库）替换技能。给定一个目标 C/C++ 项目，自动判断其中哪些数学函数可以用
  KML 库替换，选择正确的 KML 子库（KBLAS/KSVML/KFFT/KLAPACK 等），并指导大模型完成源码级或链接级的替换。
  在用户提及 KML 替换、鲲鹏数学库加速、cblas_sgemm、SLEEF 替换、OpenBLAS 替换、向量数学函数加速、
  KBLAS、KSVML、KFFT、"用 KML 加速这个项目"、"替换数学库" 等场景下触发。
  适用于鲲鹏（aarch64）平台上、有源码或可重新链接的 C/C++ 项目。
  不适用于 x86 平台、纯二进制无法重链接的项目、以及非数学密集型项目。
metadata:
  author: Kunpeng DevKit
  version: "2.0.0"
compatibility: 依赖鲲鹏 aarch64 / openEuler 22.03 LTS SP3 及以上，GCC 12.3.1+，KML 2.5.0+（boostkit-kml 或 kml rpm 包）
---

## 功能概述

本 Skill 提供端到端的 KML 替换能力：从分析目标项目的数学函数热点，到选择正确的 KML 子库，再到实施源码级或链接级替换，最终验证功能正确性和性能提升。

**KML（Kunpeng Math Library）** 是华为为鲲鹏处理器优化的数学库套件，包含以下子库：

| 子库 | .so 文件 | 功能 | 典型替换目标 |
|------|---------|------|-------------|
| KBLAS | libkblas.so | BLAS Level 1/2/3（sgemm/dgemm 等） | OpenBLAS, ATLAS, Eigen, MKL |
| KSVML | libksvml.so | 向量化数学函数（exp, log, sin, cos 等） | SLEEF, libm, ivml |
| KFFT | libkfft.so | 快速傅里叶变换 | FFTW, MKL FFT |
| KLAPACK | libklapack.so | LAPACK 高级线性代数 | LAPACK, MKL LAPACK |
| KM | libkm.so | 基础数学运行时 | libm |
| KIPL | libkipl.so | 图像处理库 | OpenCV 部分算子 |

**关键路径差异：** KML 库按指令集架构分目录：
- `/usr/local/kml/lib/neon/` — NEON 指令集（兼容所有鲲鹏处理器）
- `/usr/local/kml/lib/sve/` — SVE 指令集（鲲鹏 920+ 支持 SVE1）
- `/usr/local/kml/lib/noarch/` — 架构无关（KM、KE 等）

替换时根据目标 CPU 能力选择对应目录。

## 使用流程

### Step 1 — 安装 KML

```bash
# 方式 A：rpm 安装（root 权限）
rpm -ivh kml-2.5.0-1.aarch64.rpm
# 库文件默认在 /usr/local/kml/lib/{neon,sve,noarch}/

# 方式 B：解压到项目内（无 root 权限，推荐用于 Bazel 等沙箱构建）
mkdir -p third_party/kml
rpm2cpio kml-2.5.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D third_party/kml
# 库文件在 third_party/kml/usr/local/kml/lib/{neon,sve,noarch}/
```

验证安装：
```bash
ls /usr/local/kml/lib/neon/libksvml.so    # KSVML（NEON 版）
ls /usr/local/kml/lib/sve/libksvml.so     # KSVML（SVE 版）
ls /usr/local/kml/lib/noarch/libkm.so     # KM
ls /usr/local/kml/lib/neon/libkblas.so    # KBLAS（NEON 版）
ls /usr/local/kml/include/ksvml.h         # KSVML 头文件
```

### Step 2 — 分析目标项目的数学函数热点

#### 2.1 确认架构和 CPU 能力

```bash
# 确认是 aarch64
uname -m   # 预期：aarch64

# 确认是否支持 SVE
lscpu | grep -i sve    # 有输出则支持 SVE
# 或者
cat /proc/cpuinfo | grep -i Features | head -1 | tr ' ' '\n' | grep -i sve
```

**决策：**
- 支持 SVE → 优先使用 `/usr/local/kml/lib/sve/` 目录的库
- 仅 NEON → 使用 `/usr/local/kml/lib/neon/` 目录的库
- 不确定 → 用 NEON（兼容性最好）

#### 2.2 确认当前链接了哪些数学库

```bash
# 查看目标二进制依赖的数学库
ldd <target_binary> | grep -iE 'blas|sleef|fftw|lapack|libm|openblas|atlas|mkl|vec'
# 或查看编译参数
grep -r '\-l.*blas\|\-l.*sleef\|\-l.*fftw\|\-l.*lapack\|\-l.*m[^a-z]' <build_dir>/ 
# 或查看源码中的 include
grep -rn '#include.*blas\|#include.*sleef\|#include.*fftw\|#include.*lapack\|#include.*m\.h' <src_dir>/
```

#### 2.3 用 perf 找到数学热点函数

```bash
# 采集热点（60 秒）
perf record -F 99 -g -p <pid> -o /tmp/perf.data -- sleep 60
# 查看热点函数 Top 20
perf report -i /tmp/perf.data --stdio --no-children | grep -E '^\s+[0-9]' | head -20
```

**判断标准：**
- 热点函数名含 `sgemm`/`dgemm`/`gemm`/`matmul`/`contract` → **KBLAS 替换候选**
- 热点函数名含 `exp`/`log`/`sin`/`cos`/`tan`/`pow`/`sqrt` → **KSVML 替换候选**
- 热点函数名含 `fft`/`dft`/`rfft`/`cfft` → **KFFT 替换候选**
- 热点函数名含 `solve`/`factorize`/`inverse`/`ev`/`svd`/`qr` → **KLAPACK 替换候选**

### Step 3 — 选择替换策略

根据项目构建系统和源码可改性，选择以下一种或多种策略：

#### 策略 A：LD_PRELOAD 运行时替换（零代码修改）

**适用场景：** 项目使用了标准 BLAS/LAPACK 接口（cblas_*），且 KML 提供相同接口。

```bash
export LD_PRELOAD=/usr/local/kml/lib/neon/libkblas.so:$LD_PRELOAD
./target_binary
```

**限制：** 只能替换动态链接的库，静态链接的不行。且 KML 必须提供完全相同的符号。

**验证：**
```bash
# 确认 KML 符号被加载
ltrace -e 'cblas_sgemm' ./target_binary 2>&1 | head -5
# 或
LD_DEBUG=symbols ./target_binary 2>&1 | grep -i kblas | head -5
```

#### 策略 B：链接时替换（修改 CMakeLists.txt / Makefile）

**适用场景：** 项目使用 CMake/Make 构建，可以修改链接参数。

```cmake
# CMake 示例：将 OpenBLAS 替换为 KBLAS
# 原来：
# find_package(BLAS REQUIRED)  # 可能找到 OpenBLAS
# target_link_libraries(target ${BLAS_LIBRARIES})

# 替换为：
set(KML_ROOT "/usr/local/kml")
set(KML_LIB_DIR "${KML_ROOT}/lib/neon")
target_include_directories(target PRIVATE "${KML_ROOT}/include")
target_link_directories(target PRIVATE "${KML_LIB_DIR}")
target_link_libraries(target kblas km)
```

```makefile
# Makefile 示例：将 SLEEF 替换为 KSVML
# 原来：
# LDLIBS += -lSLEEF
# 替换为：
KML_LIB = /usr/local/kml/lib/neon
CFLAGS += -I/usr/local/kml/include
LDFLAGS += -L$(KML_LIB) -Wl,-rpath,$(KML_LIB)
LDLIBS += -lksvml -lkm
```

#### 策略 C：源码级替换（修改源代码）

**适用场景：** 项目使用了非标准接口，或 KML 接口与原库不完全兼容，需要改源码。

**KSVML 替换 SLEEF（标量函数）：**
```cpp
// 原来：
#include "sleef.h"
double result = Sleef_log10d2_u10(input);  // SLEEF 向量化 log10

// 替换为：
#include "ksvml.h"
// KSVML 向量化接口（4 个 double 一组）：
__m256d result = svml128_log10_f64(input);  // KSVML log10
// 或标量接口：
double result = svml_log10_f64(input);      // KSVML 标量 log10
```

**KBLAS 替换 Eigen GEMM（源码级）：**
```cpp
// 原来：
Eigen::MatrixXf C = A * B;  // Eigen 矩阵乘

// 替换为：
#include "cblas.h"
cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, 1.0f, A.data(), M, B.data(), K, 0.0f, C.data(), M);
```

**KFFT 替换 FFTW：**
```cpp
// 原来：
#include "fftw3.h"
fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
fftw_execute(plan);

// 替换为：
#include "kfft.h"
// KFFT 接口与 FFTW 不同，需查阅 kfft.h 确认对应接口
```

### Step 4 — 确定正确的 KML 库路径和头文件

```bash
# 头文件目录
KML_INCLUDE=/usr/local/kml/include

# 库文件目录（根据 CPU 能力选择）
if [ -d /usr/local/kml/lib/sve ] && lscpu | grep -q SVE; then
    KML_LIB=/usr/local/kml/lib/sve
else
    KML_LIB=/usr/local/kml/lib/neon
fi

# noarch 目录始终需要（KM 等基础库在这里）
KML_NOARCH=/usr/local/kml/lib/noarch

echo "KML_INCLUDE=$KML_INCLUDE"
echo "KML_LIB=$KML_LIB"
echo "KML_NOARCH=$KML_NOARCH"
```

### Step 5 — 编译和验证

#### 5.1 编译时设置环境

```bash
export PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin:$PATH
export LD_LIBRARY_PATH=$KML_LIB:$KML_NOARCH:$LD_LIBRARY_PATH
# 如果用 GCC 12.3.1，还需设置：
export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64:$LD_LIBRARY_PATH
```

#### 5.2 验证替换生效

```bash
# 1. 确认 KML 符号被引用（U = undefined，由 KML 提供）
nm -D <target_binary> | grep -E 'cblas_sgemm|svml128|kfft'
# 预期：U cblas_sgemm

# 2. 确认 ldd 能找到 KML 库
ldd <target_binary> | grep -E 'kblas|ksvml|kfft|km\.so'
# 预期：libkblas.so => /usr/local/kml/lib/...

# 3. 如果替换 SLEEF，确认 SLEEF 不再被链接
ldd <target_binary> | grep -i sleef
# 预期：无输出（已完全替换）
```

#### 5.3 性能对比

```bash
# 替换前基线（如果有备份）
./target_binary_original 2>&1 | tee /tmp/before.log

# 替换后
./target_binary 2>&1 | tee /tmp/after.log

# perf stat 对比
perf stat -p <pid> -e cycles,instructions,dTLB-load-misses,LLC-load-misses -- sleep 30 2>&1
# 关注 IPC（instructions/cycles 比值）是否提升
```

### Step 6 — 验证功能正确性

**关键：** 数学库替换后必须验证结果一致性。

```cpp
// 在关键路径上添加对比检查（临时）：
// 1. 保存替换前的计算结果
// 2. 替换后比较结果
// 3. 容差判断（浮点数不可能完全一致）
bool check_result(const float* a, const float* b, int n, float eps = 1e-4) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > eps * (1.0f + fabsf(a[i]))) {
            printf("MISMATCH at %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}
```

## KML 函数速查表

### KSVML（向量化数学函数）

| 原函数 | KSVML 向量化接口 | 数据类型 | 宽度 |
|--------|-----------------|---------|------|
| `exp` | `svml128_exp_f32` / `svml256_exp_f32` | float | 4/8 |
| `exp` | `svml128_exp_f64` / `svml256_exp_f64` | double | 2/4 |
| `log` | `svml128_log_f32` / `svml128_log_f64` | float/double | 4/2 |
| `log10` | `svml128_log10_f64` | double | 2 |
| `sin` | `svml128_sin_f32` / `svml128_sin_f64` | float/double | 4/2 |
| `cos` | `svml128_cos_f32` / `svml128_cos_f64` | float/double | 4/2 |
| `pow` | `svml128_pow_f32` / `svml128_pow_f64` | float/double | 4/2 |
| `sqrt` | `svml128_sqrt_f32` | float | 4 |

> **注意：** NEON 上 float32x4_t 对应 `svml128_*_f32`，double 一次处理 2 个对应 `svml128_*_f64`。SVE 上宽度可变。

### KBLAS（BLAS 函数）

| 原函数 | KBLAS 接口 | 说明 |
|--------|-----------|------|
| `sgemm` | `cblas_sgemm(CblasColMajor, ...)` | 单精度矩阵乘 |
| `dgemm` | `cblas_dgemm(CblasColMajor, ...)` | 双精度矩阵乘 |
| `sgemv` | `cblas_sgemv(...)` | 矩阵-向量乘 |
| `strsm` | `cblas_strsm(...)` | 三角求解 |
| `saxpy` | `cblas_saxpy(...)` | 向量运算 |

> **KBLAS 子目录：** `libkblas.so` 在 `kblas/omp/` 目录下（OpenMP 版），还有 `kblas/seq/`（串行版）。

## 常见问题速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `cannot open shared object file 'libksvml.so'` | LD_LIBRARY_PATH 未包含 KML 库目录 | `export LD_LIBRARY_PATH=$KML_LIB:$KML_NOARCH:$LD_LIBRARY_PATH` |
| `undefined symbol: cblas_sgemm` | 编译时未链接 KBLAS | 添加 `-lkblas -L$KML_LIB` |
| `undefined symbol: svml128_exp_f64` | 未包含 ksvml.h 或未链接 KSVML | `#include "ksvml.h"` + `-lksvml -lkm` |
| 替换后性能没有变化 | LD_PRELOAD 未生效或静态链接 | 用 `ldd` 确认 KML 库被加载 |
| 替换后结果不一致 | 浮点精度差异或接口参数顺序不同 | 检查 ColMajor/RowMajor、转置参数 |
| `libstdc++.so.6: cannot load` | GCC 版本不匹配 | 设置 `LD_LIBRARY_PATH` 包含 GCC 12.3.1 lib64 |
| Bazel 沙箱拒绝外部路径 | KML 在 execroot 外 | 将 KML 复制到项目 `third_party/kml/` |

## 环境要求

- **操作系统**：openEuler 22.03 LTS SP3+ / 鲲鹏 aarch64
- **编译器**：GCC 12.3.1+（gcc-toolset-12）
- **KML**：2.5.0+（kml rpm 包）或 1.7.0+（boostkit-kml）
- **CPU**：鲲鹏 920 / 920+ / 930 等

## 进阶参考

- 通用 KML 替换的实践案例（SLEEF→KSVML）：参见 [references/kml-replacement-practice.md](references/kml-replacement-practice.md)
- TF-Serving 中 Eigen GEMM 替换为 KBLAS 的疑难案例：参见 [references/tf-serving-kblas-case-study.md](references/tf-serving-kblas-case-study.md)
- Patch 技术细节（TF-Serving 专用）：参见 [references/patch-internals.md](references/patch-internals.md)
- 实际遇到的坑与解决方案：参见 [references/pitfalls.md](references/pitfalls.md)
