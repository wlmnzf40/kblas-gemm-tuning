---
name: kml-library-replacement
description: >-
  这是通用 KML（鲲鹏数学库）自动替换技能。在鲲鹏（aarch64）平台上分析任何 C/C++ 项目时自动触发，
  无需用户明确提及 KML 或数学库替换。自动完成：下载安装 KML、检测目标项目中的数学库依赖
  （SLEEF/OpenBLAS/FFTW/Eigen/libm 等）、用 perf 或源码分析定位数学热点、选择正确的 KML 子库
  （KBLAS/KSVML/KFFT/KLAPACK）、执行源码级或链接级替换、验证功能正确性和性能提升。
  在鲲鹏 aarch64 平台上进行任何 C/C++ 源码性能优化、加速、热点分析时即应触发本技能。
  不适用于 x86 平台、纯二进制无法重链接的项目、以及非数学密集型项目。
metadata:
  author: Kunpeng DevKit
  version: "2.2.0"
compatibility: 依赖鲲鹏 aarch64 / openEuler 22.03 LTS SP3 及以上，GCC 12.3.1+，KML 2.5.0+（boostkit-kml 或 kml rpm 包）
---

## 功能概述

本 Skill 提供端到端、全自动的 KML 替换能力。**无需用户明确提及 KML 替换**，只要在鲲鹏 aarch64 平台上分析 C/C++ 项目性能，本技能即自动：

1. 识别用户意图，获取目标代码仓路径和编译指令（缺失则主动询问）
2. 下载并安装 KML 库
3. 检测目标项目链接的数学库（SLEEF/OpenBLAS/FFTW/Eigen/libm 等）
4. 用 perf 或源码 grep 定位数学热点函数
5. 选择正确的 KML 子库（KBLAS/KSVML/KFFT/KLAPACK）
6. 执行源码级或链接级替换
7. 验证功能正确性和性能提升

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

### Step 0 — 识别用户意图和需求

在执行任何操作之前，**必须先确认以下信息**。如果用户未提供，应主动询问：

#### 必需信息

| 信息 | 说明 | 示例 |
|------|------|------|
| **目标代码仓路径** | 要优化的 C/C++ 项目根目录的绝对路径 | `/root/wanglimin/LiteCall_Arm` |
| **编译指令** | 项目的完整编译命令（含 cmake/make 参数） | `cd build && cmake .. -DOpenCV_DIR=... && make -j$(nproc)` |

#### 可选但推荐信息

| 信息 | 说明 | 示例 |
|------|------|------|
| 运行指令 | 编译后如何启动程序（含环境变量） | `cd install/LiteCall && ./Basecall.Server` |
| 已知热点 | 用户已知的性能瓶颈函数或模块 | `ExtractOne 占 47% cycles` |
| 精度验收标准 | 替换后需要满足的精度指标 | `Q30 ≥ 74.0` |
| 是否有 root 权限 | 影响 KML 安装方式（rpm vs 解压到项目内） | `有 root` / `无 root` |
| 构建系统 | CMake / Makefile / Bazel / 其他 | `CMake` |

#### 询问模板

如果用户未提供必需信息，按以下模板主动询问：

```
我需要以下信息来执行 KML 替换：

1. 目标代码仓路径（必需）：你的 C/C++ 项目根目录的绝对路径是什么？
2. 编译指令（必需）：完整的编译命令是什么？（包括 cmake 参数、make 参数等）

另外，以下信息如果有请一并提供（可选）：
3. 编译后如何运行程序？
4. 是否有 root 权限？
5. 已知的性能瓶颈在哪里？
6. 精度验收标准是什么？
```

> **重要：** 在用户提供目标代码仓路径和编译指令之前，不要执行后续步骤。收到信息后，先 `ls` 确认路径存在，再检查构建系统类型（CMakeLists.txt / Makefile / WORKSPACE 等）。

### Step 1 — 下载并安装 KML

#### 1.1 下载 KML rpm 包

```bash
# 下载源 A：openEuler 官方源（推荐，SP3 对应版本）
wget -O /tmp/kml-2.5.0-1.aarch64.rpm \
  "https://repo.oepkgs.net/openeuler/rpm/openEuler-22.03-LTS-SP3/extras/aarch64/Packages/k/kml-2.5.0-1.aarch64.rpm"

# 下载源 B：boostkit-kml（旧版 1.7.0，兼容 openEuler 20.03 SP3）
wget -O /tmp/boostkit-kml-1.7.0-1.aarch64.rpm \
  "https://repo.oepkgs.net/openeuler/rpm/openEuler-20.03-LTS-SP3/extras/aarch64/Packages/b/boostkit-kml-1.7.0-1.aarch64.rpm"

# 下载源 C：鲲鹏 DevKit 官网（如以上源不可用，从官网获取最新下载地址）
# https://www.hikunpeng.com/zh/developer/devkit

# 也可直接用 yum 安装（如仓库已配置）
yum install -y kml 2>/dev/null || yum install -y boostkit-kml 2>/dev/null
```

> **注意：** `kml-2.5.0` 和 `boostkit-kml-1.7.0` 的库目录结构略有不同。2.5.0 版将 KSVML 分为 `neon/` 和 `sve/` 子目录；1.7.0 版可能需要手动查找。以下步骤以 2.5.0 为准。

#### 1.2 安装 KML

```bash
# 方式 A：rpm 安装（root 权限，推荐）
rpm -ivh /tmp/kml-2.5.0-1.aarch64.rpm
# 库文件默认在 /usr/local/kml/lib/{neon,sve,noarch}/
# 头文件在 /usr/local/kml/include/

# 方式 B：解压到项目内（无 root 权限，或 Bazel 等沙箱构建需要库在 execroot 内）
mkdir -p third_party/kml
rpm2cpio /tmp/kml-2.5.0-1.aarch64.rpm | cpio -idmv --no-absolute-filenames -D third_party/kml
# 库文件在 third_party/kml/usr/local/kml/lib/{neon,sve,noarch}/

# 方式 C：yum 直接安装（如官方源可用）
yum install -y kml
```

#### 1.3 验证安装

```bash
# 确认库文件存在
ls /usr/local/kml/lib/neon/libksvml.so    # KSVML（NEON 版）
ls /usr/local/kml/lib/sve/libksvml.so     # KSVML（SVE 版）
ls /usr/local/kml/lib/noarch/libkm.so     # KM（基础数学）
ls /usr/local/kml/lib/neon/libkblas.so    # KBLAS（NEON 版）
ls /usr/local/kml/lib/noarch/libkblas.so  # KBLAS（可能在 noarch 或 neon 下）
ls /usr/local/kml/lib/neon/libkfft.so     # KFFT
ls /usr/local/kml/lib/neon/libklapack.so  # KLAPACK

# 确认头文件存在
ls /usr/local/kml/include/ksvml.h         # KSVML 头文件
ls /usr/local/kml/include/km.h            # KM 头文件
ls /usr/local/kml/include/kfft.h          # KFFT 头文件
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

#### 2.3 定位数学热点函数

**方案 A（首选）：perf 动态采集**

如果系统有 perf，优先使用 perf 采集运行时热点，精确度最高。

```bash
# 尝试安装 perf
which perf 2>/dev/null || yum install -y perf 2>/dev/null

# 如果 perf 可用，采集热点（60 秒）
if which perf >/dev/null 2>&1; then
    perf record -F 99 -g -p <pid> -o /tmp/perf.data -- sleep 60
    perf report -i /tmp/perf.data --stdio --no-children | grep -E '^\s+[0-9]' | head -20
fi
```

**判断标准：**
- 热点函数名含 `sgemm`/`dgemm`/`gemm`/`matmul`/`contract` → **KBLAS 替换候选**
- 热点函数名含 `exp`/`log`/`sin`/`cos`/`tan`/`pow`/`sqrt` → **KSVML 替换候选**
- 热点函数名含 `fft`/`dft`/`rfft`/`cfft` → **KFFT 替换候选**
- 热点函数名含 `solve`/`factorize`/`inverse`/`ev`/`svd`/`qr` → **KLAPACK 替换候选**

**方案 B（回退）：源码静态分析**

如果 perf 无法安装（无 root 权限、内核版本不匹配、容器环境等），通过源码 grep 直接定位数学函数调用点。此方案不需要运行程序，适合项目尚未编译或无法运行的情况。

```bash
# 1. 搜索 BLAS 调用（KBLAS 替换候选）
grep -rn 'cblas_sgemm\|cblas_dgemm\|cblas_sgemv\|cblas_dgemv\|sgemm_\|dgemm_' <src_dir>/ --include='*.cpp' --include='*.cc' --include='*.c' --include='*.h'
grep -rn 'Eigen::Matrix.*\*.*Matrix\|\.noalias()\|\.transpose()' <src_dir>/ --include='*.cpp'  # Eigen GEMM
grep -rn 'cublasSgemm\|cublasDgemm' <src_dir>/ --include='*.cu'  # CUDA BLAS（如适用）

# 2. 搜索向量数学函数调用（KSVML 替换候选）
grep -rn 'Sleef_\|sleef\|_expf\|_logf\|_sinf\|_cosf\|expf\|logf\|log10f\|sinf\|cosf\|powf\|sqrtf\|tanhf' <src_dir>/ --include='*.cpp' --include='*.c'
grep -rn 'Sleef_expd2\|Sleef_log10d2\|Sleef_sind2\|Sleef_cisd2' <src_dir>/  # SLEEF 向量化
grep -rn '__exp\|__log\|_mm_exp\|_mm256_exp\|ivml_exp\|vdExp\|vdLog' <src_dir>/  # 其他数学库

# 3. 搜索 FFT 调用（KFFT 替换候选）
grep -rn 'fftw_\|FFTW_\|cufft\|DftiCompute\|ne10_fft\|kiss_fft' <src_dir>/ --include='*.cpp' --include='*.c'

# 4. 搜索 LAPACK 调用（KLAPACK 替换候选）
grep -rn 'LAPACKE_\|lapacke_\|dgesv_\|sgesv_\|dpotrf_\|spotrf_\|dsyev_\|ssyev_' <src_dir>/ --include='*.cpp' --include='*.c'

# 5. 搜索 include 依赖
grep -rn '#include.*sleef\|#include.*fftw\|#include.*blas\|#include.*lapack\|#include.*mkl\|#include.*cblas' <src_dir>/ --include='*.h' --include='*.cpp'

# 6. 搜索链接参数
grep -rn '\-lSLEEF\|\-lsleef\|\-lblas\|\-lopenblas\|\-latlas\|\-lmkl\|\-lfftw\|\-llapack' <build_dir>/ 2>/dev/null
grep -rn 'find_package.*BLAS\|find_package.*FFTW\|find_package.*LAPACK\|find_package.*SLEEF' <src_dir>/ 2>/dev/null
```

> **自动决策：** 无论 perf 是否可用，都应额外执行方案 B 的源码 grep，因为 perf 只能看到运行时热点，可能遗漏未执行到的代码路径。两者结合覆盖最全。

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
