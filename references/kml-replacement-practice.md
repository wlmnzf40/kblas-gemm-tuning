# 通用 KML 替换实践案例

## 案例 1：SLEEF → KSVML 替换（LiteCall 项目）

### 背景

LiteCall 是一个基因测序数据分析程序，在鲲鹏 920 上运行。原始代码使用 SLEEF 库提供向量化数学函数（exp, log10 等），性能分析发现这些函数占热点的 5-10%。

### 分析

```bash
# perf 热点分析发现
perf report -i perf.data --stdio --no-children | grep -iE 'sleef|exp|log'
# 热点：Sleef_expd2_u10, Sleef_log10d2_u10
```

### 替换过程

**1. 安装 KML：**
```bash
rpm -ivh kml-2.5.0-1.aarch64.rpm
# KSVML 库在 /usr/local/kml/lib/neon/libksvml.so
```

**2. 源码替换（IntensityProcessor_4C.cpp）：**
```cpp
// 原来：
#include "sleef.h"
// 向量化 exp（double × 2）
__m128d result = Sleef_expd2_u10(input);
// 向量化 log10（double × 2）
__m128d result = Sleef_log10d2_u10(input);

// 替换为：
#include "ksvml.h"
// KSVML 向量化 exp（double × 2）
__m128d result = svml128_exp_f64(input);
// KSVML 向量化 log10（double × 2）
__m128d result = svml128_log10_f64(input);
```

**3. 编译参数修改：**
```makefile
# 原来：
CFLAGS += -I$(ENV)/sleef/include
LDFLAGS += -L$(ENV)/sleef/lib -lSLEEF

# 替换为：
CFLAGS += -I/usr/local/kml/include
LDFLAGS += -L/usr/local/kml/lib/neon -L/usr/local/kml/lib/noarch -lksvml -lkm
```

**4. 运行时环境：**
```bash
export LD_LIBRARY_PATH=/usr/local/kml/lib/neon:/usr/local/kml/lib/noarch:$LD_LIBRARY_PATH
```

**5. 验证：**
```bash
# 确认 SLEEF 不再被链接
ldd Basecall.Server | grep -i sleef
# 预期：无输出

# 确认 KSVML 被链接
ldd Basecall.Server | grep -i ksvml
# 预期：libksvml.so => /usr/local/kml/lib/neon/libksvml.so
```

### 结果

- 功能验证：Q30 精度指标完全一致（74.846）
- 性能：KSVML 在鲲鹏 920 上比 SLEEF 略快，IPC 从 0.99 提升到 1.04
- 主要收益：减少了一个第三方依赖（SLEEF），简化了构建

---

## 案例 2：std::min<float> → vminq_f32 NEON 内联优化（ImageProcessor.cpp）

### 背景

ImageProcessor.cpp 中的 `ConvertFloatAndIntegralFused` 函数对每个像素做 `std::min<float>(value, threshold)`，在 2048 宽度的图像行中调用 8 次（每 8 像素一组），函数调用开销大。

### 替换

```cpp
// 原来（4 次函数调用）：
Sum += std::min<float>(vgetq_lane_f32(values, 0), threshold);
Sum += std::min<float>(vgetq_lane_f32(values, 1), threshold);
Sum += std::min<float>(vgetq_lane_f32(values, 2), threshold);
Sum += std::min<float>(vgetq_lane_f32(values, 3), threshold);

// 替换为（1 次向量比较）：
float32x4_t minvals = vminq_f32(values, vdupq_n_f32(threshold));
Sum += vgetq_lane_f32(minvals, 0);
Sum += vgetq_lane_f32(minvals, 1);
Sum += vgetq_lane_f32(minvals, 2);
Sum += vgetq_lane_f32(minvals, 3);
```

### 结果

- 4 次函数调用 → 1 次 NEON 向量比较，减少分支和函数调用开销
- 精度完全一致（vminq_f32 逐 lane 比较，语义与 std::min 相同）

---

## 案例 3：jemalloc 运行时替换（零代码修改）

### 背景

新部署的鲲鹏服务器上，LiteCall 性能从 150s 暴增到 300s。perf 分析显示 83.64% 时间在 CFastQWriterEx::Write，IPC 仅 0.31。

### 分析

```bash
# perf top
83.64%  CFastQWriterEx::Write     # 写 FastQ 文件
 2.99%  ConcurrentQueue::enqueue  # 队列操作
```

根因：系统默认 malloc 在 215 线程并发时产生严重锁争用。

### 替换

```bash
# 安装 jemalloc
yum install -y jemalloc

# 运行时 LD_PRELOAD（零代码修改）
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2
./Basecall.Server
```

### 结果

- IPC 从 0.31 提升到 0.69（+123%）
- perf 热点从 83.64% CFastQWriterEx::Write 变为正常的 31.51% ExtractOne
- 整体性能恢复到接近旧服务器水平

---

## 通用替换决策流程

```
分析目标项目
  │
  ├── 用 ldd 检查链接了哪些数学库
  ├── 用 perf 找到热点函数
  │
  ├── 热点含 BLAS 函数？
  │   └── 是 → KBLAS 替换
  │       ├── 标准 cblas 接口？ → 策略 A (LD_PRELOAD) 或 策略 B (链接替换)
  │       └── 非标准接口？ → 策略 C (源码替换为 cblas_sgemm 等)
  │
  ├── 热点含 exp/log/sin/cos 等数学函数？
  │   └── 是 → KSVML 替换
  │       ├── 用 SLEEF？ → 源码替换 Sleef_* → svml128_*
  │       └── 用 libm？ → 链接替换或源码替换
  │
  ├── 热点含 FFT？
  │   └── 是 → KFFT 替换
  │
  ├── 热点是内存分配（malloc/new）？
  │   └── 是 → jemalloc LD_PRELOAD（不是 KML，但常一起用）
  │
  └── 无明显数学热点？
      └── 考虑其他优化方向（算法、缓存、SIMD）
```
