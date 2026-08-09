# KML Library Replacement Skill

通用 KML（鲲鹏数学库）替换技能 + TF-Serving KBLAS 疑难案例

## 分支说明

- `main` — 原 TF-Serving KBLAS 专用 SKILL
- `feature/general-kml-replacement` — 通用 KML 替换 SKILL（本分支）

## 文件结构

```
SKILL.md                                    # 通用 KML 替换 SKILL（主入口）
references/
  kml-replacement-practice.md               # 通用替换实践案例（SLEEF→KSVML 等）
  tf-serving-kblas-case-study.md            # TF-Serving 疑难案例（保留原内容）
  patch-internals.md                        # TF-Serving patch 技术细节（保留原内容）
  pitfalls.md                               # TF-Serving 坑与解决方案（保留原内容）
  performance-baseline.md                   # 性能基线（main 分支保留）
scripts/
  setup_kblas.sh                            # TF-Serving 专用（main 分支保留）
  apply_kblas_patch.sh                      # TF-Serving 专用（main 分支保留）
  build_backends.sh                         # TF-Serving 专用（main 分支保留）
  compare_backends.sh                       # TF-Serving 专用（main 分支保留）
assets/
  bazelrc.kml_example                       # TF-Serving 专用（main 分支保留）
deployment/
  general/                                  # 通用部署示例目录
```

## SKILL 描述

本 SKILL 提供两种能力：

1. **通用 KML 替换**（SKILL.md）：给定任意 C/C++ 项目，分析数学函数热点，选择正确的 KML 子库（KBLAS/KSVML/KFFT/KLAPACK），指导源码级或链接级替换。

2. **TF-Serving KBLAS 疑难案例**（references/tf-serving-kblas-case-study.md）：保留原 TF-Serving 中 Eigen GEMM → cblas_sgemm 替换的完整技术细节，作为 Bazel 构建系统 + 深度嵌入源码替换的参考案例。
