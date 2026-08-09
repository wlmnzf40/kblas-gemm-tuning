# 实际遇到的坑与解决方案

> 本文档保留原 TF-Serving KBLAS 替换 SKILL 中的 pitfalls 内容。
> 通用 KML 替换的常见问题参见 [SKILL.md](../SKILL.md) 的常见问题速查表。

## 坑 1：Bazel 沙箱拒绝外部绝对路径

**错误信息：**
```
The include path '<外部 KML 目录>/include' references a path outside of the execution root.
```

**原因：** Bazel 沙箱机制不允许编译时引用 execroot 外的绝对路径。当 KML 安装在仓库目录外时，所有 `-I`、`-L`、`-rpath` 都会被拒绝。

**解决：** 将 KML 复制到项目目录内：
```bash
cp -r <你的 KML 安装目录> $REPO/third_party/kml
# 然后重跑 setup_kblas.sh
```

`.bazelrc` 中应使用项目内路径或绝对路径（setup 脚本会自动用 `$(pwd)/third_party/...` 写入）。rpath 使用相对路径会导致运行时 `ldd` 报 `not found`，但通过 `LD_LIBRARY_PATH` 可以解决。

---

## 坑 2：tensorflow.patch 过时导致 Bazel fetch 失败

**错误信息：**
```
Error applying patch third_party/tensorflow/tensorflow.patch:
Content does not match target
```

**原因：** `third_party/tensorflow/tensorflow.patch` 中的代码修改（如 `tensor_testutil.cc`）与实际 TF 源码不匹配，导致 Bazel 无法正确应用 patch，org_tensorflow external 解压失败。

**解决：** **不要手动修改 WORKSPACE 或 tensorflow.patch！** 只运行 `scripts/setup_kblas.sh`，它会就地修改 Bazel 缓存中的文件，不触发 re-fetch。

---

## 坑 3：`tf_serving_vendored` 函数缺失

**错误信息：**
```
file does not contain symbol 'tf_serving_vendored'
```

**原因：** WORKSPACE 中引用了 `tf_serving_vendored`，但 `tensorflow_serving/repo.bzl` 中没有定义这个函数。

**解决：** `setup_kblas.sh` 会自动检测并追加。手动修复：
```bash
cat >> tensorflow_serving/repo.bzl << 'REPO_EOF'

def _tf_serving_vendored_impl(ctx):
    ctx.symlink(ctx.path(ctx.attr.root).dirname.get_child(ctx.attr.path), ".")

tf_serving_vendored = repository_rule(
    implementation = _tf_serving_vendored_impl,
    attrs = {
        "root": attr.label(mandatory = True),
        "path": attr.string(mandatory = True),
    },
)
REPO_EOF
```

---

## 坑 4：运行时 `libkblas.so: cannot open shared object file`

**错误信息：**
```
error while loading shared libraries: libkblas_armv8p_v1.7.0.so: cannot open shared object file
```

**原因：** KML 在非标准路径（如 `third_party/kml/...`），动态链接器找不到。

**解决：** 必须设置 `LD_LIBRARY_PATH`：
```bash
export LD_LIBRARY_PATH=$KML_LIB:$LD_LIBRARY_PATH
# 其中 KML_LIB 是 libkblas.so 所在目录，例如 $REPO/third_party/kml/lib/kblas/omp
```

`compare_backends.sh` 会自动设置这个环境变量，手动启动 server 时需自行设置。

---

## 坑 5：KML 库的符号链接指向绝对路径

**原因：** rpm2cpio 解压后，`libkblas.so` 的符号链接可能指向原始的绝对路径（如 `/usr/local/kml/lib/kblas/omp/libkblas.so`），在项目目录内无法解析。

**解决：** 修复符号链接：
```bash
cd $REPO/third_party/kml/usr/local/kml/lib
rm libkblas.so
ln -s kblas/omp/libkblas.so libkblas.so
```

---

## 坑 6：Bazel 版本不匹配

**错误信息：** WORKSPACE 要求某个 Bazel 版本，但系统装的是另一个版本

**解决：** 确保使用 WORKSPACE 要求的 Bazel 版本（通常为 7.4.1）：
```bash
export BAZEL=<bazel 可执行文件绝对路径>
$BAZEL --version  # 确认版本
```

---

## 坑 7：头文件 patch 后 Bazel cache 被重新解压

**原因：** 如果修改了 WORKSPACE 中的 `tensorflow_http_archive` 规则（如添加了新的 patch 文件），Bazel 会重新解压 org_tensorflow，覆盖我们的就地 patch。

**解决：** **绝不修改 WORKSPACE！** 所有 patch 都通过 `apply_kblas_patch.sh` 在 Bazel 缓存中就地修改。如果 cache 被重新解压，只需重跑 `bash scripts/setup_kblas.sh $BAZEL`。

---

## 坑 8：两个 backend 的 GFLOPS 几乎一样

**原因：** 头文件 patch 没真正生效（`tf_serving_kblas_enabled` 开关没编译进内核）。

**解决：**
1. 重跑 `bash scripts/apply_kblas_patch.sh $BAZEL`
2. 确认输出中没有 `WARNING: dnnl_sgemm found but regex didn't match`
3. 重新编译 `bash scripts/build_backends.sh $BAZEL`
4. 用 `nm -D bazel-bin/.../gemm_server | grep cblas_sgemm` 确认符号存在
