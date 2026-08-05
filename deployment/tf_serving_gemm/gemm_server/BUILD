load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//tensorflow:tensorflow.default.bzl",
    "tf_grpc_cc_dependencies",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(default_visibility = ["//visibility:public"])

# ── Proto + gRPC stubs ────────────────────────────────────────────────────────
#
# tf_proto_library(name = "gemm_proto") generates two targets:
#   :gemm_proto     — proto_library
#   :gemm_proto_cc  — cc_proto_library (gemm.pb.h / gemm.pb.cc)
#
# cc_grpc_library(grpc_only = True) adds just the gRPC stubs on top:
#   :gemm_grpc      — gemm.grpc.pb.h / gemm.grpc.pb.cc

tf_proto_library(
    name = "gemm_proto",
    srcs = ["proto/gemm.proto"],
    has_services = 1,
    create_java_proto = False,
    create_kotlin_proto = False,
)

cc_grpc_library(
    name = "gemm_grpc",
    srcs = [":gemm_proto"],
    grpc_only = True,
    deps = [":gemm_proto_cc"],
)

# ── Server (TF C++ Session + MatMul op kernel) ────────────────────────────────

cc_binary(
    name = "gemm_server",
    srcs = ["server.cc"],
    deps = [
        ":gemm_grpc",
        ":gemm_proto_cc",
        "//tensorflow/cc:cc_ops",        # generated standard_ops.h (MatMul, Placeholder)
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",    # TF session / executor / op kernels
        "//tensorflow/core:framework",
    ] + tf_grpc_cc_dependencies(),       # expands to ["//tensorflow:grpc++"]
)

# ── Client (gRPC only, no TF runtime dep) ────────────────────────────────────

cc_binary(
    name = "gemm_client",
    srcs = ["client.cc"],
    deps = [
        ":gemm_grpc",
        ":gemm_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)
