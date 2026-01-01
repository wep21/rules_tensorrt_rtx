load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

cc_library(
    name = "tensorrt_rtx_headers",
    hdrs = glob([
        "include/NvInfer*",
    ]),
    includes = ["include"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "tensorrt_rtx_shared",
    shared_library = "lib/libtensorrt_rtx.so.1",
    target_compatible_with = ["@platforms//os:linux"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "tensorrt_rtx",
    visibility = ["//visibility:public"],
    deps = [
        "tensorrt_rtx_headers",
        "tensorrt_rtx_shared",
        "@rules_cuda//cuda:runtime",
    ],
)

cc_library(
    name = "tensorrt_onnxparser_rtx_headers",
    hdrs = glob([
        "include/NvOnnx*",
    ]),
    includes = ["include"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "tensorrt_onnxparser_rtx_shared",
    shared_library = "lib/libtensorrt_onnxparser_rtx.so.1",
    target_compatible_with = ["@platforms//os:linux"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "tensorrt_onnxparser_rtx",
    visibility = ["//visibility:public"],
    deps = [
        "tensorrt_onnxparser_rtx_headers",
        "tensorrt_onnxparser_rtx_shared",
    ],
)

alias(
    name = "tensorrt_rtx_bin",
    actual = "//:bin/tensorrt_rtx",
    visibility = ["//visibility:public"],
)
