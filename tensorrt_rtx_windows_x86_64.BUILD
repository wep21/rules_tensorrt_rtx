load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

filegroup(
    name = "tensorrt_rtx_interface_library",
    srcs = glob(["lib/tensorrt_rtx_*.lib"]),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "tensorrt_rtx_shared_library",
    srcs = glob(["bin/tensorrt_rtx_*.dll"]),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "tensorrt_rtx_headers",
    hdrs = glob([
        "include/NvInfer*",
    ]),
    includes = ["include"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "tensorrt_rtx_dll",
    interface_library = ":tensorrt_rtx_interface_library",
    shared_library = ":tensorrt_rtx_shared_library",
    target_compatible_with = ["@platforms//os:windows"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "tensorrt_rtx",
    visibility = ["//visibility:public"],
    deps = [
        "tensorrt_rtx_dll",
        "tensorrt_rtx_headers",
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

filegroup(
    name = "tensorrt_onnxparser_rtx_interface_library",
    srcs = glob(["lib/tensorrt_onnxparser_rtx_*.lib"]),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "tensorrt_onnxparser_rtx_shared_library",
    srcs = glob(["bin/tensorrt_onnxparser_rtx_*.dll"]),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "tensorrt_onnxparser_rtx_dll",
    interface_library = ":tensorrt_onnxparser_rtx_interface_library",
    shared_library = ":tensorrt_onnxparser_rtx_shared_library",
    target_compatible_with = ["@platforms//os:windows"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "tensorrt_onnxparser_rtx",
    visibility = ["//visibility:public"],
    deps = [
        "tensorrt_onnxparser_rtx_dll",
        "tensorrt_onnxparser_rtx_headers",
    ],
)

alias(
    name = "tensorrt_rtx_bin",
    actual = "//:bin/tensorrt_rtx.exe",
    visibility = ["//visibility:public"],
)
