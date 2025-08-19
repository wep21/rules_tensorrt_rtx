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
    name = "tensorrt_rtx_dll",
    interface_library = "lib/tensorrt_rtx_1_1.lib",
    shared_library = "lib/tensorrt_rtx_1_1.dll",
    visibility = ["//visibility:private"],
    target_compatible_with = ["@platforms//os:windows"],
)

cc_library(
    name = "tensorrt_rtx",
    visibility = ["//visibility:public"],
    deps = [
        "tensorrt_rtx_headers",
        "tensorrt_rtx_dll",
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
    name = "tensorrt_onnxparser_rtx_dll",
    interface_library = "lib/tensorrt_onnxparser_rtx_1_1.lib",
    shared_library = "lib/tensorrt_onnxparser_rtx_1_1.dll",
    visibility = ["//visibility:private"],
    target_compatible_with = ["@platforms//os:windows"],
)

cc_library(
    name = "tensorrt_onnxparser_rtx",
    visibility = ["//visibility:public"],
    deps = [
        "tensorrt_onnxparser_rtx_headers",
        "tensorrt_onnxparser_rtx_dll",
    ],
)

alias(
    name = "tensorrt_rtx_bin",
    actual = "//:bin/tensorrt_rtx.exe",
    visibility = ["//visibility:public"],
)
