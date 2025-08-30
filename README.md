# rules_tensorrt_rtx 

## Usage

```
# MODULE.bazel
bazel_dep(name = "rules_tensorrt_rtx", version = "0.1.0")
tensorrt_rtx = use_extension("@rules_tensorrt_rtx//:extensions.bzl", "tensorrt_rtx")
tensorrt_rtx.fetch()
use_repo(
    tensorrt_rtx,
    "tensorrt_rtx_linux_x86_64",
    "tensorrt_rtx_windows_x86_64",
)

# BUILD.bazel
cc_binary(
    name = "tensorrt_rtx_main",
    srcs = ["main.cpp"],
    deps = [
        "@rules_tensorrt_rtx//:tensorrt_rtx",
        "@rules_tensorrt_rtx//:tensorrt_onnxparser_rtx",
    ],
)
```
