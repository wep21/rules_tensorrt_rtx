"""Module for fetching remote or local TensorRT RTX libraries using Bazel http_archive."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fetch_tensorrt_rtx():
    """function which fetch remote prebuild tensorrt rtx libraries or use local tensorrt rtx(in macos)
    """

    http_archive(
        name = "tensorrt_rtx_windows_x86_64",
        urls = [
            "https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.1/TensorRT-RTX-1.1.1.26.Windows.win10.cuda-12.9.zip",
        ],
        sha256 = "aee1ee36b320eb72d468f84c111adc0e57d79fee0d0bba7765cb035954b1a68d",
        strip_prefix = "TensorRT-RTX-1.1.1.26",
        build_file = "@rules_tensorrt_rtx//:tensorrt_rtx_windows_x86_64.BUILD",
    )

    http_archive(
        name = "tensorrt_rtx_linux_x86_64",
        urls = [
            "https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.1/TensorRT-RTX-1.1.1.26.Linux.x86_64-gnu.cuda-12.9.tar.gz",
        ],
        sha256 = "6c84e858310b071e80f89d327f33fbb93bd5637765da61c1d2c03751088ab59d",
        strip_prefix = "TensorRT-RTX-1.1.1.26",
        build_file = "@rules_tensorrt_rtx//:tensorrt_rtx_linux_x86_64.BUILD",
    )
