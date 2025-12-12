"""Module for fetching remote or local TensorRT RTX libraries using Bazel http_archive."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_DEFAULT_VERSION = "1.2.0.54"

# (version, cuda) -> sha256
_WINDOWS_SHA256 = {
    ("1.2.0.54", "12.9"): "1ea06d3a3725ef0a0607331388a99c1c17235eb76497857864a01bf0aa48ab40",
    ("1.2.0.54", "13.0"): "a0310b839e247e2c64b1614765f22f6d44a35934fa1baaf6dbde9edcfa1c05e2",
}

_LINUX_SHA256 = {
    ("1.2.0.54", "12.9"): "7917b39f5145b5dad287ad8b7f9dc4b562b685a2fa47269ebfdcfc607067f1dd",
    ("1.2.0.54", "13.0"): "a8bb8f71168c4891a618adbd7b9f8033fd3a64ea3b0e8bf26c19f472136e0cf5",
}

def _archive_name(version, cuda_version, platform):
    if platform == "windows":
        return "tensorrt-rtx-{version}-win10-amd64-cuda-{cuda}-release-external.zip".format(
            version = version,
            cuda = cuda_version,
        )
    if platform == "linux":
        return "tensorrt-rtx-{version}-linux-x86_64-cuda-{cuda}-release-external.tar.gz".format(
            version = version,
            cuda = cuda_version,
        )
    fail("Unsupported platform '{}' for TensorRT RTX archive".format(platform))

def _sha256(version, cuda_version, platform):
    key = (version, cuda_version)
    if platform == "windows":
        if key not in _WINDOWS_SHA256:
            fail("Unsupported TensorRT RTX version/cuda combo {} for windows".format(key))
        return _WINDOWS_SHA256[key]
    if platform == "linux":
        if key not in _LINUX_SHA256:
            fail("Unsupported TensorRT RTX version/cuda combo {} for linux".format(key))
        return _LINUX_SHA256[key]
    fail("Unsupported platform '{}' for TensorRT RTX archive".format(platform))

def _base_url(version):
    major_minor = ".".join(version.split(".")[0:2])
    return "https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/{}".format(major_minor)

def fetch_tensorrt_rtx(version = _DEFAULT_VERSION, cuda_version = "12.9"):
    """Fetch prebuilt TensorRT RTX libraries.

    Args:
        version: TensorRT RTX version string.
        cuda_version: CUDA upper bound (e.g. "12.9", "13.0").
    """
    strip_prefix = "TensorRT-RTX-{}".format(version)

    base_url = _base_url(version)

    http_archive(
        name = "tensorrt_rtx_windows_x86_64",
        urls = ["{}/{}".format(base_url, _archive_name(version, cuda_version, "windows"))],
        sha256 = _sha256(version, cuda_version, "windows"),
        strip_prefix = strip_prefix,
        build_file = "@rules_tensorrt_rtx//:tensorrt_rtx_windows_x86_64.BUILD",
    )

    http_archive(
        name = "tensorrt_rtx_linux_x86_64",
        urls = ["{}/{}".format(base_url, _archive_name(version, cuda_version, "linux"))],
        sha256 = _sha256(version, cuda_version, "linux"),
        strip_prefix = strip_prefix,
        build_file = "@rules_tensorrt_rtx//:tensorrt_rtx_linux_x86_64.BUILD",
    )
