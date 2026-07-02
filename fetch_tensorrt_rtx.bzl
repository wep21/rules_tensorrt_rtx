"""Module for fetching remote or local TensorRT RTX libraries using Bazel http_archive."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_DEFAULT_VERSION = "1.5.0.114"

# (version, cuda) -> sha256
_WINDOWS_SHA256 = {
    ("1.2.0.54", "12.9"): "1ea06d3a3725ef0a0607331388a99c1c17235eb76497857864a01bf0aa48ab40",
    ("1.2.0.54", "13.0"): "a0310b839e247e2c64b1614765f22f6d44a35934fa1baaf6dbde9edcfa1c05e2",
    ("1.3.0.35", "12.9"): "c0359fd7e246f76b7bfe6bf2e647fe61cb5c46f63a58c9937b623f9e2f2fbb86",
    ("1.3.0.35", "13.1"): "e64fb9c795bc7e448ef4e691d29c6eab72b7670389f89056ce94fd2b4c642e48",
    ("1.4.0.76", "12.9"): "a40bef8b2489cc987824c4c6657ec1ced6b823cabaecb78f8a53b36d769500d0",
    ("1.4.0.76", "13.2"): "0a050b10158bbe286c90b55b23dffbd3d5096c626b2ee45eccf51322795a3c29",
    ("1.5.0.114", "12.9"): "71482524e842e2826397b8ebef5a215737d3a4235f0a071522ab03709913fac3",
    ("1.5.0.114", "13.2"): "2990bf3ee42e3617e432f9afc3ae8b68d27da5103cfa5e6e8c05250e1609f1a0",
}

_LINUX_SHA256 = {
    ("1.2.0.54", "12.9"): "7917b39f5145b5dad287ad8b7f9dc4b562b685a2fa47269ebfdcfc607067f1dd",
    ("1.2.0.54", "13.0"): "a8bb8f71168c4891a618adbd7b9f8033fd3a64ea3b0e8bf26c19f472136e0cf5",
    ("1.3.0.35", "12.9"): "c653af575ee51c2d1cb23b7c54cc0dad0e12f6a7c4f3ea50e7ea80caacb7b9df",
    ("1.3.0.35", "13.1"): "d798d202f5c2cda59f45a025041c44862566ce0c042a4b0883496dc2439ab619",
    ("1.4.0.76", "12.9"): "61f66b7560d308e9eeb15a53ab629266a1628895d8a58cd7ae7863ae2d9ec4fb",
    ("1.4.0.76", "13.2"): "40de70b9b7583c9ebfc25d3399eb6986917ef0fefdde3fef67b75383a630db9c",
    ("1.5.0.114", "12.9"): "021fc649d877bf25be9d7bff72f19a326fd175d7fc5c9d6a944f88c225f8ccec",
    ("1.5.0.114", "13.2"): "2e040a7390e34ab7c57e66c4d01684d0c97f413a88562f705f210a3fd8118967",
}

_WIN10_OS_TAG_VERSIONS = ("1.2.0.54", "1.3.0.35")
_TAR_GZ_VERSIONS = ("1.2.0.54", "1.3.0.35", "1.4.0.76")

def _archive_name(version, cuda_version, platform):
    windows_os_tag = "win10" if version in _WIN10_OS_TAG_VERSIONS else "windows"
    linux_ext = "tar.gz" if version in _TAR_GZ_VERSIONS else "tar.zst"
    if platform == "windows":
        return "tensorrt-rtx-{version}-{os_tag}-amd64-cuda-{cuda}-release-external.zip".format(
            version = version,
            os_tag = windows_os_tag,
            cuda = cuda_version,
        )
    if platform == "linux":
        return "tensorrt-rtx-{version}-linux-x86_64-cuda-{cuda}-release-external.{ext}".format(
            version = version,
            cuda = cuda_version,
            ext = linux_ext,
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
        cuda_version: CUDA upper bound (e.g. "12.9", "13.2").
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
