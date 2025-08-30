"""Module extension for rules_tensorrt_rtx to fetch and register TensorRT RTX dependencies."""

load("@rules_tensorrt_rtx//:fetch_tensorrt_rtx.bzl", "fetch_tensorrt_rtx")

def _tensorrt_rtx_impl(ctx):
    for mod in ctx.modules:
        if mod.name == "rules_tensorrt_rtx":
            fetch_tensorrt_rtx()

_fetch = tag_class(attrs = {})

tensorrt_rtx = module_extension(
    implementation = _tensorrt_rtx_impl,
    tag_classes = {
        "fetch": _fetch,
    },
)
