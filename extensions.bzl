"""Module extension for rules_tensorrt_rtx to fetch and register TensorRT RTX dependencies."""

load("@rules_tensorrt_rtx//:fetch_tensorrt_rtx.bzl", "fetch_tensorrt_rtx")

def _tensorrt_rtx_impl(ctx):
    for mod in ctx.modules:
        if mod.name != "rules_tensorrt_rtx":
            continue
        for tag in mod.tags.fetch:
            fetch_tensorrt_rtx(
                version = tag.version,
                cuda_version = tag.cuda_version,
            )

_fetch = tag_class(attrs = {
    "version": attr.string(default = "1.2.0.54"),
    "cuda_version": attr.string(default = "12.9"),
})

tensorrt_rtx = module_extension(
    implementation = _tensorrt_rtx_impl,
    tag_classes = {
        "fetch": _fetch,
    },
)
