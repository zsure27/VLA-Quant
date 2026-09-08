"""拒绝悄悄更换 OFT 的双向 attention 实现。"""
import hashlib
import inspect


def assert_oft_runtime():
    from transformers.models.llama import modeling_llama
    source = inspect.getsourcefile(modeling_llama)
    from pathlib import Path
    actual = hashlib.sha256(Path(source).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != "3aac24cec583a6ef5f60b6ec634a8bd3c8377784c5c63c8cc14cb6790554c52e":
        raise RuntimeError("不是审计的 OFT Transformers fork；请先运行 bootstrap / check_runtime")
    return actual
