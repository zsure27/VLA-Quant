"""Read tensor shapes from local torch ZIP states without executing pickle globals.

This is a metadata audit, not a replacement for torch weights_only reload/finite checks.
"""
from __future__ import annotations
import io
import pickle
import zipfile
from collections import OrderedDict


def rebuild(storage, offset, size, stride, *rest):
    return {"shape": tuple(size), "stride": tuple(stride), "storage": storage, "offset": offset}


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            return rebuild
        if module == "torch" and name in ("BFloat16Storage", "FloatStorage", "HalfStorage"):
            return name
        raise ValueError(f"Unapproved metadata global: {module}.{name}")

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage":
            raise ValueError("Unsupported storage identifier")
        return {"dtype": pid[1], "key": pid[2], "numel": pid[4]}


def tensor_shapes(raw: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(members) != 1:
            raise ValueError("Expected one torch state metadata record")
        state = MetadataUnpickler(io.BytesIO(archive.read(members[0]))).load()
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty state mapping")
    return {k: tuple(v["shape"]) for k, v in state.items()}
