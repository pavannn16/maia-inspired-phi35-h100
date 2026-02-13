from __future__ import annotations

import dataclasses
import json
import os
import platform
import time
from typing import Any, Dict, Optional

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None


def now_ms() -> float:
    return time.time() * 1000.0


def atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    if orjson is not None:
        line = orjson.dumps(record) + b"\n"
    else:
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "ab") as f:
        f.write(line)


def get_env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    try:
        import torch

        snap.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            }
        )
        if torch.cuda.is_available():
            snap.update(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_cc": torch.cuda.get_device_capability(0),
                    "gpu_mem_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                }
            )
    except Exception as e:
        snap["torch_error"] = str(e)

    for k in ["COLAB_RELEASE_TAG", "NVIDIA_VISIBLE_DEVICES"]:
        if k in os.environ:
            snap[k] = os.environ[k]

    return snap


def maybe_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None
