"""评测数据集加载。支持 JSON 文件：直接数组或 {"samples": [...]}。"""

from __future__ import annotations

import json
from pathlib import Path

from raglab.schemas import EvalSample


def load_dataset(path: str | Path) -> list[EvalSample]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("samples", [])
    return [EvalSample.model_validate(item) for item in data]
