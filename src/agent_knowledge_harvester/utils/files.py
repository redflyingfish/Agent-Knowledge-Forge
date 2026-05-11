import hashlib
import json
import re
from pathlib import Path
from typing import Any


def stable_slug(value: str, max_len: int = 80) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    normalized = normalized[:max_len].strip("-") or "document"
    return f"{normalized}-{digest}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
