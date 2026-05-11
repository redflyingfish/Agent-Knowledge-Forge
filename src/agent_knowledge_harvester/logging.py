import logging
from pathlib import Path

from rich.logging import RichHandler


def configure_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, markup=True),
    ]

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "harvester.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
