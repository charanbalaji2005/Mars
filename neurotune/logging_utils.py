"""Structured logging: human-readable console output plus JSON-lines log files."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

LOGGER_NAME = "neurotune"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        text = f"[{record.levelname.lower():>7}] {record.getMessage()}"
        fields = getattr(record, "fields", None)
        if fields:
            text += "  " + " ".join(f"{k}={v}" for k, v in fields.items())
        return text


def setup_logging(level: str = "INFO", json_console: bool = False) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    for handler in list(logger.handlers):
        if getattr(handler, "_neurotune_console", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_console else ConsoleFormatter())
    handler._neurotune_console = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def add_file_handler(path: str | Path) -> None:
    """Append JSON-lines logs to `path` (one handler per path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path.resolve():
            return
    handler = logging.FileHandler(path)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
