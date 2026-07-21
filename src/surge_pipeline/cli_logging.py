"""Tee-style CLI logging: dual output to console and log file.

Provides a context manager that redirects both stdout/stderr and
Python logging output to a file while preserving console display.

Usage:
    from surge_pipeline.cli_logging import tee_output, resolve_log_path

    log_path = resolve_log_path(args.log_file, pipeline="labelling")
    with tee_output(log_path):
        # All print() and logging output goes to both console and file
        print("Hello")
        logger.info("world")
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TextIO

# Default log directory relative to project root (../output/logs from src/)
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "logs"


class TeeStream:
    """A stream that writes to both a file and the original stream (console)."""

    def __init__(self, original: TextIO, log_file: TextIO) -> None:
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")


def resolve_log_path(
    log_file_arg: str | None,
    pipeline: str = "run",
    log_dir: Path | None = None,
) -> Path | None:
    """Resolve the log file path from CLI argument.

    Args:
        log_file_arg: The --log-file argument value. Can be:
            - None or empty string: no logging (returns None)
            - "auto": auto-generate a timestamped filename
            - A path: use as-is (creates parent dirs if needed)
        pipeline: Pipeline name used in auto-generated filenames
            (e.g. "labelling", "training", "sweep").
        log_dir: Directory for auto-generated log files.
            Defaults to output/logs/.

    Returns:
        Resolved Path to log file, or None if logging is disabled.
    """
    if not log_file_arg:
        return None

    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    if log_file_arg == "auto":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{pipeline}.log"
        path = log_dir / filename
    else:
        path = Path(log_file_arg)
        # If only a filename (no directory), put it in the default log dir
        if not path.parent.exists() and path.parent == Path("."):
            path = log_dir / path

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def tee_output(log_path: Path | None) -> Generator[Path | None, None, None]:
    """Context manager that tees stdout/stderr to a log file.

    If log_path is None, this is a no-op (passes through unchanged).

    Args:
        log_path: Path to the log file, or None to disable.

    Yields:
        The log_path (for reference by the caller).
    """
    if log_path is None:
        yield None
        return

    # Open log file
    log_file = open(log_path, "w", encoding="utf-8")

    # Write header
    log_file.write(f"# Log started: {datetime.now().isoformat()}\n")
    log_file.write(f"# Command: {' '.join(sys.argv)}\n")
    log_file.write(f"# Python: {sys.version.split()[0]}\n")
    log_file.write("#" + "=" * 59 + "\n\n")
    log_file.flush()

    # Replace stdout/stderr with tee streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = TeeStream(original_stdout, log_file)
    tee_stderr = TeeStream(original_stderr, log_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    # Also add a file handler to the root logger so that logging.info() etc.
    # are captured even if they bypass print().
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        yield log_path
    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Remove our file handler
        root_logger.removeHandler(file_handler)
        file_handler.close()

        # Write footer and close
        log_file.write(f"\n# Log ended: {datetime.now().isoformat()}\n")
        log_file.close()
