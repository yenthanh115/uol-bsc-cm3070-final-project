"""Append-only experiment log (JSONL format).

Each pipeline run appends a single JSON record to output/experiment_log.jsonl.
This provides a consolidated history of all experiments without needing to
traverse individual output files.

The JSONL format is:
- Append-safe (no risk of corrupting existing data on crash)
- Git-friendly (one new line per run, clean diffs)
- Queryable via pandas: pd.read_json("experiment_log.jsonl", lines=True)
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "output" / "experiment_log.jsonl"
)


def get_git_sha() -> str | None:
    """Get current git short SHA, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def append_experiment(
    run_id: str,
    pipeline: str,
    config: dict[str, Any],
    outputs: list[str],
    summary: dict[str, Any],
    notes: str = "",
    log_path: Path | None = None,
) -> Path:
    """Append one experiment record to the JSONL log.

    Parameters
    ----------
    run_id : str
        Timestamp prefix that links to output files (e.g. "202607111714").
    pipeline : str
        Pipeline stage identifier ("labelling", "training", "sweep").
    config : dict
        Key hyperparameters used in this run.
    outputs : list[str]
        List of generated output file paths/names.
    summary : dict
        Key metrics for quick scanning without opening output files.
    notes : str, optional
        Free-text annotation for the run (e.g. "testing tau=2.0").
    log_path : Path, optional
        Override the default log file location.

    Returns
    -------
    Path
        The path to the log file that was written to.
    """
    log_path = log_path or DEFAULT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "run_id": run_id,
        "pipeline": pipeline,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": get_git_sha(),
        "config": config,
        "outputs": outputs,
        "summary": summary,
    }
    if notes:
        entry["notes"] = notes

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return log_path
