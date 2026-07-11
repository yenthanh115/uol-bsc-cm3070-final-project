"""Pipeline configuration dataclass with JSON serialisation support."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineConfig:
    """Configuration for the surge-labelling pipeline.

    All random operations use `random_seed` for reproducibility (R7-AC1).
    Configuration is serialisable to JSON for audit trail (R7-AC3).
    """

    # --- Input / Output ---
    file_path: str = ""
    output_dir: str = "output/processed"

    # --- Temporal split ---
    temporal_split_ratio: float = 0.8

    # --- Windowing ---
    # Surge method controls how the surge metric is computed:
    #   "forward_growth" (default) — uses forward/backward ratio,
    #       requires future posts to exist (excludes end-of-timeline).
    #   "backward_only" — defines surge as backward_count exceeding
    #       the ticker's historical average. No forward-looking data
    #       needed, making every record with sufficient backward
    #       history usable.
    surge_method: str = "forward_growth"

    # Minimum window count for exclusion filtering:
    #   - forward_growth mode: excludes records with
    #       forward_count < min_window_count.
    #   - backward_only mode: excludes records with
    #       backward_count < min_window_count (insufficient history).
    #
    # Default is 1 (minimal filter). Increase for stricter quality.
    min_window_count: int = 1

    # --- Threshold ---
    threshold_tau: float = 1.5

    # --- Sentiment model ---
    sentiment_model: str = "vader"  # "vader" or "textblob"

    # --- Composite score weights ---
    weight_volume: float = 0.5
    weight_sentiment: float = 0.5

    # --- Threshold sweep (batch mode) ---
    thresholds: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5]
    )

    # --- Reproducibility ---
    random_seed: int = 42

    # ------------------------------------------------------------------
    # Serialisation helpers (R7-AC3)
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise configuration to a JSON string."""
        return json.dumps(asdict(self), indent=indent)

    def save_json(self, path: Optional[str] = None) -> Path:
        """Write configuration to a JSON file for audit trail."""
        out = Path(path) if path else Path(self.output_dir) / "pipeline_config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def from_json(cls, json_str: str) -> "PipelineConfig":
        """Deserialise configuration from a JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def load_json(cls, path: str) -> "PipelineConfig":
        """Load configuration from a JSON file."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_json(text)
