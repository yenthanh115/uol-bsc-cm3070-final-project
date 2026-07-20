"""Tests for PipelineConfig JSON serialisation roundtrip and edge cases.

Validates:
  1. Roundtrip: to_json() → from_json() preserves all fields (R7-AC3)
  2. File-based roundtrip: save_json() → load_json()
  3. Custom values survive serialisation
  4. Default values are correct
  5. Unknown fields in JSON are rejected (strict schema)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surge_pipeline.config import PipelineConfig, PROJECT_ROOT


# ============================================================================
# Defaults
# ============================================================================


class TestDefaults:
    """Verify default configuration values."""

    def test_default_threshold_tau(self):
        config = PipelineConfig()
        assert config.threshold_tau == 1.5

    def test_default_weights_sum_to_one(self):
        config = PipelineConfig()
        assert config.weight_volume + config.weight_sentiment == 1.0

    def test_default_random_seed(self):
        config = PipelineConfig()
        assert config.random_seed == 42

    def test_default_temporal_split_ratio(self):
        config = PipelineConfig()
        assert config.temporal_split_ratio == 0.8

    def test_default_surge_method(self):
        config = PipelineConfig()
        assert config.surge_method == "forward_growth"

    def test_default_sentiment_model(self):
        config = PipelineConfig()
        assert config.sentiment_model == "vader"

    def test_default_thresholds_list(self):
        config = PipelineConfig()
        assert config.thresholds == [0.5, 1.0, 1.5, 2.0, 2.5]

    def test_default_min_window_count(self):
        config = PipelineConfig()
        assert config.min_window_count == 1


# ============================================================================
# JSON string roundtrip
# ============================================================================


class TestJsonRoundtrip:
    """Verify to_json() → from_json() preserves all fields."""

    def test_roundtrip_with_defaults(self):
        original = PipelineConfig()
        restored = PipelineConfig.from_json(original.to_json())

        assert restored.file_path == original.file_path
        assert restored.threshold_tau == original.threshold_tau
        assert restored.weight_volume == original.weight_volume
        assert restored.weight_sentiment == original.weight_sentiment
        assert restored.temporal_split_ratio == original.temporal_split_ratio
        assert restored.random_seed == original.random_seed
        assert restored.thresholds == original.thresholds
        assert restored.sentiment_model == original.sentiment_model
        assert restored.surge_method == original.surge_method
        assert restored.min_window_count == original.min_window_count

    def test_roundtrip_with_custom_values(self):
        original = PipelineConfig(
            file_path="/some/path/data.csv",
            output_dir="/tmp/output",
            temporal_split_ratio=0.7,
            min_window_count=5,
            threshold_tau=2.5,
            sentiment_model="textblob",
            weight_volume=0.3,
            weight_sentiment=0.7,
            thresholds=[1.0, 2.0, 3.0],
            random_seed=123,
            surge_method="backward_only",
        )
        restored = PipelineConfig.from_json(original.to_json())

        assert restored.file_path == "/some/path/data.csv"
        assert restored.output_dir == "/tmp/output"
        assert restored.temporal_split_ratio == 0.7
        assert restored.min_window_count == 5
        assert restored.threshold_tau == 2.5
        assert restored.sentiment_model == "textblob"
        assert restored.weight_volume == 0.3
        assert restored.weight_sentiment == 0.7
        assert restored.thresholds == [1.0, 2.0, 3.0]
        assert restored.random_seed == 123
        assert restored.surge_method == "backward_only"

    def test_to_json_produces_valid_json(self):
        config = PipelineConfig()
        json_str = config.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "threshold_tau" in parsed

    def test_to_json_indent(self):
        config = PipelineConfig()
        json_str = config.to_json(indent=4)
        # Indented JSON has newlines
        assert "\n" in json_str
        lines = json_str.split("\n")
        # At least one line should start with 4 spaces
        assert any(line.startswith("    ") for line in lines)


# ============================================================================
# File-based roundtrip
# ============================================================================


class TestFileRoundtrip:
    """Verify save_json() → load_json() roundtrip via filesystem."""

    def test_save_and_load(self, tmp_path):
        original = PipelineConfig(
            threshold_tau=3.0,
            weight_volume=0.8,
            weight_sentiment=0.2,
            random_seed=999,
        )
        save_path = str(tmp_path / "test_config.json")
        result_path = original.save_json(path=save_path)

        assert result_path.exists()
        assert result_path == Path(save_path)

        restored = PipelineConfig.load_json(save_path)
        assert restored.threshold_tau == 3.0
        assert restored.weight_volume == 0.8
        assert restored.weight_sentiment == 0.2
        assert restored.random_seed == 999

    def test_save_creates_parent_directories(self, tmp_path):
        config = PipelineConfig()
        nested_path = str(tmp_path / "deep" / "nested" / "config.json")
        result_path = config.save_json(path=nested_path)
        assert result_path.exists()

    def test_save_default_path_uses_output_dir(self, tmp_path):
        config = PipelineConfig(output_dir=str(tmp_path))
        result_path = config.save_json()
        assert result_path.parent == tmp_path
        assert result_path.name == "pipeline_config.json"


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_from_json_with_unknown_fields_raises(self):
        """Unknown fields should cause a TypeError from the dataclass constructor."""
        json_str = json.dumps({"threshold_tau": 1.0, "unknown_field": "value"})
        with pytest.raises(TypeError):
            PipelineConfig.from_json(json_str)

    def test_from_json_with_partial_fields(self):
        """Partial JSON should fill remaining fields with defaults."""
        json_str = json.dumps({"threshold_tau": 99.0})
        config = PipelineConfig.from_json(json_str)
        assert config.threshold_tau == 99.0
        assert config.random_seed == 42  # default

    def test_empty_thresholds_list(self):
        config = PipelineConfig(thresholds=[])
        restored = PipelineConfig.from_json(config.to_json())
        assert restored.thresholds == []

    def test_project_root_is_valid_directory(self):
        """PROJECT_ROOT should point to the actual project root."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()
