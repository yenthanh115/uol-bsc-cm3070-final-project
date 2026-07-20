"""Tests for the data loader and ticker extraction module.

Validates:
  1. Ticker extraction: dollar-sign pattern, uppercase pattern, stopword filtering
  2. Edge cases: empty strings, deleted/removed text, single-char words
  3. Multi-ticker deduplication and sorting
  4. load_data flow with synthetic data
  5. Timestamp parsing and chronological sorting
  6. Ticker explosion (one row per record-ticker pair)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surge_pipeline.config import PipelineConfig
from surge_pipeline.loader import (
    extract_tickers,
    load_data,
    generate_synthetic_data,
    TICKER_STOPWORDS,
)


# ============================================================================
# Ticker extraction — dollar-sign pattern
# ============================================================================


class TestDollarSignPattern:
    """Tests for $TICKER extraction."""

    def test_single_dollar_ticker(self):
        result = extract_tickers("Buy $AAPL now", "")
        assert "AAPL" in result

    def test_multiple_dollar_tickers(self):
        result = extract_tickers("$GME and $AMC to the moon", "")
        assert "GME" in result
        assert "AMC" in result

    def test_dollar_ticker_in_selftext(self):
        result = extract_tickers("Check this out", "I'm bullish on $TSLA")
        assert "TSLA" in result

    def test_dollar_ticker_max_length_5(self):
        result = extract_tickers("$ABCDE is valid", "")
        assert "ABCDE" in result

    def test_dollar_ticker_too_long_excluded(self):
        """Tickers longer than 5 chars should not match the regex."""
        result = extract_tickers("$ABCDEF is not a ticker", "")
        assert "ABCDEF" not in result

    def test_dollar_ticker_stopword_filtered(self):
        """Dollar-sign tickers that are stopwords should be excluded."""
        result = extract_tickers("$CEO said something", "")
        assert "CEO" not in result


# ============================================================================
# Ticker extraction — uppercase word pattern
# ============================================================================


class TestUppercaseWordPattern:
    """Tests for standalone uppercase word extraction."""

    def test_simple_uppercase_ticker(self):
        result = extract_tickers("I think PLTR is undervalued", "")
        assert "PLTR" in result

    def test_two_char_minimum(self):
        """Single uppercase letters should not be extracted."""
        result = extract_tickers("I bought A share", "")
        assert "A" not in result

    def test_five_char_maximum(self):
        result = extract_tickers("ABCDE is a ticker", "")
        assert "ABCDE" in result

    def test_six_char_word_excluded(self):
        """Words longer than 5 uppercase chars should not match."""
        result = extract_tickers("NASDAQ is not a ticker here", "")
        # NASDAQ is 6 chars so won't match the 2-5 char pattern
        assert "NASDAQ" not in result

    def test_common_english_words_filtered(self):
        """Common English words in uppercase should be filtered."""
        result = extract_tickers("THE BIG MOVE FOR THIS STOCK", "")
        assert "THE" not in result
        assert "BIG" not in result
        assert "FOR" not in result
        assert "THIS" not in result

    def test_reddit_slang_filtered(self):
        result = extract_tickers("YOLO into FOMO territory", "")
        assert "YOLO" not in result
        assert "FOMO" not in result

    def test_finance_abbreviations_filtered(self):
        result = extract_tickers("The CEO announced IPO", "")
        assert "CEO" not in result
        assert "IPO" not in result


# ============================================================================
# Ticker extraction — combined and edge cases
# ============================================================================


class TestTickerEdgeCases:
    """Edge cases for ticker extraction."""

    def test_empty_strings(self):
        result = extract_tickers("", "")
        assert result == []

    def test_no_tickers_found(self):
        result = extract_tickers("just a normal sentence with no tickers", "")
        assert result == []

    def test_deduplication_across_title_and_selftext(self):
        """Same ticker in title and selftext should appear once."""
        result = extract_tickers("$AAPL is great", "I love AAPL")
        assert result.count("AAPL") == 1

    def test_results_are_sorted(self):
        result = extract_tickers("$ZZZ and $AAA and $MMM", "")
        assert result == sorted(result)

    def test_mixed_case_not_extracted(self):
        """Mixed case words should not match the uppercase pattern."""
        result = extract_tickers("Apple is a good stock", "")
        # "Apple" is not all-uppercase, shouldn't match
        assert "APPLE" not in result
        assert "Apple" not in result

    def test_dollar_sign_overrides_stopword_check(self):
        """Dollar-sign tickers are still filtered against stopwords."""
        # Even with $ prefix, stopwords should be excluded
        result = extract_tickers("$THE stock", "")
        assert "THE" not in result

    def test_ticker_from_combined_text(self):
        """Tickers should be found in both title and selftext."""
        result = extract_tickers("Title with $NVDA", "Body mentions RIVN")
        assert "NVDA" in result
        assert "RIVN" in result


# ============================================================================
# Stopword coverage
# ============================================================================


class TestStopwords:
    """Verify stopword set properties."""

    def test_stopwords_are_uppercase(self):
        """All stopwords should be uppercase strings."""
        for word in TICKER_STOPWORDS:
            assert word == word.upper(), f"Stopword {word!r} is not uppercase"

    def test_stopwords_nonempty(self):
        assert len(TICKER_STOPWORDS) > 50

    def test_common_false_positives_in_stopwords(self):
        """Key false positives should be in the stopword set."""
        expected = {"DD", "CEO", "IPO", "ETF", "YOLO", "FOMO", "USD", "BTC"}
        assert expected.issubset(TICKER_STOPWORDS)


# ============================================================================
# Synthetic data generation
# ============================================================================


class TestSyntheticData:
    """Tests for generate_synthetic_data."""

    def test_returns_expected_columns(self):
        df = generate_synthetic_data(n_records=10, seed=42)
        expected_cols = {"id", "created_utc", "title", "selftext", "score",
                        "num_comments", "subreddit", "tickers"}
        assert expected_cols.issubset(set(df.columns))

    def test_correct_record_count(self):
        df = generate_synthetic_data(n_records=100, seed=42)
        assert len(df) == 100

    def test_deterministic_with_seed(self):
        df1 = generate_synthetic_data(n_records=50, seed=123)
        df2 = generate_synthetic_data(n_records=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_synthetic_data(n_records=50, seed=1)
        df2 = generate_synthetic_data(n_records=50, seed=2)
        # At minimum, the IDs should differ
        assert not df1["id"].equals(df2["id"])


# ============================================================================
# load_data integration (with synthetic fallback)
# ============================================================================


class TestLoadData:
    """Integration tests for load_data using synthetic data."""

    @pytest.fixture
    def config_no_file(self) -> PipelineConfig:
        """Config with no file_path → triggers synthetic data fallback."""
        return PipelineConfig(file_path="", random_seed=42)

    def test_returns_dataframe(self, config_no_file):
        df = load_data(config_no_file)
        assert isinstance(df, pd.DataFrame)

    def test_has_ticker_column(self, config_no_file):
        df = load_data(config_no_file)
        assert "ticker" in df.columns

    def test_has_created_utc_column(self, config_no_file):
        df = load_data(config_no_file)
        assert "created_utc" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["created_utc"])

    def test_sorted_chronologically(self, config_no_file):
        df = load_data(config_no_file)
        timestamps = df["created_utc"].values
        assert (timestamps[1:] >= timestamps[:-1]).all()

    def test_no_empty_tickers(self, config_no_file):
        df = load_data(config_no_file)
        assert (df["ticker"].str.len() > 0).all()

    def test_tickers_are_uppercase(self, config_no_file):
        df = load_data(config_no_file)
        assert (df["ticker"] == df["ticker"].str.upper()).all()

    def test_exploded_rows_more_than_input(self, config_no_file):
        """Multi-ticker records should produce more rows after explosion."""
        df = load_data(config_no_file)
        # Synthetic data has ~40% multi-ticker records, so exploded > 500 * 0.9
        # (minus ~10% with empty tickers)
        assert len(df) > 400

    def test_deterministic(self, config_no_file):
        df1 = load_data(config_no_file)
        df2 = load_data(config_no_file)
        pd.testing.assert_frame_equal(df1, df2)
