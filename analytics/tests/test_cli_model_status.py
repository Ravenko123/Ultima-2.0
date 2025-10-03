"""Tests for the analytics CLI model-status command."""

from __future__ import annotations

from analytics import cli
from analytics.model_training import train_trade_classifier


def test_model_status_reports_metadata(tmp_path, capsys):
    artifact_path = tmp_path / "trade_classifier.joblib"

    # Train using the synthetic fallback to produce a valid artifact quickly.
    train_trade_classifier(output_path=artifact_path)

    exit_code = cli.main(["model-status", "--model-output", str(artifact_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Model path: {artifact_path}" in captured.out
    assert "trained_at:" in captured.out
    assert "synthetic:" in captured.out
    assert "feature_columns:" in captured.out
    assert "metrics:" in captured.out


def test_model_status_missing_artifact(tmp_path, capsys):
    artifact_path = tmp_path / "missing.joblib"
    exit_code = cli.main(["model-status", "--model-output", str(artifact_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"Model artifact not found at {artifact_path}" in captured.out
