"""Scheduler-friendly wrapper for automated trade classifier retraining."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import subprocess

from analytics.auto_train import AutoTrainThresholds, run_auto_train
from analytics.config import TelemetryConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automate classifier retraining with guard rails",
    )
    parser.add_argument("--telemetry", type=Path, default=None, help="Path to telemetry JSONL file")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("analytics_output/trade_classifier.joblib"),
        help="Destination for the trained classifier artifact",
    )
    parser.add_argument("--archive-dir", type=Path, default=None, help="Folder to archive prior artifacts before overwrite")
    parser.add_argument("--min-rows", type=int, default=50, help="Minimum closed trades required for retraining")
    parser.add_argument(
        "--min-class-ratio",
        type=float,
        default=0.1,
        help="Minimum proportion of both wins and losses in the dataset",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Maximum age of trades (hours) considered fresh; use 0 to disable freshness check",
    )
    parser.add_argument(
        "--require-recent-rows",
        type=int,
        default=10,
        help="Minimum number of trades inside the freshness window",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation split ratio for classifier validation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible splits")
    parser.add_argument("--dry-run", action="store_true", help="Report decision without training")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/auto_train"),
        help="Directory for auto-train execution logs",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Explicit log file path (overrides --log-dir timestamped name)",
    )
    parser.add_argument(
        "--format-json",
        action="store_true",
        help="Emit JSON summary in addition to human-readable text",
    )
    parser.add_argument(
        "--post-command",
        type=str,
        default=None,
        help="Shell command to execute after successful training (runs only when a new model is produced)",
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> TelemetryConfig:
    base = TelemetryConfig.from_env()
    telemetry_path = args.telemetry or base.telemetry_path
    return TelemetryConfig(
        telemetry_path=telemetry_path,
        output_path=base.output_path,
        output_format=base.output_format,
    )


def _determine_log_path(args: argparse.Namespace) -> Path:
    if args.log_file:
        return args.log_file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = args.log_dir or Path("logs/auto_train")
    return directory / f"auto_train_{timestamp}.log"


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = _resolve_config(args)
    model_output: Path = args.model_output
    thresholds = AutoTrainThresholds(
        min_rows=args.min_rows,
        min_class_ratio=args.min_class_ratio,
        max_age_hours=(args.max_age_hours if args.max_age_hours > 0 else None),
        require_recent_rows=args.require_recent_rows,
    )

    log_path = _determine_log_path(args)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        result = run_auto_train(
            telemetry_path=config.telemetry_path,
            model_output=model_output,
            thresholds=thresholds,
            config=config,
            random_state=args.random_state,
            test_size=args.test_size,
            dry_run=args.dry_run,
            archive_dir=args.archive_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive guard for scheduled contexts
        lines = [
            f"[{timestamp}] ❌ auto-train failed",
            f"  error: {exc}",
        ]
        _write_lines(log_path, lines)
        for line in lines:
            print(line)
        return 1

    decision = result.decision
    stats = {key: decision.stats.get(key) for key in sorted(decision.stats)}
    lines = [
        f"[{timestamp}] ✔️ auto-train executed",
        f"  should_train: {decision.should_train}",
        f"  reason: {decision.reason}",
        f"  telemetry_path: {config.telemetry_path}",
        f"  model_output: {model_output}",
    ]
    if args.archive_dir:
        lines.append(f"  archive_dir: {args.archive_dir}")
    lines.append(f"  stats: {json.dumps(stats, default=str)}")

    if result.archived_model is not None:
        lines.append(f"  archived_model: {result.archived_model}")

    if result.training is not None:
        metrics = {key: float(value) if isinstance(value, (int, float)) else value for key, value in result.training.metrics.items()}
        trained_at = result.training.trained_at.isoformat(timespec="seconds") if result.training.trained_at else None
        lines.extend(
            [
                "  training:",
                f"    trained_at: {trained_at}",
                f"    training_rows: {result.training.training_rows}",
                f"    synthetic: {result.training.synthetic}",
                f"    sklearn_version: {result.training.sklearn_version}",
                f"    metrics: {json.dumps(metrics, default=str)}",
            ]
        )

    post_command_rc: int | None = None
    post_command_error: str | None = None
    if args.post_command and result.training is not None:
        lines.append(f"  post_command: {args.post_command}")
        try:
            completed = subprocess.run(args.post_command, shell=True, check=False)
            post_command_rc = int(completed.returncode)
            lines.append(f"  post_command_exit: {post_command_rc}")
        except Exception as exc:  # pragma: no cover - defensive guard for scheduler environments
            post_command_error = str(exc)
            lines.append(f"  post_command_error: {post_command_error}")

    _write_lines(log_path, lines)
    for line in lines:
        print(line)

    if args.format_json:
        payload = {
            "timestamp": timestamp,
            "log_path": str(log_path),
            "decision": {
                "should_train": decision.should_train,
                "reason": decision.reason,
                "stats": decision.stats,
            },
            "archived_model": str(result.archived_model) if result.archived_model else None,
            "training": None,
            "post_command": {
                "command": args.post_command,
                "exit_code": post_command_rc,
                "error": post_command_error,
            }
        }
        if result.training is not None:
            payload["training"] = {
                "trained_at": result.training.trained_at.isoformat(timespec="seconds") if result.training.trained_at else None,
                "training_rows": result.training.training_rows,
                "synthetic": result.training.synthetic,
                "sklearn_version": result.training.sklearn_version,
                "metrics": result.training.metrics,
            }
        print(json.dumps(payload, default=str))

    if post_command_error is not None:
        return 1
    if post_command_rc not in (None, 0):
        return post_command_rc

    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution entry point
    sys.exit(main())
