"""Command-line helpers for analytics workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .auto_train import AutoTrainThresholds, run_auto_train
from .config import TelemetryConfig
from .inference import load_trade_outcome_model
from .model_training import train_trade_classifier
from .telemetry_ingest import ingest_telemetry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ultima analytics utilities")
    parser.add_argument(
        "command",
        choices=["summarize", "export", "train", "auto-train", "model-status"],
        help="Action to perform",
    )
    parser.add_argument("--telemetry", type=Path, default=None, help="Path to telemetry JSONL file")
    parser.add_argument("--output", type=Path, default=Path("analytics_output"), help="Output directory or file")
    parser.add_argument("--dataset", default="guard_snapshots", help="Dataset key to export")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output file format for export")
    parser.add_argument("--model-output", type=Path, default=Path("analytics_output/trade_classifier.joblib"), help="Output path for trained classifier artifact")
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation split size for classifier training")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for classifier training")
    parser.add_argument("--min-rows", type=int, default=50, help="Minimum closed trades required for auto-train")
    parser.add_argument("--min-class-ratio", type=float, default=0.1, help="Minimum win/loss ratio required for auto-train")
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Maximum age (hours) for trades considered recent during auto-train checks; use 0 to disable",
    )
    parser.add_argument(
        "--require-recent-rows",
        type=int,
        default=10,
        help="Minimum number of trades inside freshness window for auto-train",
    )
    parser.add_argument("--archive-dir", type=Path, default=None, help="Directory to archive previous classifier artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate auto-train thresholds without training")

    args = parser.parse_args(argv)
    config = TelemetryConfig.from_env()
    if args.telemetry:
        config = TelemetryConfig(telemetry_path=args.telemetry, output_path=config.output_path, output_format=args.format)

    if args.command == "summarize":
        datasets = ingest_telemetry(config)
        for name, df in datasets.items():
            print(f"Dataset: {name}")
            print(df.head())
            print(f"Rows: {len(df)}\n")
        return 0

    if args.command == "export":
        datasets = ingest_telemetry(config)
        dataset = datasets.get(args.dataset)
        if dataset is None:
            print(f"Dataset '{args.dataset}' not available. Choices: {', '.join(datasets)}")
            return 1
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "parquet":
            dataset.to_parquet(output_path, index=False)
        else:
            dataset.to_csv(output_path, index=False)
        print(f"Exported {args.dataset} -> {output_path}")
        return 0

    if args.command == "model-status":
        model_path = args.model_output
        if not model_path.exists():
            print(f"Model artifact not found at {model_path}")
            return 1
        try:
            model = load_trade_outcome_model(model_path)
        except Exception as exc:  # noqa: BLE001 - surface artifact issues directly to CLI users
            print(f"Failed to load model artifact: {exc}")
            return 1

        artifact = model.artifact or {}
        metrics = artifact.get("metrics", {}) if isinstance(artifact, dict) else {}
        symbols = artifact.get("symbols", []) if isinstance(artifact, dict) else []
        strategies = artifact.get("strategies", []) if isinstance(artifact, dict) else []

        print(f"Model path: {model_path}")
        print(f"  trained_at: {model.trained_at or 'unknown'}")
        print(f"  synthetic: {model.synthetic}")
        print(f"  training_rows: {model.training_rows}")
        print(f"  sklearn_version: {model.sklearn_version or 'unknown'}")
        print(f"  feature_columns: {len(model.feature_columns or [])}")
        preview = (model.feature_columns or [])[:10]
        if preview:
            print("  first_features:")
            for column in preview:
                print(f"    - {column}")

        print(f"  symbols: {len(symbols)}")
        if symbols:
            print(f"    sample: {', '.join(symbols[:5])}")
        print(f"  strategies: {len(strategies)}")
        if strategies:
            print(f"    sample: {', '.join(strategies[:5])}")

        if metrics:
            print("  metrics:")
            for key in sorted(metrics):
                value = metrics[key]
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

        cost_profiles = model.cost_profiles
        print(f"  cost_profiles: {len(cost_profiles)}")
        global_cost = model.global_cost_profile
        if global_cost:
            print("  global_cost_profile:")
            for key in sorted(global_cost):
                print(f"    {key}: {global_cost[key]}")

        return 0

    if args.command == "train":
        result = train_trade_classifier(
            config,
            output_path=args.model_output,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print("Classifier training complete:")
        for metric, value in result.metrics.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
        if result.trained_at is not None:
            print(f"  trained_at: {result.trained_at.isoformat(timespec='seconds')}")
        print(f"  synthetic: {result.synthetic}")
        print(f"  training_rows: {result.training_rows}")
        print(f"Model artifact saved to {result.model_path}")
        return 0

    if args.command == "auto-train":
        max_age = args.max_age_hours if args.max_age_hours > 0 else None
        thresholds = AutoTrainThresholds(
            min_rows=args.min_rows,
            min_class_ratio=args.min_class_ratio,
            max_age_hours=max_age,
            require_recent_rows=args.require_recent_rows,
        )
        result = run_auto_train(
            telemetry_path=config.telemetry_path,
            model_output=args.model_output,
            thresholds=thresholds,
            config=config,
            random_state=args.random_state,
            test_size=args.test_size,
            dry_run=args.dry_run,
            archive_dir=args.archive_dir,
        )

        print("Auto-train decision:")
        print(f"  should_train: {result.decision.should_train}")
        print(f"  reason: {result.decision.reason}")
        for key in sorted(result.decision.stats):
            print(f"  {key}: {result.decision.stats[key]}")
        if result.archived_model is not None:
            print(f"  archived_previous_model: {result.archived_model}")
        if result.training is not None:
            print("Training metrics:")
            for metric, value in result.training.metrics.items():
                print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
            if result.training.trained_at is not None:
                print(f"  trained_at: {result.training.trained_at.isoformat(timespec='seconds')}")
            print(f"  synthetic: {result.training.synthetic}")
            print(f"  training_rows: {result.training.training_rows}")
            print(f"Model artifact saved to {result.training.model_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
