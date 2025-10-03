"""Configuration helpers for analytics pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TelemetryConfig:
    """Runtime configuration for telemetry ingestion."""

    telemetry_path: Path
    output_path: Path
    output_format: str = "parquet"

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        base_dir = Path(os.getenv("ULTIMA_ANALYTICS_DIR", "analytics_data"))
        telemetry_file = Path(os.getenv("ULTIMA_TELEMETRY_FILE", "logs/telemetry_live.jsonl"))
        output_format = os.getenv("ULTIMA_ANALYTICS_FORMAT", "parquet")
        return cls(
            telemetry_path=telemetry_file,
            output_path=base_dir,
            output_format=output_format,
        )
