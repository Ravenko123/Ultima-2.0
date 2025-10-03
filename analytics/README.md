# Analytics Telemetry Quickstart

Ultima's AI roadmap starts with rich, well-organised telemetry. This package ships the
first set of tooling for turning raw JSONL event logs into clean analytics datasets that
can feed notebooks, dashboards, and model training pipelines.

## 📡 Telemetry sources

| Event key        | Emitted by                | Purpose |
|------------------|---------------------------|---------|
| `guard_snapshot` | Risk guard state machine  | Capture every global risk scan and guard factor toggle. |
| `micro_override` | Strategy micro overrides  | Explain why individual strategies were throttled or boosted. |
| `trade_closed`   | Live runner (MT5 bridge)  | Label completed positions with PnL, duration, and decoded strategy info. |
| `risk_preset_applied` | Live risk orchestrator | Snapshot the active risk preset, persona, and guard thresholds in effect when switches occur. |

All events are appended to a newline-delimited JSON file (default
`logs/telemetry_live.jsonl`). Each event always contains:

- `timestamp`: ISO-8601 string when the event was published
- `event`: One of the keys above
- `payload`: Event specific data schema (documented below)

## ⚙️ Configuration knobs

Environment variables give you full control over where data is read from and written to:

| Variable | Default | Description |
|----------|---------|-------------|
| `ULTIMA_TELEMETRY_FILE` | `logs/telemetry_live.jsonl` | Source JSONL file to ingest |
| `ULTIMA_ANALYTICS_DIR` | `analytics_data/` | Folder used for derived datasets |
| `ULTIMA_ANALYTICS_FORMAT` | `parquet` | Default export format (`parquet` \| `csv`) |

You can override these via shell or pass explicit paths when invoking the CLI (see below).

## 🗂️ Dataset layouts

### `guard_snapshots`

One row per global guard scan. Columns closely match the runtime payload but are cast to
numerics where possible so they are ready for aggregation:

- `scan`: Monotonic scan counter
- `symbol`: Symbol currently in focus (may be `None`)
- `timestamp`: ISO-8601 timestamp copied from the event envelope
- `guard_factor`, `soft_factor`, `margin_factor`, `equity_factor`, `var_factor`, `combined`
- `risk_allowed`: Boolean flag showing if trading is permitted
- `risk_status`, `soft_status`, `margin_status`, `var_status`, `combined_bucket`
- `daily_drawdown`, `weekly_drawdown`, `guard_pressure`, `guard_relief`

### `micro_overrides`

Rows explain any strategy-level guard or micro overrides. Useful for correlating
strategy throttling with telemetry and performance:

- `scan`, `symbol`, `signal`, `regime`, `kind`
- `guard_factor`, `drawdown_atr`, `momentum`, `threshold`
- `extra_*` columns created for optional floats such as `required_alignment` and
  `tolerance`

### `trade_closed`

Emitted by `live/demo_mt5.py` whenever a ticket is closed. The ingestion pipeline normalises
these payloads and adds a few training-friendly labels:

- Identity: `ticket`, `position_id`, `magic`
- Context: `symbol`, `direction`, `volume`, `strategy_key`, `strategy_code`
- Timing: `entry_time`, `exit_time`, `duration_seconds`, `holding_minutes`
- Prices: `price_open`, `price_close`, `price_diff_points`, `stop_loss`, `take_profit`
- PnL: `profit`, `net_result`, plus derived `outcome` (`win`, `loss`, `flat`)
- Risk metrics: `rr_ratio` (reward/risk using stop loss distance)
- Metadata: `comment`

When `risk_preset_applied` and `guard_snapshot` telemetry is present, the ingestion pipeline aligns
closed trades with the most recent risk context at the time of entry. This yields additional
columns such as `context_risk_preset`, `context_persona`, guard factors at entry, and the preset
settings (e.g. `risk_account_risk`, `risk_multiplier_max`). These values power richer features for
training and let you evaluate model performance per persona or risk band.

### `risk_preset_applied`

This dataset captures every risk/ persona change emitted by the live runner. Columns include:

- `timestamp`, `preset`, `persona`, `persona_label`, `changed`
- Numeric guard configuration at the time of the switch, e.g. `account_risk`, `risk_multiplier_max`,
  `soft_guard_limit`, `margin_usage_block`, low-volatility thresholds, and micro guard clamps
- Boolean flags for `dynamic_var_enabled` and `equity_governor_enabled`

Use it to audit preset transitions, feed dashboards, or augment training data. The feature
engineering utilities automatically merge these values into every trade row.

Use this dataset to backfill supervised training labels, rank strategy health, and visualise R:R distributions across guard regimes.

## 🛠️ CLI usage

Invoke the helper with Python to summarize or export datasets:

```powershell
# Inspect the first few rows of each dataset
python -m analytics.cli summarize --telemetry logs/telemetry_live.jsonl

# Export guard snapshots to Parquet
python -m analytics.cli export --dataset guard_snapshots --telemetry logs/telemetry_live.jsonl --output analytics_output/guard_snapshots.parquet

# Export micro overrides to CSV
python -m analytics.cli export --dataset micro_overrides --telemetry logs/telemetry_live.jsonl --output analytics_output/micro_overrides.csv --format csv

# Export closed trades with derived labels
python -m analytics.cli export --dataset trade_closed --telemetry logs/telemetry_live.jsonl --output analytics_output/trade_closed.parquet

# Train (or refresh) the trade outcome classifier artifact
python -m analytics.cli train --telemetry logs/telemetry_live.jsonl --model-output analytics_output/trade_classifier.joblib
```

The CLI respects your environment variables, so you can skip explicit flags once defaults
are set.

### 🚀 Deploying the ML confidence model

1. Collect live or backtest telemetry that includes `trade_closed` events. Use the export
  commands above if you need to inspect the dataset first.
2. Train the classifier:

  ```powershell
  python -m analytics.cli train --telemetry logs/telemetry_live.jsonl --model-output analytics_output/trade_classifier.joblib
  ```

  - If no closed-trade telemetry exists yet, the training command falls back to a small
    synthetic dataset and tags the artifact accordingly. The live runner will log a one-time
    reminder to replace it once real trades are available.
    - The saved artifact now records `trained_at` (UTC timestamp), `training_rows`, and the
      `sklearn_version` used to fit the classifier so you can track provenance across refreshes.
    - Feature names are preserved end-to-end, removing the sklearn "X does not have valid feature
      names" warning during inference.
    - Context-aware features (risk preset, persona, guard factors) are engineered automatically
      and weighted during training so scarce personas or presets still influence the classifier.
      The artifact stores the context weight map and aggregated per-context metrics for quick
      drift analysis.
3. Point the live runner to the artifact (default path matches the command above):

  ```powershell
  setx ULTIMA_ML_MODEL_PATH "analytics_output\trade_classifier.joblib"
  ```

  Or set the environment variable in your process manager / `.env` file.
4. Restart `live/demo_mt5.py`. On startup you should see `🤖 Loaded ML confidence model...`
  and, once real telemetry is used, no synthetic warning.

### 🔁 Automating retraining

Once you have a steady stream of real `trade_closed` samples you can let the tooling guard and
refresh the classifier on a schedule instead of kicking it off manually.

```powershell
python -m analytics.cli auto-train --telemetry logs/telemetry_live.jsonl --model-output analytics_output/trade_classifier.joblib --archive-dir analytics_output/archive --min-rows 100 --min-class-ratio 0.15 --require-recent-rows 20 --max-age-hours 48
```

For environments where you prefer a standalone entry point (e.g. Windows Task Scheduler), use the helper script:

```powershell
python scripts/run_auto_train.py --telemetry logs/telemetry_live.jsonl --model-output analytics_output/trade_classifier.joblib --archive-dir analytics_output/archive --min-rows 100 --min-class-ratio 0.15 --require-recent-rows 20 --max-age-hours 48 --format-json
```

The script mirrors the CLI arguments, writes a timestamped log under `logs/auto_train/` by default, and can emit a JSON summary for external monitoring.

Need to restart the live bot once a fresh model is produced? Add `--post-command` to run a shell command right after a successful training cycle. Example:

```powershell
python scripts/run_auto_train.py --telemetry logs/telemetry_live.jsonl --model-output analytics_output/trade_classifier.joblib --post-command "powershell.exe -File scripts\restart_live_bot.ps1"
```

The post command is skipped when thresholds fail or when running with `--dry-run`, and the script's exit code mirrors the command's status if it runs.

#### ♻️ `scripts/restart_live_bot.ps1`

Pair the post-command hook with the PowerShell helper to safely recycle the live runner after a
successful retrain. The script will:

- Resolve the repo's virtualenv Python (`.venv\Scripts\python.exe`) and the `live/demo_mt5.py`
  entry point (override with `-PythonPath` or `-LiveScript` if your layout differs).
- Stop any existing Python process that is currently running the live script (skip this behaviour
  with `-NoKill`).
- Launch a fresh instance of the live bot from the repo root so relative paths still work.

Usage examples:

```powershell
# Standard restart
powershell.exe -File scripts\restart_live_bot.ps1

# Skip terminating an existing session (useful when only the scheduler should own restarts)
powershell.exe -File scripts\restart_live_bot.ps1 -NoKill

# Custom interpreter location (e.g., system Python) and enable verbose logging
powershell.exe -File scripts\restart_live_bot.ps1 -PythonPath "C:\Python312\python.exe" -VerboseLogging
```

When wiring this into Task Scheduler, make sure the task's "Start in" directory is the repository
root. That keeps the PowerShell helper's default paths valid and ensures MT5 credentials or `.env`
files are resolved the same way as manual runs.

### 🔍 Inspecting classifier artifacts

Need to confirm what the most recent model contains without cracking open the joblib manually?
Use the CLI's `model-status` command to print metadata, metrics, and feature snapshots:

```powershell
python -m analytics.cli model-status --model-output analytics_output/trade_classifier.joblib
```

The command reports:

- `trained_at`, `training_rows`, and whether the artifact was built from synthetic fallback data.
- The scikit-learn version used, feature count, and the first few encoded column names (handy for
  spotting unexpected categorical expansions).
- A quick sample of symbols/strategies embedded in the model, the archived training metrics, and
  any cost profiles saved during training.

If the artifact path is missing or unreadable you'll get a non-zero exit code alongside an error
message, making it easy to plug into monitoring scripts.

📎 Need the essential commands in one place? Grab the quick reference at
[`docs/automation_cheatsheet.md`](../docs/automation_cheatsheet.md).

Key behaviour:

- **Safety checks first.** The command inspects the telemetry file and only trains when the
  thresholds are satisfied:
  - `--min-rows`: minimum number of closed trades (default 50)
  - `--min-class-ratio`: minimum share of both wins and losses (default 0.10 each)
  - `--max-age-hours`: freshness window for trades (set to `0` to disable)
  - `--require-recent-rows`: minimum number of trades inside the freshness window
- **Archiving baked in.** Provide `--archive-dir` to automatically copy the previous artifact before
  overwriting it. Useful for instant rollbacks or offline analysis.
- **Dry runs.** Add `--dry-run` to see the decision and stats (row counts, class balance, recent
  trades) without producing a new model.

💡 **Scheduling tip:** Create a Windows Task Scheduler job that runs the command daily or weekly. Set
the "Start in" directory to the repo root so relative paths resolve, and capture stdout/stderr to a
log file for auditing. When using `scripts/run_auto_train.py`, point the task directly at the script
and optionally pass a fixed `--log-file` so every invocation updates the same audit trail. If you
have a process manager, re-run `live/demo_mt5.py` (or send it a reload signal) after the task
finishes so the fresh artifact is picked up.

## 🔮 Next steps

- Add lightweight notebooks that demonstrate exploratory data analysis
- Wire the CLI into scheduled jobs or backtesting runs for hands-free dataset refreshes
- Train the first supervised trade-outcome classifier using the enriched trade dataset
- Capture real trading telemetry regularly so refreshed artifacts keep metadata (timestamp,
  training rows, sklearn version) current

Feedback and pull requests welcome—this analytics layer is meant to evolve alongside the
trading agents it supports.
