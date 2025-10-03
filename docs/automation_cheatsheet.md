# Automation Command Cheat Sheet

Keep these commands handy when operating Ultima's trading and training pipeline on Windows PowerShell.

## Live runner

```powershell
# Start the live bot with the repo's virtual environment
& .venv\Scripts\python.exe live\demo_mt5.py
```

### Switch personas on launch

```powershell
# Firestorm aggressor
python live\demo_mt5.py --persona fire

# Ice preservation mode via environment
$env:ULTIMA_PERSONA = "ice"; python live\demo_mt5.py
```

See `docs/persona_profiles.md` for the full persona matrix and behaviour notes.

## Manual training refresh

```powershell
# Rebuild the classifier immediately using the latest telemetry
python -m analytics.cli train --telemetry logs/telemetry_live.jsonl --model-output analytics_output/trade_classifier.joblib
```

## Guarded auto-train with thresholds

```powershell
# Only trains when thresholds pass; archives the previous model first
python -m analytics.cli auto-train --telemetry logs/telemetry_live.jsonl --model-output analytics_output/trade_classifier.joblib --archive-dir analytics_output/archive --min-rows 100 --min-class-ratio 0.15 --require-recent-rows 20 --max-age-hours 48
```

Add `--dry-run` to preview the decision without writing a new model.

## Post-train restart helper

```powershell
# Stop any running live bot and relaunch it with the updated artifact
powershell.exe -File scripts\restart_live_bot.ps1
```

### Customisations

- Skip terminating existing sessions: add `-NoKill`.
- Use a specific interpreter: `-PythonPath "C:\\Python312\\python.exe"`.
- Enable verbose logging: append `-VerboseLogging`.

## Verify the latest artifact

```powershell
# Print metadata, metrics, and feature preview for the current model
python -m analytics.cli model-status --model-output analytics_output\trade_classifier.joblib
```

Use this after training (or on a schedule) to confirm the new classifier looks healthy before live trading.
