# Ultima Themed Personas

Ultima now ships with three curated personas that bundle a risk preset, strategy mix, and regime bias into distinct "bot flavors." Each persona flavors the same execution engine differently so you can pivot between high-octane momentum and disciplined capital preservation without editing code.

| Persona | Theme | Underlying risk preset | Personality |
| --- | --- | --- | --- |
| 🔥 **Firestorm Aggressor** (`fire`) | Breakout-driven trend rider | `high` | Leans into momentum and alpha surges, stacks positions quickly, and relaxes guardrails for explosive moves. |
| 🌍 **Terra Guardian** (`earth`) | Balanced growth | `medium` | Adaptive everyday driver that balances trend, range, and scalp systems with tempered guardrails. |
| ❄️ **Frost Sentinel** (`ice`) | Capital preservation | `low` | Slow, disciplined fade specialist that prioritizes mean reversion, tight sizing, and low-volatility harvesting. |

## Selecting a persona at launch

You can choose a persona either from the command line or via environment variables. When no explicit selection is provided, the bot defaults to **Terra Guardian (`earth`)**.

### Command-line flag

```powershell
python live/demo_mt5.py --persona fire
```

Aliases `--persona` and `-p` both work, and the value is case-insensitive. Running with the flag leaves the on-disk risk profile untouched, so it is ideal for ad-hoc sessions.

### Environment variables

Set one of the recognised variables before launching:

```powershell
$env:ULTIMA_PERSONA = "ice"
python live/demo_mt5.py
```

Supported variable names:

- `ULTIMA_PERSONA`
- `ULTIMA_PROFILE`
- `ULTIMA_THEME`

Environment settings are handy when the bot is started via scripts or schedulers.

## Persona behaviour notes

- Persona activation automatically applies the matching risk preset and retunes strategy weights. Any manual `/risk` change from Telegram clears the persona lock and reverts to raw presets.
- Strategy diagnostics now report **persona-gated** skips so you can see how often a strategy is filtered by the active persona.
- Telemetry events include the persona label, letting you correlate live guard behaviour with the themed profile.

## Persisted persona

When a persona is applied and persisted (for example by saving the risk profile state), Ultima records the persona in `risk_profile_state.json`. Subsequent runs reuse that persona unless you override it from the CLI or environment. Clearing the persona—either by selecting another persona or by issuing a manual `/risk` change—reverts to baseline behaviour.

Feel free to extend `PERSONA_PROFILES` in `live/demo_mt5.py` with your own themed variants; the builder handles priority, SL/TP, mode gating, and persona-locked strategies out of the box.
